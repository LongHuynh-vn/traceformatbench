from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pandas as pd
import pytest

from traceformatbench.data.trace import Span, Trace, load_case_traces


def _synthetic_spans() -> pd.DataFrame:
    # Deliberately shuffled rows exercise trace and span ordering independently.
    rows = [
        ("t-normal", "c-child", "normal-root", "frontend", "request", None, 1020, 300, None),
        ("t-extra", "second-root", None, "backend", "second", None, 1005, 100, None),
        ("t-orphan", "lost-child", "normal-root", "worker", "lost", None, 500, 100, None),
        ("t-normal", "normal-root", None, "frontend", "request", "GET", 1000, 100, 0),
        ("t-extra", "first-root", "", "frontend", "first", None, 1000, 40, None),
        ("t-normal", "b-child", "normal-root", "backend", "query", "POST", 1020, 30, 2),
        ("t-orphan", "attached-child", "lost-child", "worker", "child", None, 510, 30, None),
        ("t-normal", "early-child", "normal-root", "backend", "query", "GET", 1010, 20, 0),
        ("t-extra", "extra-child", "first-root", "backend", "child", None, 1010, 10, None),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "traceID",
            "spanID",
            "parentSpanID",
            "serviceName",
            "operationName",
            "methodName",
            "startTime",
            "duration",
            "statusCode",
        ],
    )
    frame["parentSpanID"] = frame["parentSpanID"].astype("string")
    frame["methodName"] = frame["methodName"].astype("string")
    frame["statusCode"] = frame["statusCode"].astype("Int64")
    return frame


def _write_case(tmp_path: Path, spans: pd.DataFrame) -> Path:
    case_dir = tmp_path / "synthetic_case"
    case_dir.mkdir()
    spans.to_parquet(case_dir / "traces.parquet", index=False)
    (case_dir / "inject_time.txt").write_text("1\n", encoding="utf-8")
    return case_dir


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    return _write_case(tmp_path, _synthetic_spans())


def test_load_case_traces_assembles_normal_trace(case_dir: Path):
    traces = {trace.trace_id: trace for trace in load_case_traces(case_dir)}
    trace = traces["t-normal"]

    assert isinstance(trace, Trace)
    assert all(isinstance(span, Span) for span in trace.spans)
    assert [span.span_id for span in trace.spans] == [
        "normal-root",
        "early-child",
        "b-child",
        "c-child",
    ]
    assert trace.root is not None
    assert trace.root.span_id == "normal-root"
    assert trace.root.parent_id is None
    assert trace.root.trace_id == "t-normal"
    assert trace.root.service == "frontend"
    assert trace.root.operation == "request"
    assert trace.root.method == "GET"
    assert trace.root.start_us == 1000
    assert trace.root.duration_us == 100
    assert trace.root.status_code == 0
    assert type(trace.root.start_us) is int
    assert type(trace.root.duration_us) is int
    assert type(trace.root.status_code) is int
    assert trace.start_us == 1000
    assert trace.end_us == 1320
    assert list(trace.services) == ["backend", "frontend"]
    assert list(trace.orphan_span_ids) == []
    assert list(trace.extra_root_ids) == []


def test_children_are_lists_in_start_and_span_id_order(case_dir: Path):
    trace = next(trace for trace in load_case_traces(case_dir) if trace.trace_id == "t-normal")

    children = trace.children("normal-root")

    assert isinstance(children, list)
    assert [span.span_id for span in children] == ["early-child", "b-child", "c-child"]
    assert all(span.parent_id == "normal-root" for span in children)
    assert trace.children("c-child") == []


def test_mutating_children_list_does_not_change_trace(case_dir: Path):
    trace = next(trace for trace in load_case_traces(case_dir) if trace.trace_id == "t-normal")

    children = trace.children("normal-root")
    children.clear()

    assert [span.span_id for span in trace.children("normal-root")] == [
        "early-child",
        "b-child",
        "c-child",
    ]


def test_trace_without_root_is_preserved_with_orphans(case_dir: Path):
    trace = next(trace for trace in load_case_traces(case_dir) if trace.trace_id == "t-orphan")

    assert trace.root is None
    assert [span.span_id for span in trace.spans] == ["lost-child", "attached-child"]
    # A parent in another trace is still absent from this trace.
    assert list(trace.orphan_span_ids) == ["lost-child"]
    assert list(trace.extra_root_ids) == []
    assert [span.span_id for span in trace.children("lost-child")] == ["attached-child"]
    assert trace.start_us == 500
    assert trace.end_us == 600
    assert list(trace.services) == ["worker"]


def test_trace_with_two_roots_keeps_spans_and_records_extra_root(case_dir: Path):
    trace = next(trace for trace in load_case_traces(case_dir) if trace.trace_id == "t-extra")

    assert [span.span_id for span in trace.spans] == ["first-root", "second-root", "extra-child"]
    assert trace.root is not None
    assert trace.root.span_id == "first-root"
    assert trace.root.parent_id is None
    assert trace.spans[1].parent_id is None
    assert list(trace.extra_root_ids) == ["second-root"]
    assert list(trace.orphan_span_ids) == []
    assert trace.end_us == 1105


def test_load_case_traces_is_stable_and_breaks_trace_start_ties(case_dir: Path):
    first = load_case_traces(case_dir)
    second = load_case_traces(case_dir)

    assert isinstance(first, list)
    assert first == second
    assert [trace.trace_id for trace in first] == ["t-orphan", "t-extra", "t-normal"]


def test_missing_method_and_status_remain_none(case_dir: Path):
    traces = {trace.trace_id: trace for trace in load_case_traces(case_dir)}
    normal_spans = {span.span_id: span for span in traces["t-normal"].spans}

    assert normal_spans["c-child"].method is None
    assert normal_spans["c-child"].status_code is None
    assert normal_spans["b-child"].method == "POST"
    assert normal_spans["b-child"].status_code == 2
    assert type(normal_spans["b-child"].status_code) is int
    assert all(span.method is None for span in traces["t-orphan"].spans)
    assert all(span.status_code is None for span in traces["t-orphan"].spans)


@pytest.mark.parametrize(
    "missing", [None, float("nan"), pd.NA, ""], ids=["none", "nan", "pd-na", "empty-string"]
)
def test_missing_values_are_normalized(tmp_path: Path, missing):
    spans = _synthetic_spans().iloc[[3]].copy()
    spans["parentSpanID"] = pd.Series([missing], dtype="string", index=spans.index)
    spans["methodName"] = pd.Series([missing], dtype="string", index=spans.index)
    status = None if isinstance(missing, str) else missing
    spans["statusCode"] = pd.Series([status], dtype="Int64", index=spans.index)
    trace = load_case_traces(_write_case(tmp_path, spans))[0]

    assert trace.root is not None
    assert trace.root.parent_id is None
    assert trace.root.method is None
    assert trace.root.status_code is None


def test_span_and_trace_are_frozen_dataclasses(case_dir: Path):
    trace = load_case_traces(case_dir)[0]
    span = trace.spans[0]

    assert is_dataclass(trace)
    assert is_dataclass(span)
    with pytest.raises(FrozenInstanceError):
        trace.trace_id = "changed"
    with pytest.raises(FrozenInstanceError):
        span.service = "changed"


@pytest.mark.parametrize("csv_only", [False, True], ids=["absent", "csv-only"])
def test_missing_traces_parquet_error_names_case(tmp_path: Path, csv_only: bool):
    case_dir = tmp_path / "re2ob_checkoutservice_cpu_1"
    case_dir.mkdir()
    if csv_only:
        _synthetic_spans().to_csv(case_dir / "traces.csv", index=False)

    with pytest.raises(FileNotFoundError, match="re2ob_checkoutservice_cpu_1"):
        load_case_traces(case_dir)


def test_empty_parquet_returns_no_traces(tmp_path: Path):
    case_dir = _write_case(tmp_path, _synthetic_spans().iloc[:0])

    assert load_case_traces(case_dir) == []
