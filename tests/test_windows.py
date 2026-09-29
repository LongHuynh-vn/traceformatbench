import json
import random
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from traceformatbench.data.trace import Span, Trace
from traceformatbench.data.windows import (
    WindowConfig,
    build_case_windows,
    build_windows,
    eligible,
    load_window_config,
    phase_of,
    sample_windows,
    select_cases,
    trace_size_chars,
)
from traceformatbench.serializers import serialize_f0

INJECT_S = 1_000
GUARD_S = 60
ROOT_CAUSE = "checkoutservice"


def _trace(
    trace_id: str,
    start_us: int,
    services: tuple[str, ...] = ("frontend",),
    *,
    root: bool = True,
    orphan: bool = False,
) -> Trace:
    """One span per service, all children of the first span.

    root=False makes the first span a child of the last one (a cycle: no root, no orphan).
    orphan=True adds a span whose parent is not in the trace.
    """
    last = f"{trace_id}-{len(services) - 1}"
    spans = [
        Span(
            span_id=f"{trace_id}-{i}",
            parent_id=f"{trace_id}-0" if i else (None if root else last),
            trace_id=trace_id,
            service=service,
            operation="GET /",
            method=None,
            start_us=start_us + i,
            duration_us=10,
            status_code=None,
        )
        for i, service in enumerate(services)
    ]
    if orphan:
        spans.append(replace(spans[-1], span_id=f"{trace_id}-lost", parent_id="missing"))
    return Trace(trace_id, tuple(spans))


def _at(seconds: float) -> int:
    return int(seconds * 1_000_000)


def _config(**overrides) -> WindowConfig:
    values = dict(
        name="test_windows",
        seed=7,
        data_dir="unused",
        datasets=["RE2-OB"],
        repetitions=[1],
        window_size=1,
        max_trace_chars=100_000,
        guard_seconds=GUARD_S,
        t1_windows_per_phase=3,
        t2_windows=3,
        output_windows="unused",
        output_stats="unused",
    )
    values.update(overrides)
    return WindowConfig(**values)


def _meta(case: str = "re2ob_checkoutservice_cpu_1", **overrides) -> dict:
    return {
        "case": case,
        "dataset": "RE2-OB",
        "system": "ob",
        "fault": "cpu",
        "root_cause_service": ROOT_CAUSE,
        "inject_time": INJECT_S,
        "has_traces": True,
        "repetition": 1,
        **overrides,
    }


def _series(prefix: str, start_s: float, n: int, services=("frontend",)) -> list[Trace]:
    return [_trace(f"{prefix}{i}", _at(start_s + i), services) for i in range(n)]


# Phase assignment


@pytest.mark.parametrize(
    ("start_us", "expected"),
    [
        (_at(INJECT_S - GUARD_S) - 1, "pre"),
        (_at(INJECT_S - GUARD_S), None),
        (_at(INJECT_S), None),
        (_at(INJECT_S + GUARD_S) - 1, None),
        (_at(INJECT_S + GUARD_S), "post"),
        (_at(0), "pre"),
        (_at(10 * INJECT_S), "post"),
    ],
)
def test_phase_of_respects_guard_band_boundaries(start_us, expected):
    assert phase_of(_trace("t", start_us), INJECT_S, GUARD_S) == expected


def test_phase_of_with_zero_guard_splits_at_inject_time():
    assert phase_of(_trace("t", _at(INJECT_S) - 1), INJECT_S, 0) == "pre"
    assert phase_of(_trace("t", _at(INJECT_S)), INJECT_S, 0) == "post"


# Eligibility


def test_trace_size_is_f0_characters_of_the_trace_alone():
    trace = _trace("t", 0, ("frontend", "cart"))
    assert trace_size_chars(trace) == len(serialize_f0([trace]))


def test_eligible_keeps_trace_at_exactly_the_cap():
    trace = _trace("t", 0, ("frontend", "cart"))
    assert eligible(trace, trace_size_chars(trace))


def test_oversize_trace_is_not_eligible():
    trace = _trace("t", 0, ("frontend", "cart"))
    assert not eligible(trace, trace_size_chars(trace) - 1)


def test_trace_without_root_is_not_eligible():
    trace = _trace("t", 0, ("frontend", "cart"), root=False)
    assert trace.root is None and not trace.orphan_span_ids
    assert not eligible(trace, 100_000)


def test_trace_with_orphans_is_not_eligible():
    trace = _trace("t", 0, ("frontend", "cart"), orphan=True)
    assert trace.root is not None and trace.orphan_span_ids
    assert not eligible(trace, 100_000)


# sample_windows


def _ids(windows: list[list[Trace]]) -> list[list[str]]:
    return [[trace.trace_id for trace in window] for window in windows]


@pytest.mark.parametrize("seed", range(20))
def test_sample_windows_are_consecutive_non_overlapping_and_sized(seed):
    traces = _series("t", 0, 20)
    shuffled = random.Random(seed).sample(traces, len(traces))
    windows = sample_windows(shuffled, "pre", 5, 3, random.Random(seed))

    assert len(windows) == 3
    starts = []
    for window in windows:
        assert len(window) == 5
        assert len({trace.trace_id for trace in window}) == 5
        start = traces.index(window[0])
        assert window == traces[start : start + 5]
        starts.append(start)
    assert starts == sorted(starts)
    assert all(b - a >= 5 for a, b in pairwise(starts))


def test_sample_windows_can_fill_the_phase_exactly():
    traces = _series("t", 0, 6)
    assert _ids(sample_windows(traces, "post", 2, 3, random.Random(0))) == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4", "t5"],
    ]


def test_sample_windows_is_deterministic_for_a_seed():
    traces = _series("t", 0, 50)
    first = _ids(sample_windows(traces, "pre", 5, 3, random.Random("20260929:case")))
    second = _ids(sample_windows(traces, "pre", 5, 3, random.Random("20260929:case")))
    assert first == second


def test_sample_windows_start_at_random_positions():
    traces = _series("t", 0, 50)
    draws = {
        tuple(map(tuple, _ids(sample_windows(traces, "pre", 5, 1, random.Random(seed)))))
        for seed in range(20)
    }
    assert len(draws) > 1


def test_sample_windows_returns_nothing_when_phase_is_shorter_than_a_window():
    assert sample_windows(_series("t", 0, 4), "pre", 5, 3, random.Random(0)) == []
    assert sample_windows([], "post", 5, 3, random.Random(0)) == []


@pytest.mark.parametrize(("n_traces", "expected"), [(5, 1), (6, 2), (7, 3), (14, 3)])
def test_sample_windows_overlap_but_stay_distinct_when_phase_is_short(n_traces, expected):
    traces = _series("t", 0, n_traces)
    windows = sample_windows(traces, "post", 5, 3, random.Random(1))
    assert len(windows) == expected
    assert len({tuple(ids) for ids in _ids(windows)}) == expected
    for window in windows:
        assert len(window) == 5 and len({trace.trace_id for trace in window}) == 5


def test_sample_windows_rejects_unknown_phase_and_duplicate_traces():
    traces = _series("t", 0, 5)
    with pytest.raises(ValueError, match="phase"):
        sample_windows(traces, "during", 1, 1, random.Random(0))
    with pytest.raises(ValueError, match="Duplicate"):
        sample_windows(traces + traces[:1], "pre", 1, 1, random.Random(0))


# build_case_windows


def test_t1_is_balanced_when_post_has_fewer_windows():
    pre = _series("p", 0, 20)
    post = _series("q", INJECT_S + GUARD_S, 4)  # only 2 distinct windows of 3
    records, stats = build_case_windows(
        _meta(), pre + post, _config(window_size=3, t2_windows=0), random.Random(0)
    )
    t1 = [r for r in records if r["task"] == "T1"]
    assert [r["phase"] for r in t1].count("pre") == 2
    assert [r["phase"] for r in t1].count("post") == 2
    assert [r["label"] for r in t1] == ["pre", "pre", "post", "post"]
    assert stats["n_t1_windows"] == 4


def test_t1_is_empty_when_one_phase_has_no_window():
    pre = _series("p", 0, 2)  # fewer than window_size
    post = _series("q", INJECT_S + GUARD_S, 20)
    records, stats = build_case_windows(
        _meta(), pre + post, _config(window_size=3, t2_windows=2), random.Random(0)
    )
    assert [r["task"] for r in records] == ["T2", "T2"]
    assert stats["n_t1_windows"] == 0 and stats["n_t2_windows"] == 2


def test_informative_flag_marks_t2_windows_with_the_root_cause_service():
    start = INJECT_S + GUARD_S
    post = [
        _trace("a", _at(start), ("frontend", ROOT_CAUSE)),
        _trace("b", _at(start + 1)),
        _trace("c", _at(start + 2)),
        _trace("d", _at(start + 3)),
    ]
    records, stats = build_case_windows(
        _meta(), post, _config(window_size=2, t2_windows=2), random.Random(0)
    )
    assert [(r["trace_ids"], r["informative"]) for r in records if r["task"] == "T2"] == [
        (["a", "b"], True),
        (["c", "d"], False),
    ]
    assert all(r["label"] == ROOT_CAUSE for r in records if r["task"] == "T2")
    assert stats["n_t2_informative"] == 1


def test_t1_records_have_no_informative_flag():
    traces = _series("p", 0, 5) + _series("q", INJECT_S + GUARD_S, 5, ("frontend", ROOT_CAUSE))
    records, _ = build_case_windows(_meta(), traces, _config(), random.Random(0))
    assert all(r["informative"] is None for r in records if r["task"] == "T1")


def test_window_record_fields():
    traces = _series("p", 0, 3) + _series("q", INJECT_S + GUARD_S, 3)
    records, _ = build_case_windows(
        _meta(),
        traces,
        _config(window_size=3, t1_windows_per_phase=1, t2_windows=1),
        random.Random(0),
    )
    assert records[0] == {
        "case": "re2ob_checkoutservice_cpu_1",
        "dataset": "RE2-OB",
        "system": "ob",
        "fault": "cpu",
        "root_cause_service": ROOT_CAUSE,
        "task": "T1",
        "phase": "pre",
        "label": "pre",
        "window_index": 0,
        "trace_ids": ["p0", "p1", "p2"],
        "n_spans": 3,
        "f0_chars": len(serialize_f0(traces[:3])),
        "informative": None,
        "config_name": "test_windows",
    }
    assert [(r["task"], r["phase"], r["window_index"]) for r in records] == [
        ("T1", "pre", 0),
        ("T1", "post", 0),
        ("T2", "post", 0),
    ]


def test_stats_row_on_small_synthetic_case():
    pre, post = INJECT_S - GUARD_S - 100, INJECT_S + GUARD_S
    traces = [
        _trace("p0", _at(pre), ("frontend", "cart")),  # eligible
        _trace("p1", _at(pre + 1), ("frontend", "cart")),  # eligible
        _trace("p2", _at(pre + 2), ("frontend", "cart"), root=False),  # no root
        _trace("g0", _at(INJECT_S - GUARD_S), ("frontend", ROOT_CAUSE)),  # eligible, guard band
        _trace("q0", _at(post), ("frontend", ROOT_CAUSE)),  # eligible, via root cause
        _trace("q1", _at(post + 1), ("frontend", "cart")),  # eligible
        _trace("q2", _at(post + 2), ("frontend", ROOT_CAUSE, "cart")),  # oversize, via root cause
        _trace("q3", _at(post + 3), ("frontend", "cart"), orphan=True),  # orphan
        _trace("q4", _at(post + 4), ("frontend", ROOT_CAUSE)),  # eligible, via root cause
    ]
    cap = max(trace_size_chars(t) for t in traces if len(t.spans) == 2)
    assert trace_size_chars(traces[6]) > cap
    records, stats = build_case_windows(
        _meta(), traces, _config(max_trace_chars=cap), random.Random(0)
    )

    assert stats == {
        "case": "re2ob_checkoutservice_cpu_1",
        "dataset": "RE2-OB",
        "n_traces": 9,
        "n_eligible": 6,
        "share_eligible": 6 / 9,
        "n_eligible_pre": 2,
        "n_eligible_post": 3,
        "share_post_via_root_cause_all": 3 / 5,
        "share_post_via_root_cause_eligible": 2 / 3,
        "n_t1_windows": 4,  # pre yields 2 windows of 1, post 3: matched to 2 + 2
        "n_t2_windows": 3,
        "n_t2_informative": 2,
    }
    assert sorted(r["trace_ids"][0] for r in records if r["task"] == "T2") == ["q0", "q1", "q4"]
    assert not {"p2", "g0", "q2", "q3"} & {tid for r in records for tid in r["trace_ids"]}


def test_stats_shares_are_none_without_post_traces():
    _, stats = build_case_windows(_meta(), _series("p", 0, 3), _config(), random.Random(0))
    assert stats["share_post_via_root_cause_all"] is None
    assert stats["share_post_via_root_cause_eligible"] is None


def test_case_without_traces_raises_with_case_name():
    with pytest.raises(ValueError, match="re2ob_empty_1"):
        build_case_windows(_meta("re2ob_empty_1"), [], _config(), random.Random(0))


# Case selection, pipeline, and per-case RNG

CASES = ["re2ob_a_cpu_1", "re2ob_b_cpu_1", "re2ob_c_cpu_1"]


def _write_case(data_dir: Path, case: str, offset: int) -> None:
    rows = []
    starts = [_at(INJECT_S - 500 + i * 20 + offset) for i in range(20)]
    starts += [_at(INJECT_S + GUARD_S + i * 20 + offset) for i in range(20)]
    for i, start in enumerate(starts):
        services = ("frontend", ROOT_CAUSE) if i % 3 == 0 else ("frontend", "cart")
        for j, service in enumerate(services):
            rows.append((f"{case}-t{i}", f"s{j}", None if j == 0 else "s0", service, start + j))
    frame = pd.DataFrame(
        rows, columns=["traceID", "spanID", "parentSpanID", "serviceName", "startTime"]
    ).assign(operationName="GET /", methodName=None, duration=10, statusCode=None)
    frame["parentSpanID"] = frame["parentSpanID"].astype("string")
    frame["methodName"] = frame["methodName"].astype("string")
    frame["statusCode"] = frame["statusCode"].astype("Int64")
    case_dir = data_dir / case
    case_dir.mkdir(parents=True)
    frame.to_parquet(case_dir / "traces.parquet", index=False)
    (case_dir / "inject_time.txt").write_text(f"{INJECT_S}\n", encoding="utf-8")


def _index(cases: list[str]) -> pd.DataFrame:
    return pd.DataFrame([_meta(case) for case in cases])


@pytest.fixture
def no_network(monkeypatch):
    download = Mock(side_effect=AssertionError("tests must not download"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return download


@pytest.fixture
def data_dir(tmp_path: Path, no_network) -> Path:
    for offset, case in enumerate(CASES):
        _write_case(tmp_path, case, offset)
    return tmp_path


def _pipeline_config(data_dir: Path) -> WindowConfig:
    return _config(data_dir=str(data_dir), window_size=3, t1_windows_per_phase=2, t2_windows=2)


def test_build_windows_uses_local_cases_and_is_deterministic(data_dir, no_network):
    config = _pipeline_config(data_dir)
    first, first_stats = build_windows(config, _index(list(reversed(CASES))))
    second, second_stats = build_windows(config, _index(CASES))

    no_network.assert_not_called()
    assert [json.dumps(r) for r in first] == [json.dumps(r) for r in second]
    pd.testing.assert_frame_equal(first_stats, second_stats)
    assert list(first_stats["case"]) == CASES
    assert len(first) == len(CASES) * (2 * 2 + 2)


def test_removing_a_case_leaves_other_cases_windows_unchanged(data_dir):
    config = _pipeline_config(data_dir)
    all_records, _ = build_windows(config, _index(CASES))
    fewer_records, _ = build_windows(config, _index([CASES[0], CASES[2]]))

    assert fewer_records == [r for r in all_records if r["case"] != CASES[1]]


def test_missing_download_raises_with_case_name(tmp_path, no_network):
    no_network.side_effect = lambda **kwargs: None  # "downloads" nothing
    with pytest.raises(FileNotFoundError, match="re2ob_missing_1"):
        build_windows(_config(data_dir=str(tmp_path)), _index(["re2ob_missing_1"]))


def test_select_cases_filters_traces_datasets_and_repetitions():
    index = pd.DataFrame(
        [
            _meta("re2ob_b_1"),
            _meta("re2ob_a_1"),
            _meta("re2ob_a_2", repetition=2),
            _meta("re2tt_a_1", dataset="RE2-TT"),
            _meta("re1ob_a_1", dataset="RE1-OB", has_traces=False),
            _meta("re2ob_untraced_1", has_traces=False),
        ]
    )
    config = _config(datasets=["RE2-OB", "RE1-OB"], repetitions=[1])
    assert list(select_cases(index, config)["case"]) == ["re2ob_a_1", "re2ob_b_1"]
    all_reps = select_cases(index, _config(datasets=["RE2-OB"], repetitions=None))
    assert list(all_reps["case"]) == ["re2ob_a_1", "re2ob_a_2", "re2ob_b_1"]


def test_committed_config_loads_with_its_file_name():
    config = load_window_config(Path(__file__).parents[1] / "configs" / "windows_v1.yaml")
    assert config.name == "windows_v1"
    assert config.repetitions == [1]
    assert config.window_size == 5 and config.max_trace_chars == 10_000
