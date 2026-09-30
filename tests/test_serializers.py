import json
from pathlib import Path

import pandas as pd
import pytest

from traceformatbench.data.trace import Trace, load_case_traces
from traceformatbench.serializers import SERIALIZERS, serialize_f0, serialize_f1


@pytest.fixture
def original_rows() -> list[dict[str, object]]:
    columns = [
        "time",
        "traceID",
        "spanID",
        "serviceName",
        "methodName",
        "operationName",
        "parentSpanID",
        "startTimeMillis",
        "startTime",
        "duration",
        "statusCode",
    ]
    # Shuffled rows straddle UTC minute and hour boundaries on 2024-01-01.
    rows = [
        (
            "01:00",
            "t2",
            "child2",
            "search",
            "POST",
            "搜索 🔎",
            "root2",
            1_704_070_800_000,
            1_704_070_800_000_001,
            50,
            2,
        ),
        (
            "00:01",
            "t1",
            "child1",
            "database",
            None,
            "query",
            "root1",
            1_704_067_260_000,
            1_704_067_260_000_001,
            30,
            None,
        ),
        (
            "00:59",
            "t2",
            "root2",
            "worker",
            None,
            "job",
            None,
            1_704_070_799_999,
            1_704_070_799_999_999,
            100,
            None,
        ),
        (
            "00:00",
            "t1",
            "root1",
            "frontend",
            "GET",
            "request",
            None,
            1_704_067_259_999,
            1_704_067_259_999_999,
            1200,
            0,
        ),
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


@pytest.fixture
def case_dir(tmp_path: Path, original_rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(original_rows).astype(
        {"parentSpanID": "string", "methodName": "string", "statusCode": "Int64"}
    )
    frame.to_parquet(tmp_path / "traces.parquet", index=False)
    return tmp_path


@pytest.fixture
def window(case_dir: Path) -> tuple[Trace, ...]:
    return tuple(load_case_traces(case_dir))


def test_f0_round_trip_preserves_original_columns_values_and_encoding(
    case_dir: Path, original_rows: list[dict[str, object]]
):
    window = load_case_traces(case_dir)
    rows_by_id = {(row["traceID"], row["spanID"]): row for row in original_rows}
    expected = [
        rows_by_id[(trace.trace_id, span.span_id)] for trace in window for span in trace.spans
    ]

    serialized = serialize_f0(window)
    actual = json.loads(serialized)

    assert actual == expected
    assert [list(row) for row in actual] == [list(row) for row in expected]
    assert serialized == json.dumps(expected, ensure_ascii=False)


def test_f1_golden_string(window: tuple[Trace, ...]):
    expected = (
        '[{"traceID":"t1","spans":['
        '{"spanID":"root1","serviceName":"frontend","operationName":"request",'
        '"methodName":"GET","startTime":1704067259999999,"duration":1200,"statusCode":0},'
        '{"spanID":"child1","parentSpanID":"root1","serviceName":"database",'
        '"operationName":"query","startTime":1704067260000001,"duration":30}]},'
        '{"traceID":"t2","spans":['
        '{"spanID":"root2","serviceName":"worker","operationName":"job",'
        '"startTime":1704070799999999,"duration":100},'
        '{"spanID":"child2","parentSpanID":"root2","serviceName":"search",'
        '"operationName":"搜索 🔎","methodName":"POST","startTime":1704070800000001,'
        '"duration":50,"statusCode":2}]}]'
    )

    assert serialize_f1(window) == expected


def test_f1_omits_null_keys_and_keeps_zero_status(window: tuple[Trace, ...]):
    traces = json.loads(serialize_f1(window))
    first_root, first_child = traces[0]["spans"]
    second_root = traces[1]["spans"][0]

    assert "parentSpanID" not in first_root
    assert "parentSpanID" not in second_root
    assert "methodName" not in first_child
    assert "statusCode" not in first_child
    assert "methodName" not in second_root
    assert "statusCode" not in second_root
    assert first_root["statusCode"] == 0
    assert all(
        value is not None for trace in traces for span in trace["spans"] for value in span.values()
    )


@pytest.mark.parametrize("serializer", [serialize_f0, serialize_f1], ids=["F0", "F1"])
def test_determinism(serializer, window: tuple[Trace, ...]):
    assert serializer(window).encode("utf-8") == serializer(window).encode("utf-8")


@pytest.mark.parametrize("serializer", [serialize_f0, serialize_f1], ids=["F0", "F1"])
def test_window_order_is_respected(serializer, window: tuple[Trace, ...]):
    original = json.loads(serializer(window))
    reversed_window = json.loads(serializer(window[::-1]))
    expected = original[2:] + original[:2] if serializer is serialize_f0 else original[::-1]

    assert reversed_window == expected


@pytest.mark.parametrize("serializer", [serialize_f0, serialize_f1], ids=["F0", "F1"])
@pytest.mark.parametrize("window", [[], ()], ids=["list", "tuple"])
def test_empty_window_raises_value_error(serializer, window):
    with pytest.raises(ValueError):
        serializer(window)


@pytest.mark.parametrize("serializer", [serialize_f0, serialize_f1], ids=["F0", "F1"])
def test_trace_without_root_is_serialized_without_special_marking(
    serializer, window: tuple[Trace, ...]
):
    rootless_trace = Trace(window[1].trace_id, window[1].spans[1:])
    assert rootless_trace.root is None
    original = json.loads(serializer(window))
    expected = (
        [original[-1]]
        if serializer is serialize_f0
        else [{"traceID": "t2", "spans": [original[1]["spans"][1]]}]
    )

    assert json.loads(serializer([rootless_trace])) == expected


@pytest.mark.parametrize("serializer", [serialize_f0, serialize_f1], ids=["F0", "F1"])
def test_unicode_operation_is_not_escaped(serializer, window: tuple[Trace, ...]):
    serialized = serializer(window)

    assert "搜索 🔎" in serialized
    assert "\\u" not in serialized


def test_serializers_registry_preserves_f0_and_f1():
    assert SERIALIZERS["F0"] is serialize_f0
    assert SERIALIZERS["F1"] is serialize_f1
