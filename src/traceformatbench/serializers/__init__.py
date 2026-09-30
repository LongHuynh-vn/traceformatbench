"""Serialize trace windows in the registered F0-F5 formats."""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from traceformatbench.data.trace import Trace
from traceformatbench.serializers.structured import (
    serialize_f2,
    serialize_f3,
    serialize_f4,
    serialize_f5,
)


def serialize_f0(window: Sequence[Trace]) -> str:
    """Rebuild all RCAEval columns in their original order, retaining nulls."""
    if not window:
        raise ValueError("Cannot serialize an empty window.")

    records = [
        {
            "time": datetime.fromtimestamp(span.start_us // 1_000_000, UTC).strftime("%H:%M"),
            "traceID": span.trace_id,
            "spanID": span.span_id,
            "serviceName": span.service,
            "methodName": span.method,
            "operationName": span.operation,
            "parentSpanID": span.parent_id,
            "startTimeMillis": span.start_us // 1000,
            "startTime": span.start_us,
            "duration": span.duration_us,
            "statusCode": span.status_code,
        }
        for trace in window
        for span in trace.spans
    ]
    return json.dumps(records, ensure_ascii=False)


def serialize_f1(window: Sequence[Trace]) -> str:
    """Group spans by trace, dropping derivable fields and null-valued keys."""
    if not window:
        raise ValueError("Cannot serialize an empty window.")

    records = []
    for trace in window:
        spans = []
        for span in trace.spans:
            record = {
                "spanID": span.span_id,
                "parentSpanID": span.parent_id,
                "serviceName": span.service,
                "operationName": span.operation,
                "methodName": span.method,
                "startTime": span.start_us,
                "duration": span.duration_us,
                "statusCode": span.status_code,
            }
            spans.append({key: value for key, value in record.items() if value is not None})
        records.append({"traceID": trace.trace_id, "spans": spans})
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


SERIALIZERS: dict[str, Callable[[Sequence[Trace]], str]] = {
    "F0": serialize_f0,
    "F1": serialize_f1,
    "F2": serialize_f2,
    "F3": serialize_f3,
    "F4": serialize_f4,
    "F5": serialize_f5,
}
