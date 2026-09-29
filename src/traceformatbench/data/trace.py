"""Assemble RCAEval span rows into deterministic, in-memory traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class Span:
    """One span, with timestamps and durations in microseconds."""

    span_id: str
    parent_id: str | None
    trace_id: str
    service: str
    operation: str
    method: str | None
    start_us: int
    duration_us: int
    status_code: int | None


@dataclass(frozen=True, slots=True)
class Trace:
    """Ordered spans and a parent index that stores each child once."""

    trace_id: str
    spans: tuple[Span, ...]
    root: Span | None = field(init=False)
    extra_root_ids: tuple[str, ...] = field(init=False)
    orphan_span_ids: tuple[str, ...] = field(init=False)
    start_us: int = field(init=False)
    end_us: int = field(init=False)
    services: tuple[str, ...] = field(init=False)
    _children: dict[str, tuple[Span, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        spans = tuple(sorted(self.spans, key=lambda span: (span.start_us, span.span_id)))
        span_ids = {span.span_id for span in spans}
        roots = [span for span in spans if span.parent_id is None]
        children: dict[str, list[Span]] = {}
        for span in spans:
            if span.parent_id is not None:
                children.setdefault(span.parent_id, []).append(span)

        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "root", roots[0] if roots else None)
        object.__setattr__(self, "extra_root_ids", tuple(span.span_id for span in roots[1:]))
        object.__setattr__(
            self,
            "orphan_span_ids",
            tuple(
                span.span_id
                for span in spans
                if span.parent_id is not None and span.parent_id not in span_ids
            ),
        )
        object.__setattr__(self, "start_us", spans[0].start_us)
        object.__setattr__(self, "end_us", max(span.start_us + span.duration_us for span in spans))
        object.__setattr__(self, "services", tuple(sorted({span.service for span in spans})))
        object.__setattr__(
            self, "_children", {parent: tuple(items) for parent, items in children.items()}
        )

    def children(self, span_id: str) -> list[Span]:
        """Return children in span order without exposing the internal index."""
        return list(self._children.get(span_id, ()))


def _optional_text(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def load_case_traces(case_dir: str | Path) -> list[Trace]:
    """Read traces.parquet and assemble traces sorted by start time and ID."""
    case_dir = Path(case_dir)
    path = case_dir / "traces.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Case {case_dir.name!r}: traces.parquet was not found in {case_dir}."
        )

    rows = pd.read_parquet(
        path,
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
    grouped: dict[str, list[Span]] = {}
    for row in rows.itertuples(index=False):
        span = Span(
            span_id=str(row.spanID),
            parent_id=_optional_text(row.parentSpanID) or None,
            trace_id=str(row.traceID),
            service=str(row.serviceName),
            operation=str(row.operationName),
            method=_optional_text(row.methodName) or None,
            start_us=int(row.startTime),
            duration_us=int(row.duration),
            status_code=None if pd.isna(row.statusCode) else int(row.statusCode),
        )
        grouped.setdefault(span.trace_id, []).append(span)

    traces = [Trace(trace_id, tuple(spans)) for trace_id, spans in grouped.items()]
    return sorted(traces, key=lambda trace: (trace.start_us, trace.trace_id))
