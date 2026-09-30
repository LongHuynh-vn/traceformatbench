"""Deterministic span trees, heaviest paths, service statistics, and top-k spans."""

from collections.abc import Iterator, Sequence

from traceformatbench.data.trace import Span, Trace
from traceformatbench.serializers.structure import fmt_ms, self_time_us

F5_K = 5


def _require_window(window: Sequence[Trace]) -> None:
    if not window:
        raise ValueError("Cannot serialize an empty window.")


def _tree_order(trace: Trace) -> Iterator[tuple[Span, int]]:
    """Visit every root's tree without depending on Python's recursion limit."""
    pending = [(span, 1) for span in reversed(trace.spans) if span.parent_id is None]
    while pending:
        span, depth = pending.pop()
        yield span, depth
        pending.extend((child, depth + 1) for child in reversed(trace.children(span.span_id)))


def _header(index: int, trace: Trace) -> str:
    return f"Trace {index} ({len(trace.spans)} spans)"


def _tree_line(trace: Trace, span: Span, depth: int) -> str:
    return (
        f"{'  ' * depth}{span.service} {span.operation} dur={fmt_ms(span.duration_us)}ms "
        f"at=+{fmt_ms(span.start_us - trace.start_us)}ms"
    )


def serialize_f2(window: Sequence[Trace]) -> str:
    """Render every span in root and child tree order."""
    _require_window(window)
    lines = [
        "# Span tree per trace. dur = span duration, at = start offset from the trace start, in ms."
    ]
    for index, trace in enumerate(window, 1):
        lines.append(_header(index, trace))
        lines.extend(_tree_line(trace, span, depth) for span, depth in _tree_order(trace))
    return "\n".join(lines)


def serialize_f3(window: Sequence[Trace]) -> str:
    """Follow the longest child from the primary root of each trace (spec 006, revised)."""
    _require_window(window)
    lines = [
        "# Heaviest path per trace: from the root, "
        "always into the child span with the longest duration. "
        "self = span duration minus time covered by its child spans, in ms."
    ]
    for index, trace in enumerate(window, 1):
        lines.append(_header(index, trace))
        span = trace.root
        depth = 0
        while span is not None:
            depth += 1
            lines.append(
                f"{'  ' * depth}{span.service} {span.operation} "
                f"dur={fmt_ms(span.duration_us)}ms self={fmt_ms(self_time_us(trace, span))}ms"
            )
            # Longest duration, then later end; max keeps the first child in
            # Trace.children() order for equal keys.
            span = max(
                trace.children(span.span_id),
                key=lambda child: (child.duration_us, child.start_us + child.duration_us),
                default=None,
            )
        other_spans = len(trace.spans) - depth
        if other_spans:
            lines.append(f"  (+{other_spans} other spans)")
    return "\n".join(lines)


def serialize_f4(window: Sequence[Trace]) -> str:
    """Aggregate span count, mean self-time, and max self-time by service."""
    _require_window(window)
    lines = [
        f"# Per-service statistics over all spans of the {len(window)} traces in this window. "
        "self = span duration minus time covered by its child spans, in ms.",
        "service spans mean_self_ms max_self_ms",
    ]
    by_service: dict[str, list[int]] = {}
    for trace in window:
        for span in trace.spans:
            by_service.setdefault(span.service, []).append(self_time_us(trace, span))
    for service, times in sorted(by_service.items()):
        lines.append(
            f"{service} {len(times)} {fmt_ms(sum(times) / len(times))} {fmt_ms(max(times))}"
        )
    return "\n".join(lines)


def serialize_f5(window: Sequence[Trace]) -> str:
    """Render the highest self-time spans and their ancestors in tree order."""
    _require_window(window)
    ranked = sorted(
        (-self_time_us(trace, span), trace_index, span_index)
        for trace_index, trace in enumerate(window)
        for span_index, span in enumerate(trace.spans)
    )[:F5_K]
    lines = [
        f"# The {len(ranked)} spans with the highest self-time in this window (marked *) "
        "and their ancestors. dur = span duration, at = start offset from the trace start, "
        "self = span duration minus time covered by its child spans, in ms."
    ]
    selected: dict[int, set[str]] = {}
    for _, trace_index, span_index in ranked:
        selected.setdefault(trace_index, set()).add(window[trace_index].spans[span_index].span_id)
    for trace_index, trace in enumerate(window):
        if trace_index not in selected:
            continue
        marked = selected[trace_index]
        kept = set(marked)
        by_id = {span.span_id: span for span in trace.spans}
        for span_id in marked:
            parent_id = by_id[span_id].parent_id
            while parent_id is not None and parent_id not in kept:
                kept.add(parent_id)
                parent_id = by_id[parent_id].parent_id
        lines.append(_header(trace_index + 1, trace))
        for span, depth in _tree_order(trace):
            if span.span_id in kept:
                line = _tree_line(trace, span, depth)
                if span.span_id in marked:
                    line += f" self={fmt_ms(self_time_us(trace, span))}ms *"
                lines.append(line)
    return "\n".join(lines)
