"""Shared timing definitions for the structured trace formats (spec 006)."""

from traceformatbench.data.trace import Span, Trace


def self_time_us(trace: Trace, span: Span) -> int:
    """Subtract the union of direct child intervals clipped to this span."""
    parent_end = span.start_us + span.duration_us
    covered_end = span.start_us
    covered_us = 0
    for child in trace.children(span.span_id):
        start = max(span.start_us, child.start_us)
        end = min(parent_end, child.start_us + child.duration_us)
        if end > max(start, covered_end):
            covered_us += end - max(start, covered_end)
            covered_end = end
    return max(0, span.duration_us - covered_us)


def fmt_ms(us: int | float) -> str:
    """Format microseconds as milliseconds with two decimal places."""
    return f"{us / 1000:.2f}"
