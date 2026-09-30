"""Time alignment for paired T1 windows (spec 005b, Revision 1)."""

from collections.abc import Sequence
from dataclasses import replace

from traceformatbench.data.trace import Trace


def align_pair_times(
    pre: Sequence[Trace], post: Sequence[Trace]
) -> tuple[list[Trace], list[Trace]]:
    """Shift copies of both windows to their earlier start, preserving within-window offsets."""
    if not pre or not post:
        raise ValueError("Time alignment requires two non-empty windows.")
    starts = [
        min(span.start_us for trace in window for span in trace.spans) for window in (pre, post)
    ]
    anchor = min(starts)
    aligned = [
        [
            replace(
                trace,
                spans=tuple(
                    replace(span, start_us=span.start_us + anchor - start) for span in trace.spans
                ),
            )
            for trace in window
        ]
        for window, start in zip((pre, post), starts, strict=True)
    ]
    return aligned[0], aligned[1]
