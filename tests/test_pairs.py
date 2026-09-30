from dataclasses import replace

import pytest

from traceformatbench.data.trace import Span, Trace
from traceformatbench.serializers import SERIALIZERS
from traceformatbench.tasks.pairs import align_pair_times


def _window(prefix, start):
    return [
        Trace(
            f"{prefix}-{i}",
            (
                Span(
                    f"root-{i}", None, f"{prefix}-{i}", "a", "op", "GET", start + offset, 130, 200
                ),
                Span(
                    f"child-{i}",
                    f"root-{i}",
                    f"{prefix}-{i}",
                    "b",
                    "child",
                    None,
                    start + offset + 11,
                    35,
                    None,
                ),
            ),
        )
        for i, offset in enumerate((107, 0))
    ]


@pytest.mark.parametrize("pre_start, post_start", [(1001, 9017), (9017, 1001), (1001, 1001)])
def test_alignment_preserves_offsets_durations_fields_order_and_inputs(pre_start, post_start):
    pre, post = _window("pre", pre_start), _window("post", post_start)
    before = [tuple(window) for window in (pre, post)]
    original_texts = [
        serializer(window) for serializer in SERIALIZERS.values() for window in (pre, post)
    ]
    aligned = align_pair_times(pre, post)
    anchor = min(pre_start, post_start)
    for original, shifted, start in zip((pre, post), aligned, (pre_start, post_start), strict=True):
        assert shifted is not original
        assert min(span.start_us for trace in shifted for span in trace.spans) == anchor
        assert [trace.trace_id for trace in shifted] == [trace.trace_id for trace in original]
        for trace, copy in zip(original, shifted, strict=True):
            assert copy is not trace
            assert copy.end_us - copy.start_us == trace.end_us - trace.start_us
            assert copy.extra_root_ids == trace.extra_root_ids
            assert copy.orphan_span_ids == trace.orphan_span_ids
            assert copy.services == trace.services
            for span, shifted_span in zip(trace.spans, copy.spans, strict=True):
                assert shifted_span is not span
                assert shifted_span.start_us - anchor == span.start_us - start
                assert shifted_span.duration_us == span.duration_us
                assert replace(shifted_span, start_us=span.start_us) == span
                assert [s.span_id for s in copy.children(shifted_span.span_id)] == [
                    s.span_id for s in trace.children(span.span_id)
                ]
    assert before == [tuple(window) for window in (pre, post)]
    assert original_texts == [
        serializer(window) for serializer in SERIALIZERS.values() for window in (pre, post)
    ]
    assert min(t.start_us for t in pre) == pre_start
    assert min(t.start_us for t in post) == post_start


@pytest.mark.parametrize("empty_phase", ["pre", "post", "both"])
def test_alignment_rejects_empty_windows(empty_phase):
    pre = [] if empty_phase in {"pre", "both"} else _window("pre", 1000)
    post = [] if empty_phase in {"post", "both"} else _window("post", 9000)
    with pytest.raises(ValueError, match="two non-empty windows"):
        align_pair_times(pre, post)
