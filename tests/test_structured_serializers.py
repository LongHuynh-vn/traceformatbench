from dataclasses import replace

import pytest

from traceformatbench.data.trace import Span, Trace
from traceformatbench.serializers import (
    SERIALIZERS,
    serialize_f0,
    serialize_f1,
    serialize_f2,
    serialize_f3,
    serialize_f4,
    serialize_f5,
)
from traceformatbench.serializers.structure import fmt_ms, self_time_us
from traceformatbench.serializers.structured import F5_K


def _span(span_id, parent_id, service, operation, start, duration, trace_id="t1"):
    return Span(span_id, parent_id, trace_id, service, operation, None, start, duration, None)


@pytest.fixture
def window():
    # Hand-built times in microseconds. The worker ends after its gateway parent.
    return (
        Trace(
            "t1",
            (
                _span("r1", None, "gateway", "GET", 1_000_000, 10_000),
                _span("a1", "r1", "api", "handle", 1_001_000, 3_000),
                _span("d1", "a1", "db", "query", 1_002_000, 1_000),
                _span("w1", "r1", "worker", "async", 1_008_000, 5_000),
            ),
        ),
        Trace(
            "t2",
            (
                _span("r2", None, "gateway", "GET", 2_000_000, 8_000, "t2"),
                _span("a2", "r2", "api", "handle", 2_001_000, 2_000, "t2"),
                _span("d2", "a2", "db", "搜索 🔎", 2_001_500, 500, "t2"),
                _span("c2", "r2", "cache", "get", 2_004_000, 1_000, "t2"),
            ),
        ),
    )


def test_f2_golden(window):
    # These literal strings are written by hand from spec 006, not serializer output.
    expected = (
        "# Span tree per trace. dur = span duration, "
        "at = start offset from the trace start, in ms.\n"
        "Trace 1 (4 spans)\n"
        "  gateway GET dur=10.00ms at=+0.00ms\n"
        "    api handle dur=3.00ms at=+1.00ms\n"
        "      db query dur=1.00ms at=+2.00ms\n"
        "    worker async dur=5.00ms at=+8.00ms\n"
        "Trace 2 (4 spans)\n"
        "  gateway GET dur=8.00ms at=+0.00ms\n"
        "    api handle dur=2.00ms at=+1.00ms\n"
        "      db 搜索 🔎 dur=0.50ms at=+1.50ms\n"
        "    cache get dur=1.00ms at=+4.00ms"
    )
    assert serialize_f2(window) == expected


def test_f3_golden(window):
    expected = (
        "# Heaviest path per trace: from the root, "
        "always into the child span with the longest duration. "
        "self = span duration minus time covered by its child spans, in ms.\n"
        "Trace 1 (4 spans)\n"
        "  gateway GET dur=10.00ms self=5.00ms\n"
        "    worker async dur=5.00ms self=5.00ms\n"
        "  (+2 other spans)\n"
        "Trace 2 (4 spans)\n"
        "  gateway GET dur=8.00ms self=5.00ms\n"
        "    api handle dur=2.00ms self=1.50ms\n"
        "      db 搜索 🔎 dur=0.50ms self=0.50ms\n"
        "  (+1 other spans)"
    )
    assert serialize_f3(window) == expected


def test_f4_golden(window):
    expected = (
        "# Per-service statistics over all spans of the 2 traces in this window. "
        "self = span duration minus time covered by its child spans, in ms.\n"
        "service spans mean_self_ms max_self_ms\n"
        "api 2 1.75 2.00\n"
        "cache 1 1.00 1.00\n"
        "db 2 0.75 1.00\n"
        "gateway 2 5.00 5.00\n"
        "worker 1 5.00 5.00"
    )
    assert serialize_f4(window) == expected


def test_f5_golden(window):
    expected = (
        "# The 5 spans with the highest self-time in this window (marked *) and their ancestors. "
        "dur = span duration, at = start offset from the trace start, "
        "self = span duration minus time covered by its child spans, in ms.\n"
        "Trace 1 (4 spans)\n"
        "  gateway GET dur=10.00ms at=+0.00ms self=5.00ms *\n"
        "    api handle dur=3.00ms at=+1.00ms self=2.00ms *\n"
        "    worker async dur=5.00ms at=+8.00ms self=5.00ms *\n"
        "Trace 2 (4 spans)\n"
        "  gateway GET dur=8.00ms at=+0.00ms self=5.00ms *\n"
        "    api handle dur=2.00ms at=+1.00ms self=1.50ms *"
    )
    assert F5_K == 5
    assert serialize_f5(window) == expected


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([], 10_000),
        ([(1_000, 2_000), (4_000, 3_000)], 5_000),
        ([(8_000, 5_000)], 8_000),
        ([(-2_000, 5_000)], 7_000),
        ([(1_000, 5_000), (4_000, 5_000)], 2_000),
        ([(1_000, 8_000), (2_000, 1_000), (7_000, 3_000)], 1_000),
        ([(-1_000, 20_000), (0, 10_000)], 0),
        ([(-5_000, 1_000), (12_000, 1_000), (5_000, 0)], 10_000),
        ([(0, 10_000), (0, 10_000)], 0),
    ],
    ids=[
        "leaf",
        "sequential",
        "async-end",
        "early-start",
        "overlap",
        "nested-overlap",
        "clipped-full-coverage",
        "outside-and-empty",
        "never-negative",
    ],
)
def test_self_time_union_and_clipping(intervals, expected):
    root = _span("root", None, "svc", "root", 100_000, 10_000)
    children = [
        _span(str(index), "root", "svc", "child", 100_000 + start, duration)
        for index, (start, duration) in enumerate(intervals)
    ]
    trace = Trace("t1", (root, *reversed(children)))
    assert self_time_us(trace, root) == expected


def test_self_time_only_subtracts_direct_children(window):
    assert self_time_us(window[0], window[0].root) == 5_000
    assert self_time_us(window[0], window[0].spans[1]) == 2_000


@pytest.mark.parametrize(
    ("us", "expected"), [(0, "0.00"), (1, "0.00"), (49, "0.05"), (1234.5, "1.23")]
)
def test_fmt_ms(us, expected):
    assert fmt_ms(us) == expected


@pytest.mark.parametrize("fmt", ["F2", "F3", "F4", "F5"])
def test_determinism_and_no_trailing_newline(fmt, window):
    serializer = SERIALIZERS[fmt]
    assert serializer(window).encode("utf-8") == serializer(window).encode("utf-8")
    assert not serializer(window).endswith("\n")


@pytest.mark.parametrize("fmt", list(SERIALIZERS))
@pytest.mark.parametrize("window", [[], ()])
def test_empty_window(fmt, window):
    with pytest.raises(ValueError, match="empty window"):
        SERIALIZERS[fmt](window)


def test_f3_prefers_longer_duration_over_later_end():
    # Sequential calls: the slow early call must win over the short trailing one.
    trace = Trace(
        "t1",
        (
            _span("r", None, "svc", "root", 0, 20_000),
            _span("long", "r", "svc", "long", 1_000, 10_000),
            _span("late", "r", "svc", "late", 12_000, 1_000),
        ),
    )
    text = serialize_f3([trace])
    assert "    svc long dur=10.00ms self=10.00ms" in text
    assert "svc late" not in text


@pytest.mark.parametrize(
    ("z_start", "z_duration", "chosen"),
    [
        (4_000, 6_000, "a"),  # a is longer
        (3_000, 8_000, "z"),  # same duration, z ends later
        (2_000, 8_000, "a"),  # full tie: first in Trace.children() order
    ],
)
def test_f3_ties_by_end_then_children_order(z_start, z_duration, chosen):
    trace = Trace(
        "t1",
        (
            _span("r", None, "svc", "root", 0, 12_000),
            _span("z", "r", "svc", "op-z", z_start, z_duration),
            _span("a", "r", "svc", "op-a", 2_000, 8_000),
        ),
    )
    text = serialize_f3([trace])
    other = "z" if chosen == "a" else "a"
    assert f"    svc op-{chosen} " in text
    assert f"svc op-{other}" not in text


@pytest.mark.parametrize("fmt", ["F2", "F5"])
def test_multiple_roots_and_tree_order(fmt):
    trace = Trace(
        "t1",
        (
            _span("z", "r", "svc", "last-child", 2_000, 1_000),
            _span("s", None, "svc", "second-root", 1_000, 1_000),
            _span("a", "r", "svc", "first-child", 2_000, 1_000),
            _span("r", None, "svc", "first-root", 0, 4_000),
        ),
    )
    lines = SERIALIZERS[fmt]([trace]).splitlines()[2:]
    assert [line.split()[1] for line in lines] == [
        "first-root",
        "first-child",
        "last-child",
        "second-root",
    ]
    assert [len(line) - len(line.lstrip()) for line in lines] == [2, 4, 4, 2]


def test_f3_counts_extra_roots_and_omits_zero_other_spans():
    root = _span("r", None, "svc", "root", 0, 4_000)
    child = _span("c", "r", "svc", "child", 1_000, 1_000)
    second = _span("s", None, "svc", "second-root", 2_000, 5_000)
    text = serialize_f3([Trace("t1", (second, child, root))])
    assert text.endswith("  (+1 other spans)")
    assert "second-root" not in text
    assert "other spans" not in serialize_f3([Trace("t1", (root, child))])


def test_f4_sorts_services_and_uses_self_time(window):
    rows = [line.split() for line in serialize_f4(window).splitlines()[2:]]
    assert [row[0] for row in rows] == ["api", "cache", "db", "gateway", "worker"]
    assert rows[0] == ["api", "2", "1.75", "2.00"]
    assert rows[3] == ["gateway", "2", "5.00", "5.00"]


@pytest.mark.parametrize("n_spans", [1, 3, 5, 8])
def test_f5_marks_exactly_min_k_spans(n_spans):
    trace = Trace(
        "t1",
        tuple(_span(str(i), None, "svc", str(i), i * 10_000, 1_000) for i in range(n_spans)),
    )
    text = serialize_f5([trace])
    assert sum(line.endswith(" *") for line in text.splitlines()) == min(F5_K, n_spans)
    assert text.startswith(f"# The {min(F5_K, n_spans)} spans ")


def test_f5_keeps_all_ancestors_and_skips_unselected_traces():
    small = Trace("t1", (_span("r", None, "svc", "small", 0, 10),))
    busy = Trace(
        "t2",
        (
            _span("r", None, "svc", "root", 0, 50_000, "t2"),
            _span("p", "r", "svc", "parent", 0, 50_000, "t2"),
            *(_span(str(i), "p", "svc", f"child{i}", i * 10_000, 10_000, "t2") for i in range(5)),
        ),
    )
    text = serialize_f5([small, busy, small])
    assert "Trace 1" not in text and "Trace 3" not in text
    assert "Trace 2 (7 spans)" in text
    assert "  svc root dur=50.00ms at=+0.00ms\n" in text
    assert "    svc parent dur=50.00ms at=+0.00ms\n" in text
    assert sum(line.endswith(" *") for line in text.splitlines()) == 5
    assert "      svc child4 dur=10.00ms at=+40.00ms self=10.00ms *" in text


def test_f5_ties_by_window_position_then_span_index():
    first = Trace(
        "z-trace",
        tuple(
            _span(str(i), None, "svc", f"first{i}", i * 10_000, 1_000, "z-trace")
            for i in reversed(range(3))
        ),
    )
    second = Trace(
        "a-trace",
        tuple(
            _span(str(i), None, "svc", f"second{i}", i * 10_000, 1_000, "a-trace")
            for i in reversed(range(3))
        ),
    )
    lines = [line for line in serialize_f5([first, second]).splitlines() if line.endswith(" *")]
    assert [line.split()[1] for line in lines] == [
        "first0",
        "first1",
        "first2",
        "second0",
        "second1",
    ]


@pytest.mark.parametrize("fmt", ["F2", "F3", "F4", "F5"])
def test_unicode_names_are_preserved(fmt):
    trace = Trace("t1", (_span("r", None, "服务 🌍", "搜索 🔎", 0, 1_000),))
    text = SERIALIZERS[fmt]([trace])
    assert "服务 🌍" in text
    if fmt != "F4":
        assert "搜索 🔎" in text
    assert "\\u" not in text


@pytest.mark.parametrize("fmt", ["F2", "F3", "F4", "F5"])
def test_ids_method_status_and_absolute_start_do_not_affect_output(fmt, window):
    changed = tuple(
        Trace(
            "hidden-trace-id",
            tuple(
                replace(
                    span,
                    span_id="hidden-" + span.span_id,
                    parent_id="hidden-" + span.parent_id if span.parent_id else None,
                    trace_id="hidden-trace-id",
                    method="hidden-method",
                    status_code=2,
                    start_us=span.start_us + 99_000_000,
                )
                for span in trace.spans
            ),
        )
        for trace in window
    )
    assert SERIALIZERS[fmt](window) == SERIALIZERS[fmt](changed)


def test_registry_contains_exactly_f0_through_f5():
    assert SERIALIZERS == {
        "F0": serialize_f0,
        "F1": serialize_f1,
        "F2": serialize_f2,
        "F3": serialize_f3,
        "F4": serialize_f4,
        "F5": serialize_f5,
    }
