from traceformatbench.analysis.token_probe import (
    ProbeConfig,
    needs_remeasure,
    pick_windows,
    probe_case,
    record_key,
    summarize,
)
from traceformatbench.data.trace import Span, Trace
from traceformatbench.runners.ollama import OllamaClient, PromptTokenCount


def _trace(i: int, n_spans: int = 2) -> Trace:
    spans = tuple(
        Span(
            span_id=f"s{i}-{j}",
            parent_id=None if j == 0 else f"s{i}-0",
            trace_id=f"t{i}",
            service="frontend",
            operation="GET /",
            method=None,
            start_us=1_000_000 * i + j,
            duration_us=10,
            status_code=None,
        )
        for j in range(n_spans)
    )
    return Trace(f"t{i}", spans)


def _config(**overrides) -> ProbeConfig:
    values = dict(
        seed=1,
        ollama_url="http://unused",
        models=["m1", "m2"],
        num_ctx=1000,
        data_dir="unused",
        cases=["c"],
        formats=["F0", "F1"],
        window_sizes=[1, 3],
        windows_per_size=2,
        skip_if_chars_over=10_000,
        output="unused",
    )
    values.update(overrides)
    return ProbeConfig(**values)


class FakeClient(OllamaClient):
    def __init__(self, tokens_per_char: float = 0.25) -> None:
        self.calls: list[tuple[str, int]] = []
        self.tokens_per_char = tokens_per_char

    def count_prompt_tokens(self, model, text, num_ctx, seed):
        self.calls.append((model, num_ctx))
        return PromptTokenCount(int(len(text) * self.tokens_per_char), 0, 1)


def test_pick_windows_is_deterministic_and_consecutive():
    traces = [_trace(i) for i in range(10)]
    a = pick_windows(traces, [1, 3], per_size=2, seed=7)
    b = pick_windows(traces, [1, 3], per_size=2, seed=7)
    assert [(k, s) for k, s, _ in a] == [(k, s) for k, s, _ in b]
    for k, start, window in a:
        assert [t.trace_id for t in window] == [f"t{i}" for i in range(start, start + k)]


def test_pick_windows_skips_sizes_larger_than_case():
    picked = pick_windows([_trace(0)], [1, 5], per_size=3, seed=0)
    assert [k for k, _, _ in picked] == [1, 1, 1]


def test_probe_records_every_model_format_window_and_sends_num_ctx():
    client = FakeClient()
    config = _config()
    records = list(
        probe_case(
            "c", [_trace(i) for i in range(5)], config, client, {"m1": "d1", "m2": "d2"}, "v"
        )
    )
    assert len(records) == 2 * 2 * (2 + 2)  # models x formats x windows
    assert all(r["status"] == "ok" and r["prompt_tokens"] > 0 for r in records)
    assert {num_ctx for _, num_ctx in client.calls} == {1000}
    assert {r["model_digest"] for r in records} == {"d1", "d2"}


def test_long_texts_are_skipped_without_calling_the_model():
    client = FakeClient()
    records = list(
        probe_case(
            "c",
            [_trace(i) for i in range(5)],
            _config(skip_if_chars_over=10),
            client,
            {"m1": "d", "m2": "d"},
            "v",
        )
    )
    assert client.calls == []
    assert all(r["status"] == "skipped_too_long" and r["prompt_tokens"] is None for r in records)


def test_counts_at_num_ctx_are_flagged():
    client = FakeClient(tokens_per_char=100.0)
    records = list(
        probe_case(
            "c", [_trace(i) for i in range(5)], _config(), client, {"m1": "d", "m2": "d"}, "v"
        )
    )
    assert all(r["status"] == "at_or_over_num_ctx" for r in records)


def test_summary_counts_and_medians():
    records = list(
        probe_case(
            "c", [_trace(i) for i in range(5)], _config(), FakeClient(), {"m1": "d", "m2": "d"}, "v"
        )
    )
    summary = summarize(records)
    assert len(summary) == 2 * 2 * 2  # models x formats x window sizes
    assert (summary["windows"] == 2).all() and (summary["too_long"] == 0).all()
    assert (summary["median_tokens_per_trace"] > 0).all()


def test_window_ids_are_unique_even_if_starts_repeat():
    records = list(
        probe_case(
            "c",
            [_trace(i) for i in range(2)],
            _config(window_sizes=[2], windows_per_size=3),
            FakeClient(),
            {"m1": "d", "m2": "d"},
            "v",
        )
    )
    keys = [record_key(r) for r in records]
    assert len(keys) == len(set(keys)) == 2 * 2 * 3


def test_done_keys_are_skipped_without_calling_the_model():
    traces = [_trace(i) for i in range(5)]
    first = list(probe_case("c", traces, _config(), FakeClient(), {"m1": "d", "m2": "d"}, "v"))
    done = frozenset(record_key(r) for r in first[:5])
    client = FakeClient()
    rest = list(probe_case("c", traces, _config(), client, {"m1": "d", "m2": "d"}, "v", done))
    assert len(rest) == len(first) - 5
    assert len(client.calls) == len(first) - 5
    assert {record_key(r) for r in rest} | done == {record_key(r) for r in first}


class TruncatingClient(FakeClient):
    """Mimics Ollama: a prompt longer than num_ctx is cut and reports ~num_ctx // 2 + 2."""

    def count_prompt_tokens(self, model, text, num_ctx, seed):
        self.calls.append((model, num_ctx))
        true_tokens = int(len(text) * self.tokens_per_char)
        reported = true_tokens if true_tokens <= num_ctx else num_ctx // 2 + 2
        return PromptTokenCount(reported, 0, 1)


def _one_window_config(**overrides):
    return _config(models=["m1"], formats=["F0"], window_sizes=[1], windows_per_size=1, **overrides)


def test_silent_truncation_is_detected_and_count_discarded():
    # true length ~ 10x num_ctx, so Ollama would truncate to num_ctx // 2 + 2
    client = TruncatingClient(tokens_per_char=1.0)
    (record,) = probe_case(
        "c", [_trace(0)], _one_window_config(num_ctx=64), client, {"m1": "d"}, "v"
    )
    assert record["status"] == "truncated_over_num_ctx"
    assert record["prompt_tokens"] is None
    assert record["reported_tokens"] == 64 // 2 + 2
    assert [ctx for _, ctx in client.calls] == [64, 48]


def test_genuine_count_near_half_context_is_kept():
    config = _one_window_config(num_ctx=100_000)
    (first,) = probe_case("c", [_trace(0)], config, TruncatingClient(1.0), {"m1": "d"}, "v")
    num_ctx = first["prompt_tokens"] * 2  # true count sits exactly at num_ctx // 2
    client = TruncatingClient(tokens_per_char=1.0)
    (record,) = probe_case(
        "c", [_trace(0)], _one_window_config(num_ctx=num_ctx), client, {"m1": "d"}, "v"
    )
    assert record["status"] == "ok"
    assert record["prompt_tokens"] == num_ctx // 2
    assert record["verify_tokens"] == record["prompt_tokens"]


def test_needs_remeasure_only_for_unverified_counts_at_the_signature():
    base = {"status": "ok", "num_ctx": 32768}
    assert needs_remeasure({**base, "prompt_tokens": 16386})
    assert not needs_remeasure({**base, "prompt_tokens": 9000})
    assert not needs_remeasure({**base, "prompt_tokens": 16386, "verify_tokens": 16386})
    assert not needs_remeasure({**base, "status": "skipped_too_long", "prompt_tokens": None})
