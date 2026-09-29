"""Exploratory token probe used to choose window parameters for spec 003.

Not a benchmark result. Windows here are k consecutive traces (by start time) from a
random start position; the real window sampler is defined separately in spec 003.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traceformatbench.data.trace import Trace, load_case_traces
from traceformatbench.runners.ollama import OllamaClient
from traceformatbench.serializers import SERIALIZERS

# Ollama does not report truncation. When a prompt exceeds num_ctx it silently cuts it,
# and prompt_eval_count lands just above num_ctx // 2 (observed: 16386 and 16387 for
# num_ctx 32768 with qwen3.5:9b and gemma4:12b). A count within this distance of
# num_ctx // 2 is re-checked with a smaller context (see probe_case).
TRUNCATION_TOLERANCE = 16


@dataclass(frozen=True)
class ProbeConfig:
    seed: int
    ollama_url: str
    models: list[str]
    num_ctx: int
    data_dir: str
    cases: list[str]
    formats: list[str]
    window_sizes: list[int]
    windows_per_size: int
    skip_if_chars_over: int
    output: str


def pick_windows(
    traces: Sequence[Trace], window_sizes: Sequence[int], per_size: int, seed: int
) -> list[tuple[int, int, list[Trace]]]:
    """Return (window_size, start_index, window) tuples; deterministic for a given seed."""
    rng = random.Random(seed)
    picked = []
    for k in window_sizes:
        if k > len(traces):
            continue
        for _ in range(per_size):
            start = rng.randrange(len(traces) - k + 1)
            picked.append((k, start, list(traces[start : start + k])))
    return picked


def probe_case(
    case: str,
    traces: Sequence[Trace],
    config: ProbeConfig,
    client: OllamaClient,
    model_digests: dict[str, str],
    ollama_version: str,
    done: frozenset[tuple] = frozenset(),
) -> Iterator[dict]:
    """Yield one record per (model, format, window), skipping keys already in `done`."""
    windows = pick_windows(traces, config.window_sizes, config.windows_per_size, config.seed)
    texts = [
        (window_id, k, start, fmt, len(window), SERIALIZERS[fmt](window))
        for window_id, (k, start, window) in enumerate(windows)
        for fmt in config.formats
    ]
    for model in config.models:
        for window_id, k, start, fmt, n_traces, text in texts:
            if (case, model, fmt, window_id) in done:
                continue
            record = {
                "case": case,
                "model": model,
                "model_digest": model_digests[model],
                "ollama_version": ollama_version,
                "num_ctx": config.num_ctx,
                "format": fmt,
                "window_id": window_id,
                "window_size": k,
                "start_index": start,
                "n_traces": n_traces,
                "chars": len(text),
                "prompt_tokens": None,
                "cached_tokens": None,
                "status": "ok",
            }
            if len(text) > config.skip_if_chars_over:
                record["status"] = "skipped_too_long"
            else:
                count = client.count_prompt_tokens(model, text, config.num_ctx, config.seed)
                record["prompt_tokens"] = count.prompt_tokens
                record["cached_tokens"] = count.cached_tokens
                if count.prompt_tokens >= config.num_ctx:
                    record["status"] = "at_or_over_num_ctx"
                elif near_truncation_point(count.prompt_tokens, config.num_ctx):
                    # A genuine count does not change with a smaller (but sufficient) context;
                    # a truncated one does, because the true length exceeds both contexts.
                    check_ctx = config.num_ctx * 3 // 4
                    check = client.count_prompt_tokens(model, text, check_ctx, config.seed)
                    record["verify_num_ctx"] = check_ctx
                    record["verify_tokens"] = check.prompt_tokens
                    if check.prompt_tokens != count.prompt_tokens:
                        record["status"] = "truncated_over_num_ctx"
                        record["reported_tokens"] = count.prompt_tokens
                        record["prompt_tokens"] = None
            yield record


def near_truncation_point(tokens: int, num_ctx: int) -> bool:
    return abs(tokens - num_ctx // 2) <= TRUNCATION_TOLERANCE


def needs_remeasure(record: dict) -> bool:
    """Records from before the truncation check that sit at the truncation signature."""
    return (
        record.get("status") == "ok"
        and record.get("prompt_tokens") is not None
        and "verify_tokens" not in record
        and near_truncation_point(record["prompt_tokens"], record["num_ctx"])
    )


def record_key(record: dict) -> tuple:
    """Identity of one measurement; used to resume an interrupted run."""
    return (record["case"], record["model"], record["format"], record["window_id"])


def run_probe(
    config: ProbeConfig, client: OllamaClient, done: frozenset[tuple] = frozenset()
) -> Iterator[dict]:
    """Stream records case by case (one case in memory at a time), skipping `done` keys."""
    version = client.version()
    digests = {model: client.model_digest(model) for model in config.models}
    for case in config.cases:
        traces = load_case_traces(Path(config.data_dir) / case)
        yield from probe_case(case, traces, config, client, digests, version, done)
        del traces


def summarize(records: Sequence[dict]) -> pd.DataFrame:
    """Median/max tokens per (case, model, format, window size), plus skip/overflow counts."""
    df = pd.DataFrame(records)
    df["measured"] = df["status"] == "ok"
    ok = df[df["measured"]].assign(tokens_per_trace=lambda d: d["prompt_tokens"] / d["n_traces"])
    keys = ["case", "model", "format", "window_size"]
    counts = df.groupby(keys).agg(
        windows=("status", "size"),
        measured=("measured", "sum"),
        too_long=("status", lambda s: int((s != "ok").sum())),
    )
    stats = ok.groupby(keys).agg(
        median_tokens=("prompt_tokens", "median"),
        max_tokens=("prompt_tokens", "max"),
        median_tokens_per_trace=("tokens_per_trace", "median"),
    )
    return counts.join(stats).reset_index()
