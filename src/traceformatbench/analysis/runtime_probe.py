"""Exploratory runtime probe for spec-003 windows; answers are never scored."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd

from traceformatbench.analysis.token_probe import near_truncation_point
from traceformatbench.data.rcaeval import INDEX_FILE
from traceformatbench.data.trace import Trace, load_case_traces
from traceformatbench.data.windows import WindowConfig, build_case_windows, load_window_config
from traceformatbench.runners.ollama import ChatResult, OllamaClient
from traceformatbench.serializers import SERIALIZERS

SYSTEM_PROMPT = (
    "You analyze distributed traces from a microservice system. Reply with one JSON object only."
)
ANSWER_INSTRUCTION = 'Answer as JSON: {"phase": "pre" or "post", "service": "<service name>"}'
# The warm-up prompt must differ from every measured prompt right after the system message:
# Ollama reuses the KV cache of a shared prompt prefix, so an identical warm-up would make the
# first measured call look almost free.
WARMUP_PREFIX = "Warm-up request; its timing is discarded.\n"
# The observed minimum in the approved spec 003, not an exact tokenizer bound.
PROBED_MIN_CHARS_PER_TOKEN = 1.95
SUMMARY_COLUMNS = [
    "model",
    "format",
    "system",
    "n",
    "median_call_s",
    "max_call_s",
    "median_prompt_tokens",
    "median_eval_count",
    "median_prefill_tokens_per_s",
    "median_decode_tokens_per_s",
    "length_share",
    "n_cached_calls",
    "sum_prompt_eval_cached_count",
]
ESTIMATE_COLUMNS = [
    "t1_windows_per_phase",
    "t2_windows",
    "hours_f1_bound",
    "hours_f0_bound",
    "n_calls",
    "fits_48h_f1_bound",
    "fits_48h_f0_bound",
]


@dataclass(frozen=True)
class RuntimeProbeConfig:
    seed: int
    ollama_url: str
    models: list[str]
    num_ctx: int
    num_predict: int
    temperature: float
    windows_config: str
    cases: list[str]
    formats: list[str]
    output: str

    def __post_init__(self) -> None:
        for name in ("models", "cases", "formats"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and contain no duplicates.")
        if not set(self.formats) <= set(SERIALIZERS):
            raise ValueError(f"Unknown format ID; expected one of {sorted(SERIALIZERS)}.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")


def serialize_windows(
    windows: Sequence[dict], traces: Sequence[Trace], formats: Sequence[str]
) -> list[dict]:
    """Attach format texts to the exact trace IDs chosen by build_case_windows."""
    by_id = {trace.trace_id: trace for trace in traces}
    return [
        {
            **window,
            "texts": {
                fmt: SERIALIZERS[fmt]([by_id[trace_id] for trace_id in window["trace_ids"]])
                for fmt in formats
            },
        }
        for window in windows
    ]


def build_call_list(
    windows: Sequence[dict], models: Sequence[str], formats: Sequence[str]
) -> list[dict]:
    """One warm-up per model, then every window and format, without interleaving models."""
    if not windows or not formats:
        raise ValueError("The runtime probe requires at least one window and format.")
    calls = []
    for model in models:
        model_calls = [
            {
                "case": window["case"],
                "dataset": window["dataset"],
                "model": model,
                "format": fmt,
                "task": window["task"],
                "phase": window["phase"],
                "window_index": window["window_index"],
                "f0_chars": window["f0_chars"],
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": window["texts"][fmt] + "\n" + ANSWER_INSTRUCTION,
                "warmup": False,
            }
            for window in windows
            for fmt in formats
        ]
        first = model_calls[0]
        calls.append({**first, "warmup": True, "user_prompt": WARMUP_PREFIX + first["user_prompt"]})
        calls.extend(model_calls)
    return calls


def record_key(record: Mapping) -> tuple:
    """Phase separates T1 pre/post indices; warm-up is separate from its measured twin."""
    return tuple(
        record[key]
        for key in (
            "case",
            "model",
            "format",
            "task",
            "phase",
            "window_index",
            "warmup",
        )
    )


def call_status(result: ChatResult, num_ctx: int) -> str:
    """Flag the truncation signature and leaked thinking, without parsing the answer."""
    if (
        near_truncation_point(result.prompt_eval_count, num_ctx)
        or result.prompt_eval_count >= num_ctx
    ):
        return "truncated"
    if result.thinking or "<think>" in result.content:
        return "thinking_leak"
    return "ok"


def check_prompt_length(call: Mapping, config: RuntimeProbeConfig) -> None:
    """Screen input before sending using the prior probe's observed character/token ratio.

    This estimate is not exact tokenization. The chat request also disables server truncation,
    and returned token counts are checked for the legacy silent-truncation signature.
    """
    prompt_chars = len(call["system_prompt"]) + len(call["user_prompt"])
    estimated_tokens = ceil(prompt_chars / PROBED_MIN_CHARS_PER_TOKEN)
    if estimated_tokens + config.num_predict >= config.num_ctx:
        raise ValueError(
            f"Prompt for {call['case']!r} ({call['format']}) may exceed num_ctx={config.num_ctx}: "
            f"{prompt_chars} characters, approximately {estimated_tokens} input tokens "
            f"plus num_predict={config.num_predict}. No chat request was sent."
        )


def make_record(
    call: Mapping,
    result: ChatResult,
    config: RuntimeProbeConfig,
    *,
    model_digest: str,
    ollama_version: str,
    think_sent: bool | None,
    wall_s: float,
) -> dict:
    """Store exact prompts and seed alongside the fields specified for each timed call."""
    return {
        **call,
        "model_digest": model_digest,
        "ollama_version": ollama_version,
        "seed": config.seed,
        "prompt_chars": len(call["system_prompt"]) + len(call["user_prompt"]),
        "num_ctx": config.num_ctx,
        "num_predict": config.num_predict,
        "temperature": config.temperature,
        "think_sent": think_sent,
        "prompt_eval_count": result.prompt_eval_count,
        "prompt_eval_cached_count": result.prompt_eval_cached_count,
        "eval_count": result.eval_count,
        "done_reason": result.done_reason,
        "load_s": result.load_duration_ns / 1e9,
        "prompt_eval_s": result.prompt_eval_duration_ns / 1e9,
        "eval_s": result.eval_duration_ns / 1e9,
        "total_s": result.total_duration_ns / 1e9,
        "wall_s": wall_s,
        "thinking_chars": len(result.thinking),
        "content": result.content,
        "status": call_status(result, config.num_ctx),
    }


def load_probe_windows(config: RuntimeProbeConfig, windows_config: WindowConfig) -> list[dict]:
    """Read only cached data; build and serialize one selected case at a time."""
    data_dir = Path(windows_config.data_dir)
    index = pd.read_parquet(data_dir / INDEX_FILE)
    windows = []
    for case in config.cases:
        matches = index[index["case"] == case]
        if len(matches) != 1:
            raise ValueError(f"Case {case!r} must occur exactly once in the local {INDEX_FILE}.")
        case_meta = matches.iloc[0].to_dict()
        traces = load_case_traces(data_dir / case)
        case_windows, _ = build_case_windows(
            case_meta, traces, windows_config, random.Random(f"{windows_config.seed}:{case}")
        )
        if not case_windows:
            raise ValueError(f"Case {case!r} produced no windows.")
        windows.extend(serialize_windows(case_windows, traces, config.formats))
        del traces
    return windows


def run_probe(
    config: RuntimeProbeConfig, client: OllamaClient, done: frozenset[tuple] = frozenset()
) -> Iterator[dict]:
    """Preflight all models before the first call; stream results and unload between models."""
    version = client.version()
    digests = {model: client.model_digest(model) for model in config.models}
    capabilities = {model: client.model_capabilities(model) for model in config.models}
    windows_config = load_window_config(config.windows_config)
    windows = load_probe_windows(config, windows_config)
    calls = build_call_list(windows, config.models, config.formats)
    for call in calls:
        check_prompt_length(call, config)
    for model in config.models:
        pending = [
            call for call in calls if call["model"] == model and record_key(call) not in done
        ]
        if not pending:
            continue
        try:
            for call in pending:
                started = time.monotonic()
                result = client.chat(
                    model=model,
                    system=call["system_prompt"],
                    user=call["user_prompt"],
                    num_ctx=config.num_ctx,
                    num_predict=config.num_predict,
                    temperature=config.temperature,
                    seed=config.seed,
                    think=False,
                )
                wall_s = time.monotonic() - started
                yield make_record(
                    call,
                    result,
                    config,
                    model_digest=digests[model],
                    ollama_version=version,
                    think_sent=False if "thinking" in capabilities[model] else None,
                    wall_s=wall_s,
                )
        finally:
            client.unload_model(model)


def summarize(records: Sequence[dict]) -> pd.DataFrame:
    """Valid measured calls only; cached prefill rates use the uncached token count.

    Call time is total_s minus load_s, so a model reload (for example after a resume) does not
    inflate the timing of the first call that follows it.
    """
    valid = [record for record in records if not record["warmup"] and record["status"] == "ok"]
    if not valid:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = pd.DataFrame(valid)
    df["call_s"] = df["total_s"] - df["load_s"]
    df["system"] = df["dataset"].str.rsplit("-", n=1).str[-1].str.upper()
    df["prefill_tokens_per_s"] = (df["prompt_eval_count"] - df["prompt_eval_cached_count"]) / df[
        "prompt_eval_s"
    ].where(df["prompt_eval_s"] > 0)
    df["decode_tokens_per_s"] = df["eval_count"] / df["eval_s"].where(df["eval_s"] > 0)
    df["length"] = df["done_reason"] == "length"
    df["cached"] = df["prompt_eval_cached_count"] > 0
    return (
        df.groupby(["model", "format", "system"])
        .agg(
            n=("call_s", "size"),
            median_call_s=("call_s", "median"),
            max_call_s=("call_s", "max"),
            median_prompt_tokens=("prompt_eval_count", "median"),
            median_eval_count=("eval_count", "median"),
            median_prefill_tokens_per_s=("prefill_tokens_per_s", "median"),
            median_decode_tokens_per_s=("decode_tokens_per_s", "median"),
            length_share=("length", "mean"),
            n_cached_calls=("cached", "sum"),
            sum_prompt_eval_cached_count=("prompt_eval_cached_count", "sum"),
        )
        .reset_index()[SUMMARY_COLUMNS]
    )


def estimate(summary: pd.DataFrame, n_cases: Mapping[str, int] | None = None) -> pd.DataFrame:
    """The spec-004 1..4 window grid and 48-hour bounds, using paired F0/F1 medians."""
    if summary.empty:
        return pd.DataFrame(columns=ESTIMATE_COLUMNS)
    n_cases = {"OB": 40, "TT": 38} if n_cases is None else n_cases
    times = summary.pivot(index=["model", "system"], columns="format", values="median_call_s")
    if not {"F0", "F1"} <= set(times.columns) or times[["F0", "F1"]].isna().any().any():
        raise ValueError("Runtime estimates require valid F0 and F1 timings for each model/system.")
    f1_seconds = f0_seconds = calls_per_window = 0.0
    for (_, system), row in times.iterrows():
        cases = n_cases[system]
        f1_seconds += cases * (row["F0"] + 5 * row["F1"])
        f0_seconds += cases * 6 * row["F0"]
        calls_per_window += cases * 6
    rows = []
    for t1 in range(1, 5):
        for t2 in range(1, 5):
            windows = 2 * t1 + t2
            hours_f1 = windows * f1_seconds / 3600
            hours_f0 = windows * f0_seconds / 3600
            rows.append(
                {
                    "t1_windows_per_phase": t1,
                    "t2_windows": t2,
                    "hours_f1_bound": hours_f1,
                    "hours_f0_bound": hours_f0,
                    "n_calls": int(windows * calls_per_window),
                    "fits_48h_f1_bound": hours_f1 <= 48,
                    "fits_48h_f0_bound": hours_f0 <= 48,
                }
            )
    return pd.DataFrame(rows, columns=ESTIMATE_COLUMNS)
