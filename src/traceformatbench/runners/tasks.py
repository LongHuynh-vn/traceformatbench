"""Resumable task calls on fixed windows, with spec-005 preflight and provenance."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traceformatbench.analysis.runtime_probe import (
    call_status,
    check_prompt_length,
    serialize_windows,
)
from traceformatbench.data.rcaeval import INDEX_FILE
from traceformatbench.data.trace import load_case_traces
from traceformatbench.data.windows import WindowConfig, build_case_windows, load_window_config
from traceformatbench.runners.ollama import ChatResult, OllamaClient
from traceformatbench.serializers import SERIALIZERS
from traceformatbench.tasks.pairs import align_pair_times
from traceformatbench.tasks.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_t1_prompt,
    build_t1p_prompt,
    build_t2_prompt,
    case_candidates,
)
from traceformatbench.tasks.schemas import t1_schema, t1p_schema, t2_schema


@dataclass(frozen=True)
class TaskConfig:
    seed: int
    prompt_version: str
    ollama_url: str
    models: list[str]
    num_ctx: int
    num_predict: int
    temperature: float
    windows_config: str
    cases: list[str]
    formats: list[str]
    tasks: list[str]
    output: str

    def __post_init__(self) -> None:
        for name in ("models", "cases", "formats", "tasks"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and contain no duplicates.")
        if not set(self.formats) <= set(SERIALIZERS):
            raise ValueError(f"Unknown format ID; expected one of {sorted(SERIALIZERS)}.")
        if not set(self.tasks) <= {"T1", "T1P", "T2"}:
            raise ValueError("Specs 005/005b support only T1, T1P and T2.")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"Unsupported prompt_version {self.prompt_version!r}.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")
        if self.temperature != 0:
            raise ValueError("Spec 005 requires temperature 0.")


def window_key(window: Mapping) -> tuple:
    """T1 phase separates the pre/post windows with the same index."""
    return tuple(window[key] for key in ("case", "task", "phase", "window_index"))


def check_saved_windows(windows: Sequence[dict], path: Path, cases: Sequence[str]) -> None:
    if not path.exists():
        return
    saved = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["case"] in cases:
            key = window_key(record)
            if key in saved:
                raise ValueError(f"Duplicate saved window {key!r} in {path}.")
            saved[key] = record["trace_ids"]
    rebuilt = {window_key(window): window["trace_ids"] for window in windows}
    mismatches = sorted(
        key for key in saved.keys() | rebuilt.keys() if saved.get(key) != rebuilt.get(key)
    )
    if mismatches:
        raise ValueError(f"Trace IDs do not match {path} for windows: {mismatches!r}.")


def load_task_windows(config: TaskConfig, windows_config: WindowConfig) -> list[dict]:
    """Load local data only, retaining all services before window filtering."""
    data_dir = Path(windows_config.data_dir)
    index = pd.read_parquet(data_dir / INDEX_FILE)
    windows = []
    missing_root = []
    too_few = []
    for case in config.cases:
        matches = index[index["case"] == case]
        if len(matches) != 1:
            raise ValueError(f"Case {case!r} must occur exactly once in the local {INDEX_FILE}.")
        metadata = matches.iloc[0].to_dict()
        traces = load_case_traces(data_dir / case)
        candidates = case_candidates(traces)
        case_windows, _ = build_case_windows(
            metadata, traces, windows_config, random.Random(f"{windows_config.seed}:{case}")
        )
        has_t2 = any(window["task"] == "T2" for window in case_windows)
        if has_t2 and metadata["root_cause_service"] not in candidates:
            missing_root.append(case)
        if has_t2 and len(candidates) < 3:
            too_few.append(case)
        if case in missing_root or case in too_few:
            del traces
            continue
        if "T1P" in config.tasks:
            pair_t1_windows(case_windows, case)
        if not any(
            w["task"] == ("T1" if task == "T1P" else task)
            for task in config.tasks
            for w in case_windows
        ):
            raise ValueError(f"Case {case!r} produced no windows for any requested task.")
        serialized = serialize_windows(case_windows, traces, config.formats)
        if "T1P" in config.tasks:
            by_id = {trace.trace_id: trace for trace in traces}
            for window in serialized:
                if window["task"] == "T1":
                    window["traces"] = [by_id[trace_id] for trace_id in window["trace_ids"]]
            del by_id
        windows.extend({**window, "candidates": candidates} for window in serialized)
        del traces
    errors = []
    if missing_root:
        errors.append(f"Root-cause service not in candidates for cases: {missing_root!r}.")
    if too_few:
        errors.append(f"T2 needs at least 3 candidates for cases: {too_few!r}.")
    if errors:
        raise ValueError(" ".join(errors))
    for path in dict.fromkeys(
        (Path("data/windows/windows_v1.jsonl"), Path(windows_config.output_windows))
    ):
        check_saved_windows(windows, path, config.cases)
    return windows


def prompt_sha256(system: str, user: str) -> str:
    """Hash UTF-8 canonical JSON of the exact ordered system/user chat messages."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pair_t1_windows(windows: Sequence[dict], case: str) -> list[tuple[dict, dict]]:
    """Pair existing T1 pre/post windows by index; omit incomplete pairs."""
    phases = {"pre": {}, "post": {}}
    for window in windows:
        if window["case"] == case and window["task"] == "T1":
            phase = phases[window["phase"]]
            index = window["window_index"]
            assert index not in phase, f"Duplicate T1 {window['phase']} window {index} in {case!r}."
            phase[index] = window
    pairs = [
        (phases["pre"][index], phases["post"][index])
        for index in sorted(phases["pre"].keys() & phases["post"].keys())
    ]
    if not pairs:
        raise ValueError(f"Case {case!r} produced no complete T1P pairs.")
    return pairs


def build_call_list(windows: Sequence[dict], config: TaskConfig) -> list[dict]:
    calls = []
    for model in config.models:
        for case in config.cases:
            for task in config.tasks:
                task_windows = (
                    pair_t1_windows(windows, case)
                    if task == "T1P"
                    else [
                        (window, None)
                        for window in windows
                        if window["case"] == case and window["task"] == task
                    ]
                )
                for window, post in task_windows:
                    if task == "T1P":
                        aligned_pre, aligned_post = align_pair_times(
                            window["traces"], post["traces"]
                        )
                    for fmt in config.formats:
                        n = len(window["trace_ids"])
                        if task == "T1P":
                            assert window["case"] == post["case"] == case
                            assert window["window_index"] == post["window_index"]
                            assert fmt in window["texts"] and fmt in post["texts"]
                            assert n == len(post["trace_ids"])
                            a, b = (
                                (window, post)
                                if window["window_index"] % 2 == 0
                                else (post, window)
                            )
                            aligned_a, aligned_b = (
                                (aligned_pre, aligned_post)
                                if window["window_index"] % 2 == 0
                                else (aligned_post, aligned_pre)
                            )
                            user = build_t1p_prompt(
                                SERIALIZERS[fmt](aligned_a), SERIALIZERS[fmt](aligned_b), n
                            )
                            schema = t1p_schema()
                        else:
                            text = window["texts"][fmt]
                            user = (
                                build_t1_prompt(text, n)
                                if task == "T1"
                                else build_t2_prompt(text, n, window["candidates"])
                            )
                            schema = (
                                t1_schema() if task == "T1" else t2_schema(window["candidates"])
                            )
                        call = {
                            key: window[key]
                            for key in (
                                "case",
                                "dataset",
                                "system",
                                "fault",
                                "root_cause_service",
                                "task",
                                "window_index",
                                "informative",
                                "label",
                            )
                        }
                        call.update(
                            model=model,
                            format=fmt,
                            prompt_version=config.prompt_version,
                            seed=config.seed,
                            num_ctx=config.num_ctx,
                            num_predict=config.num_predict,
                            temperature=config.temperature,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=user,
                            prompt_sha256=prompt_sha256(SYSTEM_PROMPT, user),
                            prompt_chars=len(SYSTEM_PROMPT) + len(user),
                            output_schema=schema,
                        )
                        if task == "T1P":
                            call.update(
                                task="T1P",
                                label="B" if a["phase"] == "pre" else "A",
                                a_phase=a["phase"],
                                informative=None,
                                time_aligned=True,
                            )
                        if task == "T2":
                            call["candidates"] = window["candidates"]
                        calls.append(call)
    return calls


def record_key(record: Mapping) -> tuple:
    """Resume the same prompt and generation settings, including both T1 labels."""
    return tuple(
        record[key]
        for key in (
            "case",
            "task",
            "label",
            "window_index",
            "model",
            "format",
            "prompt_version",
            "seed",
            "num_ctx",
            "num_predict",
            "temperature",
            "prompt_sha256",
        )
    )


def make_record(
    call: Mapping,
    result: ChatResult,
    *,
    model_digest: str,
    ollama_version: str,
    think_sent: bool | None,
) -> dict:
    """Raw records omit prompts and raw traces, keeping reproducible prompt hashes."""
    return {
        **{
            key: value
            for key, value in call.items()
            if key not in {"system_prompt", "user_prompt", "output_schema"}
        },
        "model_digest": model_digest,
        "ollama_version": ollama_version,
        "think_sent": think_sent,
        "prompt_eval_count": result.prompt_eval_count,
        "prompt_eval_cached_count": result.prompt_eval_cached_count,
        "eval_count": result.eval_count,
        "done_reason": result.done_reason,
        "total_s": result.total_duration_ns / 1e9,
        "load_s": result.load_duration_ns / 1e9,
        "content": result.content,
        "status": call_status(result, call["num_ctx"]),
    }


def run_tasks(
    config: TaskConfig,
    client: OllamaClient,
    done: frozenset[tuple] = frozenset(),
) -> Iterator[dict]:
    """Preflight before generation, stream calls, and unload each model before the next."""
    version = client.version()
    digests = {model: client.model_digest(model) for model in config.models}
    capabilities = {model: client.model_capabilities(model) for model in config.models}
    windows = load_task_windows(config, load_window_config(config.windows_config))
    calls = build_call_list(windows, config)
    pending = [call for call in calls if record_key(call) not in done]
    for call in pending:
        check_prompt_length(call, config)
    for model in config.models:
        batch = [call for call in pending if call["model"] == model]
        if not batch:
            continue
        try:
            for call in batch:
                check_prompt_length(call, config)
                result = client.chat(
                    model=model,
                    system=call["system_prompt"],
                    user=call["user_prompt"],
                    num_ctx=config.num_ctx,
                    num_predict=config.num_predict,
                    temperature=config.temperature,
                    seed=config.seed,
                    think=False,
                    output_schema=call["output_schema"],
                )
                yield make_record(
                    call,
                    result,
                    model_digest=digests[model],
                    ollama_version=version,
                    think_sent=False if "thinking" in capabilities[model] else None,
                )
        finally:
            client.unload_model(model)


def read_records(path: Path, *, allow_incomplete_tail: bool = False) -> list[dict]:
    """A resume may recover one interrupted final line; never discard interior corruption."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_incomplete_tail and index == len(lines) - 1:
                break
            raise
    return records
