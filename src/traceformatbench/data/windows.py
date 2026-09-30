"""Fixed trace windows shared by every experiment, plus per-case size-cap statistics (spec 003).

A window is `window_size` consecutive eligible traces of one phase (pre or post fault injection),
ordered by start time. Each case gets its own RNG seeded with f"{seed}:{case}", so adding or
removing a case never changes the windows of any other case.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from traceformatbench.data.rcaeval import download_traced_cases
from traceformatbench.data.trace import Trace, load_case_traces
from traceformatbench.serializers import serialize_f0
from traceformatbench.tasks.prompts import case_candidates

Phase = Literal["pre", "post"]
PHASES: tuple[Phase, ...] = ("pre", "post")


@dataclass(frozen=True)
class WindowConfig:
    name: str
    seed: int
    data_dir: str
    datasets: list[str]
    repetitions: list[int] | None
    window_size: int
    max_trace_chars: int
    guard_seconds: int
    t1_windows_per_phase: int
    t2_windows: int
    output_windows: str
    output_stats: str


def load_window_config(path: str | Path) -> WindowConfig:
    """Read a window config; its file stem becomes `config_name` in every record."""
    path = Path(path)
    return WindowConfig(name=path.stem, **yaml.safe_load(path.read_text()))


def trace_size_chars(trace: Trace) -> int:
    """Size of a trace in F0 characters, the largest format."""
    return len(serialize_f0([trace]))


def phase_of(trace: Trace, inject_time_s: int, guard_seconds: int) -> Phase | None:
    """Assign a trace to pre/post by its start time; traces inside the guard band get None."""
    if trace.start_us < (inject_time_s - guard_seconds) * 1_000_000:
        return "pre"
    if trace.start_us >= (inject_time_s + guard_seconds) * 1_000_000:
        return "post"
    return None


def eligible(trace: Trace, max_trace_chars: int) -> bool:
    """Has a root span, no orphan spans, and fits the per-trace size cap."""
    return (
        trace.root is not None
        and not trace.orphan_span_ids
        and trace_size_chars(trace) <= max_trace_chars
    )


def sample_windows(
    traces: Sequence[Trace], phase: Phase, window_size: int, n: int, rng: random.Random
) -> list[list[Trace]]:
    """Draw up to `n` distinct windows of consecutive traces, returned in start-time order.

    `traces` are the eligible traces of `phase`. Windows do not overlap when `n` of them fit;
    otherwise they may overlap but are still distinct, so fewer than `n` come back when fewer
    distinct windows exist (none when there are fewer than `window_size` traces).
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}.")
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}.")
    ordered = sorted(traces, key=lambda trace: (trace.start_us, trace.trace_id))
    if len({trace.trace_id for trace in ordered}) != len(ordered):
        raise ValueError(f"Duplicate trace IDs in the {phase} traces.")

    n_positions = len(ordered) - window_size + 1
    if n <= 0 or n_positions <= 0:
        return []
    if n * window_size <= len(ordered):
        # Choosing n non-overlapping windows is choosing n of (slack + n) slots, where slack is
        # the number of traces left outside all windows; this is uniform over all placements.
        slack = len(ordered) - n * window_size
        slots = sorted(rng.sample(range(slack + n), n))
        starts = [slot + i * (window_size - 1) for i, slot in enumerate(slots)]
    else:
        starts = sorted(rng.sample(range(n_positions), min(n, n_positions)))
    return [ordered[start : start + window_size] for start in starts]


def _window_record(
    case_meta: Mapping,
    config: WindowConfig,
    task: Literal["T1", "T2"],
    phase: Phase,
    window_index: int,
    window: list[Trace],
) -> dict:
    root_cause = str(case_meta["root_cause_service"])
    return {
        "case": str(case_meta["case"]),
        "dataset": str(case_meta["dataset"]),
        "system": str(case_meta["system"]),
        "fault": str(case_meta["fault"]),
        "root_cause_service": root_cause,
        "task": task,
        "phase": phase,
        "label": phase if task == "T1" else root_cause,
        "window_index": window_index,
        "trace_ids": [trace.trace_id for trace in window],
        "n_spans": sum(len(trace.spans) for trace in window),
        "f0_chars": len(serialize_f0(window)),
        "informative": (
            any(root_cause in trace.services for trace in window) if task == "T2" else None
        ),
        "config_name": config.name,
    }


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_case_windows(
    case_meta: Mapping, traces: Sequence[Trace], config: WindowConfig, rng: random.Random
) -> tuple[list[dict], dict]:
    """Window records (T1 pre/post, then T2 post) and one stats row for a single case."""
    case = str(case_meta["case"])
    if not traces:
        raise ValueError(f"Case {case!r} has no traces.")
    inject_time_s = int(case_meta["inject_time"])
    root_cause = str(case_meta["root_cause_service"])

    by_phase: dict[Phase, list[Trace]] = {"pre": [], "post": []}
    eligible_by_phase: dict[Phase, list[Trace]] = {"pre": [], "post": []}
    n_eligible = 0
    for trace in traces:
        is_eligible = eligible(trace, config.max_trace_chars)
        n_eligible += is_eligible
        phase = phase_of(trace, inject_time_s, config.guard_seconds)
        if phase is None:
            continue
        by_phase[phase].append(trace)
        if is_eligible:
            eligible_by_phase[phase].append(trace)

    t1 = {
        phase: sample_windows(
            eligible_by_phase[phase], phase, config.window_size, config.t1_windows_per_phase, rng
        )
        for phase in PHASES
    }
    # Keep T1 labels 50/50 per case: trim the phase with more windows to a random subset.
    matched = min(len(windows) for windows in t1.values())
    for phase in PHASES:
        if len(t1[phase]) > matched:
            keep = sorted(rng.sample(range(len(t1[phase])), matched))
            t1[phase] = [t1[phase][i] for i in keep]
    # T2 is answerable only if the root cause is a T2 candidate (a service seen in the case's
    # traces); otherwise the case gets no T2 windows (spec 003 amendment, 2026-09-30).
    # T2 sampling is the last use of the per-case RNG, so skipping it leaves T1 unchanged.
    t2_answerable = root_cause in case_candidates(traces)
    t2 = (
        sample_windows(
            eligible_by_phase["post"], "post", config.window_size, config.t2_windows, rng
        )
        if t2_answerable
        else []
    )

    records = [
        _window_record(case_meta, config, "T1", phase, index, window)
        for phase in PHASES
        for index, window in enumerate(t1[phase])
    ]
    records += [
        _window_record(case_meta, config, "T2", "post", index, window)
        for index, window in enumerate(t2)
    ]

    def via_root_cause(group: list[Trace]) -> int:
        return sum(root_cause in trace.services for trace in group)

    stats = {
        "case": case,
        "dataset": str(case_meta["dataset"]),
        "n_traces": len(traces),
        "n_eligible": n_eligible,
        "share_eligible": n_eligible / len(traces),
        "n_eligible_pre": len(eligible_by_phase["pre"]),
        "n_eligible_post": len(eligible_by_phase["post"]),
        "share_post_via_root_cause_all": _share(
            via_root_cause(by_phase["post"]), len(by_phase["post"])
        ),
        "share_post_via_root_cause_eligible": _share(
            via_root_cause(eligible_by_phase["post"]), len(eligible_by_phase["post"])
        ),
        "n_t1_windows": 2 * matched,
        "t2_answerable": t2_answerable,
        "n_t2_windows": len(t2),
        "n_t2_informative": sum(
            record["informative"] for record in records if record["task"] == "T2"
        ),
    }
    return records, stats


def select_cases(case_index: pd.DataFrame, config: WindowConfig) -> pd.DataFrame:
    """Traced cases matching the config's datasets and repetitions, sorted by case name."""
    selected = case_index[
        case_index["has_traces"].astype(bool) & case_index["dataset"].isin(config.datasets)
    ]
    if config.repetitions is not None:
        selected = selected[selected["repetition"].isin(config.repetitions)]
    return selected.sort_values("case").reset_index(drop=True)


def build_windows(
    config: WindowConfig, case_index: pd.DataFrame
) -> tuple[list[dict], pd.DataFrame]:
    """Download missing cases and build windows one case at a time (a TT case needs ~1.8 GB)."""
    records: list[dict] = []
    stats_rows: list[dict] = []
    for case_meta in select_cases(case_index, config).to_dict(orient="records"):
        case = str(case_meta["case"])
        (case_dir,) = download_traced_cases([case], config.data_dir)
        traces = load_case_traces(case_dir)
        rng = random.Random(f"{config.seed}:{case}")
        case_records, case_stats = build_case_windows(case_meta, traces, config, rng)
        records += case_records
        stats_rows.append(case_stats)
        del traces
    return records, pd.DataFrame(stats_rows)
