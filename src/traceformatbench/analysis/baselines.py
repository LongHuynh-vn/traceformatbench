"""Total self-time baseline; real windows_v1 evaluation is subject to the ORDER RULE."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from traceformatbench.data.trace import Trace, load_case_traces
from traceformatbench.data.windows import WindowConfig
from traceformatbench.serializers.structure import self_time_us
from traceformatbench.tasks.prompts import case_candidates

MODEL = "heuristic-selftime"


def expected_hit(scores: Mapping[str, int], label: str, k: int) -> float:
    """Credit a tied label by the fraction of its group's slots inside the top k."""
    if label not in scores:
        return 0.0
    score = scores[label]
    above = sum(value > score for value in scores.values())
    tied = sum(value == score for value in scores.values())
    return max(0.0, min(1.0, (k - above) / tied))


def tie_at_cut(scores: Mapping[str, int], k: int) -> bool:
    ordered = sorted(scores.values(), reverse=True)
    return 0 < k < len(ordered) and ordered[k - 1] == ordered[k]


def score_window(
    window: Mapping, traces: Sequence[Trace], candidates: Sequence[str], seed: int
) -> dict:
    """Use all case candidates, including those absent from these window traces."""
    if window["task"] != "T2" or len(candidates) < 3 or len(set(candidates)) != len(candidates):
        raise ValueError(
            "The self-time baseline needs a T2 window and at least 3 unique candidates."
        )
    start = time.perf_counter()
    scores = dict.fromkeys(candidates, 0)
    for trace in traces:
        for span in trace.spans:
            scores[span.service] += self_time_us(trace, span)
    ranking = sorted(candidates, key=lambda service: (-scores[service], service))[:3]
    tie1, tie3 = tie_at_cut(scores, 1), tie_at_cut(scores, 3)
    return {
        **window,
        "model": MODEL,
        "format": "-",
        "seed": seed,
        "status": "ok",
        "parse_status": "ok",
        "fenced": False,
        "content": json.dumps({"ranking": ranking}),
        "answer": ranking,
        "candidates": list(candidates),
        "n_candidates": len(candidates),
        "hit_at_1": expected_hit(scores, window["label"], 1),
        "hit_at_3": expected_hit(scores, window["label"], 3),
        "chance_at_1": 1 / len(candidates),
        "chance_at_3": 3 / len(candidates),
        "tie_at_1": tie1,
        "tie_at_3": tie3,
        "prompt_eval_count": None,
        "prompt_eval_cached_count": 0,
        "total_s": time.perf_counter() - start,
    }


def run_baseline(config: WindowConfig) -> Iterator[dict]:
    """Score saved fixed windows, loading and releasing one local case at a time."""
    path = Path(config.output_windows)
    windows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_case: dict[str, list[dict]] = {}
    for window in windows:
        if window["task"] == "T2":
            by_case.setdefault(window["case"], []).append(window)
    for case, case_windows in sorted(by_case.items()):
        traces = load_case_traces(Path(config.data_dir) / case)
        candidates = case_candidates(traces)
        by_id = {trace.trace_id: trace for trace in traces}
        for window in sorted(case_windows, key=lambda r: (r["label"], r["window_index"])):
            yield score_window(
                window,
                [by_id[trace_id] for trace_id in window["trace_ids"]],
                candidates,
                config.seed,
            )
        del by_id, traces
