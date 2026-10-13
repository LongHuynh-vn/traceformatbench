"""Summed per-call model time per model, in hours, from the saved calls and the summary table.

The sum of ``total_s`` (server-side time of each call) is not wall-clock time: the main grid was
stopped and resumed, and the time between calls is not counted.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

# Summary rows that hold one mean per model, format and task over all windows of the task.
SUMMARY_ROWS = {("T1", "T1-all", "accuracy"), ("T2", "T2-all", "acc_at_1")}


@dataclass(frozen=True)
class ModelTimeConfig:
    calls: str
    summary: str
    tolerance_h: float


@dataclass(frozen=True)
class ModelHours:
    model: str
    hours_from_calls: float
    hours_from_summary: float
    agree: bool


def load_model_time_config(path: str | Path) -> ModelTimeConfig:
    return ModelTimeConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def hours_from_calls(path: str | Path) -> dict[str, float]:
    """Sum of total_s over every saved call, per model, in hours."""
    seconds: dict[str, float] = defaultdict(float)
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                record = json.loads(line)
                seconds[record["model"]] += float(record["total_s"])
    return {model: total / 3600 for model, total in seconds.items()}


def hours_from_summary(path: str | Path, models: list[str]) -> dict[str, float]:
    """Sum of mean_total_s x n_windows over the all-window T1 and T2 Acc@1 rows, per model, in h."""
    seconds = dict.fromkeys(models, 0.0)
    with Path(path).open(newline="", encoding="utf-8") as table:
        for row in csv.DictReader(table):
            key = (row["task"], row["set"], row["metric"])
            if row["model"] in seconds and key in SUMMARY_ROWS:
                seconds[row["model"]] += float(row["mean_total_s"]) * int(row["n_windows"])
    return {model: total / 3600 for model, total in seconds.items()}


def model_hours(config: ModelTimeConfig) -> list[ModelHours]:
    """Both sums per model, sorted by model, and whether they agree within the tolerance."""
    from_calls = hours_from_calls(config.calls)
    models = sorted(from_calls)
    from_summary = hours_from_summary(config.summary, models)
    return [
        ModelHours(
            model=model,
            hours_from_calls=from_calls[model],
            hours_from_summary=from_summary[model],
            agree=abs(from_calls[model] - from_summary[model]) <= config.tolerance_h,
        )
        for model in models
    ]
