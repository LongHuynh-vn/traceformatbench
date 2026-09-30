"""Deterministic spec-007 summary/comparison tables and within-model Pareto panels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traceformatbench.analysis.baselines import MODEL as BASELINE
from traceformatbench.analysis.bootstrap import (
    SETS,
    check_paired,
    draw_resamples,
    in_set,
    paired_difference,
    pooled_statistic,
    verdict,
    window_key,
)

METRICS = {
    "T1": (("accuracy", "correct"),),
    "T2": (("acc_at_1", "hit_at_1"), ("acc_at_3", "hit_at_3")),
}
SUMMARY_COLUMNS = [
    "model",
    "format",
    "task",
    "set",
    "metric",
    "n_windows",
    "n_cases",
    "estimate",
    "ci_low",
    "ci_high",
    "mean_prompt_tokens",
    "mean_prompt_tokens_ci_low",
    "mean_prompt_tokens_ci_high",
    "tokens_reliable",
    "acc_per_1k_tokens",
    "mean_total_s",
    "invalid_rate",
    "truncated_rate",
    "post_rate",
    "tie_rate",
]
COMPARISON_COLUMNS = [
    "model",
    "format",
    "reference_model",
    "reference_format",
    "task",
    "set",
    "metric",
    "is_main",
    "n_windows",
    "n_cases",
    "estimate_diff",
    "ci_low",
    "ci_high",
    "verdict",
    "n_only_x_correct",
    "n_only_reference_correct",
]
STRATA_COLUMNS = [
    "set",
    "original_stratum",
    "original_n_cases",
    "merged_stratum",
    "merged_n_cases",
]


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    n_boot: int
    ci: float
    control: str
    min_stratum_cases: int
    scores: str | list[str]
    baseline: str | None
    output_dir: str

    def __post_init__(self) -> None:
        if self.n_boot < 1 or self.min_stratum_cases < 1 or not 0 < self.ci < 1:
            raise ValueError("Require n_boot >= 1, min_stratum_cases >= 1, and 0 < ci < 1.")


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    return AnalysisConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def group_records(records: Sequence[Mapping]) -> dict[tuple, list[Mapping]]:
    """Check uniqueness before computing anything, and completeness before resampling."""
    groups: dict[tuple, dict[tuple, Mapping]] = {}
    canonical: dict[tuple, Mapping] = {}
    for record in records:
        condition = (record["model"], record["format"], record["task"])
        key = window_key(record)
        group = groups.setdefault(condition, {})
        if key in group:
            raise ValueError(f"Duplicate window keys for {condition!r}: {key!r}.")
        group[key] = record
        if key in canonical:
            previous = canonical[key]
            for field in ("dataset", "system", "informative"):
                if previous[field] != record[field]:
                    raise ValueError(f"Inconsistent {field} for window key {key!r}.")
        canonical[key] = record
    for condition, group in sorted(groups.items()):
        task = condition[2]
        expected = {key for key in canonical if key[1] == task}
        missing = expected - group.keys()
        if missing:
            raise ValueError(f"Missing window keys for {condition!r}: {sorted(missing)!r}.")
    return {
        condition: [group[key] for key in sorted(group)]
        for condition, group in sorted(groups.items())
        if condition[2] in METRICS
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _descriptive(
    group: Sequence[Mapping], condition_records: Sequence[Mapping], resamples, ci: float
) -> dict:
    n = len(group)
    tokens = [
        r for r in group if r["status"] != "truncated" and r.get("prompt_eval_count") is not None
    ]
    token_stat = pooled_statistic(tokens, [r["prompt_eval_count"] for r in tokens], resamples, ci)
    llm = bool(condition_records) and condition_records[0]["model"] != BASELINE
    reliable = (
        not any(
            r["status"] == "truncated" or r.get("prompt_eval_cached_count", 0) > 0
            for r in condition_records
        )
        if llm
        else None
    )
    # The efficiency numerator uses exactly the non-truncated windows counted in its denominator.
    efficiency = float("nan")
    if tokens and tokens[0]["task"] == "T2" and token_stat.estimate > 0:
        efficiency = _mean([r["hit_at_1"] for r in tokens]) / (token_stat.estimate / 1000)
    return {
        "mean_prompt_tokens": token_stat.estimate,
        "mean_prompt_tokens_ci_low": token_stat.ci_low,
        "mean_prompt_tokens_ci_high": token_stat.ci_high,
        "tokens_reliable": reliable,
        "acc_per_1k_tokens": efficiency,
        "mean_total_s": _mean([r["total_s"] for r in group if r.get("total_s") is not None]),
        "invalid_rate": sum(r["parse_status"] != "ok" for r in group) / n if n else float("nan"),
        "truncated_rate": sum(r["status"] == "truncated" for r in group) / n if n else float("nan"),
        "post_rate": (
            sum(r["status"] == r["parse_status"] == "ok" and r["answer"] == "post" for r in group)
            / n
            if n and group[0]["task"] == "T1"
            else float("nan")
        ),
    }


def _comparison(
    condition,
    reference_condition,
    x,
    reference,
    metric,
    field,
    set_name,
    resamples,
    config,
    is_main,
) -> dict:
    stat = paired_difference(x, reference, field, resamples, config.ci)
    by_key = {window_key(r): r for r in reference}
    # For fractional baseline hits these are expected discordant counts under random tie-breaking.
    only_x = sum(r[field] * (1 - by_key[window_key(r)][field]) for r in x)
    only_ref = sum((1 - r[field]) * by_key[window_key(r)][field] for r in x)
    return {
        "model": condition[0],
        "format": condition[1],
        "reference_model": reference_condition[0],
        "reference_format": reference_condition[1],
        "task": condition[2],
        "set": set_name,
        "metric": metric,
        "is_main": is_main,
        "n_windows": len(x),
        "n_cases": len({r["case"] for r in x}),
        "estimate_diff": stat.estimate,
        "ci_low": stat.ci_low,
        "ci_high": stat.ci_high,
        "verdict": verdict(stat.ci_low, stat.ci_high),
        "n_only_x_correct": only_x,
        "n_only_reference_correct": only_ref,
    }


def analyze_records(records: Sequence[Mapping], config: AnalysisConfig) -> tuple[pd.DataFrame, ...]:
    """Use the same set-level case multiplicities for metrics, tokens, chance and differences."""
    groups = group_records(records)
    for condition, group in groups.items():
        if condition[0] != BASELINE and condition[1] != config.control:
            check_paired(group, groups.get((condition[0], config.control, condition[2]), []))
    canonical = {window_key(r): r for group in groups.values() for r in group}
    summary, comparisons, strata = [], [], []
    for set_name in SETS:
        resamples = draw_resamples(
            [canonical[key] for key in sorted(canonical)],
            set_name,
            config.seed,
            config.n_boot,
            config.min_stratum_cases,
        )
        strata.extend(resamples.strata)
        task = "T1" if set_name == "T1-all" else "T2"
        selected = {
            condition: [r for r in group if in_set(r, set_name)]
            for condition, group in groups.items()
            if condition[2] == task
        }
        for condition, group in selected.items():
            descriptive = _descriptive(group, groups[condition], resamples, config.ci)
            for metric, field in METRICS[task]:
                stat = pooled_statistic(group, [r[field] for r in group], resamples, config.ci)
                summary.append(
                    {
                        "model": condition[0],
                        "format": condition[1],
                        "task": task,
                        "set": set_name,
                        "metric": metric,
                        "n_windows": len(group),
                        "n_cases": len({r["case"] for r in group}),
                        "estimate": stat.estimate,
                        "ci_low": stat.ci_low,
                        "ci_high": stat.ci_high,
                        **descriptive,
                        "tie_rate": _mean([int(r[f"tie_at_{field[-1]}"]) for r in group])
                        if condition[0] == BASELINE
                        else float("nan"),
                    }
                )
                if condition[0] != BASELINE and condition[1] != config.control:
                    reference_condition = (condition[0], config.control, task)
                    comparisons.append(
                        _comparison(
                            condition,
                            reference_condition,
                            group,
                            selected.get(reference_condition, []),
                            metric,
                            field,
                            set_name,
                            resamples,
                            config,
                            set_name == "T2-informative" and metric == "acc_at_1",
                        )
                    )
            baseline_condition = (BASELINE, "-", "T2")
            if task == "T2" and condition[0] != BASELINE and baseline_condition in selected:
                comparisons.append(
                    _comparison(
                        condition,
                        baseline_condition,
                        group,
                        selected[baseline_condition],
                        "acc_at_1",
                        "hit_at_1",
                        set_name,
                        resamples,
                        config,
                        False,
                    )
                )
        if task == "T2":
            windows = [
                canonical[key] for key in sorted(canonical) if in_set(canonical[key], set_name)
            ]
            for metric, field in METRICS[task]:
                stat = pooled_statistic(
                    windows,
                    [r[field.replace("hit", "chance")] for r in windows],
                    resamples,
                    config.ci,
                )
                summary.append(
                    {
                        "model": "chance",
                        "format": "-",
                        "task": "T2",
                        "set": set_name,
                        "metric": metric,
                        "n_windows": len(windows),
                        "n_cases": len({r["case"] for r in windows}),
                        "estimate": stat.estimate,
                        "ci_low": stat.ci_low,
                        "ci_high": stat.ci_high,
                    }
                )
    return (
        pd.DataFrame(summary)
        .reindex(columns=SUMMARY_COLUMNS)
        .sort_values(["set", "model", "format", "task", "metric"], ignore_index=True),
        pd.DataFrame(comparisons)
        .reindex(columns=COMPARISON_COLUMNS)
        .sort_values(
            ["set", "model", "format", "reference_model", "reference_format", "metric"],
            ignore_index=True,
        ),
        pd.DataFrame(strata)
        .reindex(columns=STRATA_COLUMNS)
        .sort_values(["set", "original_stratum"], ignore_index=True),
    )


def pareto_optimal(tokens: Sequence[float], accuracy: Sequence[float]) -> np.ndarray:
    """Spec 007 uses strict improvement in BOTH coordinates to dominate a format."""
    x, y = np.asarray(tokens), np.asarray(accuracy)
    finite = np.isfinite(x) & np.isfinite(y)
    return finite & ~((x[:, None] > x[None, :]) & (y[:, None] < y[None, :]) & finite[None, :]).any(
        axis=1
    )


def plot_pareto(summary: pd.DataFrame, path: str | Path) -> None:
    # A noninteractive backend also supports subprocess tests and headless reproduction.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = summary[(summary["set"] == "T2-informative") & (summary["metric"] == "acc_at_1")]
    models = sorted(set(primary["model"]) - {BASELINE, "chance"})
    fig, axes = plt.subplots(
        1, max(1, len(models)), figsize=(6 * max(1, len(models)), 5), squeeze=False
    )
    if not models:
        axes[0, 0].text(0.5, 0.5, "No T2-informative model calls", ha="center", va="center")
    for ax, model in zip(axes[0], models, strict=bool(models)):
        group = primary[primary["model"] == model]
        optimal = pareto_optimal(group["mean_prompt_tokens"], group["estimate"])
        for is_optimal, (_, row) in zip(optimal, group.iterrows(), strict=True):
            x, y = row["mean_prompt_tokens"], row["estimate"]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            ax.scatter(
                x,
                y,
                marker="*" if is_optimal else "o",
                s=120 if is_optimal else 60,
                color="tab:blue" if row["tokens_reliable"] else "tab:red",
            )
            if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
                ax.vlines(x, row["ci_low"], row["ci_high"], color="0.4")
            ax.annotate(
                row["format"] + (" !" if not row["tokens_reliable"] else ""),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
            )
        for name, style in ((BASELINE, "--"), ("chance", ":")):
            line = primary[primary["model"] == name]
            if not line.empty and np.isfinite(line.iloc[0]["estimate"]):
                ax.axhline(line.iloc[0]["estimate"], linestyle=style, color="0.5", label=name)
        ax.set(
            title=model,
            xlabel="Mean prompt tokens (model's tokenizer)",
            ylabel="T2-informative Acc@1",
            ylim=(-0.05, 1.05),
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.text(
        0.5,
        0.02,
        "Stars: Pareto-optimal formats. Red / !: truncated or cached prompt tokens.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(config: AnalysisConfig) -> tuple[pd.DataFrame, ...]:
    paths = [config.scores] if isinstance(config.scores, str) else list(config.scores)
    if config.baseline is not None:
        paths.append(config.baseline)
    records = []
    for path in paths:
        records.extend(
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        )
    summary, comparisons, strata = analyze_records(records, config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for table, name in ((summary, "summary"), (comparisons, "comparisons"), (strata, "strata")):
        table.to_csv(output / f"{name}.csv", index=False, na_rep="NaN", lineterminator="\n")
    plot_pareto(summary, output / "pareto_t2_informative.png")
    print(
        f"Main comparisons: {int(comparisons['is_main'].sum())}; "
        "95% CIs without multiplicity correction."
    )
    print(
        "Limitation: under the null, about 0.5 false clear differences are expected among "
        "10 comparisons at 95% (at least one with about 40% probability if independent)."
    )
    print(
        "F4/F5 expose self-time directly; comparison with self-time mainly tests "
        "reading the supplied table."
    )
    print(f"Analysis outputs: {output}")
    return summary, comparisons, strata
