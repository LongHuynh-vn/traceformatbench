"""Forest plot of the pre-registered main comparisons, drawn from the saved comparisons table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INK = "#1f2933"
MUTED = "#7b8794"


@dataclass(frozen=True)
class FigureConfig:
    comparisons: str
    output: str


def load_figure_config(path: str | Path) -> FigureConfig:
    return FigureConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def main_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """The rows flagged is_main, which must share one task, set, metric and reference format."""
    main = comparisons[comparisons["is_main"].astype(str) == "True"]
    if main.empty:
        raise ValueError("No main comparisons (is_main) in the comparisons table.")
    for column in ("task", "set", "metric", "reference_format"):
        if main[column].nunique() != 1:
            raise ValueError(f"Main comparisons mix several values of {column!r}.")
    if (main["model"] != main["reference_model"]).any():
        raise ValueError("Main comparisons must be within-model.")
    if main.duplicated(["model", "format"]).any():
        raise ValueError("Duplicate model and format in the main comparisons.")
    return main.sort_values(["model", "format"], ignore_index=True)


def plot_main_comparisons(comparisons: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    main = main_comparisons(comparisons)
    first = main.iloc[0]
    models = sorted(main["model"].unique())
    formats = sorted(main["format"].unique())
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.2 * len(models), 0.45 * len(formats) + 2.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes[0], models, strict=True):
        rows = main[main["model"] == model].set_index("format")
        for y, fmt in enumerate(formats):
            if fmt not in rows.index:
                continue
            row = rows.loc[fmt]
            if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
                ax.hlines(y, row["ci_low"], row["ci_high"], color=INK, linewidth=2)
            clear = row["verdict"] == "clear difference"
            ax.plot(
                row["estimate_diff"],
                y,
                marker="o",
                markersize=8,
                color=INK,
                markerfacecolor=INK if clear else "white",
                markeredgewidth=2,
                linestyle="none",
            )
        ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#e4e7eb", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        n_windows, n_cases = rows["n_windows"].iloc[0], rows["n_cases"].iloc[0]
        ax.set_title(f"{model}  (n = {n_windows} windows, {n_cases} cases)", fontsize=10)
        ax.set_xlabel(f"{first['metric']} difference vs {first['reference_format']}")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_yticks(
        range(len(formats)), [f"{f} - {first['reference_format']}" for f in formats]
    )
    axes[0, 0].invert_yaxis()
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            markerfacecolor=fill,
            markeredgewidth=2,
            linestyle="none",
            label=label,
        )
        for fill, label in ((INK, "clear difference"), ("white", "no clear difference"))
    ]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncols=2, fontsize=9)
    fig.text(
        0.5,
        0.01,
        f"{first['set']}. Points: paired difference; lines: 95% stratified case-"
        "bootstrap CI, no multiplicity correction.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)
