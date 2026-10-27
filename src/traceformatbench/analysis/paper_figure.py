"""Validated, fixed-size paper figure from saved analysis tables (spec 011)."""

from __future__ import annotations

import hashlib
import json
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import blended_transform_factory


@dataclass(frozen=True)
class PaperFigureConfig:
    comparisons: str
    summary: str
    output_pdf: str
    output_png: str
    provenance: str
    data: dict
    font: dict
    style: dict
    metadata: dict


def load_paper_figure_config(path: str | Path) -> PaperFigureConfig:
    """Load the complete presentation contract; paths are relative to the working directory."""
    return PaperFigureConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _booleans(series: pd.Series, name: str) -> pd.Series:
    # Do not use astype(bool): it turns strings such as "False" into True.
    values = series.astype(str)
    if not values.isin(["True", "False"]).all():
        raise ValueError(f"{name} must contain only True or False.")
    return values.eq("True")


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _finite(table: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} must be finite numeric values.")
        result[column] = values
    return result


def paper_figure_data(
    comparisons: pd.DataFrame, summary: pd.DataFrame, config: PaperFigureConfig
) -> pd.DataFrame:
    """Validate the ten primary comparisons and join exactly one token row per operand."""
    expected = config.data
    keys = ["model", "format"]
    scope = ["task", "set", "metric"]
    counts = ["n_windows", "n_cases"]
    numeric = ["estimate_diff", "ci_low", "ci_high"]
    _require_columns(
        comparisons,
        keys
        + scope
        + counts
        + numeric
        + ["is_main", "reference_model", "reference_format", "verdict"],
        "comparisons",
    )
    _require_columns(
        summary, keys + scope + counts + ["mean_prompt_tokens", "tokens_reliable"], "summary"
    )
    rows = comparisons.loc[_booleans(comparisons["is_main"], "is_main")].copy()
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate primary model/format rows.")
    wanted = [(model, fmt) for model in expected["models"] for fmt in expected["formats"]]
    if len(wanted) != 10 or len(set(wanted)) != 10:
        raise ValueError("The paper configuration must describe ten distinct comparisons.")
    if set(rows[keys].itertuples(index=False, name=None)) != set(wanted):
        raise ValueError("Primary rows must match exactly the ten configured model/format pairs.")
    for column in scope + ["reference_format"]:
        if not rows[column].eq(expected[column]).all():
            raise ValueError(f"Primary rows have unexpected {column}.")
    if not rows["reference_model"].eq(rows["model"]).all():
        raise ValueError("Primary comparisons must be within-model.")
    rows = _finite(rows, numeric + counts, "comparisons")
    for column in counts:
        if not rows[column].eq(expected[column]).all():
            raise ValueError(f"Primary rows have inconsistent {column}.")
    if not ((rows.ci_low <= rows.estimate_diff) & (rows.estimate_diff <= rows.ci_high)).all():
        raise ValueError("Primary intervals must be ordered and contain the estimate.")
    if not rows["verdict"].isin(["clear difference", "no clear difference"]).all():
        raise ValueError("Unsupported primary verdict (including not enough cases).")
    clear = (rows.ci_low > 0) | (rows.ci_high < 0)
    if not rows["verdict"].eq("clear difference").eq(clear).all():
        raise ValueError("Primary verdict is inconsistent with the interval's exclusion of zero.")
    xmin, xmax = config.style["xlim"]
    if (rows.ci_low < xmin).any() or (rows.ci_high > xmax).any():
        raise ValueError("Primary intervals would be clipped by the configured x limits.")

    selected = summary["model"].isin(expected["models"])
    for column in scope:
        selected &= summary[column].eq(expected[column])
    tokens = summary.loc[selected].copy()
    token_keys = set(wanted) | {
        (model, expected["reference_format"]) for model in expected["models"]
    }
    if tokens.duplicated(keys).any():
        raise ValueError("Duplicate matching summary model/format rows.")
    if set(tokens[keys].itertuples(index=False, name=None)) != token_keys:
        raise ValueError("Summary must match exactly the configured primary formats and F0.")
    tokens = _finite(tokens, counts + ["mean_prompt_tokens"], "summary")
    for column in counts:
        if not tokens[column].eq(expected[column]).all():
            raise ValueError(f"Summary has inconsistent {column}.")
    if not tokens["mean_prompt_tokens"].gt(0).all():
        raise ValueError("Summary mean_prompt_tokens must be positive.")
    tokens["tokens_reliable"] = _booleans(tokens["tokens_reliable"], "tokens_reliable")
    tokens = tokens.set_index(keys)
    rows = rows.set_index(keys).loc[wanted].reset_index()
    for column in ("mean_prompt_tokens", "tokens_reliable"):
        rows[column] = [tokens.loc[key, column] for key in wanted]
        rows[f"reference_{column}"] = [
            tokens.loc[(model, expected["reference_format"]), column] for model, _ in wanted
        ]
    rows["token_ratio"] = rows.reference_mean_prompt_tokens / rows.mean_prompt_tokens
    if not np.isfinite(rows.token_ratio).all():
        raise ValueError("Token ratios must be finite.")
    rows["ratio_reliable"] = rows.tokens_reliable & rows.reference_tokens_reliable
    # Keep only validated presentation fields: unrelated CSV annotations may be blank.
    return rows[
        keys
        + scope
        + counts
        + numeric
        + [
            "reference_model",
            "reference_format",
            "verdict",
            "mean_prompt_tokens",
            "tokens_reliable",
            "reference_mean_prompt_tokens",
            "reference_tokens_reliable",
            "token_ratio",
            "ratio_reliable",
        ]
    ]


def _font_path(config: PaperFigureConfig) -> Path:
    path = Path(mpl.get_data_path()) / "fonts" / "ttf" / config.font["bundled_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != config.font["sha256"]:
        raise ValueError("Configured font SHA-256 does not match the bundled font.")
    return path


@contextmanager
def _style(config: PaperFigureConfig):
    # rc_context alone inherits ambient settings; reset every built-in default first.
    defaults = {key: value for key, value in mpl.rcParamsDefault.items() if key != "backend"}
    with mpl.rc_context(rc=defaults):
        mpl.rcParams.update(config.style["rc"])
        yield


def build_paper_figure(rows: pd.DataFrame, config: PaperFigureConfig) -> Figure:
    """Build artists from validated rows; export_paper_figure handles validation and saving."""
    style = config.style
    font = FontProperties(fname=_font_path(config), size=style["font_size"])
    with _style(config):
        fig = Figure(figsize=style["size_inches"], dpi=style["dpi"])
        FigureCanvasAgg(fig)
        ax = fig.add_axes(style["axes_bounds"])
        formats_per_model = len(config.data["formats"])
        positions = []
        labels = []
        for index, row in rows.iterrows():
            model_index = index // formats_per_model
            y = index + model_index * style["model_gap"]
            positions.append(y)
            labels.append(f"{style['row_labels'][model_index]} {row['format']}")
            color = style["colors"][model_index]
            interval = ax.errorbar(
                row.estimate_diff,
                y,
                xerr=[[row.estimate_diff - row.ci_low], [row.ci_high - row.estimate_diff]],
                fmt=style["markers"][model_index],
                color=color,
                markerfacecolor=color if row.verdict == "clear difference" else "white",
                markeredgewidth=style["marker_edge_width"],
                markersize=style["marker_size"],
                elinewidth=style["ci_width"],
                capsize=style["cap_size"],
                capthick=style["ci_width"],
                linestyle="none",
                zorder=3,
            )
            # errorbar reconstructs endpoints from distances, which can round by an ULP.
            # Keep the saved CSV endpoints exactly in both cap and interval artists.
            _, caps, bars = interval.lines
            caps[0].set_xdata([row.ci_low])
            caps[1].set_xdata([row.ci_high])
            # Matplotlib's markeredgewidth overrides capthick in errorbar().
            for cap in caps:
                cap.set_markeredgewidth(style["ci_width"])
            bars[0].set_segments([[(row.ci_low, y), (row.ci_high, y)]])
            ratio = row.token_ratio
            label = f"{ratio:.0f}" if ratio >= style["ratio_integer_threshold"] else f"{ratio:.1f}"
            label += style["ratio_suffix"]
            if not row.ratio_reliable:
                label += style["unreliable_suffix"]
            # Figure x coordinates keep the text column independent of the numeric axis.
            ax.text(
                style["token_column_x"],
                y,
                label,
                transform=blended_transform_factory(fig.transFigure, ax.transData),
                fontproperties=font,
                color=style["text_color"],
                ha="center",
                va="center",
                clip_on=False,
            )
        ax.axvline(
            0,
            color=style["zero_color"],
            linewidth=style["zero_width"],
            linestyle=(0, style["zero_dashes"]),
            zorder=1,
        )
        ax.set_xlim(style["xlim"])
        ax.set_ylim(positions[-1] + style["row_margin"], -style["row_margin"])
        ax.set_xticks(style["xticks"], style["xticklabels"], fontproperties=font)
        ax.set_yticks(positions, labels, fontproperties=font)
        ax.tick_params(colors=style["text_color"])
        ax.set_xlabel(
            style["xlabel"],
            fontproperties=font,
            color=style["text_color"],
            labelpad=style["xlabel_pad"],
        )
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        fig.text(
            style["token_column_x"],
            style["token_heading_y"],
            style["token_heading"],
            fontproperties=font,
            color=style["text_color"],
            ha="center",
            va="top",
            linespacing=style["token_linespacing"],
        )
        return fig


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_paper_figure(config_path: str | Path) -> dict:
    """Validate, render PDF/PNG, and record hashes without timestamps or local paths."""
    config = load_paper_figure_config(config_path)
    # Round-trip parsing preserves CSV floats exactly in the plotted coordinates.
    rows = paper_figure_data(
        pd.read_csv(config.comparisons, float_precision="round_trip"),
        pd.read_csv(config.summary, float_precision="round_trip"),
        config,
    )
    with _style(config):
        fig = build_paper_figure(rows, config)
        try:
            for path in (config.output_pdf, config.output_png, config.provenance):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(config.output_pdf, format="pdf", metadata=config.metadata)
            fig.savefig(
                config.output_png,
                format="png",
                dpi=config.style["dpi"],
                metadata={"Software": "TraceFormatBench paper figure"},
            )
        finally:
            fig.clear()
    from matplotlib import ft2font

    provenance = {
        "schema_version": 1,
        "inputs": {
            "comparisons_sha256": _digest(config.comparisons),
            "summary_sha256": _digest(config.summary),
            "config_sha256": _digest(config_path),
        },
        "font": {"file": config.font["bundled_file"], "sha256": _digest(_font_path(config))},
        "renderer": {
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "freetype": ft2font.__freetype_version__,
        },
        "outputs": {
            "pdf_sha256": _digest(config.output_pdf),
            "png_sha256": _digest(config.output_png),
        },
        "rows": rows.to_dict(orient="records"),
    }
    Path(config.provenance).write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return provenance
