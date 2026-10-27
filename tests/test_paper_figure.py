"""The paper figure is a strict presentation of the saved primary comparison table."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.text import Text

from traceformatbench.analysis.paper_figure import (
    build_paper_figure,
    export_paper_figure,
    load_paper_figure_config,
    paper_figure_data,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/paper_figure_v1.yaml"


@pytest.fixture
def inputs():
    config = load_paper_figure_config(CONFIG)
    comparisons = pd.read_csv(ROOT / config.comparisons, float_precision="round_trip")
    summary = pd.read_csv(ROOT / config.summary, float_precision="round_trip")
    return comparisons, summary, config


def test_exact_rows_ratios_and_source_reliability(inputs):
    comparisons, summary, config = inputs
    rows = paper_figure_data(comparisons.sample(frac=1, random_state=11), summary, config)
    assert list(rows[["model", "format"]].itertuples(index=False, name=None)) == [
        (model, fmt) for model in config.data["models"] for fmt in config.data["formats"]
    ]
    for row in rows.itertuples():
        source = comparisons[
            comparisons.is_main
            & comparisons.model.eq(row.model)
            & comparisons.format.eq(row.format)
        ].iloc[0]
        for column in ("estimate_diff", "ci_low", "ci_high"):
            assert getattr(row, column) == source[column]
        selected = summary[
            summary.model.eq(row.model)
            & summary["set"].eq("T2-informative")
            & summary.task.eq("T2")
            & summary.metric.eq("acc_at_1")
        ]
        f0 = selected[selected.format.eq("F0")].iloc[0]
        fmt = selected[selected.format.eq(row.format)].iloc[0]
        assert row.token_ratio == f0.mean_prompt_tokens / fmt.mean_prompt_tokens
        assert row.tokens_reliable == fmt.tokens_reliable
        assert row.reference_tokens_reliable == f0.tokens_reliable
        assert row.ratio_reliable == bool(f0.tokens_reliable and fmt.tokens_reliable)
    assert rows.verdict.eq("clear difference").sum() == 1


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("model", "unknown", "exactly"),
        ("format", "F6", "exactly"),
        ("reference_model", "wrong", "within-model"),
        ("reference_format", "F1", "reference_format"),
        ("task", "T1", "task"),
        ("set", "T2-all", "set"),
        ("metric", "acc_at_3", "metric"),
        ("n_windows", 192, "n_windows"),
        ("n_cases", 53, "n_cases"),
        ("n_windows", np.nan, "finite"),
        ("estimate_diff", np.nan, "finite"),
        ("ci_low", -np.inf, "finite"),
        ("ci_high", np.inf, "finite"),
        ("ci_low", 0.2, "ordered"),
        ("ci_high", -0.2, "ordered"),
        ("estimate_diff", 0.2, "ordered"),
        ("ci_low", -0.22, "clipped"),
        ("verdict", "not enough cases", "Unsupported"),
        ("verdict", "significant", "Unsupported"),
        ("verdict", "clear difference", "inconsistent"),
        ("is_main", "yes", "True or False"),
    ],
)
def test_bad_comparisons_rejected(inputs, column, value, message):
    comparisons, summary, config = inputs
    index = comparisons.index[comparisons.is_main][0]
    comparisons[column] = comparisons[column].astype(object)
    comparisons.loc[index, column] = value
    with pytest.raises(ValueError, match=message):
        paper_figure_data(comparisons, summary, config)


@pytest.mark.parametrize(
    "change,message",
    [("missing", "exactly"), ("duplicate", "Duplicate"), ("column", "missing columns")],
)
def test_missing_duplicate_comparisons(inputs, change, message):
    comparisons, summary, config = inputs
    index = comparisons.index[comparisons.is_main][0]
    if change == "missing":
        comparisons = comparisons.drop(index)
    elif change == "duplicate":
        comparisons = pd.concat([comparisons, comparisons.loc[[index]]])
    else:
        comparisons = comparisons.drop(columns="ci_low")
    with pytest.raises(ValueError, match=message):
        paper_figure_data(comparisons, summary, config)


@pytest.mark.parametrize(
    "change,message",
    [
        ("missing", "exactly"),
        ("duplicate", "Duplicate"),
        ("wrong_task", "exactly"),
        ("wrong_metric", "exactly"),
        ("wrong_set", "exactly"),
        ("extra_format", "exactly"),
        ("counts", "n_windows"),
        ("cases", "n_cases"),
        ("nan", "finite"),
        ("infinity", "finite"),
        ("zero", "positive"),
        ("negative", "positive"),
        ("flag", "True or False"),
    ],
)
def test_bad_summary_rejected(inputs, change, message):
    comparisons, summary, config = inputs
    index = summary.index[
        summary.model.eq("qwen3.5:9b")
        & summary.format.eq("F0")
        & summary["set"].eq("T2-informative")
        & summary.metric.eq("acc_at_1")
    ][0]
    if change == "missing":
        summary = summary.drop(index)
    elif change == "duplicate":
        summary = pd.concat([summary, summary.loc[[index]]])
    elif change == "extra_format":
        extra = summary.loc[[index]].copy()
        extra["format"] = "F6"
        summary = pd.concat([summary, extra])
    else:
        column, value = {
            "wrong_task": ("task", "T1"),
            "wrong_metric": ("metric", "acc_at_3"),
            "wrong_set": ("set", "T2-all"),
            "counts": ("n_windows", 192),
            "cases": ("n_cases", 53),
            "nan": ("mean_prompt_tokens", np.nan),
            "infinity": ("mean_prompt_tokens", np.inf),
            "zero": ("mean_prompt_tokens", 0),
            "negative": ("mean_prompt_tokens", -1),
            "flag": ("tokens_reliable", "unknown"),
        }[change]
        summary.loc[index, column] = value
    with pytest.raises(ValueError, match=message):
        paper_figure_data(comparisons, summary, config)


def test_zero_endpoint_is_not_clear_and_negative_interval_is_clear(inputs):
    comparisons, summary, config = inputs
    index = comparisons.index[comparisons.is_main][0]
    comparisons.loc[index, ["estimate_diff", "ci_low", "ci_high"]] = [0.02, 0, 0.04]
    paper_figure_data(comparisons, summary, config)
    comparisons.loc[index, ["estimate_diff", "ci_low", "ci_high"]] = [-0.02, -0.04, -0.01]
    comparisons.loc[index, "verdict"] = "clear difference"
    paper_figure_data(comparisons, summary, config)


def test_either_unreliable_operand_flags_ratio(inputs):
    comparisons, summary, config = inputs
    selected = summary.model.eq("qwen3.5:9b") & summary["set"].eq("T2-informative")
    summary.loc[selected, "tokens_reliable"] = True
    summary.loc[selected & summary.format.eq("F1"), "tokens_reliable"] = False
    rows = paper_figure_data(comparisons, summary, config)
    assert rows.ratio_reliable.tolist() == [False, True, True, True, True] + [False] * 5
    summary.loc[selected & summary.format.eq("F0"), "tokens_reliable"] = False
    assert not paper_figure_data(comparisons, summary, config).ratio_reliable.any()


def test_artists_match_data_and_final_size_layout(inputs):
    comparisons, summary, config = inputs
    rows = paper_figure_data(comparisons, summary, config)
    fig = build_paper_figure(rows, config)
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert len(fig.axes) == 1
        np.testing.assert_array_equal(fig.get_size_inches(), [3.487, 2.05])
        np.testing.assert_array_equal(ax.get_xlim(), [-0.21, 0.21])
        np.testing.assert_array_equal(ax.get_xticks(), [-0.2, -0.1, 0, 0.1, 0.2])
        assert ax.get_xlabel() == "Acc@1 difference vs. F0 (95% CI)"
        assert not fig.legends and not ax.get_title()
        assert [text.get_text() for text in ax.get_yticklabels()] == [
            f"{model} F{i}" for model in ("qwen", "gemma") for i in range(1, 6)
        ]
        for i, (row, container) in enumerate(zip(rows.itertuples(), ax.containers, strict=True)):
            point, caps, bars = container.lines
            assert point.get_xdata()[0] == row.estimate_diff
            assert point.get_marker() == ("o" if i < 5 else "s")
            assert point.get_markersize() == 4.5
            assert point.get_markeredgewidth() == config.style["marker_edge_width"]
            assert point.get_markerfacecolor() == (
                config.style["colors"][i // 5] if row.verdict == "clear difference" else "white"
            )
            assert [cap.get_xdata()[0] for cap in caps] == [row.ci_low, row.ci_high]
            assert all(cap.get_markeredgewidth() == config.style["ci_width"] for cap in caps)
            assert bars[0].get_linewidths().tolist() == [config.style["ci_width"]]
            np.testing.assert_allclose(
                bars[0].get_segments()[0][:, 0], [row.ci_low, row.ci_high], rtol=0, atol=3e-17
            )
        assert [text.get_text() for text in ax.texts] == [
            "1.9×!",
            "5.0×!",
            "13×!",
            "28×!",
            "17×!",
            "1.8×!",
            "4.9×!",
            "12×!",
            "26×!",
            "16×!",
        ]
        text = [
            artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text()
        ]
        assert all(artist.get_fontsize() == 8 for artist in text)
        bounds = [artist.get_window_extent(fig.canvas.get_renderer()) for artist in text]
        for box in bounds:
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
        for i, first in enumerate(bounds):
            assert not any(first.overlaps(second) for second in bounds[i + 1 :])
    finally:
        fig.clear()


def _temporary_config(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    for key in ("comparisons", "summary"):
        raw[key] = str(ROOT / raw[key])
    for key in ("output_pdf", "output_png", "provenance"):
        raw[key] = str(tmp_path / Path(raw[key]).name)
    path = tmp_path / "figure.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def test_pdf_size_embedded_font_metadata_and_determinism(tmp_path):
    config_path, raw = _temporary_config(tmp_path)
    result = export_paper_figure(config_path)
    pdf = Path(raw["output_pdf"]).read_bytes()
    assert result == export_paper_figure(config_path)
    assert pdf == Path(raw["output_pdf"]).read_bytes()
    box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf)
    assert box
    assert [float(x) for x in box.group(1).split()] == pytest.approx([0, 0, 251.064, 147.6])
    assert b"/Subtype /Type3" not in pdf
    assert b"/Subtype /CIDFontType2" in pdf and b"/FontFile2" in pdf
    assert b"STIXGeneral" in pdf
    assert b"/CreationDate" not in pdf and b"/ModDate" not in pdf and b"/Author" not in pdf
    assert b"/Users/" not in pdf
    streams = []
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.DOTALL):
        try:
            streams.append(zlib.decompress(stream))
        except zlib.error:
            pass
    sizes = re.findall(rb"/F\d+ ([\d.]+) Tf", b"\n".join(streams))
    assert sizes and set(map(float, sizes)) == {8.0}
    assert result["outputs"]["pdf_sha256"] == hashlib.sha256(pdf).hexdigest()
    assert json.loads(Path(raw["provenance"]).read_text()) == result


def test_unrelated_blank_csv_columns_do_not_break_export(tmp_path):
    config_path, raw = _temporary_config(tmp_path)
    expected = export_paper_figure(config_path)
    comparisons = pd.read_csv(raw["comparisons"], float_precision="round_trip")
    comparisons["optional_note"] = np.nan
    annotated = tmp_path / "annotated.csv"
    comparisons.to_csv(annotated, index=False)
    raw["comparisons"] = str(annotated)
    config_path.write_text(yaml.safe_dump(raw))
    actual = export_paper_figure(config_path)
    assert actual["rows"] == expected["rows"]
    assert actual["outputs"] == expected["outputs"]
    assert all("optional_note" not in row for row in actual["rows"])


def test_font_hash_is_enforced(inputs):
    comparisons, summary, config = inputs
    rows = paper_figure_data(comparisons, summary, config)
    config = replace(config, font={**config.font, "sha256": "wrong"})
    with pytest.raises(ValueError, match="font SHA-256"):
        build_paper_figure(rows, config)


def test_hostile_rc_does_not_change_exports_or_global_settings(tmp_path):
    config_path, raw = _temporary_config(tmp_path)
    expected = export_paper_figure(config_path)["outputs"]
    hostile = {
        "text.usetex": True,
        "font.size": 31,
        "font.family": "fantasy",
        "savefig.bbox": "tight",
        "savefig.transparent": True,
        "axes.grid": True,
        "axes.xmargin": 0.9,
        "pdf.fonttype": 3,
        "figure.autolayout": True,
    }
    with mpl.rc_context(hostile):
        before = copy.deepcopy(dict(mpl.rcParams))
        assert export_paper_figure(config_path)["outputs"] == expected
        assert dict(mpl.rcParams) == before
    # Also exercise rc-file loading in a fresh process, not only mutated rcParams.
    rc_dir = tmp_path / "mpl"
    rc_dir.mkdir()
    (rc_dir / "matplotlibrc").write_text(
        "font.size: 31\ntext.usetex: True\npdf.fonttype: 3\n"
        "savefig.bbox: tight\nfigure.autolayout: True\n"
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/plot_paper_figure.py"), str(config_path)],
        cwd=ROOT,
        env={**os.environ, "MPLCONFIGDIR": str(rc_dir)},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "SHA-256:" in completed.stdout
    assert (
        hashlib.sha256(Path(raw["output_pdf"]).read_bytes()).hexdigest() == expected["pdf_sha256"]
    )
    assert (
        hashlib.sha256(Path(raw["output_png"]).read_bytes()).hexdigest() == expected["png_sha256"]
    )
