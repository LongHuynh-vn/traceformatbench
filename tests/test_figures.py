"""Checks for the forest plot of the main comparisons."""

import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
import yaml

from traceformatbench.analysis import figures
from traceformatbench.analysis.figures import main_comparisons, plot_main_comparisons

ROOT = Path(__file__).parents[1]


def _row(model="m1", fmt="F1", is_main=True, **overrides):
    return {
        "model": model,
        "format": fmt,
        "reference_model": model,
        "reference_format": "F0",
        "task": "T2",
        "set": "T2-informative",
        "metric": "acc_at_1",
        "is_main": is_main,
        "n_windows": 10,
        "n_cases": 4,
        "estimate_diff": 0.1,
        "ci_low": -0.05,
        "ci_high": 0.2,
        "verdict": "no clear difference",
        **overrides,
    }


def test_only_main_rows_are_kept_and_sorted():
    table = pd.DataFrame(
        [
            _row("m2", "F2"),
            _row("m1", "F2"),
            _row("m1", "F1", is_main=False, set="T2-all"),
            _row("m1", "F1"),
        ]
    )
    main = main_comparisons(table)
    assert list(zip(main["model"], main["format"], strict=True)) == [
        ("m1", "F1"),
        ("m1", "F2"),
        ("m2", "F2"),
    ]


def test_is_main_read_back_from_csv_as_text_is_accepted(tmp_path):
    path = tmp_path / "comparisons.csv"
    pd.DataFrame([_row(), _row(fmt="F2", is_main=False)]).to_csv(path, index=False)
    assert len(main_comparisons(pd.read_csv(path, dtype={"is_main": str}))) == 1


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_row(is_main=False)], "No main comparisons"),
        ([_row(), _row(fmt="F2", metric="acc_at_3")], "metric"),
        ([_row(), _row(fmt="F2", reference_format="F1")], "reference_format"),
        ([_row(reference_model="m2")], "within-model"),
        ([_row(), _row()], "Duplicate"),
    ],
)
def test_inconsistent_main_rows_raise(rows, message):
    with pytest.raises(ValueError, match=message):
        main_comparisons(pd.DataFrame(rows))


def test_one_panel_per_model_with_filled_marker_only_for_clear_difference(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_made = []
    monkeypatch.setattr(plt, "close", figures_made.append)
    table = pd.DataFrame(
        [
            _row("m1", "F1", verdict="clear difference", ci_low=0.01),
            _row("m1", "F2"),
            _row("m2", "F1", ci_low=float("nan"), ci_high=float("nan"), verdict="not enough cases"),
        ]
    )
    plot_main_comparisons(table, tmp_path / "forest.png")
    fig = figures_made[0]
    try:
        axes = fig.axes
        assert [ax.get_title() for ax in axes] == [
            "m1  (n = 10 windows, 4 cases)",
            "m2  (n = 10 windows, 4 cases)",
        ]
        faces = [line.get_markerfacecolor() for line in axes[0].lines if line.get_marker() == "o"]
        assert faces == [figures.INK, "white"]
        assert len(axes[0].collections) == 2  # one CI line per format with a finite CI
        assert len(axes[1].collections) == 0  # no CI line when the CI is NaN
        assert [t.get_text() for t in axes[0].get_yticklabels()] == ["F1 - F0", "F2 - F0"]
    finally:
        monkeypatch.undo()
        plt.close("all")
    assert (tmp_path / "forest.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_draws_the_committed_main_comparisons(tmp_path):
    config = tmp_path / "figures.yaml"
    output = tmp_path / "out" / "main_comparisons.png"
    config.write_text(
        yaml.safe_dump(
            {
                "comparisons": str(ROOT / "results" / "analysis_v1" / "comparisons.csv"),
                "output": str(output),
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_main_comparisons.py"), str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.strip() == f"Figure: {output}"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    main = main_comparisons(pd.read_csv(ROOT / "results" / "analysis_v1" / "comparisons.csv"))
    assert len(main) == 10


def test_cli_missing_table_exits_with_message(tmp_path):
    config = tmp_path / "figures.yaml"
    config.write_text(
        yaml.safe_dump({"comparisons": str(tmp_path / "missing.csv"), "output": "x.png"})
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_main_comparisons.py"), str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 1 and "missing.csv" in completed.stderr
