"""Checks for the one-command verification of committed outputs."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from traceformatbench.analysis.report import AnalysisConfig, write_report
from traceformatbench.analysis.verify import (
    VerifyConfig,
    compare_records,
    compare_tables,
    load_verify_config,
    verify,
)
from traceformatbench.runners.tasks import read_records
from traceformatbench.tasks.scoring import metrics

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "results" / "tasks_smoke"


def _table(**columns):
    return pd.DataFrame({"model": ["a", "b"], "estimate": [0.25, float("nan")], **columns})


def test_identical_tables_pass_without_drift():
    check = compare_tables("t.csv", _table(), _table(), 1e-9, 1e-12)
    assert check.ok and check.detail == "2 rows match"


def test_last_digit_float_drift_passes_and_is_reported():
    drifted = _table(estimate=[np.nextafter(0.25, 1.0), float("nan")])
    check = compare_tables("t.csv", _table(), drifted, 1e-9, 1e-12)
    assert check.ok and "1 float cells differ within tolerance" in check.detail


def test_float_change_beyond_tolerance_fails_and_names_the_cell():
    check = compare_tables("t.csv", _table(), _table(estimate=[0.26, float("nan")]), 1e-9, 1e-12)
    assert not check.ok and "row 2 estimate: 0.25 != 0.26" in check.detail


def test_nan_against_number_fails():
    check = compare_tables("t.csv", _table(), _table(estimate=[0.25, 0.5]), 1e-9, 1e-12)
    assert not check.ok and "row 3 estimate: NaN != 0.5" in check.detail


def test_text_cells_must_match_exactly():
    check = compare_tables("t.csv", _table(), _table(model=["a", "c"]), 1e-9, 1e-12)
    assert not check.ok and "row 3 model: b != c" in check.detail


def test_column_or_row_count_mismatch_fails():
    assert not compare_tables("t.csv", _table(), _table(extra=[1, 2]), 0, 0).ok
    assert not compare_tables("t.csv", _table(), _table().iloc[:1], 0, 0).ok


def test_records_must_match_exactly_and_in_order():
    records = [{"hit_at_1": 1}, {"hit_at_1": 0}]
    assert compare_records("s.jsonl", records, list(records)).ok
    assert not compare_records("s.jsonl", records, records[::-1]).ok
    assert not compare_records("s.jsonl", records, records[:1]).ok


def test_negative_tolerance_raises():
    with pytest.raises(ValueError):
        VerifyConfig("c", "s", "m", "a", float_rtol=-1, float_atol=0)


def _smoke_setup(tmp_path):
    """Committed smoke calls and scores, plus metrics and analysis tables from the current code.

    The committed smoke metrics.csv predates later metric columns, so it is rebuilt here.
    """
    for name in ("calls.jsonl", "scores.jsonl"):
        shutil.copy(SMOKE / name, tmp_path / name)
    metrics(read_records(tmp_path / "scores.jsonl")).to_csv(tmp_path / "metrics.csv", index=False)
    analysis = AnalysisConfig(
        seed=7,
        n_boot=10,
        ci=0.95,
        control="F0",
        min_stratum_cases=5,
        scores=str(tmp_path / "scores.jsonl"),
        baseline=None,
        output_dir=str(tmp_path / "analysis"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        write_report(analysis)
    (tmp_path / "analysis.yaml").write_text(yaml.safe_dump(analysis.__dict__))
    config = {
        "calls": str(tmp_path / "calls.jsonl"),
        "scores": str(tmp_path / "scores.jsonl"),
        "metrics": str(tmp_path / "metrics.csv"),
        "analysis_config": str(tmp_path / "analysis.yaml"),
        "float_rtol": 1e-9,
        "float_atol": 1e-12,
    }
    (tmp_path / "verify.yaml").write_text(yaml.safe_dump(config))
    return tmp_path / "verify.yaml"


def _run_cli(config_path):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_results.py"), str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_smoke_outputs_verify_end_to_end_and_leave_inputs_untouched(tmp_path):
    config_path = _smoke_setup(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    checks = verify(load_verify_config(config_path))
    assert [c.name for c in checks] == [
        "scores.jsonl",
        "metrics.csv",
        "summary.csv",
        "comparisons.csv",
        "strata.csv",
    ]
    assert all(c.ok for c in checks), checks
    after = {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    completed = _run_cli(config_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines()[-1] == "5 of 5 checks passed."


def test_cli_exits_nonzero_when_a_committed_number_changes(tmp_path):
    config_path = _smoke_setup(tmp_path)
    summary = tmp_path / "analysis" / "summary.csv"
    table = pd.read_csv(summary, keep_default_na=False, na_values=["NaN", ""])
    table.loc[0, "estimate"] += 0.01
    table.to_csv(summary, index=False, na_rep="NaN", lineterminator="\n")
    completed = _run_cli(config_path)
    assert completed.returncode == 1
    assert "FAIL  summary.csv: 1 cells differ (row 2 estimate:" in completed.stdout
    assert completed.stdout.splitlines()[-1] == "4 of 5 checks passed."


def test_cli_fails_when_a_committed_score_changes(tmp_path):
    config_path = _smoke_setup(tmp_path)
    scores = tmp_path / "scores.jsonl"
    records = [json.loads(line) for line in scores.read_text().splitlines()]
    records[3]["correct"] = 1 - records[3]["correct"]
    scores.write_text("".join(json.dumps(r) + "\n" for r in records))
    completed = _run_cli(config_path)
    assert completed.returncode == 1
    assert "FAIL  scores.jsonl: 1 records differ (lines 4)" in completed.stdout


def test_missing_calls_file_exits_with_message(tmp_path):
    config_path = _smoke_setup(tmp_path)
    (tmp_path / "calls.jsonl").unlink()
    completed = _run_cli(config_path)
    assert completed.returncode == 1 and "No saved calls found" in completed.stderr


def test_analysis_config_reading_other_scores_raises(tmp_path):
    config_path = _smoke_setup(tmp_path)
    analysis = yaml.safe_load((tmp_path / "analysis.yaml").read_text())
    analysis["scores"] = [analysis["scores"], str(tmp_path / "other.jsonl")]
    (tmp_path / "analysis.yaml").write_text(yaml.safe_dump(analysis))
    with pytest.raises(ValueError, match="must read exactly"):
        verify(load_verify_config(config_path))
