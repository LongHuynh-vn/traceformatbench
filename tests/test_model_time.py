"""Checks for the summed per-call model time per model."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from traceformatbench.analysis.model_time import (
    ModelTimeConfig,
    hours_from_calls,
    hours_from_summary,
    load_model_time_config,
    model_hours,
)

ROOT = Path(__file__).parents[1]
SUMMARY_FIELDS = ["model", "format", "task", "set", "metric", "n_windows", "mean_total_s"]


def _write_calls(path, calls):
    lines = [json.dumps({"model": model, "total_s": total_s}) for model, total_s in calls]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    return path


def _write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(SUMMARY_FIELDS, row, strict=True)))
    return path


def _consistent_inputs(tmp_path):
    # m1: 2 T1 calls of 900 s and 3 T2 calls of 1200 s = 1.5 h.
    # m2: 1 T1 call and 1 T2 call of 1800 s = 1.0 h.
    calls = [("m1", 900.0)] * 2 + [("m1", 1200.0)] * 3 + [("m2", 1800.0)] * 2
    summary = [
        ("m1", "F0", "T1", "T1-all", "accuracy", 2, 900.0),
        ("m1", "F0", "T2", "T2-all", "acc_at_1", 3, 1200.0),
        ("m1", "F0", "T2", "T2-all", "acc_at_3", 3, 1200.0),
        ("m1", "F0", "T2", "T2-informative", "acc_at_1", 2, 1200.0),
        ("m2", "F0", "T1", "T1-all", "accuracy", 1, 1800.0),
        ("m2", "F0", "T2", "T2-all", "acc_at_1", 1, 1800.0),
        ("chance", "-", "T2", "T2-all", "acc_at_1", 3, "NaN"),
        ("heuristic-selftime", "-", "T2", "T2-all", "acc_at_1", 3, 0.5),
    ]
    calls_path = _write_calls(tmp_path / "calls.jsonl", calls)
    return calls_path, _write_summary(tmp_path / "summary.csv", summary)


def test_hours_from_calls_sums_total_s_per_model(tmp_path):
    calls, _ = _consistent_inputs(tmp_path)
    assert hours_from_calls(calls) == pytest.approx({"m1": 1.5, "m2": 1.0})


def test_hours_from_summary_counts_each_call_once_and_only_for_the_given_models(tmp_path):
    _, summary = _consistent_inputs(tmp_path)
    assert hours_from_summary(summary, ["m1", "m2"]) == pytest.approx({"m1": 1.5, "m2": 1.0})
    assert hours_from_summary(summary, ["m2", "m3"]) == pytest.approx({"m2": 1.0, "m3": 0.0})


def test_model_hours_agree_on_consistent_inputs(tmp_path):
    calls, summary = _consistent_inputs(tmp_path)
    rows = model_hours(ModelTimeConfig(calls=str(calls), summary=str(summary), tolerance_h=0.001))
    assert [row.model for row in rows] == ["m1", "m2"]
    assert all(row.agree for row in rows)


def test_model_hours_flag_a_difference_above_the_tolerance(tmp_path):
    calls, summary = _consistent_inputs(tmp_path)
    _write_calls(
        calls, [("m1", 900.0)] * 2 + [("m1", 1200.0)] * 3 + [("m1", 36.0)] + [("m2", 1800.0)] * 2
    )
    rows = model_hours(ModelTimeConfig(calls=str(calls), summary=str(summary), tolerance_h=0.001))
    assert {row.model: row.agree for row in rows} == {"m1": False, "m2": True}
    rows = model_hours(ModelTimeConfig(calls=str(calls), summary=str(summary), tolerance_h=0.02))
    assert all(row.agree for row in rows)


def test_a_model_missing_from_the_summary_disagrees(tmp_path):
    calls, summary = _consistent_inputs(tmp_path)
    _write_calls(calls, [("m1", 900.0)] * 2 + [("m1", 1200.0)] * 3 + [("m3", 1800.0)])
    rows = model_hours(ModelTimeConfig(calls=str(calls), summary=str(summary), tolerance_h=0.001))
    assert {row.model: (row.hours_from_summary, row.agree) for row in rows} == {
        "m1": (1.5, True),
        "m3": (0.0, False),
    }


def _run_cli(config):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "model_time.py"), str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_prints_the_committed_model_times():
    config = ROOT / "configs" / "model_time_v1.yaml"
    assert load_model_time_config(config).tolerance_h == 0.001
    completed = _run_cli(config)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "model,hours_from_calls,hours_from_summary",
        "gemma4:12b,12.8786,12.8786",
        "qwen3.5:9b,7.7991,7.7991",
    ]


def test_cli_fails_when_the_sums_disagree(tmp_path):
    calls, summary = _consistent_inputs(tmp_path)
    _write_calls(calls, [("m1", 3600.0)])
    config = tmp_path / "model_time.yaml"
    config.write_text(
        yaml.safe_dump({"calls": str(calls), "summary": str(summary), "tolerance_h": 0.001})
    )
    completed = _run_cli(config)
    assert completed.returncode != 0
    assert "disagree" in completed.stderr
