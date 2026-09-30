"""Independent numeric expectations for spec 007, plus saved-score CLI smoke checks."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from traceformatbench.analysis import report
from traceformatbench.analysis.report import AnalysisConfig, analyze_records, pareto_optimal

ROOT = Path(__file__).parents[1]


def _config(**overrides):
    return AnalysisConfig(
        **{
            "seed": 7,
            "n_boot": 200,
            "ci": 0.95,
            "control": "F0",
            "min_stratum_cases": 1,
            "scores": "unused",
            "baseline": None,
            "output_dir": "unused",
            **overrides,
        }
    )


def _record(case="a", fmt="F0", **overrides):
    return {
        "case": case,
        "dataset": "RE2-OB",
        "system": "ob",
        "task": "T2",
        "label": "fault",
        "window_index": 0,
        "informative": True,
        "model": "m1",
        "format": fmt,
        "status": "ok",
        "parse_status": "ok",
        "prompt_eval_count": 100,
        "prompt_eval_cached_count": 0,
        "total_s": 1,
        "answer": ["fault", "b", "c"],
        "hit_at_1": 1,
        "hit_at_3": 1,
        "chance_at_1": 1 / 4,
        "chance_at_3": 3 / 4,
        **overrides,
    }


def _row(summary, model="m1", fmt="F0", set_name="T2-informative", metric="acc_at_1"):
    selected = summary[
        (summary["model"] == model)
        & (summary["format"] == fmt)
        & (summary["set"] == set_name)
        & (summary["metric"] == metric)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_pooled_summary_and_paired_discordance_have_hand_computed_values():
    records = [
        _record("a", "F0", hit_at_1=0),
        _record("a", "F1"),
        _record("b", "F0"),
        _record("b", "F1", hit_at_1=0),
    ]
    summary, comparisons, _ = analyze_records(records, _config())
    assert _row(summary)["estimate"] == _row(summary, fmt="F1")["estimate"] == 1 / 2
    main = comparisons[comparisons["is_main"]].iloc[0]
    assert main["estimate_diff"] == 0
    assert main["ci_low"] == -1 and main["ci_high"] == 1
    assert main["verdict"] == "no clear difference"
    assert main["n_only_x_correct"] == main["n_only_reference_correct"] == 1


def test_all_main_and_secondary_comparisons_including_baseline_dash_format():
    records = []
    for model in ("m1", "m2"):
        for fmt in ("F0", "F1", "F2", "F3", "F4", "F5"):
            for case in ("a", "b"):
                hit = int(fmt != "F0")
                records.append(_record(case, fmt, model=model, hit_at_1=hit, hit_at_3=hit))
                for label in ("pre", "post"):
                    records.append(
                        _record(
                            case,
                            fmt,
                            model=model,
                            task="T1",
                            informative=None,
                            label=label,
                            answer=label,
                            correct=hit,
                        )
                    )
    for case in ("a", "b"):
        records.append(
            _record(
                case,
                "-",
                model="heuristic-selftime",
                hit_at_1=1 / 2,
                hit_at_3=1,
                prompt_eval_count=None,
                tie_at_1=True,
                tie_at_3=False,
            )
        )
    summary, comparisons, _ = analyze_records(records, _config())
    # 2 models * 5 formats * (4 T2 set/metric pairs + 1 T1 accuracy) = 50.
    # 2 models * 6 formats * 2 T2 sets vs baseline = 24. Ten of the 74 are main.
    assert len(comparisons) == 74 and comparisons["is_main"].sum() == 10
    main = comparisons[comparisons["is_main"]]
    assert set(main["format"]) == {"F1", "F2", "F3", "F4", "F5"}
    assert set(main["set"]) == {"T2-informative"} and set(main["metric"]) == {"acc_at_1"}
    assert set(main["reference_format"]) == {"F0"}
    assert (main["estimate_diff"] == 1).all() and (main["ci_low"] == 1).all()
    assert (main["ci_high"] == 1).all() and (main["verdict"] == "clear difference").all()
    baseline = comparisons[comparisons["reference_model"] == "heuristic-selftime"]
    assert len(baseline) == 24 and set(baseline["reference_format"]) == {"-"}
    assert not baseline["is_main"].any() and set(baseline["metric"]) == {"acc_at_1"}
    assert set(baseline["estimate_diff"]) == {-1 / 2, 1 / 2}
    f0 = baseline[baseline["format"] == "F0"]
    assert (f0["n_only_x_correct"] == 0).all() and (f0["n_only_reference_correct"] == 1).all()
    row = _row(summary, model="heuristic-selftime", fmt="-")
    assert row["estimate"] == 1 / 2 and row["tie_rate"] == 1
    assert _row(summary, model="heuristic-selftime", fmt="-", metric="acc_at_3")["tie_rate"] == 0


def test_tokens_include_parse_failure_and_thinking_leak_but_exclude_truncated():
    records = [
        _record("a", prompt_eval_count=100, total_s=1),
        _record(
            "a",
            window_index=1,
            prompt_eval_count=300,
            total_s=3,
            parse_status="invalid_json",
            hit_at_1=0,
            hit_at_3=0,
            answer=None,
        ),
        _record(
            "b", prompt_eval_count=500, total_s=5, status="thinking_leak", hit_at_1=0, hit_at_3=0
        ),
        _record(
            "b",
            window_index=1,
            prompt_eval_count=9999,
            total_s=9,
            status="truncated",
            hit_at_1=0,
            hit_at_3=0,
        ),
    ]
    summary, _, _ = analyze_records(records, _config())
    row = _row(summary)
    assert row["n_windows"] == 4 and row["n_cases"] == 2
    assert row["estimate"] == 1 / 4
    assert row["mean_prompt_tokens"] == 300  # (100 + 300 + 500) / 3
    assert row["mean_prompt_tokens_ci_low"] == 200 and row["mean_prompt_tokens_ci_high"] == 500
    assert row["acc_per_1k_tokens"] == pytest.approx(10 / 9)  # (1/3) / (300/1000)
    assert row["mean_total_s"] == 9 / 2
    assert row["invalid_rate"] == row["truncated_rate"] == 1 / 4
    assert not row["tokens_reliable"]


def test_reliability_flips_on_one_cached_call_and_tokens_remain_counted():
    records = [_record("a", prompt_eval_count=100), _record("b", prompt_eval_count=200)]
    clean, _, _ = analyze_records(records, _config())
    cached, _, _ = analyze_records(
        [records[0], {**records[1], "prompt_eval_cached_count": 1}], _config()
    )
    assert _row(clean)["tokens_reliable"]
    assert not _row(cached)["tokens_reliable"]
    assert _row(cached)["mean_prompt_tokens"] == 150
    assert _row(cached)["acc_per_1k_tokens"] == pytest.approx(20 / 3)


def test_all_truncated_calls_have_nan_tokens_and_efficiency():
    summary, _, _ = analyze_records(
        [
            _record("a", status="truncated", hit_at_1=0),
            _record("b", status="truncated", hit_at_1=0),
        ],
        _config(),
    )
    row = _row(summary)
    assert row["estimate"] == 0 and row["truncated_rate"] == 1 and not row["tokens_reliable"]
    assert np.isnan(row["mean_prompt_tokens"]) and np.isnan(row["mean_prompt_tokens_ci_low"])
    assert np.isnan(row["acc_per_1k_tokens"])


@pytest.mark.parametrize("failure", [{"status": "truncated"}, {"prompt_eval_cached_count": 1}])
def test_reliability_checks_any_call_in_the_condition(failure):
    summary, _, _ = analyze_records(
        [_record("a"), _record("b", informative=False, **failure)], _config()
    )
    primary = _row(summary)
    assert primary["n_windows"] == 1 and primary["mean_prompt_tokens"] == 100
    assert not primary["tokens_reliable"]


def test_chance_rows_use_case_varying_candidate_counts_with_shared_ci():
    summary, _, _ = analyze_records(
        [
            _record("a", chance_at_1=1 / 7, chance_at_3=3 / 7),
            _record("b", chance_at_1=1 / 10, chance_at_3=3 / 10),
        ],
        _config(),
    )
    for set_name in ("T2-informative", "T2-all"):
        row1 = _row(summary, "chance", "-", set_name, "acc_at_1")
        row3 = _row(summary, "chance", "-", set_name, "acc_at_3")
        assert row1["estimate"] == pytest.approx(17 / 140)
        assert row3["estimate"] == pytest.approx(51 / 140)
        assert row1["ci_low"] == 1 / 10 and row1["ci_high"] == 1 / 7
        assert row3["ci_low"] == 3 / 10 and row3["ci_high"] == 3 / 7
        assert row1["n_windows"] == row3["n_windows"] == 2


def test_t1_pre_and_post_same_index_are_distinct_and_post_rate_is_descriptive():
    records = [
        _record("a", task="T1", informative=None, label="pre", answer="post", correct=0),
        _record("a", task="T1", informative=None, label="post", answer="post", correct=1),
        _record(
            "b",
            task="T1",
            informative=None,
            label="pre",
            answer="post",
            correct=0,
            parse_status="schema_violation",
        ),
        _record("b", task="T1", informative=None, label="post", answer="pre", correct=0),
    ]
    summary, _, _ = analyze_records(records, _config())
    row = _row(summary, set_name="T1-all", metric="accuracy")
    assert row["n_windows"] == 4 and row["n_cases"] == 2
    assert row["estimate"] == 1 / 4 and row["post_rate"] == 1 / 2
    assert not ((summary["model"] == "m1") & (summary["task"] == "T2")).any()


def test_empty_informative_condition_and_none_informative_exclusion():
    summary, comparisons, _ = analyze_records(
        [
            _record("a", "F0", informative=False),
            _record("a", "F1", informative=False),
            _record("b", "F0", informative=None),
            _record("b", "F1", informative=None),
        ],
        _config(),
    )
    for fmt in ("F0", "F1"):
        empty = _row(summary, fmt=fmt)
        assert empty["n_windows"] == empty["n_cases"] == 0 and np.isnan(empty["estimate"])
        assert np.isnan(empty["ci_low"]) and np.isnan(empty["ci_high"])
        all_t2 = _row(summary, fmt=fmt, set_name="T2-all")
        assert all_t2["n_windows"] == all_t2["n_cases"] == 1
    assert set(comparisons["verdict"]) == {"not enough cases"}


def test_one_case_retains_estimate_and_difference_with_no_ci():
    summary, comparisons, _ = analyze_records(
        [
            _record("a", "F0", hit_at_1=0),
            _record("a", "F1", hit_at_1=1),
        ],
        _config(),
    )
    assert _row(summary, fmt="F1")["estimate"] == 1
    main = comparisons[comparisons["is_main"]].iloc[0]
    assert main["estimate_diff"] == 1 and np.isnan(main["ci_low"]) and np.isnan(main["ci_high"])
    assert main["verdict"] == "not enough cases"


def test_duplicate_keys_raise_before_any_bootstrap_computation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Bootstrap must not run before uniqueness validation.")

    monkeypatch.setattr(report, "draw_resamples", fail)
    with pytest.raises(ValueError, match="Duplicate.*'a', 'T2', 'fault', 0"):
        analyze_records([_record(), _record()], _config())


def test_missing_keys_name_the_missing_window_before_computation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Bootstrap must not run before missing-key validation.")

    monkeypatch.setattr(report, "draw_resamples", fail)
    with pytest.raises(ValueError, match="Missing.*'b', 'T2', 'fault', 0"):
        analyze_records([_record("a"), _record("b"), _record("a", "F1")], _config())


def test_missing_control_names_all_window_keys_before_computation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Bootstrap must not run before control-key validation.")

    monkeypatch.setattr(report, "draw_resamples", fail)
    with pytest.raises(ValueError, match="missing in reference=.*'a', 'T2', 'fault', 0"):
        analyze_records([_record("a", "F1")], _config())


def test_condition_and_record_order_do_not_change_tables_or_resample_count(monkeypatch):
    calls = []
    original = report.draw_resamples

    def draw(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "draw_resamples", draw)
    records = [
        _record(case, fmt, hit_at_1=int(case == "a" or fmt == "F1"))
        for fmt in ("F0", "F1")
        for case in ("a", "b", "c")
    ]
    first = analyze_records(records, _config())
    second = analyze_records(list(reversed(records)), _config())
    assert calls == ["T2-informative", "T2-all", "T1-all"] * 2
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_pareto_marks_only_strictly_dominated_points_false():
    # F0 dominates F1. Equal tokens do not dominate F3 under this spec's strict rule.
    np.testing.assert_array_equal(
        pareto_optimal([100, 200, 200, 100, np.nan], [0.8, 0.7, 0.9, 0.7, 1]),
        [True, False, True, True, False],
    )


def test_plot_keeps_model_panels_separate_and_marks_unreliable_points(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    # Direct, hand-built plotting rows avoid deriving expected marks from analysis output.
    points = pd.DataFrame(
        [
            {
                "model": "m1",
                "format": "F0",
                "mean_prompt_tokens": 100,
                "estimate": 0.8,
                "ci_low": 0.7,
                "ci_high": 0.9,
                "tokens_reliable": True,
            },
            {
                "model": "m1",
                "format": "F1",
                "mean_prompt_tokens": 200,
                "estimate": 0.7,
                "ci_low": 0.6,
                "ci_high": 0.8,
                "tokens_reliable": False,
            },
            {
                "model": "m2",
                "format": "F0",
                "mean_prompt_tokens": 300,
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1,
                "tokens_reliable": True,
            },
            {"model": "heuristic-selftime", "format": "-", "estimate": 0.6},
            {"model": "chance", "format": "-", "estimate": 0.25},
        ]
    ).assign(set="T2-informative", metric="acc_at_1")
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    report.plot_pareto(points, tmp_path / "synthetic.png")
    fig = figures[-1]
    try:
        assert [ax.get_title() for ax in fig.axes] == ["m1", "m2"]
        scatter1 = [c for c in fig.axes[0].collections if isinstance(c, PathCollection)]
        scatter2 = [c for c in fig.axes[1].collections if isinstance(c, PathCollection)]
        np.testing.assert_allclose(scatter1[0].get_offsets(), [[100, 0.8]])
        np.testing.assert_allclose(scatter1[1].get_offsets(), [[200, 0.7]])
        np.testing.assert_allclose(scatter2[0].get_offsets(), [[300, 0.9]])
        assert scatter1[0].get_sizes().tolist() == [120]  # Pareto star
        assert scatter1[1].get_sizes().tolist() == [60]  # dominated circle
        assert [text.get_text() for text in fig.axes[0].texts] == ["F0", "F1 !"]
        for ax in fig.axes:
            assert [line.get_ydata() for line in ax.lines] == [[0.6, 0.6], [0.25, 0.25]]
        assert (tmp_path / "synthetic.png").stat().st_size > 0
    finally:
        close(fig)


def _run_cli(config_path, hash_seed):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze.py"), str(config_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def test_separate_processes_produce_byte_identical_csvs(tmp_path):
    records = [
        _record(case, fmt, hit_at_1=int(fmt == "F1"), chance_at_1=1 / 7, chance_at_3=3 / 7)
        for fmt in ("F0", "F1")
        for case in ("a", "b", "c")
    ]
    baseline = [
        _record(
            case,
            "-",
            model="heuristic-selftime",
            hit_at_1=1 / 2,
            hit_at_3=1,
            prompt_eval_count=None,
            chance_at_1=1 / 7,
            chance_at_3=3 / 7,
            tie_at_1=True,
            tie_at_3=False,
        )
        for case in ("a", "b", "c")
    ]
    baseline_path = tmp_path / "synthetic_baseline.jsonl"
    baseline_path.write_text("".join(json.dumps(r) + "\n" for r in baseline))
    for index, ordered in enumerate((records, list(reversed(records)))):
        scores = tmp_path / f"scores{index}.jsonl"
        scores.write_text("".join(json.dumps(r) + "\n" for r in ordered))
        config = tmp_path / f"analysis{index}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    **_config().__dict__,
                    "scores": str(scores),
                    "baseline": str(baseline_path),
                    "output_dir": str(tmp_path / f"out{index}"),
                }
            )
        )
        _run_cli(config, index + 1)
    for name in ("summary.csv", "comparisons.csv", "strata.csv"):
        assert (tmp_path / "out0" / name).read_bytes() == (tmp_path / "out1" / name).read_bytes()
    assert (
        (tmp_path / "out0" / "pareto_t2_informative.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )


def test_saved_tasks_smoke_runs_end_to_end_without_baseline(tmp_path):
    config = tmp_path / "analysis_smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **replace(_config(), n_boot=10, min_stratum_cases=5).__dict__,
                "scores": str(ROOT / "results" / "tasks_smoke" / "scores.jsonl"),
                "output_dir": str(tmp_path / "smoke"),
            }
        )
    )
    completed = _run_cli(config, 17)
    assert "Main comparisons:" in completed.stdout
    assert (
        "Limitation: under the null, about 0.5 false clear differences are expected among "
        "10 comparisons at 95% (at least one with about 40% probability if independent)."
    ) in completed.stdout.splitlines()
    assert "about one false clear difference" not in completed.stdout
    assert set(p.name for p in (tmp_path / "smoke").iterdir()) == {
        "summary.csv",
        "comparisons.csv",
        "strata.csv",
        "pareto_t2_informative.png",
    }


def test_missing_input_raises_without_running_or_creating_baseline(tmp_path):
    config = _config(
        scores=str(tmp_path / "missing_scores.jsonl"),
        baseline=str(tmp_path / "selftime_v1.jsonl"),
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(FileNotFoundError):
        report.write_report(config)
    assert not Path(config.baseline).exists() and not Path(config.output_dir).exists()


@pytest.mark.parametrize("overrides", [{"n_boot": 0}, {"ci": 1}, {"min_stratum_cases": 0}])
def test_invalid_analysis_parameters_raise(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)
