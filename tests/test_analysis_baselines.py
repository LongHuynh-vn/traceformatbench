"""Only synthetic traces/windows: never evaluate the windows_v1 baseline in tests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from traceformatbench.analysis import baselines
from traceformatbench.analysis.baselines import expected_hit, score_window, tie_at_cut
from traceformatbench.data.trace import Span, Trace
from traceformatbench.data.windows import WindowConfig


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Baseline tests must use synthetic data, without network access.")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)


def _span(name, service, duration, parent=None, start=0, trace="t1"):
    return Span(name, parent, trace, service, "op", None, start, duration, None)


def _window(**overrides):
    return {
        "case": "synthetic",
        "dataset": "RE2-OB",
        "system": "ob",
        "task": "T2",
        "window_index": 0,
        "label": "api",
        "informative": True,
        "trace_ids": ["t1", "t2"],
        **overrides,
    }


def test_golden_ranking_sums_spans_subtracts_child_time_and_scores_absent_zero():
    # t1: gateway=70, api=20, db=10. t2: api=60, cache=70.
    # Totals: api=80, cache=gateway=70, db=10, absent worker=0.
    traces = [
        Trace(
            "t1",
            (
                _span("g", "gateway", 100),
                _span("a", "api", 30, "g", 20),
                _span("d", "db", 10, "a", 30),
            ),
        ),
        Trace(
            "t2",
            (
                _span("a2", "api", 60, trace="t2"),
                _span("c2", "cache", 70, start=100, trace="t2"),
            ),
        ),
    ]
    candidates = ["api", "cache", "db", "gateway", "worker"]
    record = score_window(_window(), traces, candidates, 7)
    assert record["answer"] == ["api", "cache", "gateway"]
    assert record["hit_at_1"] == record["hit_at_3"] == 1
    assert record["chance_at_1"] == 1 / 5 and record["chance_at_3"] == 3 / 5
    assert record["tie_at_1"] is False and record["tie_at_3"] is False
    absent = score_window(_window(label="worker", informative=False), traces, candidates, 7)
    assert absent["hit_at_1"] == absent["hit_at_3"] == 0
    assert absent["informative"] is False
    required = {
        "case",
        "dataset",
        "system",
        "task",
        "window_index",
        "informative",
        "label",
        "model",
        "format",
        "seed",
        "status",
        "parse_status",
        "prompt_eval_count",
        "prompt_eval_cached_count",
        "total_s",
        "answer",
        "hit_at_1",
        "hit_at_3",
        "chance_at_1",
        "chance_at_3",
    }
    assert required <= record.keys()
    assert record["model"] == "heuristic-selftime" and record["format"] == "-"
    assert record["status"] == record["parse_status"] == "ok" and record["seed"] == 7
    assert record["prompt_eval_count"] is None and record["total_s"] >= 0


@pytest.mark.parametrize(
    ("scores", "label", "k", "credit", "tie"),
    [
        ({"a": 10, "b": 10, "c": 0, "d": 0}, "b", 1, 1 / 2, True),
        ({"a": 10, "b": 10, "c": 10, "d": 0}, "c", 1, 1 / 3, True),
        ({"a": 20, "b": 10, "c": 10, "d": 10}, "d", 3, 2 / 3, True),
        ({"a": 20, "b": 15, "c": 10, "d": 10}, "d", 3, 1 / 2, True),
        ({"a": 20, "b": 10, "c": 10, "d": 0}, "c", 3, 1, False),
        ({"a": 20, "b": 10, "c": 10, "d": 0}, "d", 3, 0, False),
        ({"a": 0, "b": 0, "c": 0, "d": 0}, "d", 1, 1 / 4, True),
        ({"a": 0, "b": 0, "c": 0, "d": 0}, "d", 3, 3 / 4, True),
        ({"a": 0, "b": 0, "c": 0}, "c", 3, 1, False),
    ],
)
def test_hand_built_top_one_and_top_three_cut_ties(scores, label, k, credit, tie):
    assert expected_hit(scores, label, k) == credit
    assert tie_at_cut(scores, k) is tie


def test_alphabetical_display_does_not_determine_hits():
    traces = [Trace("t1", tuple(_span(service, service, 10) for service in ("a", "b", "c", "d")))]
    record = score_window(_window(label="d", trace_ids=["t1"]), traces, ["d", "c", "b", "a"], 7)
    assert record["answer"] == ["a", "b", "c"]
    assert record["hit_at_1"] == 1 / 4 and record["hit_at_3"] == 3 / 4
    assert record["tie_at_1"] is True and record["tie_at_3"] is True


def test_baseline_calls_the_serializer_self_time_function(monkeypatch):
    trace = Trace("t1", (_span("a", "a", 100), _span("b", "b", 1)))
    calls = []

    def shared(trace, span):
        calls.append((trace.trace_id, span.span_id))
        return {"a": 1, "b": 9}[span.service]

    monkeypatch.setattr(baselines, "self_time_us", shared)
    result = score_window(_window(label="b"), [trace], ["a", "b", "c"], 7)
    assert result["answer"] == ["b", "a", "c"]
    assert calls == [("t1", "a"), ("t1", "b")]


def test_baseline_runner_loads_one_synthetic_case_at_a_time_and_all_case_candidates(
    tmp_path, monkeypatch
):
    windows_file = tmp_path / "synthetic.jsonl"
    windows_file.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                _window(case="b", label="absent", trace_ids=["t1"], informative=False),
                _window(case="a", label="a", trace_ids=["t1"]),
                _window(case="a", task="T1", label="pre", trace_ids=["t1"]),
            ]
        )
    )
    config = WindowConfig(
        "synthetic",
        7,
        str(tmp_path),
        ["RE2-OB"],
        [1],
        1,
        1000,
        60,
        1,
        1,
        str(windows_file),
        str(tmp_path / "stats.csv"),
    )
    loaded = []

    def load(path):
        loaded.append(Path(path).name)
        return [
            Trace("t1", (_span("a", "a", 10), _span("b", "b", 5))),
            Trace("t2", (_span("x", "absent", 100, trace="t2"),)),
        ]

    monkeypatch.setattr(baselines, "load_case_traces", load)
    runner = baselines.run_baseline(config)
    first = next(runner)
    assert loaded == ["a"]
    assert first["candidates"] == ["a", "absent", "b"]
    assert first["answer"] == ["a", "b", "absent"] and first["hit_at_1"] == 1
    second = next(runner)
    assert loaded == ["a", "b"] and second["hit_at_3"] == 1 and second["hit_at_1"] == 0
    assert list(runner) == []


def test_baseline_script_writes_synthetic_scored_jsonl(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("run_baseline", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = tmp_path / "windows_synthetic.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "data_dir": str(tmp_path),
                "datasets": ["RE2-OB"],
                "repetitions": [1],
                "window_size": 1,
                "max_trace_chars": 1000,
                "guard_seconds": 60,
                "t1_windows_per_phase": 1,
                "t2_windows": 1,
                "output_windows": "unused",
                "output_stats": str(tmp_path / "results" / "window_stats" / "synthetic.csv"),
            }
        )
    )
    trace = Trace("t1", tuple(_span(s, s, 10) for s in ("a", "b", "c", "d")))
    record = score_window(_window(label="d"), [trace], ["a", "b", "c", "d"], 7)
    monkeypatch.setattr(script, "run_baseline", lambda config: iter([record]))
    monkeypatch.setattr(sys, "argv", [str(path), str(config)])
    script.main()
    output = tmp_path / "results" / "baselines" / "selftime_synthetic.jsonl"
    saved = json.loads(output.read_text())
    assert saved["answer"] == ["a", "b", "c"] and saved["hit_at_1"] == 1 / 4
    assert saved["hit_at_3"] == 3 / 4 and saved["seed"] == 7


def test_baseline_script_keeps_existing_output_when_scoring_fails(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("run_baseline", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = tmp_path / "windows_synthetic.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "data_dir": str(tmp_path / "missing"),
                "datasets": ["RE2-OB"],
                "repetitions": [1],
                "window_size": 1,
                "max_trace_chars": 1000,
                "guard_seconds": 60,
                "t1_windows_per_phase": 1,
                "t2_windows": 1,
                "output_windows": "unused",
                "output_stats": str(tmp_path / "results" / "window_stats" / "synthetic.csv"),
            }
        )
    )
    output = tmp_path / "results" / "baselines" / "selftime_synthetic.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text('{"committed": true}\n')
    trace = Trace("t1", tuple(_span(s, s, 10) for s in ("a", "b", "c", "d")))
    record = score_window(_window(label="d"), [trace], ["a", "b", "c", "d"], 7)

    def fail_after_first(config):
        yield record
        raise FileNotFoundError("Case 'b': traces.parquet was not found.")

    monkeypatch.setattr(script, "run_baseline", fail_after_first)
    monkeypatch.setattr(sys, "argv", [str(path), str(config)])
    with pytest.raises(FileNotFoundError, match="traces.parquet"):
        script.main()
    assert output.read_text() == '{"committed": true}\n'
