import hashlib
import importlib.util
import json
import random
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from traceformatbench.data.trace import Span, Trace
from traceformatbench.data.windows import WindowConfig, build_case_windows
from traceformatbench.runners import tasks as runner
from traceformatbench.runners.ollama import ChatResult
from traceformatbench.tasks.prompts import SYSTEM_PROMPT


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_script = _script("run_tasks")
score_script = _script("score_tasks")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Task tests must not connect to Ollama or the network.")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)


def _config(**overrides):
    return runner.TaskConfig(
        **{
            "seed": 7,
            "prompt_version": "v1",
            "ollama_url": "http://unused",
            "models": ["m1", "m2"],
            "num_ctx": 32768,
            "num_predict": 256,
            "temperature": 0,
            "windows_config": "unused",
            "cases": ["case"],
            "formats": ["F0", "F1"],
            "tasks": ["T1", "T2"],
            "output": "unused",
            **overrides,
        }
    )


def _result(**overrides):
    return replace(
        ChatResult(
            content='{"phase":"pre"}',
            thinking="",
            done_reason="stop",
            prompt_eval_count=100,
            prompt_eval_cached_count=3,
            eval_count=10,
            load_duration_ns=1_000_000_000,
            prompt_eval_duration_ns=2_000_000_000,
            eval_duration_ns=4_000_000_000,
            total_duration_ns=7_000_000_000,
        ),
        **overrides,
    )


def _windows(case="case"):
    return [
        {
            "case": case,
            "dataset": "RE2-OB",
            "system": "ob",
            "fault": "cpu",
            "root_cause_service": "a",
            "task": task,
            "phase": phase,
            "window_index": 0,
            "informative": True if task == "T2" else None,
            "label": phase if task == "T1" else "a",
            "trace_ids": ["trace"],
            "texts": {fmt: f"{fmt} {case} {phase}" for fmt in ("F0", "F1")},
            "candidates": ["a", "b", "c"],
        }
        for task, phase in (("T1", "pre"), ("T1", "post"), ("T2", "post"))
    ]


class FakeClient:
    def __init__(self):
        self.events = []
        self.calls = []

    def version(self):
        self.events.append(("version",))
        return "v"

    def model_digest(self, model):
        self.events.append(("digest", model))
        return f"digest-{model}"

    def model_capabilities(self, model):
        self.events.append(("capabilities", model))
        return ["thinking"] if model == "m1" else []

    def chat(self, **kwargs):
        self.events.append(("chat", kwargs["model"]))
        self.calls.append(kwargs)
        return _result()

    def unload_model(self, model):
        self.events.append(("unload", model))


@pytest.fixture
def fake_inputs(monkeypatch):
    monkeypatch.setattr(runner, "load_window_config", lambda path: None)
    monkeypatch.setattr(runner, "load_task_windows", lambda *args: _windows())


def test_call_order_and_label_keys_separate_pre_post():
    config = _config(cases=["case2", "case"])
    calls = runner.build_call_list(_windows() + _windows("case2"), config)
    assert [(c["model"], c["case"], c["task"], c["label"], c["format"]) for c in calls] == [
        (model, case, task, label, fmt)
        for model in config.models
        for case in config.cases
        for task, label in (("T1", "pre"), ("T1", "post"), ("T2", "a"))
        for fmt in config.formats
    ]
    assert len({runner.record_key(c) for c in calls}) == len(calls)
    assert all(("candidates" in c) == (c["task"] == "T2") for c in calls)


def test_runner_preflight_options_schemas_and_sequential_models(fake_inputs):
    client = FakeClient()
    records = list(runner.run_tasks(_config(), client))
    assert client.events[:5] == [
        ("version",),
        ("digest", "m1"),
        ("digest", "m2"),
        ("capabilities", "m1"),
        ("capabilities", "m2"),
    ]
    assert client.events[5:] == (
        [("chat", "m1")] * 6 + [("unload", "m1")] + [("chat", "m2")] * 6 + [("unload", "m2")]
    )
    assert len(records) == 12
    for call in client.calls:
        assert (
            call["num_ctx"],
            call["num_predict"],
            call["temperature"],
            call["seed"],
            call["think"],
        ) == (32768, 256, 0, 7, False)
        assert call["system"] == SYSTEM_PROMPT
        properties = call["output_schema"]["properties"]
        if "ranking" in properties:
            assert properties["ranking"]["items"]["enum"] == ["a", "b", "c"]
    assert all(r["think_sent"] is False for r in records[:6])
    assert all(r["think_sent"] is None for r in records[6:])


def test_raw_record_fields_hash_and_no_trace_text(fake_inputs):
    client = FakeClient()
    records = list(runner.run_tasks(_config(), client))
    expected = set(
        "case dataset system fault root_cause_service task window_index informative label "
        "model model_digest ollama_version format prompt_version seed num_ctx num_predict "
        "temperature think_sent prompt_sha256 prompt_chars prompt_eval_count "
        "prompt_eval_cached_count eval_count done_reason total_s load_s content status".split()
    )
    for record, call in zip(records, client.calls, strict=True):
        assert set(record) == expected | ({"candidates"} if record["task"] == "T2" else set())
        messages = [
            {"role": "system", "content": call["system"]},
            {"role": "user", "content": call["user"]},
        ]
        digest = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert record["prompt_sha256"] == digest
        assert record["prompt_chars"] == len(call["system"]) + len(call["user"])
        assert record["total_s"] == 7 and record["load_s"] == 1
        assert record["prompt_eval_cached_count"] == 3


@pytest.mark.parametrize(
    "result, status",
    [
        (_result(prompt_eval_count=32768), "truncated"),
        (_result(thinking="reasoning"), "thinking_leak"),
        (_result(content="<think>reasoning</think>"), "thinking_leak"),
        (_result(), "ok"),
    ],
)
def test_call_status_is_retained(result, status):
    call = runner.build_call_list(_windows(), _config())[0]
    record = runner.make_record(
        call, result, model_digest="d", ollama_version="v", think_sent=False
    )
    assert record["status"] == status


@pytest.mark.parametrize("failure", ["server", "model"])
def test_preflight_failure_sends_no_chat(fake_inputs, failure):
    client = FakeClient()
    if failure == "server":
        client.version = Mock(side_effect=urllib.error.URLError("offline"))
        error = urllib.error.URLError
    else:
        client.model_digest = Mock(side_effect=["digest", LookupError("Model m2 not pulled")])
        error = LookupError
    with pytest.raises(error):
        list(runner.run_tasks(_config(), client))
    assert client.calls == []


def test_overlong_later_prompt_stops_before_first_chat(monkeypatch, fake_inputs):
    windows = _windows()
    windows[-1]["texts"]["F1"] = "x" * 100_000
    monkeypatch.setattr(runner, "load_task_windows", lambda *args: windows)
    client = FakeClient()
    with pytest.raises(ValueError, match="may exceed num_ctx"):
        list(runner.run_tasks(_config(), client))
    assert client.calls == []


def test_failed_chat_unloads_current_model(fake_inputs):
    client = FakeClient()
    client.chat = Mock(side_effect=TimeoutError("interrupted"))
    with pytest.raises(TimeoutError):
        list(runner.run_tasks(_config(), client))
    assert client.events[-1] == ("unload", "m1")
    assert ("unload", "m2") not in client.events


def test_resume_skips_previous_calls_including_failed_calls(tmp_path, fake_inputs):
    config = _config()
    first = list(runner.run_tasks(config, FakeClient()))
    previous = first[:7]
    previous[1]["status"] = "truncated"
    previous[2]["status"] = "thinking_leak"
    output = tmp_path / "calls.jsonl"
    output.write_text("".join(json.dumps(r) + "\n" for r in previous) + '{"interrupted":')
    done = frozenset(
        runner.record_key(r) for r in runner.read_records(output, allow_incomplete_tail=True)
    )
    client = FakeClient()
    rest = list(runner.run_tasks(config, client, done))
    assert len(rest) == len(client.calls) == len(first) - len(previous)
    assert all(c["model"] == "m2" for c in client.calls)
    assert {runner.record_key(r) for r in rest}.isdisjoint(done)
    assert {runner.record_key(r) for r in rest} | done == {runner.record_key(r) for r in first}
    client = FakeClient()
    assert (
        list(runner.run_tasks(config, client, frozenset(runner.record_key(r) for r in first))) == []
    )
    assert client.calls == []


def test_resume_keys_change_when_prompt_or_generation_settings_change():
    call = runner.build_call_list(_windows(), _config())[0]
    for key, value in (("prompt_sha256", "changed"), ("seed", 42), ("num_ctx", 8192)):
        assert runner.record_key({**call, key: value}) != runner.record_key(call)


def test_run_and_score_scripts_resume_and_rescore_offline(tmp_path, monkeypatch, fake_inputs):
    config = _config(output=str(tmp_path / "results" / "calls.jsonl"))
    output = Path(config.output)
    previous = list(runner.run_tasks(config, FakeClient()))[:2]
    output.parent.mkdir()
    output.write_text("".join(json.dumps(r) + "\n" for r in previous) + '{"interrupted":')
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__))
    client = FakeClient()
    monkeypatch.setattr(run_script, "OllamaClient", lambda url: client)
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", str(config_path)])
    run_script.main()
    records = runner.read_records(output)
    assert records[:2] == previous and len(records) == 12 and len(client.calls) == 10
    raw_bytes = output.read_bytes()
    monkeypatch.setattr(sys, "argv", ["score_tasks.py", str(output)])
    score_script.main()
    assert len(runner.read_records(output.with_name("scores.jsonl"))) == 12
    assert len(pd.read_csv(output.with_name("metrics.csv"))) == 16
    assert output.read_bytes() == raw_bytes
    before = output.with_name("scores.jsonl").read_bytes()
    score_script.main()
    assert output.with_name("scores.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "error, message",
    [
        (urllib.error.URLError("offline"), "Start Ollama"),
        (LookupError("Model m2 is not pulled"), "not pulled"),
    ],
)
def test_script_clear_errors_before_first_call(tmp_path, monkeypatch, fake_inputs, error, message):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(_config(output=str(tmp_path / "calls.jsonl")).__dict__))
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", str(config_path)])
    client = FakeClient()
    client.version = Mock(side_effect=error)
    monkeypatch.setattr(run_script, "OllamaClient", lambda url: client)
    with pytest.raises(SystemExit, match=message):
        run_script.main()
    assert client.calls == []


def test_record_reader_rejects_interior_corruption_and_strict_incomplete_tail(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("{}\nbroken\n{}\n")
    with pytest.raises(json.JSONDecodeError):
        runner.read_records(path, allow_incomplete_tail=True)
    path.write_text('{}\n{"interrupted":')
    with pytest.raises(json.JSONDecodeError):
        runner.read_records(path)
    assert runner.read_records(path, allow_incomplete_tail=True) == [{}]


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "local"
    data_dir.mkdir()
    metadata = [
        {
            "case": case,
            "dataset": "RE2-OB",
            "system": "ob",
            "fault": "cpu",
            "root_cause_service": "a",
            "inject_time": 1000,
        }
        for case in ("case", "other")
    ]
    pd.DataFrame(metadata).to_parquet(data_dir / "cases.parquet", index=False)
    traces = [
        Trace(
            f"{phase}{i}",
            tuple(
                Span(
                    f"{phase}{i}-{service}",
                    None if service == "a" else f"{phase}{i}-a",
                    f"{phase}{i}",
                    service,
                    "op",
                    None,
                    start * 1_000_000,
                    10,
                    None,
                )
                for service in ("a", "b", "c")
            ),
        )
        for phase, start in (("pre", 100), ("post", 1100))
        for i in range(6)
    ]
    # A service only visible in a guard-band orphan must still be a T2 candidate.
    traces.append(
        Trace(
            "orphan",
            (Span("orphan-span", "missing", "orphan", "z", "op", None, 1000_000_000, 1, None),),
        )
    )
    monkeypatch.setattr(runner, "load_case_traces", lambda path: traces)
    config = WindowConfig(
        name="windows_v1",
        seed=19,
        data_dir=str(data_dir),
        datasets=["RE2-OB"],
        repetitions=[1],
        window_size=2,
        max_trace_chars=10000,
        guard_seconds=60,
        t1_windows_per_phase=2,
        t2_windows=3,
        output_windows="data/windows/windows_v1.jsonl",
        output_stats="unused",
    )
    return metadata, traces, config


def test_local_loader_rebuilds_exact_windows_rng_and_all_trace_candidates(local_inputs):
    metadata, traces, windows_config = local_inputs
    config = _config(cases=["other", "case"])
    windows = runner.load_task_windows(config, windows_config)
    assert len(windows) == 14
    for case in config.cases:
        meta = next(m for m in metadata if m["case"] == case)
        expected, _ = build_case_windows(meta, traces, windows_config, random.Random(f"19:{case}"))
        actual = [w for w in windows if w["case"] == case]
        assert [
            {k: v for k, v in w.items() if k not in {"texts", "candidates"}} for w in actual
        ] == expected
        assert all(w["candidates"] == ["a", "b", "c", "z"] for w in actual)


@pytest.mark.parametrize("mismatch", ["trace", "missing", "extra", "duplicate"])
def test_saved_windows_mismatch_stops_before_first_call(local_inputs, monkeypatch, mismatch):
    _, _, windows_config = local_inputs
    config = _config()
    windows = runner.load_task_windows(config, windows_config)
    path = Path(windows_config.output_windows)
    path.parent.mkdir(parents=True)
    saved = [{k: v for k, v in w.items() if k not in {"texts", "candidates"}} for w in windows]
    path.write_text("".join(json.dumps(w) + "\n" for w in saved))
    assert runner.load_task_windows(config, windows_config) == windows
    if mismatch == "trace":
        saved[-1]["trace_ids"] = ["different"]
    elif mismatch == "missing":
        saved.pop()
    elif mismatch == "extra":
        saved.append({**saved[-1], "window_index": 99})
    else:
        saved.append(saved[-1])
    path.write_text("".join(json.dumps(w) + "\n" for w in saved))
    monkeypatch.setattr(runner, "load_window_config", lambda path: windows_config)
    client = FakeClient()
    with pytest.raises(ValueError, match="match|Duplicate"):
        list(runner.run_tasks(config, client))
    assert client.calls == []


def test_root_cause_mismatch_lists_all_cases_before_first_call(local_inputs, monkeypatch):
    metadata, _, windows_config = local_inputs
    for meta in metadata:
        meta["root_cause_service"] = "absent"
    pd.DataFrame(metadata).to_parquet(Path(windows_config.data_dir) / "cases.parquet", index=False)
    monkeypatch.setattr(runner, "load_window_config", lambda path: windows_config)
    client = FakeClient()
    with pytest.raises(ValueError, match="Root-cause.*case.*other"):
        list(runner.run_tasks(_config(cases=["case", "other"]), client))
    assert client.calls == []


def test_fewer_than_three_candidates_stops_before_first_call(local_inputs, monkeypatch):
    _, traces, windows_config = local_inputs
    monkeypatch.setattr(
        runner,
        "load_case_traces",
        lambda path: [
            Trace(t.trace_id, tuple(s for s in t.spans if s.service in {"a", "b"}))
            for t in traces
            if t.trace_id != "orphan"
        ],
    )
    monkeypatch.setattr(runner, "load_window_config", lambda path: windows_config)
    client = FakeClient()
    with pytest.raises(ValueError, match="T2 needs at least 3 candidates.*case"):
        list(runner.run_tasks(_config(), client))
    assert client.calls == []


def test_candidate_errors_are_listed_even_when_cases_have_no_eligible_windows(
    local_inputs, monkeypatch
):
    metadata, traces, windows_config = local_inputs
    for meta in metadata:
        meta["root_cause_service"] = "absent"
    pd.DataFrame(metadata).to_parquet(Path(windows_config.data_dir) / "cases.parquet", index=False)
    monkeypatch.setattr(runner, "load_case_traces", lambda path: [traces[-1]])
    with pytest.raises(ValueError, match="Root-cause.*case.*other.*T2 needs at least 3"):
        runner.load_task_windows(_config(cases=["case", "other"]), windows_config)


def test_smoke_config_matches_spec_without_running_it():
    path = Path(__file__).parents[1] / "configs" / "tasks_smoke.yaml"
    config = runner.TaskConfig(**yaml.safe_load(path.read_text()))
    assert config == runner.TaskConfig(
        seed=20260930,
        prompt_version="v1",
        ollama_url="http://localhost:11434",
        models=["qwen3.5:9b", "gemma4:12b"],
        num_ctx=32768,
        num_predict=256,
        temperature=0,
        windows_config="configs/windows_v1.yaml",
        cases=["re2ob_checkoutservice_cpu_1", "re2tt_ts-auth-service_cpu_1"],
        formats=["F0", "F1"],
        tasks=["T1", "T2"],
        output="results/tasks_smoke/calls.jsonl",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_version": "v2"},
        {"formats": ["F2"]},
        {"tasks": ["T3"]},
        {"models": []},
        {"cases": ["case", "case"]},
        {"num_ctx": 0},
        {"num_predict": 0},
        {"temperature": 0.5},
    ],
)
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)
