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
from traceformatbench.serializers import SERIALIZERS
from traceformatbench.tasks.pairs import align_pair_times
from traceformatbench.tasks.prompts import SYSTEM_PROMPT, build_t1p_prompt
from traceformatbench.tasks.schemas import t1p_schema
from traceformatbench.tasks.scoring import score_record


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_script = _script("run_tasks")
score_script = _script("score_tasks")
check_script = _script("check_grid")


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
            "traces": [
                Trace(
                    "trace",
                    (
                        Span(
                            "span",
                            None,
                            "trace",
                            "a",
                            "op",
                            None,
                            100_000_000 if phase == "pre" else 1100_000_000,
                            10,
                            None,
                        ),
                    ),
                )
            ],
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
        "prompt_eval_cached_count eval_count done_reason total_s load_s content status "
        "recount_num_ctx recount_prompt_eval_count status_before_recount".split()
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


class BandClient(FakeClient):
    """The first request returns `first` tokens; a re-count returns `recount` and another answer."""

    def __init__(self, first=16384, recount=16384):
        super().__init__()
        self.first, self.recount = first, recount

    def chat(self, **kwargs):
        super().chat(**kwargs)
        if kwargs["num_ctx"] == 32768:
            return _result(prompt_eval_count=self.first)
        return _result(prompt_eval_count=self.recount, content='{"phase":"post"}')


def _t1_pre_config():
    return _config(models=["m1"], formats=["F0"], tasks=["T1"])


@pytest.mark.parametrize("count", [16368, 16384, 16400])
def test_band_hit_with_equal_recount_is_ok_and_scores_original_answer(fake_inputs, count):
    client = BandClient(first=count, recount=count)
    records = list(runner.run_tasks(_t1_pre_config(), client))
    assert [call["num_ctx"] for call in client.calls] == [32768, 24576] * 2
    assert client.calls[1] == {**client.calls[0], "num_ctx": 24576}
    record = records[0]
    assert record["label"] == "pre"
    assert (
        record["status"],
        record["status_before_recount"],
        record["recount_num_ctx"],
        record["recount_prompt_eval_count"],
        record["prompt_eval_count"],
        record["num_ctx"],
        record["content"],
    ) == ("ok", "truncated", 24576, count, count, 32768, '{"phase":"pre"}')
    scored = score_record(record)
    assert scored["answer"] == "pre" and scored["correct"] == 1


def test_band_hit_with_different_recount_stays_truncated(fake_inputs):
    client = BandClient(first=16384, recount=16383)
    record = list(runner.run_tasks(_t1_pre_config(), client))[0]
    assert len(client.calls) == 4
    assert (
        record["status"],
        record["status_before_recount"],
        record["recount_num_ctx"],
        record["recount_prompt_eval_count"],
        record["content"],
    ) == ("truncated", "truncated", 24576, 16383, '{"phase":"pre"}')
    assert score_record(record)["correct"] == 0


@pytest.mark.parametrize("count", [100, 16367, 16401, 32767])
def test_calls_outside_band_send_no_second_request(fake_inputs, count):
    client = BandClient(first=count)
    records = list(runner.run_tasks(_t1_pre_config(), client))
    assert [call["num_ctx"] for call in client.calls] == [32768, 32768]
    for record in records:
        assert record["status"] == "ok"
        assert record["recount_num_ctx"] is None
        assert record["recount_prompt_eval_count"] is None
        assert record["status_before_recount"] is None


@pytest.mark.parametrize("count", [32768, 40000])
def test_count_at_or_above_num_ctx_is_truncated_without_recount(fake_inputs, count):
    client = BandClient(first=count)
    records = list(runner.run_tasks(_t1_pre_config(), client))
    assert [call["num_ctx"] for call in client.calls] == [32768, 32768]
    assert all(r["status"] == "truncated" and r["recount_num_ctx"] is None for r in records)


def test_normal_calls_send_one_request_each(fake_inputs):
    client = FakeClient()
    records = list(runner.run_tasks(_config(), client))
    assert len(client.calls) == len(records) == 12
    assert all(call["num_ctx"] == 32768 for call in client.calls)
    assert all(r["recount_num_ctx"] is None for r in records)


def test_equal_recount_keeps_thinking_leak_status():
    call = runner.build_call_list(_windows(), _config())[0]
    record = runner.make_record(
        call,
        _result(prompt_eval_count=16384, thinking="reasoning"),
        model_digest="d",
        ollama_version="v",
        think_sent=False,
        recount=_result(prompt_eval_count=16384),
    )
    assert (record["status"], record["status_before_recount"]) == ("thinking_leak", "truncated")


@pytest.mark.parametrize("count", [100, 32768])
def test_recount_outside_band_is_rejected(count):
    call = runner.build_call_list(_windows(), _config())[0]
    with pytest.raises(ValueError, match="band alone"):
        runner.make_record(
            call,
            _result(prompt_eval_count=count),
            model_digest="d",
            ollama_version="v",
            think_sent=False,
            recount=_result(prompt_eval_count=count),
        )


def test_recount_checks_prompt_length_against_smaller_num_ctx(monkeypatch, fake_inputs):
    windows = _windows()
    windows[0]["texts"]["F0"] = "x" * 50_000
    monkeypatch.setattr(runner, "load_task_windows", lambda *args: windows)
    client = BandClient()
    with pytest.raises(ValueError, match="may exceed num_ctx=24576"):
        list(runner.run_tasks(_t1_pre_config(), client))
    assert [call["num_ctx"] for call in client.calls] == [32768]
    assert client.events[-1] == ("unload", "m1")


def test_resume_does_not_repeat_completed_recount(tmp_path, fake_inputs):
    config = _t1_pre_config()
    first = list(runner.run_tasks(config, BandClient()))
    output = tmp_path / "calls.jsonl"
    output.write_text(json.dumps(first[0]) + "\n")
    saved = runner.read_records(output)
    assert saved[0]["recount_num_ctx"] == 24576 and saved[0]["status"] == "ok"
    client = BandClient()
    rest = list(runner.run_tasks(config, client, frozenset(runner.record_key(r) for r in saved)))
    assert [runner.record_key(r) for r in rest] == [runner.record_key(first[1])]
    assert [call["num_ctx"] for call in client.calls] == [32768, 24576]
    assert all(
        runner.prompt_sha256(call["system"], call["user"]) == first[1]["prompt_sha256"]
        for call in client.calls
    )
    client = BandClient()
    done = frozenset(runner.record_key(r) for r in first)
    assert list(runner.run_tasks(config, client, done)) == []
    assert client.calls == []


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


@pytest.mark.parametrize("existing_output", [False, True])
def test_dry_run_counts_full_call_list_without_client_or_output(
    tmp_path, monkeypatch, capsys, existing_output
):
    monkeypatch.setattr(run_script, "load_window_config", lambda path: None)
    monkeypatch.setattr(run_script, "load_task_windows", lambda *args: _windows())
    config = _config(output=str(tmp_path / "results" / "calls.jsonl"))
    output = Path(config.output)
    if existing_output:
        output.parent.mkdir()
        output.write_bytes(b"Existing output must not be read, resumed or rewritten.\n")
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__))
    client = Mock(side_effect=AssertionError("Dry run must not create an Ollama client."))
    run = Mock(side_effect=AssertionError("Dry run must not start task execution."))
    reader = Mock(side_effect=AssertionError("Dry run must not read output for resume."))
    monkeypatch.setattr(run_script, "OllamaClient", client)
    monkeypatch.setattr(run_script, "run_tasks", run)
    monkeypatch.setattr(run_script, "read_records", reader)
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", str(config_path), "--dry-run"])
    run_script.main()
    lines = capsys.readouterr().out.splitlines()
    for model in config.models:
        for task, count in (("T1", 2), ("T2", 1)):
            for fmt in config.formats:
                assert f"{model} {task} {fmt}: {count}" in lines
    assert "Total calls: 12" in lines
    assert "T1: 0 cases skipped" in lines
    assert "T2: 0 cases skipped" in lines
    calls = runner.build_call_list(_windows(), config)
    assert f"Largest prompt in characters: {max(c['prompt_chars'] for c in calls)}" in lines
    client.assert_not_called()
    run.assert_not_called()
    reader.assert_not_called()
    if existing_output:
        assert output.read_bytes() == b"Existing output must not be read, resumed or rewritten.\n"
    else:
        assert not output.parent.exists()


def test_dry_run_prints_skipped_cases_per_task_and_reasons(tmp_path, monkeypatch, capsys):
    windows = [w for w in _windows("no-root") if w["task"] == "T1"]
    for window in windows:
        window["root_cause_service"] = "absent"
    windows += [w for w in _windows("no-t1") if w["task"] == "T2"]
    monkeypatch.setattr(run_script, "load_window_config", lambda path: None)
    monkeypatch.setattr(run_script, "load_task_windows", lambda *args: windows)
    config = _config(cases=["no-root", "no-t1"], output=str(tmp_path / "calls.jsonl"))
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__))
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", str(config_path), "--dry-run"])
    client = Mock(side_effect=AssertionError("Dry run must not create an Ollama client."))
    monkeypatch.setattr(run_script, "OllamaClient", client)
    run_script.main()
    lines = capsys.readouterr().out.splitlines()
    assert "Total calls: 12" in lines
    assert "T1: 1 cases skipped" in lines
    assert "  no-t1: no T1 windows" in lines
    assert "T2: 1 cases skipped" in lines
    assert "  no-root: root cause not in traces" in lines
    client.assert_not_called()
    assert not Path(config.output).exists()


def test_dry_run_checks_prompt_length_without_client_or_output(tmp_path, monkeypatch):
    windows = _windows()
    windows[-1]["texts"]["F1"] = "x" * 100_000
    monkeypatch.setattr(run_script, "load_window_config", lambda path: None)
    monkeypatch.setattr(run_script, "load_task_windows", lambda *args: windows)
    config = _config(output=str(tmp_path / "calls.jsonl"))
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__))
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", str(config_path), "--dry-run"])
    client = Mock(side_effect=AssertionError("Dry run must not create an Ollama client."))
    monkeypatch.setattr(run_script, "OllamaClient", client)
    with pytest.raises(SystemExit, match="may exceed num_ctx"):
        run_script.main()
    client.assert_not_called()
    assert not Path(config.output).exists()


@pytest.fixture
def grid(tmp_path, monkeypatch, fake_inputs):
    """A complete synthetic grid of 12 records and a config file, planned without Ollama."""
    monkeypatch.setattr(check_script, "load_window_config", lambda path: None)
    monkeypatch.setattr(check_script, "load_task_windows", lambda *args: _windows())
    config = _config(output=str(tmp_path / "calls.jsonl"))
    config_path = tmp_path / "grid.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__))
    return config_path, Path(config.output), list(runner.run_tasks(config, FakeClient()))


def _check_grid(grid, records, monkeypatch, capsys):
    config_path, output, _ = grid
    output.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(sys, "argv", ["check_grid.py", str(config_path), str(output)])
    with pytest.raises(SystemExit) as exit_info:
        check_script.main()
    return exit_info.value.code, capsys.readouterr().out.splitlines()


def test_check_grid_passes_on_complete_grid(grid, monkeypatch, capsys):
    records = grid[2]
    records[0].update(
        status="ok",
        recount_num_ctx=24576,
        recount_prompt_eval_count=16384,
        status_before_recount="truncated",
    )
    records[1]["status"] = "truncated"
    records[2]["status"] = "thinking_leak"
    code, lines = _check_grid(grid, records, monkeypatch, capsys)
    assert code == 0
    assert lines == [
        "PASS  calls: 12 planned, 12 records; missing 0, extra 0, duplicate keys 0",
        "PASS  model m1: 6 records; model_digest digest-m1; ollama_version v",
        "PASS  model m2: 6 records; model_digest digest-m2; ollama_version v",
        "PASS  statuses: ok 10, truncated 1, thinking_leak 1, other 0; "
        "re-counted 1 (1 no longer truncated)",
        "All checks passed.",
    ]


def _extra(records):
    return [*records, {**records[0], "seed": 8}]


def _two_digests(records):
    records[0]["model_digest"] = "other"
    return records


def _two_versions(records):
    records[6]["ollama_version"] = "v2"
    return records


def _unknown_status(records):
    records[0]["status"] = "error"
    return records


@pytest.mark.parametrize(
    "change, failed",
    [
        (lambda r: r[:-1], "calls: 12 planned, 11 records; missing 1, extra 0, duplicate keys 0"),
        (_extra, "calls: 12 planned, 13 records; missing 0, extra 1, duplicate keys 0"),
        (
            lambda r: [*r, r[0]],
            "calls: 12 planned, 13 records; missing 0, extra 0, duplicate keys 1",
        ),
        (_two_digests, "model m1: 6 records; model_digest digest-m1, other; ollama_version v"),
        (_two_versions, "model m2: 6 records; model_digest digest-m2; ollama_version v, v2"),
        (lambda r: r[:6], "model m2: 0 records; model_digest -; ollama_version -"),
        (
            _unknown_status,
            "statuses: ok 11, truncated 0, thinking_leak 0, other 1; "
            "re-counted 0 (0 no longer truncated)",
        ),
    ],
    ids=["missing", "extra", "duplicate", "two-digests", "two-versions", "no-model", "status"],
)
def test_check_grid_fails(grid, monkeypatch, capsys, change, failed):
    code, lines = _check_grid(grid, change(grid[2]), monkeypatch, capsys)
    assert code == 1
    assert f"FAIL  {failed}" in lines
    assert lines[-1] == "Some checks failed."


def test_check_grid_without_calls_file_exits_with_message(grid, monkeypatch):
    config_path, output, _ = grid
    monkeypatch.setattr(sys, "argv", ["check_grid.py", str(config_path), str(output)])
    with pytest.raises(SystemExit, match="No calls file"):
        check_script.main()


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


def test_t1p_pairs_reuse_t1_windows_and_counterbalance_per_case(local_inputs):
    _, _, windows_config = local_inputs
    config = _config(tasks=["T1P"], cases=["other", "case"])
    windows = runner.load_task_windows(config, windows_config)
    pairs = {case: runner.pair_t1_windows(windows, case) for case in config.cases}
    calls = runner.build_call_list(windows, config)
    assert len(calls) == 2 * len(config.models) * len(config.cases) * len(config.formats)
    assert [(c["model"], c["case"], c["window_index"], c["format"]) for c in calls] == [
        (model, case, index, fmt)
        for model in config.models
        for case in config.cases
        for index in (0, 1)
        for fmt in config.formats
    ]
    for call in calls:
        pre, post = pairs[call["case"]][call["window_index"]]
        assert pre in windows and post in windows
        assert all(trace.startswith("pre") for trace in pre["trace_ids"])
        assert all(trace.startswith("post") for trace in post["trace_ids"])
        assert pre["case"] == post["case"] == call["case"]
        assert pre["window_index"] == post["window_index"] == call["window_index"]
        assert call["a_phase"] == ("pre" if call["window_index"] == 0 else "post")
        assert call["label"] == ("B" if call["window_index"] == 0 else "A")
        assert call["task"] == "T1P" and call["informative"] is None
        assert call["time_aligned"] is True
        assert call["output_schema"] == t1p_schema()
        assert "candidates" not in call
        aligned_pre, aligned_post = align_pair_times(pre["traces"], post["traces"])
        aligned_a, aligned_b = (
            (aligned_pre, aligned_post)
            if call["window_index"] == 0
            else (aligned_post, aligned_pre)
        )
        assert call["user_prompt"] == build_t1p_prompt(
            SERIALIZERS[call["format"]](aligned_a),
            SERIALIZERS[call["format"]](aligned_b),
            windows_config.window_size,
        )
    original = runner.build_call_list(windows, replace(config, tasks=["T1", "T2"]))
    mixed = runner.build_call_list(windows, replace(config, tasks=["T1", "T1P", "T2"]))
    assert [c for c in mixed if c["task"] != "T1P"] == original
    assert [c for c in mixed if c["task"] == "T1P"] == calls


@pytest.mark.parametrize("missing_phase", ["pre", "post"])
def test_t1p_builds_only_complete_pairs(local_inputs, missing_phase):
    _, _, windows_config = local_inputs
    config = _config(tasks=["T1P"])
    windows = runner.load_task_windows(config, windows_config)
    windows = [
        w
        for w in windows
        if not (w["task"] == "T1" and w["phase"] == missing_phase and w["window_index"] == 0)
    ]
    calls = runner.build_call_list(windows, config)
    assert len(calls) == len(config.models) * len(config.formats)
    assert all(
        c["window_index"] == 1 and c["a_phase"] == "post" and c["label"] == "A" for c in calls
    )


def test_t1p_no_pair_stops_before_first_chat(local_inputs, monkeypatch):
    _, _, windows_config = local_inputs
    monkeypatch.setattr(
        runner, "load_window_config", lambda path: replace(windows_config, window_size=7)
    )
    client = FakeClient()
    with pytest.raises(ValueError, match="Case 'case'.*no complete T1P pairs"):
        list(runner.run_tasks(_config(tasks=["T1P"]), client))
    assert client.calls == []


def test_t1p_pair_without_matching_index_is_not_complete():
    windows = _windows()
    windows[1]["window_index"] = 1
    with pytest.raises(ValueError, match="no complete T1P pairs"):
        runner.build_call_list(windows, _config(tasks=["T1P"]))


def test_t1p_asserts_shared_format():
    windows = _windows()
    del windows[1]["texts"]["F1"]
    with pytest.raises(AssertionError):
        runner.build_call_list(windows, _config(tasks=["T1P"]))


def test_t1p_runner_records_resume_and_prompt_length_guard(monkeypatch, fake_inputs):
    config = _config(tasks=["T1P"], num_ctx=65536)
    client = FakeClient()
    records = list(runner.run_tasks(config, client))
    assert len(records) == len(client.calls) == 4
    for record, call in zip(records, client.calls, strict=True):
        assert record["a_phase"] == "pre" and record["label"] == "B"
        assert record["informative"] is None
        assert record["num_ctx"] == call["num_ctx"] == 65536
        assert record["task"] == "T1P" and call["output_schema"] == t1p_schema()
        assert record["time_aligned"] is True
        assert record["prompt_sha256"] == runner.prompt_sha256(call["system"], call["user"])
        assert (
            not {"system_prompt", "user_prompt", "output_schema", "texts", "trace_ids", "traces"}
            & record.keys()
        )
    done = frozenset(runner.record_key(r) for r in records[:1])
    resumed = list(runner.run_tasks(config, FakeClient(), done))
    assert resumed == records[1:]
    windows = _windows()
    trace = windows[1]["traces"][0]
    windows[1]["traces"] = [
        replace(trace, spans=(replace(trace.spans[0], operation="x" * 150_000),))
    ]
    monkeypatch.setattr(runner, "load_task_windows", lambda *args: windows)
    client = FakeClient()
    with pytest.raises(ValueError, match="may exceed num_ctx"):
        list(runner.run_tasks(config, client))
    assert client.calls == []


@pytest.mark.parametrize("fmt", ["F0", "F1"])
def test_t1p_prompt_windows_share_minimum_start_time(local_inputs, fmt):
    _, _, windows_config = local_inputs
    config = _config(tasks=["T1P"], formats=[fmt])
    windows = runner.load_task_windows(config, windows_config)
    for call in runner.build_call_list(windows, config):
        starts = []
        for letter in ("a", "b"):
            text = (
                call["user_prompt"]
                .split(f"<window_{letter}>\n")[1]
                .split(f"\n</window_{letter}>")[0]
            )
            records = json.loads(text)
            spans = (
                records if fmt == "F0" else [span for trace in records for span in trace["spans"]]
            )
            starts.append(min(span["startTime"] for span in spans))
            if fmt == "F0":
                assert min(span["startTimeMillis"] for span in spans) == starts[-1] // 1000
        pre, post = runner.pair_t1_windows(windows, "case")[call["window_index"]]
        anchor = min(t.start_us for w in (pre, post) for t in w["traces"])
        assert starts == [anchor, anchor]


def test_t1_t2_texts_and_calls_match_pre_revision_baseline(local_inputs):
    _, _, windows_config = local_inputs
    config = _config(tasks=["T1", "T2"])
    original_windows = runner.load_task_windows(config, windows_config)
    paired_config = replace(config, tasks=["T1", "T1P", "T2"])
    paired_windows = runner.load_task_windows(paired_config, windows_config)
    original_calls = runner.build_call_list(original_windows, config)
    paired_calls = [
        c for c in runner.build_call_list(paired_windows, paired_config) if c["task"] != "T1P"
    ]
    assert [w["texts"] for w in original_windows] == [w["texts"] for w in paired_windows]
    assert paired_calls == original_calls

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    # Captured from this synthetic case before implementing Revision 1.
    assert digest([w["texts"] for w in original_windows]) == (
        "32d6720c15fd4dc5b3493221c9cc7a0519201ad8c6aaacbc2e50c7fe637486a9"
    )
    assert (
        digest(original_calls) == "28cc71df89b95441e734ac5b64195be8439e3fc06c4e142b5e148e4252c7b96d"
    )


def test_t1p_pilot_config_matches_spec_without_running_it():
    root = Path(__file__).parents[1]
    pilot = runner.TaskConfig(**yaml.safe_load((root / "configs/t1p_pilot.yaml").read_text()))
    smoke = runner.TaskConfig(**yaml.safe_load((root / "configs/tasks_smoke.yaml").read_text()))
    assert pilot == replace(
        smoke, tasks=["T1P"], num_ctx=65536, output="results/t1p_pilot/calls.jsonl"
    )


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
    # The real builder omits T2 for an absent root cause; simulate invalid T2 windows.
    monkeypatch.setattr(
        runner,
        "build_case_windows",
        lambda meta, *args: (
            [{**w, "root_cause_service": "absent"} for w in _windows(meta["case"])],
            {},
        ),
    )
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


def test_no_eligible_windows_raises_without_global_candidate_errors(local_inputs, monkeypatch):
    metadata, traces, windows_config = local_inputs
    for meta in metadata:
        meta["root_cause_service"] = "absent"
    pd.DataFrame(metadata).to_parquet(Path(windows_config.data_dir) / "cases.parquet", index=False)
    monkeypatch.setattr(runner, "load_case_traces", lambda path: [traces[-1]])
    with pytest.raises(ValueError, match="Case 'case'.*no windows for any requested task"):
        runner.load_task_windows(_config(cases=["case", "other"]), windows_config)


@pytest.mark.parametrize("tasks", [["T1", "T2"], ["T1"]])
def test_absent_root_cause_keeps_t1_and_skips_t2(local_inputs, tasks):
    metadata, _, windows_config = local_inputs
    metadata[0]["root_cause_service"] = "absent"
    pd.DataFrame(metadata).to_parquet(Path(windows_config.data_dir) / "cases.parquet", index=False)
    config = _config(tasks=tasks)
    windows = runner.load_task_windows(config, windows_config)
    assert len(windows) == 4 and all(w["task"] == "T1" for w in windows)
    assert all(w["root_cause_service"] not in w["candidates"] for w in windows)
    calls = runner.build_call_list(windows, config)
    assert len(calls) == 4 * len(config.models) * len(config.formats)
    assert {call["task"] for call in calls} == {"T1"}


def test_no_t1_windows_keeps_t2(local_inputs, monkeypatch):
    _, traces, windows_config = local_inputs
    monkeypatch.setattr(runner, "load_case_traces", lambda path: traces[6:])
    config = _config()
    windows = runner.load_task_windows(config, windows_config)
    assert len(windows) == 3 and all(w["task"] == "T2" for w in windows)
    calls = runner.build_call_list(windows, config)
    assert len(calls) == 3 * len(config.models) * len(config.formats)
    assert {call["task"] for call in calls} == {"T2"}


@pytest.mark.parametrize("tasks", [["T1"], ["T2"], ["T1", "T2"]])
def test_no_windows_for_any_requested_task_raises(local_inputs, monkeypatch, tasks):
    metadata, traces, windows_config = local_inputs
    config = _config(tasks=tasks)
    if tasks == ["T1"]:
        # T2 windows exist, but none for the requested T1 task.
        monkeypatch.setattr(runner, "load_case_traces", lambda path: traces[6:])
    elif tasks == ["T2"]:
        # T1 windows exist, but none for the requested T2 task.
        metadata[0]["root_cause_service"] = "absent"
        pd.DataFrame(metadata).to_parquet(
            Path(windows_config.data_dir) / "cases.parquet", index=False
        )
    else:
        windows_config = replace(windows_config, window_size=7)
    with pytest.raises(ValueError, match="Case 'case'.*no windows for any requested task"):
        runner.load_task_windows(config, windows_config)


@pytest.mark.parametrize("n_candidates", [1, 2])
def test_fewer_than_three_candidates_allowed_without_t2_windows(
    local_inputs, monkeypatch, n_candidates
):
    _, traces, windows_config = local_inputs
    services = {"a", "b"} if n_candidates == 2 else {"a"}
    monkeypatch.setattr(
        runner,
        "load_case_traces",
        lambda path: [
            Trace(t.trace_id, tuple(s for s in t.spans if s.service in services))
            for t in traces
            if t.trace_id != "orphan"
        ],
    )
    config = _config()
    windows = runner.load_task_windows(config, replace(windows_config, t2_windows=0))
    assert len(windows) == 4 and all(w["task"] == "T1" for w in windows)
    assert all(len(w["candidates"]) == n_candidates for w in windows)
    assert {call["task"] for call in runner.build_call_list(windows, config)} == {"T1"}


def test_main_grid_config_matches_spec_and_all_sorted_windows_v1_cases():
    root = Path(__file__).parents[1]
    path = root / "configs/main_grid.yaml"
    config = runner.TaskConfig(**yaml.safe_load(path.read_text()))
    windows = [
        json.loads(line)
        for line in (root / "data/windows/windows_v1.jsonl").read_text().splitlines()
    ]
    cases = sorted({window["case"] for window in windows})
    assert len(cases) == 78
    smoke = runner.TaskConfig(**yaml.safe_load((root / "configs/tasks_smoke.yaml").read_text()))
    assert config == replace(
        smoke,
        cases=cases,
        formats=["F0", "F1", "F2", "F3", "F4", "F5"],
        output="results/main_grid/calls.jsonl",
    )
    assert path.read_text().splitlines()[0] == "# Run only after PREREGISTRATION.md is committed."


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
        {"formats": ["F6"]},
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


@pytest.mark.parametrize("fmt", ["F2", "F3", "F4", "F5"])
def test_task_config_accepts_structured_formats(fmt):
    assert _config(formats=[fmt]).formats == [fmt]
