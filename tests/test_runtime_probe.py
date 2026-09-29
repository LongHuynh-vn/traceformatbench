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

from traceformatbench.analysis import runtime_probe as probe
from traceformatbench.data.trace import Span, Trace
from traceformatbench.data.windows import WindowConfig, build_case_windows
from traceformatbench.runners.ollama import ChatResult
from traceformatbench.serializers import SERIALIZERS

script_spec = importlib.util.spec_from_file_location(
    "runtime_probe_script", Path(__file__).parents[1] / "scripts" / "runtime_probe.py"
)
script = importlib.util.module_from_spec(script_spec)
script_spec.loader.exec_module(script)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Runtime-probe tests must not use the network or real Ollama.")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)


def _config(**overrides):
    return probe.RuntimeProbeConfig(
        **{
            "seed": 7,
            "ollama_url": "http://unused",
            "models": ["m1", "m2"],
            "num_ctx": 32768,
            "num_predict": 256,
            "temperature": 0,
            "windows_config": "unused",
            "cases": ["case"],
            "formats": ["F0", "F1"],
            "output": "unused",
            **overrides,
        }
    )


def _result(**overrides):
    return replace(
        ChatResult(
            content='{"phase":"pre","service":"frontend"}',
            thinking="",
            done_reason="stop",
            prompt_eval_count=100,
            prompt_eval_cached_count=0,
            eval_count=20,
            load_duration_ns=1_000_000_000,
            prompt_eval_duration_ns=2_000_000_000,
            eval_duration_ns=4_000_000_000,
            total_duration_ns=10_000_000_000,
        ),
        **overrides,
    )


def _windows(case="case", dataset="RE2-OB"):
    return [
        {
            "case": case,
            "dataset": dataset,
            "task": task,
            "phase": phase,
            "window_index": 0,
            "f0_chars": 10,
            "texts": {"F0": f"F0 {case} {phase}", "F1": f"F1 {case} {phase}"},
        }
        for task, phase in (("T1", "pre"), ("T1", "post"), ("T2", "post"))
    ]


def _record(**overrides):
    call = probe.build_call_list(_windows(), ["m1"], ["F0"])[1]
    return {
        **probe.make_record(
            call,
            _result(),
            _config(),
            model_digest="d1",
            ollama_version="v",
            think_sent=False,
            wall_s=11.0,
        ),
        **overrides,
    }


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
        return ["thinking"] if model == "m1" else ["completion"]

    def chat(self, **kwargs):
        self.events.append(("chat", kwargs["model"]))
        self.calls.append(kwargs)
        return _result()

    def unload_model(self, model):
        self.events.append(("unload", model))


@pytest.fixture
def fake_inputs(monkeypatch):
    monkeypatch.setattr(probe, "load_window_config", lambda path: None)
    windows = _windows("ob", "RE2-OB") + _windows("tt", "RE2-TT")
    monkeypatch.setattr(probe, "load_probe_windows", lambda config, windows_config: windows)
    return windows


def test_call_list_warms_first_window_once_per_model_and_keeps_model_batches():
    windows = _windows("ob", "RE2-OB") + _windows("tt", "RE2-TT")
    calls = probe.build_call_list(windows, ["m1", "m2"], ["F0", "F1"])
    assert len(calls) == 2 * (1 + 6 * 2)
    assert [call["model"] for call in calls] == ["m1"] * 13 + ["m2"] * 13
    assert [i for i, call in enumerate(calls) if call["warmup"]] == [0, 13]
    warmup, first = calls[0], calls[1]
    assert warmup["user_prompt"] == probe.WARMUP_PREFIX + first["user_prompt"]
    assert {**warmup, "warmup": False, "user_prompt": first["user_prompt"]} == first
    assert calls[0]["case"] == "ob" and calls[0]["format"] == "F0"
    assert all(call["system_prompt"] == probe.SYSTEM_PROMPT for call in calls)
    assert all(call["user_prompt"].endswith("\n" + probe.ANSWER_INSTRUCTION) for call in calls)
    assert len({probe.record_key(call) for call in calls}) == len(calls)
    assert windows[0].get("warmup") is None  # pure call-list construction


@pytest.mark.parametrize("offset", [-16, 0, 2, 16])
def test_truncation_signature_is_flagged(offset):
    assert probe.call_status(_result(prompt_eval_count=32768 // 2 + offset), 32768) == "truncated"


@pytest.mark.parametrize("offset", [-17, 17])
def test_counts_just_outside_truncation_signature_are_ok(offset):
    assert probe.call_status(_result(prompt_eval_count=32768 // 2 + offset), 32768) == "ok"


@pytest.mark.parametrize(
    "result",
    [
        _result(thinking="reasoning"),
        _result(thinking=" "),
        _result(content="<think>reasoning</think>{}"),
    ],
)
def test_nonempty_thinking_and_content_tags_are_flagged(result):
    assert probe.call_status(result, 32768) == "thinking_leak"


def test_make_record_keeps_metadata_exact_prompts_seed_and_all_timings():
    call = probe.build_call_list(_windows(), ["m1"], ["F0"])[1]
    result = _result(thinking="private reasoning", prompt_eval_cached_count=5)
    record = probe.make_record(
        call,
        result,
        _config(),
        model_digest="exact-digest",
        ollama_version="exact-version",
        think_sent=False,
        wall_s=11.5,
    )
    assert record["model_digest"] == "exact-digest" and record["ollama_version"] == "exact-version"
    assert (record["num_ctx"], record["num_predict"], record["seed"], record["temperature"]) == (
        32768,
        256,
        7,
        0,
    )
    assert record["think_sent"] is False
    assert record["prompt_chars"] == len(call["system_prompt"]) + len(call["user_prompt"])
    assert (
        record["load_s"],
        record["prompt_eval_s"],
        record["eval_s"],
        record["total_s"],
        record["wall_s"],
    ) == (1, 2, 4, 10, 11.5)
    assert record["thinking_chars"] == len(result.thinking)
    assert record["content"] == result.content and record["status"] == "thinking_leak"
    assert record["prompt_eval_cached_count"] == 5


def test_summary_excludes_warmup_truncation_and_thinking_leaks_and_reports_cache():
    records = [
        _record(prompt_eval_cached_count=20),
        _record(
            dataset="RE3-OB",
            total_s=30,
            prompt_eval_count=200,
            eval_count=40,
            prompt_eval_s=4,
            eval_s=2,
            done_reason="length",
        ),
        _record(warmup=True, total_s=9999, prompt_eval_cached_count=100),
        _record(status="truncated", total_s=9999),
        _record(status="thinking_leak", total_s=9999),
    ]
    summary = probe.summarize(records)
    assert list(summary.columns) == probe.SUMMARY_COLUMNS
    assert summary.to_dict(orient="records") == [
        {
            "model": "m1",
            "format": "F0",
            "system": "OB",
            "n": 2,
            "median_call_s": 19,
            "max_call_s": 29,
            "median_prompt_tokens": 150,
            "median_eval_count": 30,
            "median_prefill_tokens_per_s": 45,
            "median_decode_tokens_per_s": 12.5,
            "length_share": 0.5,
            "n_cached_calls": 1,
            "sum_prompt_eval_cached_count": 20,
        }
    ]


def test_summary_separates_models_formats_and_systems():
    records = [
        _record(model=model, format=fmt, dataset=dataset)
        for model in ("m1", "m2")
        for fmt in ("F0", "F1")
        for dataset in ("RE2-OB", "RE2-TT")
    ]
    summary = probe.summarize(records)
    assert len(summary) == 8 and (summary["n"] == 1).all()
    assert set(summary["system"]) == {"OB", "TT"}


def test_call_time_excludes_model_load_time():
    row = probe.summarize([_record(total_s=25, load_s=5)]).iloc[0]
    assert row["median_call_s"] == 20 and row["max_call_s"] == 20


def test_zero_duration_rates_are_missing_instead_of_infinite():
    row = probe.summarize([_record(prompt_eval_s=0, eval_s=0)]).iloc[0]
    assert pd.isna(row["median_prefill_tokens_per_s"])
    assert pd.isna(row["median_decode_tokens_per_s"])


@pytest.mark.parametrize("records", [[], [_record(warmup=True)], [_record(status="truncated")]])
def test_empty_summary_and_estimate_have_stable_csv_headers(records):
    summary = probe.summarize(records)
    assert summary.empty and list(summary.columns) == probe.SUMMARY_COLUMNS
    estimates = probe.estimate(summary)
    assert estimates.empty and list(estimates.columns) == probe.ESTIMATE_COLUMNS


def test_estimate_formula_on_hand_computed_one_model_one_system():
    summary = pd.DataFrame(
        [
            {"model": "m1", "system": "OB", "format": "F0", "median_call_s": 30},
            {"model": "m1", "system": "OB", "format": "F1", "median_call_s": 12},
        ]
    )
    estimates = probe.estimate(summary)
    assert len(estimates) == 16
    row = estimates.iloc[0].to_dict()
    # 40 OB cases, 2*1+1 windows: 40*3*(30+5*12)/3600 = 3 hours;
    # F0 bound: 40*3*6*30/3600 = 6 hours; calls: 40*3*6 = 720.
    assert row == {
        "t1_windows_per_phase": 1,
        "t2_windows": 1,
        "hours_f1_bound": 3,
        "hours_f0_bound": 6,
        "n_calls": 720,
        "fits_48h_f1_bound": True,
        "fits_48h_f0_bound": True,
    }
    assert set(zip(estimates["t1_windows_per_phase"], estimates["t2_windows"], strict=True)) == {
        (t1, t2) for t1 in range(1, 5) for t2 in range(1, 5)
    }


def test_full_grid_case_weights_call_count_and_48_hour_flags():
    summary = pd.DataFrame(
        [
            {"model": model, "system": system, "format": fmt, "median_call_s": seconds}
            for model in ("m1", "m2")
            for system in ("OB", "TT")
            for fmt, seconds in (("F0", 30), ("F1", 12))
        ]
    )
    row = probe.estimate(summary).query("t1_windows_per_phase == 3 and t2_windows == 3").iloc[0]
    assert row["n_calls"] == 8424
    assert row["hours_f1_bound"] == pytest.approx(35.1)
    assert row["hours_f0_bound"] == pytest.approx(70.2)
    assert bool(row["fits_48h_f1_bound"]) is True
    assert bool(row["fits_48h_f0_bound"]) is False


def test_estimate_does_not_silently_drop_missing_format_timings():
    summary = pd.DataFrame(
        [
            {"model": "m1", "system": "OB", "format": "F0", "median_call_s": 30},
            {"model": "m1", "system": "OB", "format": "F1", "median_call_s": 12},
            {"model": "m1", "system": "TT", "format": "F0", "median_call_s": 30},
        ]
    )
    with pytest.raises(ValueError, match="F0 and F1"):
        probe.estimate(summary)


def test_runner_preflights_all_models_and_unloads_between_model_batches(fake_inputs):
    client = FakeClient()
    records = list(probe.run_probe(_config(), client))
    assert client.events[:5] == [
        ("version",),
        ("digest", "m1"),
        ("digest", "m2"),
        ("capabilities", "m1"),
        ("capabilities", "m2"),
    ]
    timed = [event for event in client.events if event[0] in ("chat", "unload")]
    assert timed == [("chat", "m1")] * 13 + [("unload", "m1")] + [("chat", "m2")] * 13 + [
        ("unload", "m2")
    ]
    assert all(record["think_sent"] is False for record in records[:13])
    assert all(record["think_sent"] is None for record in records[13:])
    for call in client.calls:
        assert (
            call["num_ctx"],
            call["num_predict"],
            call["temperature"],
            call["seed"],
            call["think"],
        ) == (32768, 256, 0, 7, False)


@pytest.mark.parametrize("failure", ["server", "model"])
def test_preflight_failures_stop_before_the_first_call(fake_inputs, failure):
    client = FakeClient()
    if failure == "server":
        client.version = Mock(side_effect=urllib.error.URLError("offline"))
        error = urllib.error.URLError
    else:

        def digest(model):
            if model == "m2":
                raise LookupError("Model 'm2' is not pulled. Run: ollama pull m2")
            return "d1"

        client.model_digest = digest
        error = LookupError
    with pytest.raises(error):
        list(probe.run_probe(_config(), client))
    assert client.calls == []


def test_runner_unloads_on_failed_chat_and_does_not_load_next_model(fake_inputs):
    client = FakeClient()
    client.chat = Mock(side_effect=TimeoutError("interrupted"))
    with pytest.raises(TimeoutError):
        list(probe.run_probe(_config(), client))
    assert client.events[-1] == ("unload", "m1")
    assert ("unload", "m2") not in client.events


def test_long_prompt_is_rejected_before_any_chat_request(monkeypatch, fake_inputs):
    windows = _windows()
    windows[-1]["texts"]["F1"] = "x" * 100_000
    monkeypatch.setattr(probe, "load_probe_windows", lambda config, windows_config: windows)
    client = FakeClient()
    with pytest.raises(ValueError, match="may exceed num_ctx=32768"):
        list(probe.run_probe(_config(), client))
    assert client.calls == []


def test_length_screen_includes_system_message_and_output_allowance():
    call = {"case": "case", "format": "F0", "system_prompt": "x" * 1000, "user_prompt": "y" * 600}
    # ceil(1600/1.95) + 256 = 1077, even though the user text alone would fit.
    with pytest.raises(ValueError, match="No chat request was sent"):
        probe.check_prompt_length(call, _config(num_ctx=1000))
    probe.check_prompt_length(call, _config(num_ctx=2000))


def test_resume_skips_keys_read_from_output_including_warmup_and_invalid_records(
    tmp_path, fake_inputs
):
    config = _config()
    first = list(probe.run_probe(config, FakeClient()))
    previous = first[:15]
    previous[2] = {**previous[2], "status": "truncated"}
    previous[3] = {**previous[3], "status": "thinking_leak"}
    output = tmp_path / "calls.jsonl"
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in previous) + '{"cutoff":', encoding="utf-8"
    )
    done = frozenset(probe.record_key(record) for record in script._read_records(output))
    client = FakeClient()
    rest = list(probe.run_probe(config, client, done))
    assert len(rest) == len(client.calls) == len(first) - len(previous)
    assert {probe.record_key(record) for record in rest}.isdisjoint(done)
    assert {probe.record_key(record) for record in rest} | done == {
        probe.record_key(record) for record in first
    }
    assert all(call["model"] == "m2" for call in client.calls)
    assert all(not record["warmup"] for record in rest)
    complete_client = FakeClient()
    assert (
        list(
            probe.run_probe(
                config, complete_client, frozenset(probe.record_key(record) for record in first)
            )
        )
        == []
    )
    assert complete_client.calls == []


def test_script_repairs_interrupted_tail_appends_calls_and_writes_both_csvs(
    tmp_path, monkeypatch, fake_inputs, capsys
):
    config = _config(output=str(tmp_path / "results" / "calls.jsonl"))
    previous = list(probe.run_probe(config, FakeClient()))[:2]
    output = Path(config.output)
    output.parent.mkdir()
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in previous) + '{"cutoff":', encoding="utf-8"
    )
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runtime_probe.py", str(config_path)])
    client = FakeClient()
    monkeypatch.setattr(script, "OllamaClient", lambda url: client)
    script.main()
    records = script._read_records(output)
    assert records[:2] == previous
    assert len(records) == 26 and len(client.calls) == 24
    assert len({probe.record_key(record) for record in records}) == 26
    assert len(pd.read_csv(output.with_name("summary.csv"))) == 8
    assert len(pd.read_csv(output.with_name("estimate.csv"))) == 16
    assert "Resuming with 2" in capsys.readouterr().out


def test_script_warns_for_truncation_and_thinking_leaks_and_keeps_raw_calls(
    tmp_path, monkeypatch, fake_inputs, capsys
):
    config = _config(output=str(tmp_path / "calls.jsonl"))
    records = list(probe.run_probe(config, FakeClient()))
    records[1]["status"] = "truncated"
    records[2]["status"] = "thinking_leak"
    records[3]["prompt_eval_cached_count"] = 12
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runtime_probe.py", str(config_path)])
    monkeypatch.setattr(script, "run_probe", lambda *args: iter(records))
    script.main()
    assert script._read_records(Path(config.output)) == records
    console = capsys.readouterr().out
    assert "WARNING: truncated" in console and "WARNING: thinking_leak" in console
    assert "WARNING: 12 cached prompt tokens" in console
    assert pd.read_csv(Path(config.output).with_name("summary.csv"))["n"].sum() == 22


@pytest.mark.parametrize(
    "error, message",
    [
        (urllib.error.URLError("offline"), "Start Ollama"),
        (LookupError("Model 'm2' is not pulled. Run: ollama pull m2"), "ollama pull m2"),
    ],
)
def test_script_reports_clear_preflight_errors(tmp_path, monkeypatch, fake_inputs, error, message):
    config = _config(output=str(tmp_path / "calls.jsonl"))
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(yaml.safe_dump(config.__dict__), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runtime_probe.py", str(config_path)])
    client = FakeClient()
    client.version = Mock(side_effect=error)
    monkeypatch.setattr(script, "OllamaClient", lambda url: client)
    with pytest.raises(SystemExit, match=message):
        script.main()
    assert client.calls == []


def _trace(trace_id, start_s):
    return Trace(
        trace_id,
        (
            Span(
                span_id=f"{trace_id}-root",
                parent_id=None,
                trace_id=trace_id,
                service="frontend",
                operation="GET /",
                method=None,
                start_us=start_s * 1_000_000,
                duration_us=10,
                status_code=None,
            ),
        ),
    )


def test_local_window_loader_uses_exact_spec003_rng_and_never_downloads(tmp_path, monkeypatch):
    data_dir = tmp_path / "local-data"
    data_dir.mkdir()
    metadata = [
        {
            "case": case,
            "dataset": dataset,
            "system": dataset[-2:].lower(),
            "fault": "cpu",
            "root_cause_service": "frontend",
            "inject_time": 1000,
        }
        for case, dataset in (("ob", "RE2-OB"), ("tt", "RE2-TT"))
    ]
    pd.DataFrame(metadata).to_parquet(data_dir / "cases.parquet", index=False)
    traces = [_trace(f"pre{i}", 100 + i) for i in range(8)] + [
        _trace(f"post{i}", 1100 + i) for i in range(8)
    ]
    loaded = []

    def load(case_dir):
        loaded.append(case_dir)
        return traces

    monkeypatch.setattr(probe, "load_case_traces", load)
    windows_config = WindowConfig(
        name="windows",
        seed=19,
        data_dir=str(data_dir),
        datasets=["RE2-OB", "RE2-TT"],
        repetitions=[1],
        window_size=2,
        max_trace_chars=10000,
        guard_seconds=60,
        t1_windows_per_phase=3,
        t2_windows=3,
        output_windows="unused",
        output_stats="unused",
    )
    config = _config(cases=["tt", "ob"])
    windows = probe.load_probe_windows(config, windows_config)
    assert loaded == [data_dir / "tt", data_dir / "ob"] and len(windows) == 18
    by_id = {trace.trace_id: trace for trace in traces}
    for case, meta in (("tt", metadata[1]), ("ob", metadata[0])):
        expected, _ = build_case_windows(meta, traces, windows_config, random.Random(f"19:{case}"))
        actual = [window for window in windows if window["case"] == case]
        assert [
            {key: value for key, value in window.items() if key != "texts"} for window in actual
        ] == expected
        for window in actual:
            for fmt in config.formats:
                assert window["texts"][fmt] == SERIALIZERS[fmt](
                    [by_id[trace_id] for trace_id in window["trace_ids"]]
                )
    only_ob = probe.load_probe_windows(replace(config, cases=["ob"]), windows_config)
    assert only_ob == [window for window in windows if window["case"] == "ob"]


def test_runtime_config_rejects_out_of_scope_formats():
    with pytest.raises(ValueError, match="only F0 and F1"):
        _config(formats=["F2"])
