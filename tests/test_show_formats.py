import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from traceformatbench.data.trace import load_case_traces
from traceformatbench.serializers import SERIALIZERS

script_spec = importlib.util.spec_from_file_location(
    "show_formats", Path(__file__).parents[1] / "scripts" / "show_formats.py"
)
script = importlib.util.module_from_spec(script_spec)
script_spec.loader.exec_module(script)


@pytest.fixture
def review_case(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "inject_time.txt").write_text("100\n")
    rows = [
        {
            "traceID": trace_id,
            "spanID": "root",
            "parentSpanID": parent_id,
            "serviceName": service,
            "operationName": "op",
            "methodName": None,
            "startTime": seconds * 1_000_000,
            "duration": 1_000,
            "statusCode": None,
        }
        for trace_id, seconds, parent_id, service in (
            ("pre0", 90, None, "svc"),
            ("pre1", 91, None, "svc"),
            ("guard", 100, None, "svc"),
            ("post0", 101, None, "svc"),
            ("rootless", 102, "missing", "svc"),
            ("oversized", 103, None, "x" * 2_000),
            ("post1", 104, None, "服务 🌍"),
            ("post2", 105, None, "svc"),
        )
    ]
    pd.DataFrame(rows).to_parquet(case_dir / "traces.parquet", index=False)
    defaults = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "windows_v1.yaml").read_text()
    )
    defaults.update(window_size=2, max_trace_chars=1_000, guard_seconds=1)
    config_path = tmp_path / "windows.yaml"
    config_path.write_text(yaml.safe_dump(defaults))
    return case_dir, config_path


@pytest.mark.parametrize(
    ("options", "trace_ids"),
    [
        ([], ["post0", "post1"]),
        (["--index", "1"], ["post1", "post2"]),
        (["--phase", "pre"], ["pre0", "pre1"]),
    ],
)
def test_helper_prints_consecutive_eligible_window_and_counts(
    review_case, options, trace_ids, monkeypatch, capsys
):
    case_dir, config_path = review_case
    monkeypatch.setattr(
        sys, "argv", ["show_formats.py", str(case_dir), "--config", str(config_path), *options]
    )
    script.main()
    by_id = {trace.trace_id: trace for trace in load_case_traces(case_dir)}
    window = [by_id[trace_id] for trace_id in trace_ids]
    # Integration check only; the serializer golden strings live in their own tests.
    expected = "".join(
        f"{fmt} ({len(text)} characters)\n{text}\n\n"
        for fmt, serializer in SERIALIZERS.items()
        for text in [serializer(window)]
    )
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("index", ["-1", "2", "99"])
def test_helper_rejects_invalid_or_incomplete_window(review_case, index, monkeypatch, capsys):
    case_dir, config_path = review_case
    monkeypatch.setattr(
        sys,
        "argv",
        ["show_formats.py", str(case_dir), "--config", str(config_path), "--index", index],
    )
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert capsys.readouterr().out == ""


def test_helper_uses_default_window_config(review_case, monkeypatch, capsys):
    case_dir, config_path = review_case
    monkeypatch.chdir(config_path.parent)
    (config_path.parent / "configs").mkdir()
    (config_path.parent / "configs" / "windows_v1.yaml").write_text(config_path.read_text())
    monkeypatch.setattr(sys, "argv", ["show_formats.py", str(case_dir)])
    script.main()
    assert capsys.readouterr().out.startswith("F0 (")
