"""Call fixed T1/T2 tasks with local Ollama models; append and resume JSONL calls.

Usage: uv run python scripts/run_tasks.py configs/tasks_smoke.yaml
Dry run: uv run python scripts/run_tasks.py configs/main_grid.yaml --dry-run
"""

import argparse
import json
import urllib.error
from collections import Counter
from pathlib import Path

import yaml

from traceformatbench.analysis.runtime_probe import check_prompt_length
from traceformatbench.data.windows import load_window_config
from traceformatbench.runners.ollama import OllamaClient
from traceformatbench.runners.tasks import (
    TaskConfig,
    build_call_list,
    load_task_windows,
    read_records,
    record_key,
    run_tasks,
)


def dry_run(config: TaskConfig) -> None:
    """Build and check the full call list using local data, without a client or output file."""
    windows = load_task_windows(config, load_window_config(config.windows_config))
    calls = build_call_list(windows, config)
    for call in calls:
        check_prompt_length(call, config)
    counts = Counter((call["model"], call["task"], call["format"]) for call in calls)
    print("Dry run: no Ollama requests or output files.")
    print("Calls per (model, task, format):")
    for model in config.models:
        for task in config.tasks:
            for fmt in config.formats:
                print(f"{model} {task} {fmt}: {counts[model, task, fmt]}")
    print(f"Total calls: {len(calls)}")
    print("Cases skipped per task:")
    by_case = {case: [w for w in windows if w["case"] == case] for case in config.cases}
    for task in config.tasks:
        called_cases = {call["case"] for call in calls if call["task"] == task}
        skipped = [case for case in config.cases if case not in called_cases]
        print(f"{task}: {len(skipped)} cases skipped")
        for case in skipped:
            window = by_case[case][0]
            reason = f"no {task} windows"
            if task == "T2" and window["root_cause_service"] not in window["candidates"]:
                reason = "root cause not in traces"
            print(f"  {case}: {reason}")
    print(f"Largest prompt in characters: {max(call['prompt_chars'] for call in calls)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize calls without contacting Ollama.",
    )
    args = parser.parse_args()
    try:
        config = TaskConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
        if args.dry_run:
            dry_run(config)
            return
        output = Path(config.output)
        previous = read_records(output, allow_incomplete_tail=True)
        done = frozenset(record_key(record) for record in previous)
        print(f"Resuming with {len(done)} recorded calls." if done else "Starting fresh.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in previous),
            encoding="utf-8",
        )
        with output.open("a", encoding="utf-8") as stream:
            for index, record in enumerate(
                run_tasks(config, OllamaClient(config.ollama_url), done),
                len(done) + 1,
            ):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"[{index}] {record['model']} {record['case']} {record['task']} "
                    f"{record['label']} window={record['window_index']} {record['format']} "
                    f"{record['status']}",
                    flush=True,
                )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama request failed ({error.code}): {detail}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise SystemExit(
            f"Cannot reach Ollama at {config.ollama_url}. Start Ollama: {error}"
        ) from error
    except (LookupError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Raw calls: {output}")


if __name__ == "__main__":
    main()
