"""Check saved main-grid calls against the config's call list; make no Ollama requests.

Usage: uv run python scripts/check_grid.py configs/main_grid.yaml results/main_grid/calls.jsonl
Exit code 0 only if every check passes (PREREGISTRATION.md section 14, K2 and K3).
"""

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from traceformatbench.data.windows import load_window_config
from traceformatbench.runners.tasks import (
    TaskConfig,
    build_call_list,
    load_task_windows,
    read_records,
    record_key,
)

STATUSES = ("ok", "truncated", "thinking_leak")


def check_completeness(planned: Sequence[Mapping], records: Sequence[Mapping]) -> tuple[bool, str]:
    planned_keys = Counter(record_key(call) for call in planned)
    if any(count > 1 for count in planned_keys.values()):
        raise ValueError("The config's call list contains duplicate keys.")
    record_keys = Counter(record_key(record) for record in records)
    missing = len(planned_keys.keys() - record_keys.keys())
    extra = len(record_keys.keys() - planned_keys.keys())
    duplicate = sum(count > 1 for count in record_keys.values())
    return missing == extra == duplicate == 0, (
        f"calls: {len(planned)} planned, {len(records)} records; "
        f"missing {missing}, extra {extra}, duplicate keys {duplicate}"
    )


def check_builds(models: Sequence[str], records: Sequence[Mapping]) -> list[tuple[bool, str]]:
    """One model digest and one Ollama version per model across all its records."""
    found = list(dict.fromkeys([*models, *(record["model"] for record in records)]))
    lines = []
    for model in found:
        own = [record for record in records if record["model"] == model]
        digests = sorted({str(record["model_digest"]) for record in own})
        versions = sorted({str(record["ollama_version"]) for record in own})
        lines.append(
            (
                len(digests) == len(versions) == 1,
                f"model {model}: {len(own)} records; model_digest {', '.join(digests) or '-'}; "
                f"ollama_version {', '.join(versions) or '-'}",
            )
        )
    return lines


def check_statuses(records: Sequence[Mapping]) -> tuple[bool, str]:
    counts = Counter(record["status"] for record in records)
    other = sum(count for status, count in counts.items() if status not in STATUSES)
    recounted = [record for record in records if record.get("recount_num_ctx") is not None]
    cleared = sum(record["status"] != "truncated" for record in recounted)
    return other == 0, (
        "statuses: "
        + ", ".join(f"{status} {counts[status]}" for status in STATUSES)
        + f", other {other}; re-counted {len(recounted)} ({cleared} no longer truncated)"
    )


def check_grid(config: TaskConfig, records: Sequence[Mapping]) -> list[tuple[bool, str]]:
    """Build the call list as in `run_tasks.py --dry-run`, from local data only."""
    windows = load_task_windows(config, load_window_config(config.windows_config))
    planned = build_call_list(windows, config)
    return [
        check_completeness(planned, records),
        *check_builds(config.models, records),
        check_statuses(records),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("calls", type=Path)
    args = parser.parse_args()
    try:
        config = TaskConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
        if not args.calls.exists():
            raise FileNotFoundError(f"No calls file at {args.calls}.")
        results = check_grid(config, read_records(args.calls))
    except (LookupError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for passed, line in results:
        print(f"{'PASS' if passed else 'FAIL'}  {line}")
    passed = all(passed for passed, _ in results)
    print("All checks passed." if passed else "Some checks failed.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
