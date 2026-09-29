"""Measure exploratory runtime on fixed spec-003 windows with local Ollama models.

Usage:
    uv run python scripts/runtime_probe.py configs/runtime_probe.yaml

Uses only already-downloaded cases. Appends and flushes every call, resumes complete
records after interruption, and never parses or scores the placeholder answers.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path

import yaml

from traceformatbench.analysis.runtime_probe import (
    RuntimeProbeConfig,
    estimate,
    record_key,
    run_probe,
    summarize,
)
from traceformatbench.runners.ollama import OllamaClient


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # discard an incomplete tail before appending more records
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = RuntimeProbeConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
    output = Path(config.output)
    previous = _read_records(output)
    done = frozenset(record_key(record) for record in previous)
    print(f"Resuming with {len(done)} recorded calls." if done else "Starting fresh.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record) + "\n" for record in previous), encoding="utf-8")
    started = time.monotonic()
    try:
        with output.open("a", encoding="utf-8") as stream:
            for new, record in enumerate(
                run_probe(config, OllamaClient(config.ollama_url), done), 1
            ):
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                previous.append(record)
                print(
                    f"[{len(done) + new}] {time.monotonic() - started:6.0f}s "
                    f"{record['case']} {record['model']} {record['format']} "
                    f"{record['task']}/{record['phase']} window={record['window_index']} "
                    f"warmup={record['warmup']} total_s={record['total_s']:.3f} {record['status']}",
                    flush=True,
                )
                if record["status"] != "ok":
                    print(f"WARNING: {record['status']}; excluded from summary.", flush=True)
                if not record["warmup"] and record["prompt_eval_cached_count"] > 0:
                    print(
                        f"WARNING: {record['prompt_eval_cached_count']} cached prompt tokens.",
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

    summary = summarize(previous)
    summary_path = output.with_name("summary.csv")
    summary.to_csv(summary_path, index=False)
    try:
        systems = {call["dataset"].rsplit("-", 1)[-1].upper() for call in previous}
        expected = {
            (model, fmt, system)
            for model in config.models
            for fmt in ("F0", "F1")
            for system in systems
        }
        measured = set(summary[["model", "format", "system"]].itertuples(index=False, name=None))
        if not expected or expected - measured:
            raise ValueError(
                "Cannot estimate the grid: valid F0/F1 timings are missing for a model/system."
            )
        estimate(summary).to_csv(output.with_name("estimate.csv"), index=False)
    except ValueError as error:
        raise SystemExit(f"{error} Raw calls and summary were saved.") from error
    print(summary.to_string(index=False))
    print(
        f"\nRaw calls: {output}\nSummary: {summary_path}"
        f"\nEstimate: {output.with_name('estimate.csv')}"
    )


if __name__ == "__main__":
    main()
