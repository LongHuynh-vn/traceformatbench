"""Measure real prompt-token counts of serialized windows with local Ollama models.

Usage:
    uv run python scripts/measure_tokens.py configs/token_probe.yaml

Each record is appended to the output JSONL as soon as it is measured, so an interrupted
run loses nothing: running the same command again resumes where it stopped.
Requires the Ollama server running and the configured models pulled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from traceformatbench.analysis.token_probe import (
    ProbeConfig,
    needs_remeasure,
    record_key,
    run_probe,
    summarize,
)
from traceformatbench.runners.ollama import OllamaClient


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a line cut off by an interruption; everything after it is re-measured
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = ProbeConfig(**yaml.safe_load(args.config.read_text()))
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    previous = _read_records(output)
    stale = [r for r in previous if needs_remeasure(r)]
    if stale:
        print(f"Re-measuring {len(stale)} records that may have been truncated by Ollama.")
        previous = [r for r in previous if not needs_remeasure(r)]
    # Rewrite the file with only complete lines, then append new records.
    output.write_text("".join(json.dumps(r) + "\n" for r in previous))
    done = frozenset(record_key(r) for r in previous)
    print(f"Resuming with {len(done)} records already measured." if done else "Starting fresh.")

    started = time.monotonic()
    new = 0
    with output.open("a") as f:
        for record in run_probe(config, OllamaClient(config.ollama_url), done):
            f.write(json.dumps(record) + "\n")
            f.flush()
            new += 1
            tokens = record["prompt_tokens"] if record["prompt_tokens"] is not None else "-"
            print(
                f"[{len(done) + new}] {time.monotonic() - started:6.0f}s "
                f"{record['case']} {record['model']} {record['format']} "
                f"k={record['window_size']} chars={record['chars']} tokens={tokens} "
                f"{record['status']}",
                flush=True,
            )

    summary = summarize(_read_records(output))
    summary.to_csv(output.with_name("summary.csv"), index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"\nRaw records: {output}\nSummary: {output.with_name('summary.csv')}")


if __name__ == "__main__":
    main()
