"""Run the self-time baseline on saved T2 windows, one local case at a time.

Usage: uv run python scripts/run_baseline.py configs/windows_v1.yaml
ORDER RULE: first evaluate windows_v1 only after PREREGISTRATION.md is committed.
"""

import argparse
import json
from pathlib import Path

from traceformatbench.analysis.baselines import run_baseline
from traceformatbench.data.windows import load_window_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_window_config(args.config)
    # Derive the output version/location from the window config rather than fixing a run here.
    version = config.name.removeprefix("windows_")
    output = Path(config.output_stats).parent.parent / "baselines" / f"selftime_{version}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in run_baseline(config):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Baseline records: {output}")


if __name__ == "__main__":
    main()
