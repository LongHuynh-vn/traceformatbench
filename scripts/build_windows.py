"""Build the fixed trace windows and per-case size-cap statistics (spec 003).

Usage:
    uv run python scripts/build_windows.py configs/windows_v1.yaml

Downloads missing cases (traces and injection time only) and processes one case at a time.
The same config always produces a byte-identical windows file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traceformatbench.data.rcaeval import load_case_index
from traceformatbench.data.windows import build_windows, load_window_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_window_config(args.config)
    records, stats = build_windows(config, load_case_index(config.data_dir))

    output_windows = Path(config.output_windows)
    output_windows.parent.mkdir(parents=True, exist_ok=True)
    output_windows.write_text("".join(json.dumps(record) + "\n" for record in records))
    output_stats = Path(config.output_stats)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_stats, index=False)

    short = stats[
        (stats["n_t1_windows"] < 2 * config.t1_windows_per_phase)
        | (stats["n_t2_windows"] < config.t2_windows)
    ]
    print(f"Cases: {len(stats)}")
    print(f"T1 windows: {int(stats['n_t1_windows'].sum())} (half pre, half post)")
    print(
        f"T2 windows: {int(stats['n_t2_windows'].sum())}, "
        f"informative: {int(stats['n_t2_informative'].sum())}"
    )
    print(f"Cases with fewer windows than configured: {list(short['case'])}")
    print(f"\nWindows: {output_windows}\nStats: {output_stats}")


if __name__ == "__main__":
    main()
