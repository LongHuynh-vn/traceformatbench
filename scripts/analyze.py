"""Analyze saved scored calls and baseline records; make no model or baseline calls.

Usage: uv run python scripts/analyze.py configs/analysis_v1.yaml
"""

import argparse
from pathlib import Path

from traceformatbench.analysis.report import load_analysis_config, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        write_report(load_analysis_config(args.config))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
