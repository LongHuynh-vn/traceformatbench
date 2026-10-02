"""Count labeled reasons, automatic flags, rater agreement and Q2 for spec 010.

Usage: uv run python scripts/summarize_errors.py configs/error_analysis_v1.yaml
EXPLORATORY: reported as counts next to, never instead of, the registered results.
"""

import argparse
from pathlib import Path

from traceformatbench.analysis.errors import load_error_config, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        lines = write_summary(load_error_config(args.config))
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("\n".join(lines))


if __name__ == "__main__":
    main()
