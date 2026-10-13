"""Print the summed per-call model time per model, in hours, checked against the summary table.

Usage: uv run python scripts/model_time.py configs/model_time_v1.yaml
Exit code 0 only if the sum over the saved calls and the sum from the summary table agree.
"""

import argparse
from pathlib import Path

from traceformatbench.analysis.model_time import load_model_time_config, model_hours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        rows = model_hours(load_model_time_config(args.config))
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("model,hours_from_calls,hours_from_summary")
    for row in rows:
        print(f"{row.model},{row.hours_from_calls:.4f},{row.hours_from_summary:.4f}")
    if not all(row.agree for row in rows):
        raise SystemExit("The two sums disagree by more than the configured tolerance.")


if __name__ == "__main__":
    main()
