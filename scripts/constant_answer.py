"""Count the constant-answer reference of the spec 010 notes (section 5).

Usage: uv run python scripts/constant_answer.py configs/error_analysis_v1.yaml
EXPLORATORY, post hoc and not registered: always rank the config's habitual service first.
"""

import argparse
from pathlib import Path

from traceformatbench.analysis.errors import (
    constant_answer_counts,
    load_error_config,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = load_error_config(args.config)
        counts = constant_answer_counts(read_jsonl(config.scores), config.habitual)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("Constant answer, Acc@1 on informative T2 windows:")
    for name, count in counts.items():
        what = f"{name}, always {config.habitual[name]}" if name in config.habitual else name
        hits, windows = count["hits"], count["windows"]
        acc = f" = {hits / windows:.3f}" if windows else ""
        print(f"  {what}: {hits} of {windows}{acc}")


if __name__ == "__main__":
    main()
