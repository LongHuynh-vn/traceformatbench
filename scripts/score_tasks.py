"""Score saved task calls without model requests.

Usage: uv run python scripts/score_tasks.py results/tasks_smoke/calls.jsonl
"""

import argparse
import json
from pathlib import Path

from traceformatbench.runners.tasks import read_records
from traceformatbench.tasks.scoring import metrics, score_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("calls", type=Path)
    args = parser.parse_args()
    if not args.calls.is_file():
        raise SystemExit(f"Calls file not found: {args.calls}")
    scored = [score_record(record) for record in read_records(args.calls)]
    scores_path = args.calls.with_name("scores.jsonl")
    scores_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in scored),
        encoding="utf-8",
    )
    metrics_path = args.calls.with_name("metrics.csv")
    metrics(scored).to_csv(metrics_path, index=False)
    print(f"Scores: {scores_path}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
