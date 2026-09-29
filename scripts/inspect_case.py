"""Download one RCAEval case and describe its traces.

Purpose: see the real trace schema and sizes before writing any serializer spec.

Usage:
    uv run python scripts/inspect_case.py                     # first RE2-OB case
    uv run python scripts/inspect_case.py --dataset RE2-TT
    uv run python scripts/inspect_case.py --case <case name from cases.parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traceformatbench.data.rcaeval import (
    TRACE_ID_COL,
    download_case,
    find_traces_file,
    load_case_index,
    load_traces,
    summarize_traces,
    trace_quality,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="RE2-OB", help="suite + system, e.g. RE2-OB, RE3-TT")
    parser.add_argument("--case", default=None, help="case name; default: first case of --dataset")
    parser.add_argument("--data-dir", default="data/raw/rcaeval", type=Path)
    parser.add_argument("--show-rows", default=5, type=int, help="rows of one trace to print")
    args = parser.parse_args()

    index = load_case_index(args.data_dir)
    print(f"Index: {len(index)} cases, columns: {list(index.columns)}")

    subset = index[index["dataset"] == args.dataset]
    print(f"{args.dataset}: {len(subset)} cases")
    if subset.empty:
        raise SystemExit(f"No cases for dataset {args.dataset!r}")

    case = args.case or sorted(subset["case"].astype(str))[0]
    print(f"\nCase: {case}")
    print(index[index["case"].astype(str) == case].T.to_string(header=False))

    case_dir = download_case(case, args.data_dir)
    print(f"\nTrace file: {find_traces_file(case_dir)}")

    spans = load_traces(case_dir)
    if spans is None:
        raise SystemExit("This case has no traces.")

    summary = summarize_traces(spans)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    meta = index[index["case"].astype(str) == case].iloc[0]
    quality = trace_quality(
        spans,
        inject_time_s=int(meta["inject_time"]),
        root_cause_service=str(meta["root_cause_service"]),
    )
    print("\nData quality:")
    print(json.dumps(quality, indent=2))

    if TRACE_ID_COL in spans.columns:
        first_trace = spans[TRACE_ID_COL].iloc[0]
        sample = spans[spans[TRACE_ID_COL] == first_trace].head(args.show_rows)
    else:
        sample = spans.head(args.show_rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(f"\nFirst {len(sample)} rows of one trace:")
        print(sample.to_string())


if __name__ == "__main__":
    main()
