"""Loader for the RCAEval benchmark (Parquet copy on Hugging Face).

Dataset: https://huggingface.co/datasets/phamquiluan/RCAEval (MIT License).
Only the RE2 and RE3 suites carry traces, and only for Online Boutique (OB) and
Train Ticket (TT). Sock Shop is not traced.

This module downloads single cases on demand and describes their trace tables.
It does not decide sampling, labels, or task definitions: those belong in specs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HF_REPO_ID = "phamquiluan/RCAEval"
INDEX_FILE = "cases.parquet"
TRACE_FILES = ("traces.parquet", "traces.csv")

# Column names used by RCAEval's own trace-based baselines. The inspect script
# prints the real schema, so verify these against actual data before relying on them.
TRACE_ID_COL = "traceID"
SERVICE_COL = "serviceName"
SPAN_ID_COL = "spanID"
PARENT_ID_COL = "parentSpanID"
STATUS_COL = "statusCode"
# startTime is in microseconds: startTime // 1000 == startTimeMillis on
# re2ob_checkoutservice_cpu_1. Re-check on other suites before relying on it.
START_US_COL = "startTime"


def load_case_index(cache_dir: str | Path) -> pd.DataFrame:
    """Download (once) and read the index of all cases with their ground truth."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=INDEX_FILE,
        local_dir=str(cache_dir),
    )
    return pd.read_parquet(path)


def download_case(case: str, dest: str | Path) -> Path:
    """Download one case directory and return its local path."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{case}/*", f"{case}/**"],
        local_dir=str(dest),
    )
    case_dir = Path(dest) / case
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"Case {case!r} was not found after download. Check the 'case' column "
            f"in {INDEX_FILE} against the folder names in the Hugging Face repo."
        )
    return case_dir


def find_traces_file(case_dir: str | Path) -> Path | None:
    """Return the trace file of a case (Parquet preferred), or None if there is none."""
    for name in TRACE_FILES:
        path = Path(case_dir) / name
        if path.exists():
            return path
    return None


def load_traces(case_dir: str | Path) -> pd.DataFrame | None:
    """Read a case's spans as one row per span, or None when the case has no traces."""
    path = find_traces_file(case_dir)
    if path is None:
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    return None if df.empty else df


def _describe(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def summarize_traces(
    spans: pd.DataFrame,
    trace_id_col: str = TRACE_ID_COL,
    service_col: str = SERVICE_COL,
) -> dict:
    """Describe a span table: schema, size, and per-trace size distribution.

    Per-trace size is measured in characters of the trace's rows dumped as JSON
    records. This is a rough size signal only; token counts depend on each
    model's tokenizer and are measured later by the runners.
    """
    summary: dict = {
        "n_spans": int(len(spans)),
        "columns": {col: str(dtype) for col, dtype in spans.dtypes.items()},
    }

    if service_col in spans.columns:
        summary["services"] = sorted(spans[service_col].dropna().astype(str).unique().tolist())

    if trace_id_col not in spans.columns:
        summary["note"] = f"column {trace_id_col!r} not found; per-trace stats skipped"
        return summary

    groups = spans.groupby(trace_id_col, sort=False)
    summary["n_traces"] = int(groups.ngroups)
    summary["spans_per_trace"] = _describe(groups.size())
    json_chars = groups.apply(
        lambda g: len(json.dumps(g.to_dict(orient="records"), default=str)),
        include_groups=False,
    )
    summary["json_chars_per_trace"] = _describe(json_chars)
    return summary


def _is_root(parent_ids: pd.Series) -> pd.Series:
    return parent_ids.isna() | (parent_ids.astype("string") == "")


def trace_quality(
    spans: pd.DataFrame,
    inject_time_s: int,
    root_cause_service: str | None = None,
) -> dict:
    """Data-quality facts that serializer and label specs depend on.

    - status codes present (including missing values)
    - traces by number of root spans (0 means the root is missing)
    - traces containing spans whose parent is not in the same trace
    - traces starting before / after the fault injection time
    - traces that pass through the root-cause service
    """
    status = spans[STATUS_COL].astype("string").fillna("<NA>").value_counts()
    is_root = _is_root(spans[PARENT_ID_COL])

    per_trace = spans.assign(_is_root=is_root).groupby(TRACE_ID_COL, sort=False)
    n_roots = per_trace["_is_root"].sum()
    root_hist = {
        "0": int((n_roots == 0).sum()),
        "1": int((n_roots == 1).sum()),
        ">1": int((n_roots > 1).sum()),
    }

    span_ids_by_trace = spans.groupby(TRACE_ID_COL, sort=False)[SPAN_ID_COL].agg(set)
    known = spans[TRACE_ID_COL].map(span_ids_by_trace)
    orphan = [
        (not r) and (p not in ids)
        for r, p, ids in zip(is_root, spans[PARENT_ID_COL], known, strict=True)
    ]
    n_traces_with_orphans = int(spans.loc[orphan, TRACE_ID_COL].nunique())

    trace_start_s = per_trace[START_US_COL].min() // 1_000_000
    quality: dict = {
        "status_codes": {str(k): int(v) for k, v in status.items()},
        "traces_by_root_count": root_hist,
        "traces_with_orphan_spans": n_traces_with_orphans,
        "traces_start_before_inject": int((trace_start_s < inject_time_s).sum()),
        "traces_start_at_or_after_inject": int((trace_start_s >= inject_time_s).sum()),
    }
    if root_cause_service is not None:
        touches = per_trace[SERVICE_COL].agg(lambda x: root_cause_service in set(x))
        quality["traces_touching_root_cause_service"] = int(touches.sum())
    return quality
