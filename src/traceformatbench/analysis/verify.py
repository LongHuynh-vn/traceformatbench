"""Recompute committed main-grid outputs from committed calls and compare; write only to a temp dir.

Scores and metrics are recomputed from the saved calls, then the analysis is re-run on the
recomputed scores and the committed baseline records. Text cells must match exactly; float cells
may differ within a small tolerance, because the bootstrap's last digits can differ across
platforms (seen for 4 chance-row CI bounds between macOS and Linux).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traceformatbench.analysis.report import load_analysis_config, write_report
from traceformatbench.runners.tasks import read_records
from traceformatbench.tasks.scoring import metrics, score_record

ANALYSIS_TABLES = ("summary.csv", "comparisons.csv", "strata.csv")
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class VerifyConfig:
    calls: str
    scores: str
    metrics: str
    analysis_config: str
    float_rtol: float
    float_atol: float

    def __post_init__(self) -> None:
        if self.float_rtol < 0 or self.float_atol < 0:
            raise ValueError("Require float_rtol >= 0 and float_atol >= 0.")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def load_verify_config(path: str | Path) -> VerifyConfig:
    return VerifyConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def compare_records(name: str, expected: Sequence[dict], actual: Sequence[dict]) -> Check:
    """Scores come from plain Python arithmetic, so records must match exactly and in order."""
    if len(expected) != len(actual):
        return Check(name, False, f"{len(expected)} committed records, {len(actual)} recomputed")
    differing = [i for i, (e, a) in enumerate(zip(expected, actual, strict=True)) if e != a]
    if differing:
        lines = ", ".join(str(i + 1) for i in differing[:MAX_EXAMPLES])
        return Check(name, False, f"{len(differing)} records differ (lines {lines})")
    return Check(name, True, f"{len(expected)} records identical")


def _cell(value) -> str:
    return "NaN" if pd.isna(value) else str(value)


def compare_tables(
    name: str, expected: pd.DataFrame, actual: pd.DataFrame, rtol: float, atol: float
) -> Check:
    """Numeric columns use np.isclose (NaN equals NaN); every other column is compared as text."""
    if list(expected.columns) != list(actual.columns):
        return Check(
            name, False, f"columns differ: {list(expected.columns)} vs {list(actual.columns)}"
        )
    if len(expected) != len(actual):
        return Check(name, False, f"{len(expected)} committed rows, {len(actual)} recomputed")
    mismatches, drift, max_rel = [], 0, 0.0
    for column in expected.columns:
        left, right = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            x, y = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            close = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
            unequal = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            drift += int((close & unequal).sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(x - y) / np.maximum(np.abs(x), np.abs(y))
            if (close & unequal).any():
                max_rel = max(max_rel, float(np.nanmax(rel[close & unequal])))
            rows = np.flatnonzero(~close)
        else:
            rows = [
                i for i, (a, b) in enumerate(zip(left, right, strict=True)) if _cell(a) != _cell(b)
            ]
        mismatches.extend((int(row), column) for row in rows)
    if mismatches:
        examples = ", ".join(
            f"row {row + 2} {column}: {_cell(expected[column].iloc[row])} != "
            f"{_cell(actual[column].iloc[row])}"
            for row, column in sorted(mismatches)[:MAX_EXAMPLES]
        )
        return Check(name, False, f"{len(mismatches)} cells differ ({examples})")
    detail = f"{len(expected)} rows match"
    if drift:
        detail += f"; {drift} float cells differ within tolerance (max relative {max_rel:.1e})"
    return Check(name, True, detail)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, keep_default_na=False, na_values=["NaN", ""], float_precision="round_trip"
    )


def verify(config: VerifyConfig) -> list[Check]:
    analysis = load_analysis_config(config.analysis_config)
    analysis_scores = [analysis.scores] if isinstance(analysis.scores, str) else analysis.scores
    if [Path(p).resolve() for p in analysis_scores] != [Path(config.scores).resolve()]:
        raise ValueError(
            f"The analysis config must read exactly {config.scores}; it reads {analysis_scores}."
        )
    calls = read_records(Path(config.calls))
    if not calls:
        raise FileNotFoundError(f"No saved calls found: {config.calls}")
    committed_scores = read_records(Path(config.scores))
    scored = [score_record(record) for record in calls]
    # A JSON round trip gives recomputed records the same types as the committed file.
    scored = [json.loads(json.dumps(record, ensure_ascii=False)) for record in scored]
    checks = [compare_records(Path(config.scores).name, committed_scores, scored)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        metrics(scored).to_csv(tmp_dir / "metrics.csv", index=False)
        checks.append(
            compare_tables(
                Path(config.metrics).name,
                _read_csv(Path(config.metrics)),
                _read_csv(tmp_dir / "metrics.csv"),
                config.float_rtol,
                config.float_atol,
            )
        )
        scores_path = tmp_dir / "scores.jsonl"
        scores_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in scored),
            encoding="utf-8",
        )
        rerun = replace(analysis, scores=[str(scores_path)], output_dir=str(tmp_dir / "analysis"))
        with contextlib.redirect_stdout(io.StringIO()):
            write_report(rerun)
        for table in ANALYSIS_TABLES:
            checks.append(
                compare_tables(
                    table,
                    _read_csv(Path(analysis.output_dir) / table),
                    _read_csv(tmp_dir / "analysis" / table),
                    config.float_rtol,
                    config.float_atol,
                )
            )
    return checks
