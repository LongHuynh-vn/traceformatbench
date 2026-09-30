"""Shared, dataset-stratified case resamples and pooled statistics (spec 007)."""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

SETS = ("T2-informative", "T2-all", "T1-all")
WindowKey = tuple[str, str, str, int]


def window_key(record: Mapping) -> WindowKey:
    return (record["case"], record["task"], record["label"], record["window_index"])


def in_set(record: Mapping, set_name: str) -> bool:
    if set_name == "T1-all":
        return record["task"] == "T1"
    if set_name not in SETS:
        raise ValueError(f"Unknown analysis set {set_name!r}.")
    return (
        record["task"] == "T2"
        and record["informative"] is not None
        and (set_name == "T2-all" or record["informative"] == True)  # noqa: E712
    )


@dataclass(frozen=True)
class Resamples:
    cases: tuple[str, ...]
    counts: np.ndarray
    strata: tuple[dict, ...]


@dataclass(frozen=True)
class Estimate:
    estimate: float
    ci_low: float
    ci_high: float
    samples: np.ndarray


def draw_resamples(
    records: Sequence[Mapping],
    set_name: str,
    seed: int,
    n_boot: int,
    min_stratum_cases: int,
) -> Resamples:
    """Draw once per set; sorted cases/strata and SHA-256 make order immaterial."""
    if n_boot < 1 or min_stratum_cases < 1:
        raise ValueError("n_boot and min_stratum_cases must be positive.")
    case_datasets: dict[str, str] = {}
    for record in records:
        if in_set(record, set_name):
            case, dataset = record["case"], record["dataset"]
            if case in case_datasets and case_datasets[case] != dataset:
                raise ValueError(f"Case {case!r} occurs in multiple datasets.")
            case_datasets[case] = dataset
    original_counts = Counter(case_datasets.values())
    merged = {dataset: dataset for dataset in original_counts}
    # Decide from original counts in one pass; when both are small, keep RE2 as the target.
    for system in sorted({dataset.rsplit("-", 1)[1] for dataset in original_counts}):
        re2, re3 = f"RE2-{system}", f"RE3-{system}"
        if re2 in original_counts and re3 in original_counts:
            if original_counts[re3] < min_stratum_cases:
                merged[re3] = re2
            elif original_counts[re2] < min_stratum_cases:
                merged[re2] = re3
    merged_counts = Counter(merged[dataset] for dataset in case_datasets.values())
    for dataset, count in sorted(merged_counts.items()):
        if count < min_stratum_cases:
            warnings.warn(
                f"{set_name}: stratum {dataset} has {count} cases after merging "
                f"(minimum {min_stratum_cases}); it stays in that system.",
                stacklevel=2,
            )
    strata = tuple(
        {
            "set": set_name,
            "original_stratum": dataset,
            "original_n_cases": original_counts[dataset],
            "merged_stratum": merged[dataset],
            "merged_n_cases": merged_counts[merged[dataset]],
        }
        for dataset in sorted(original_counts)
    )
    cases = tuple(sorted(case_datasets))
    counts = np.zeros((n_boot, len(cases)), dtype=np.int64)
    digest = hashlib.sha256(f"{seed}:{set_name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    for dataset in sorted(merged_counts):
        indices = np.array(
            [i for i, case in enumerate(cases) if merged[case_datasets[case]] == dataset]
        )
        drawn = indices[rng.integers(len(indices), size=(n_boot, len(indices)))]
        np.add.at(counts, (np.arange(n_boot)[:, None], drawn), 1)
    return Resamples(cases, counts, strata)


def interval(samples: np.ndarray, n_cases: int, ci: float) -> tuple[float, float]:
    """Empty/one-case populations have no CI; undefined draws remain undefined."""
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")
    if n_cases < 2 or not np.isfinite(samples).any():
        return float("nan"), float("nan")
    low, high = np.percentile(samples, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], method="linear")
    return float(low), float(high)


def pooled_statistic(
    records: Sequence[Mapping], values: Sequence[float], resamples: Resamples, ci: float
) -> Estimate:
    """Multiply per-case window sums/counts by case multiplicities, never case means."""
    sums = np.zeros(len(resamples.cases))
    counts = np.zeros(len(resamples.cases))
    indices = {case: i for i, case in enumerate(resamples.cases)}
    for record, value in zip(records, values, strict=True):
        index = indices[record["case"]]
        sums[index] += value
        counts[index] += 1
    denominators = resamples.counts @ counts
    samples = np.full(len(resamples.counts), np.nan)
    np.divide(resamples.counts @ sums, denominators, out=samples, where=denominators > 0)
    estimate = float(sums.sum() / counts.sum()) if counts.sum() else float("nan")
    low, high = interval(samples, int(np.count_nonzero(counts)), ci)
    return Estimate(estimate, low, high, samples)


def check_paired(x: Sequence[Mapping], reference: Sequence[Mapping]) -> None:
    x_keys, ref_keys = {window_key(r) for r in x}, {window_key(r) for r in reference}
    if x_keys != ref_keys:
        raise ValueError(
            f"Paired window keys differ: missing in X={sorted(ref_keys - x_keys)!r}; "
            f"missing in reference={sorted(x_keys - ref_keys)!r}."
        )


def paired_difference(
    x: Sequence[Mapping], reference: Sequence[Mapping], metric: str, resamples: Resamples, ci: float
) -> Estimate:
    check_paired(x, reference)
    stat_x = pooled_statistic(x, [r[metric] for r in x], resamples, ci)
    stat_ref = pooled_statistic(reference, [r[metric] for r in reference], resamples, ci)
    samples = stat_x.samples - stat_ref.samples
    low, high = interval(samples, len({r["case"] for r in x}), ci)
    return Estimate(stat_x.estimate - stat_ref.estimate, low, high, samples)


def verdict(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "not enough cases"
    return "clear difference" if ci_low > 0 or ci_high < 0 else "no clear difference"
