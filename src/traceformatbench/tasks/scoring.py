"""Strict answer parsing and the student-defined spec-005 metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class ParsedAnswer:
    status: Literal["ok", "invalid_json", "schema_violation"]
    value: str | list[str] | None = None
    fenced: bool = False


def parse_answer(task: str, content: str, candidates: Sequence[str]) -> ParsedAnswer:
    """Strip one whole Markdown fence; perform no other JSON repair."""
    if task not in {"T1", "T1P", "T2"}:
        raise ValueError(f"Unknown task {task!r}.")
    text = content.strip()
    fence = re.fullmatch(r"```[^\n`]*\n(.*?)\n```", text, flags=re.DOTALL)
    fenced = fence is not None and "```" not in fence[1]
    if fenced:
        text = fence[1].strip()
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return ParsedAnswer("invalid_json", fenced=fenced)
    key = {"T1": "phase", "T1P": "after", "T2": "ranking"}[task]
    if not isinstance(answer, dict) or set(answer) != {key}:
        return ParsedAnswer("schema_violation", fenced=fenced)
    value = answer[key]
    if task == "T1":
        valid = isinstance(value, str) and value in {"pre", "post"}
    elif task == "T1P":
        valid = isinstance(value, str) and value in {"A", "B"}
    else:
        valid = (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(item, str) and item in candidates for item in value)
            and len(set(value)) == 3
        )
    return (
        ParsedAnswer("ok", value, fenced)
        if valid
        else ParsedAnswer("schema_violation", fenced=fenced)
    )


def score_record(call_record: Mapping) -> dict:
    """Preserve call status separately from parsing; any failed call scores zero."""
    task = call_record["task"]
    candidates = call_record.get("candidates", [])
    parsed = parse_answer(task, call_record["content"], candidates)
    ok = call_record["status"] == "ok" and parsed.status == "ok"
    record = {
        **call_record,
        "parse_status": parsed.status,
        "fenced": parsed.fenced,
        "answer": parsed.value,
    }
    if task in {"T1", "T1P"}:
        record["correct"] = int(ok and parsed.value == call_record["label"])
    else:
        n = len(candidates)
        if n < 3:
            raise ValueError("T2 needs at least 3 candidates.")
        record.update(
            hit_at_1=int(ok and parsed.value[0] == call_record["label"]),
            hit_at_3=int(ok and call_record["label"] in parsed.value),
            n_candidates=n,
            chance_at_1=1 / n,
            chance_at_3=3 / n,
        )
    return record


METRIC_COLUMNS = [
    "model",
    "format",
    "task",
    "system",
    "n",
    "invalid_rate",
    "fence_rate",
    "n_truncated",
    "truncated_rate",
    "n_thinking_leak",
    "thinking_leak_rate",
    "accuracy",
    "precision_post",
    "recall_post",
    "f1_post",
    "post_rate",
    "a_rate",
    "acc_at_1",
    "acc_at_3",
    "chance_at_1",
    "chance_at_3",
    "n_informative",
    "acc_at_1_informative",
    "acc_at_3_informative",
]


def metrics(scored_records: Sequence[Mapping]) -> pd.DataFrame:
    """Overall and per-system metrics; system=None identifies the overall rows.

    invalid_rate measures parsing failures; call-status failures have their own counts/rates.
    Undefined precision/recall/F1 are zero. An empty informative subset has NaN accuracies.
    """
    rows = []
    for keys in (("model", "format", "task"), ("model", "format", "task", "system")):
        groups: dict[tuple, list[Mapping]] = {}
        for record in scored_records:
            groups.setdefault(tuple(record[key] for key in keys), []).append(record)
        for values, group in groups.items():
            n = len(group)
            row = dict(zip(keys, values, strict=True))
            row.update(
                n=n,
                invalid_rate=sum(r["parse_status"] != "ok" for r in group) / n,
                fence_rate=sum(r["fenced"] for r in group) / n,
            )
            for status in ("truncated", "thinking_leak"):
                count = sum(r["status"] == status for r in group)
                row[f"n_{status}"] = count
                row[f"{status}_rate"] = count / n
            if row["task"] == "T1":
                positive = [
                    r
                    for r in group
                    if r["status"] == r["parse_status"] == "ok" and r["answer"] == "post"
                ]
                tp = sum(r["label"] == "post" for r in positive)
                actual_post = sum(r["label"] == "post" for r in group)
                precision = tp / len(positive) if positive else 0.0
                recall = tp / actual_post if actual_post else 0.0
                row.update(
                    accuracy=sum(r["correct"] for r in group) / n,
                    precision_post=precision,
                    recall_post=recall,
                    f1_post=2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    post_rate=len(positive) / n,
                )
            elif row["task"] == "T1P":
                valid = [r for r in group if r["status"] == r["parse_status"] == "ok"]
                row.update(
                    accuracy=sum(r["correct"] for r in group) / n,
                    a_rate=sum(r["answer"] == "A" for r in valid) / len(valid)
                    if valid
                    else float("nan"),
                )
            else:
                informative = [r for r in group if r["informative"] == True]  # noqa: E712
                row.update(
                    acc_at_1=sum(r["hit_at_1"] for r in group) / n,
                    acc_at_3=sum(r["hit_at_3"] for r in group) / n,
                    chance_at_1=sum(r["chance_at_1"] for r in group) / n,
                    chance_at_3=sum(r["chance_at_3"] for r in group) / n,
                    n_informative=len(informative),
                    acc_at_1_informative=sum(r["hit_at_1"] for r in informative) / len(informative)
                    if informative
                    else float("nan"),
                    acc_at_3_informative=sum(r["hit_at_3"] for r in informative) / len(informative)
                    if informative
                    else float("nan"),
                )
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=METRIC_COLUMNS)
