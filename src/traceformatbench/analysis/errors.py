"""Spec-010 error analysis: sample wrong T2 answers, compute automatic flags, summarize labels.

EXPLORATORY: nothing here changes registered scores. The student assigns every reason by hand.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from traceformatbench.data.trace import Span, Trace
from traceformatbench.serializers.structure import self_time_us

SYSTEMS = ("ob", "tt")
FLAG_COLUMNS = [
    "contract",
    "habitual",
    "true_rank",
    "top1_is_ancestor",
    "top1_owns_longest_root",
    "root_span_count",
    "root_max_self_ms",
]
LABEL_COLUMNS = [
    "item_id",
    "system",
    "format",
    "root_cause_service",
    "answer",
    *FLAG_COLUMNS,
    "reason",
    "note",
]
SUMMARY_COLUMNS = ["section", "model", "format", "name", "value"]


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    seed: int
    scores: str
    task_config: str
    models: list[str]
    cell_sizes: dict[str, dict[str, int]]
    habitual: dict[str, str]
    excluded: list[dict]
    rater2_size: int
    reasons: list[str]
    codebook: str
    q2: dict[str, str]
    output_dir: str

    def __post_init__(self) -> None:
        if set(self.habitual) != set(SYSTEMS):
            raise ValueError(f"habitual must name one service for each of {SYSTEMS}.")
        for fmt, sizes in self.cell_sizes.items():
            if set(sizes) != set(SYSTEMS) or min(sizes.values()) < 0:
                raise ValueError(f"cell_sizes[{fmt!r}] needs non-negative sizes for {SYSTEMS}.")
        if not 0 < self.rater2_size <= len(self.models) * self.n_per_model:
            raise ValueError("rater2_size must be between 1 and the sample size.")
        if not self.reasons or len(self.reasons) != len(set(self.reasons)):
            raise ValueError("reasons must be non-empty and unique.")

    @property
    def n_per_model(self) -> int:
        return sum(sum(sizes.values()) for sizes in self.cell_sizes.values())


def load_error_config(path: str | Path) -> ErrorAnalysisConfig:
    return ErrorAnalysisConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def is_error(record: Mapping) -> bool:
    """Design 1: a wrong rank-1 answer on an informative T2 window."""
    return record["task"] == "T2" and record["informative"] is True and record["hit_at_1"] == 0


def _item_key(record: Mapping) -> tuple:
    return (record["model"], record["format"], record["case"], record["window_index"])


def sample_items(records: Sequence[Mapping], config: ErrorAnalysisConfig) -> list[dict]:
    """Draw each model x format x system cell independently, avoiding repeated cases in a cell."""
    excluded = {_item_key(item) for item in config.excluded}
    sampled = []
    for model in config.models:
        for fmt, sizes in config.cell_sizes.items():
            for system in SYSTEMS:
                eligible = sorted(
                    (
                        dict(r)
                        for r in records
                        if is_error(r)
                        and r["model"] == model
                        and r["format"] == fmt
                        and r["system"] == system
                        and _item_key(r) not in excluded
                    ),
                    key=lambda r: (r["case"], r["window_index"]),
                )
                n = sizes[system]
                if len(eligible) < n:
                    print(
                        f"WARNING: {model} {fmt} {system}: only {len(eligible)} eligible records,"
                        f" {n} requested; taking all."
                    )
                rng = random.Random(f"{config.seed}:{model}:{fmt}:{system}")
                order = rng.sample(eligible, len(eligible))
                chosen, used_cases = [], set()
                for record in order:
                    if len(chosen) < n and record["case"] not in used_cases:
                        chosen.append(record)
                        used_cases.add(record["case"])
                for record in order:
                    if len(chosen) < n and record not in chosen:
                        chosen.append(record)
                sampled.extend(chosen)
    return sampled


def assign_item_ids(items: Sequence[dict], seed: int) -> list[dict]:
    """Shuffle so that model and format are not grouped, then number E01, E02, ..."""
    ordered = sorted(items, key=lambda r: (r["model"], r["format"], r["system"], *_item_key(r)[2:]))
    random.Random(f"{seed}:item_order").shuffle(ordered)
    width = max(2, len(str(len(ordered))))
    return [{"item_id": f"E{i:0{width}d}", **item} for i, item in enumerate(ordered, 1)]


def rater2_item_ids(item_ids: Sequence[str], seed: int, size: int) -> list[str]:
    return sorted(random.Random(f"{seed}:rater2").sample(sorted(item_ids), size))


def raw_ranking(record: Mapping) -> list[str] | None:
    """The registered answer, or for invalid answers the ranking list as written by the model."""
    if record["answer"] is not None:
        return list(record["answer"])
    text = record["content"].strip()
    fence = re.fullmatch(r"```[^\n`]*\n(.*?)\n```", text, flags=re.DOTALL)
    try:
        value = json.loads(fence[1] if fence else text)
    except json.JSONDecodeError:
        return None
    ranking = value.get("ranking") if isinstance(value, dict) else None
    if isinstance(ranking, list) and all(isinstance(item, str) for item in ranking):
        return ranking
    return None


def _ancestors(trace: Trace, span: Span) -> list[Span]:
    by_id = {s.span_id: s for s in trace.spans}
    chain, seen = [], {span.span_id}
    parent = by_id.get(span.parent_id) if span.parent_id is not None else None
    while parent is not None and parent.span_id not in seen:
        chain.append(parent)
        seen.add(parent.span_id)
        parent = by_id.get(parent.parent_id) if parent.parent_id is not None else None
    return chain


def automatic_flags(
    record: Mapping, ranking: Sequence[str] | None, traces: Sequence[Trace], habitual: str
) -> dict:
    """Flags from the rebuilt window traces; contract items use the first listed service."""
    root = record["root_cause_service"]
    top1 = ranking[0] if ranking else None
    root_spans = [(t, s) for t in traces for s in t.spans if s.service == root]
    if not root_spans:
        raise ValueError(
            f"Root-cause service {root!r} owns no span in informative window "
            f"{record['case']} {record['window_index']}."
        )
    roots = [t.root for t in traces if t.root is not None]
    longest = max((s.duration_us for s in roots), default=None)
    return {
        "contract": record["parse_status"] != "ok",
        "habitual": top1 is not None and top1 == habitual and top1 != root,
        "true_rank": (list(ranking).index(root) + 1) if ranking and root in ranking[:3] else 0,
        "top1_is_ancestor": top1 is not None
        and top1 != root
        and any(a.service == top1 for t, s in root_spans for a in _ancestors(t, s)),
        "top1_owns_longest_root": top1 is not None
        and any(s.service == top1 and s.duration_us == longest for s in roots),
        "root_span_count": len(root_spans),
        "root_max_self_ms": max(self_time_us(t, s) for t, s in root_spans) / 1000,
    }


def check_prompt_hash(item: Mapping, rebuilt_sha256: str) -> None:
    if rebuilt_sha256 != item["prompt_sha256"]:
        raise ValueError(
            f"Rebuilt prompt hash differs for item {item['item_id']} ({item['model']} "
            f"{item['format']} {item['case']} window {item['window_index']}); refusing to "
            "label a prompt the model did not see."
        )


def _csv_text(rows: Sequence[Mapping], columns: Sequence[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return stream.getvalue()


def _answer_text(ranking: Sequence[str] | None) -> str:
    return " | ".join(ranking) if ranking else "(no ranking)"


def label_row(item: Mapping) -> dict:
    return {
        "item_id": item["item_id"],
        "system": item["system"],
        "format": item["format"],
        "root_cause_service": item["root_cause_service"],
        "answer": _answer_text(item["raw_ranking"]),
        **{column: item[column] for column in FLAG_COLUMNS},
        "reason": "",
        "note": "",
    }


def item_markdown(item: Mapping, system_prompt: str, user_prompt: str, f2_text: str) -> str:
    """The labeler's view; the model name stays in key.csv only."""
    flags = "\n".join(f"- `{column}`: {item[column]}" for column in FLAG_COLUMNS)
    contract = " (invalid answer: scored 0 under the registered rule)" if item["contract"] else ""
    return (
        f"# {item['item_id']}\n\n"
        f"- System: {item['system']}\n"
        f"- Format the model saw: {item['format']}\n"
        f"- True root cause: `{item['root_cause_service']}`\n"
        f"- Model answer: {_answer_text(item['raw_ranking'])}{contract}\n\n"
        f"## Automatic flags\n\n{flags}\n\n"
        f"## F2 view of the same window\n\n````\n{f2_text}\n````\n\n"
        f"<details>\n<summary>Full prompt the model saw ({item['format']})</summary>\n\n"
        f"System prompt:\n\n````\n{system_prompt}\n````\n\n"
        f"User prompt:\n\n````\n{user_prompt}\n````\n\n</details>\n"
    )


def output_files(
    items: Sequence[Mapping],
    rater2_ids: Sequence[str],
    item_texts: Mapping[str, str],
    codebook: str,
) -> dict[str, str]:
    """Every output as text, keyed by its path relative to the output directory."""
    rows = [label_row(item) for item in items]
    files = {
        "sample.jsonl": "".join(
            json.dumps({k: v for k, v in item.items() if k != "content"}, ensure_ascii=False) + "\n"
            for item in items
        ),
        "labels.csv": _csv_text(rows, LABEL_COLUMNS),
        "key.csv": _csv_text(items, ["item_id", "model"]),
        "rater2_items.txt": "".join(f"{item_id}\n" for item_id in rater2_ids),
        "labels_rater2.csv": _csv_text(
            [row for row in rows if row["item_id"] in set(rater2_ids)], LABEL_COLUMNS
        ),
        "CODEBOOK.md": codebook,
    }
    for item_id, text in item_texts.items():
        files[f"items/{item_id}.md"] = text
    return files


def write_files(files: Mapping[str, str], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    for name, text in sorted(files.items()):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted Cohen's kappa; NaN when expected agreement is 1."""
    if len(a) != len(b) or not a:
        raise ValueError("Kappa needs two equally long, non-empty label lists.")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in count_a) / n**2
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def q2_counts(records: Sequence[Mapping], model: str, fmt: str, reference: str) -> dict[str, int]:
    """Informative T2 windows where only one of the two formats is right at rank 1."""
    hits: dict[str, dict[tuple, int]] = {fmt: {}, reference: {}}
    for r in records:
        if (
            r["task"] == "T2"
            and r["informative"] is True
            and r["model"] == model
            and r["format"] in hits
        ):
            hits[r["format"]][(r["case"], r["window_index"])] = r["hit_at_1"]
    if hits[fmt].keys() != hits[reference].keys():
        raise ValueError(f"{fmt} and {reference} cover different windows for {model}.")
    keys = hits[fmt].keys()
    return {
        f"{fmt}_right_{reference}_wrong": sum(
            hits[fmt][k] == 1 and hits[reference][k] == 0 for k in keys
        ),
        f"{reference}_right_{fmt}_wrong": sum(
            hits[reference][k] == 1 and hits[fmt][k] == 0 for k in keys
        ),
        "n_windows": len(keys),
    }


def constant_answer_counts(
    records: Sequence[Mapping], habitual: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    """Rank-1 hits of always naming the system's habitual service, per system and in total.

    Post hoc reference, not registered. Each informative T2 window counts once, however many
    model x format records it has.
    """
    windows: dict[tuple, tuple[str, str]] = {}
    for r in records:
        if r["task"] == "T2" and r["informative"] is True:
            if r["system"] not in habitual:
                raise ValueError(f"No habitual service for system {r['system']!r}.")
            key = (r["case"], r["window_index"])
            if windows.setdefault(key, (r["system"], r["label"])) != (r["system"], r["label"]):
                raise ValueError(f"Window {key!r} has different systems or labels across records.")
    if not windows:
        raise ValueError("No informative T2 windows in the scores.")
    counts = {system: {"hits": 0, "windows": 0} for system in habitual}
    for system, label in windows.values():
        counts[system]["hits"] += label == habitual[system]
        counts[system]["windows"] += 1
    counts["all"] = {name: sum(c[name] for c in counts.values()) for name in ("hits", "windows")}
    return counts


def read_labels(path: str | Path, reasons: Sequence[str]) -> dict[str, str]:
    """item_id -> reason ('' when unlabeled); unknown reasons are an error."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unknown = sorted({row["reason"].strip() for row in rows} - {"", *reasons})
    if unknown:
        raise ValueError(f"Unknown reason(s) {unknown!r} in {path}; allowed: {list(reasons)!r}.")
    return {row["item_id"]: row["reason"].strip() for row in rows}


def summarize(config: ErrorAnalysisConfig) -> tuple[list[dict], list[str]]:
    """Summary rows plus printable lines; the kappa is skipped while labels are missing."""
    out = Path(config.output_dir)
    items = read_jsonl(out / "sample.jsonl")
    with (out / "key.csv").open(encoding="utf-8", newline="") as stream:
        models = {row["item_id"]: row["model"] for row in csv.DictReader(stream)}
    labels = read_labels(out / "labels.csv", config.reasons)
    labels2 = read_labels(out / "labels_rater2.csv", config.reasons)
    rows, lines = [], []

    labeled = [item for item in items if labels.get(item["item_id"])]
    lines.append(f"Reasons: {len(labeled)} of {len(items)} items labeled.")
    reason_counts = Counter(
        (models[i["item_id"]], i["format"], labels[i["item_id"]]) for i in labeled
    )
    for (model, fmt, reason), count in sorted(reason_counts.items()):
        rows.append(dict(section="reason", model=model, format=fmt, name=reason, value=count))
    for model in config.models:
        for reason in config.reasons:
            total = sum(c for (m, _, r), c in reason_counts.items() if m == model and r == reason)
            if labeled:
                lines.append(f"  {model} {reason}: {total}")

    lines.append("Automatic flags (true counts):")
    for flag in ("contract", "habitual", "top1_is_ancestor", "top1_owns_longest_root"):
        per_cell = Counter((models[i["item_id"]], i["format"]) for i in items if i[flag] is True)
        for (model, fmt), count in sorted(per_cell.items()):
            rows.append(dict(section="flag", model=model, format=fmt, name=flag, value=count))
        by_model = {m: sum(c for (mm, _), c in per_cell.items() if mm == m) for m in config.models}
        parts = ", ".join(f"{m} {c}" for m, c in by_model.items())
        lines.append(f"  {flag}: {sum(by_model.values())} of {len(items)} ({parts})")
    ranks = Counter(i["true_rank"] for i in items)
    for rank, count in sorted(ranks.items()):
        rows.append(
            dict(section="flag", model="all", format="all", name=f"true_rank_{rank}", value=count)
        )
    lines.append("  true_rank: " + ", ".join(f"{r}: {c}" for r, c in sorted(ranks.items())))

    pairs = [(labels.get(i, ""), labels2.get(i, "")) for i in sorted(labels2)]
    if pairs and all(a and b for a, b in pairs):
        first, second = zip(*pairs, strict=True)
        agreement = sum(a == b for a, b in pairs) / len(pairs)
        kappa = cohen_kappa(first, second)
        rows.append(
            dict(
                section="reliability",
                model="all",
                format="all",
                name="raw_agreement",
                value=agreement,
            )
        )
        rows.append(
            dict(section="reliability", model="all", format="all", name="cohen_kappa", value=kappa)
        )
        lines.append(
            f"Rater 2 ({len(pairs)} items): raw agreement {agreement:.3f}, kappa {kappa:.3f}"
        )
    else:
        lines.append("Rater 2: skipped (labels.csv or labels_rater2.csv not fully labeled).")

    records = read_jsonl(config.scores)
    q2 = q2_counts(records, config.q2["model"], config.q2["format"], config.q2["reference"])
    for name, value in q2.items():
        rows.append(
            dict(section="q2", model=config.q2["model"], format="-", name=name, value=value)
        )
    lines.append(
        f"Q2 ({config.q2['model']}, informative): " + ", ".join(f"{k} {v}" for k, v in q2.items())
    )
    return rows, lines


def write_summary(config: ErrorAnalysisConfig) -> list[str]:
    rows, lines = summarize(config)
    path = Path(config.output_dir) / "summary.csv"
    path.write_text(_csv_text(rows, SUMMARY_COLUMNS), encoding="utf-8")
    return [*lines, f"Summary: {path}"]
