import json
from dataclasses import replace

import pandas as pd
import pytest

from traceformatbench.data.trace import Span, Trace
from traceformatbench.serializers import SERIALIZERS
from traceformatbench.tasks.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_t1_prompt,
    build_t2_prompt,
    case_candidates,
)
from traceformatbench.tasks.schemas import t1_schema, t2_schema
from traceformatbench.tasks.scoring import metrics, parse_answer, score_record

CANDIDATES = ["a", "b", "c", "d"]


def test_exact_prompt_strings():
    assert PROMPT_VERSION == "v1"
    assert SYSTEM_PROMPT == (
        "You analyze distributed traces from a microservice system. "
        "Reply with one JSON object only."
    )
    assert build_t1_prompt("TRACE", 2) == (
        "Below are 2 traces recorded during one time window of a microservice system.\n\n"
        "<traces>\nTRACE\n</traces>\n\n"
        "At some point, a fault was injected into one service of this system. "
        "Were these traces recorded before or after the fault injection?\n"
        'Answer with {"phase": "pre"} if before, or {"phase": "post"} if after.'
    )
    assert build_t2_prompt("TRACE", 2, ["c", "a", "b"]) == (
        "Below are 2 traces recorded after a fault was injected into exactly one service "
        "of a microservice system.\n\n"
        "<traces>\nTRACE\n</traces>\n\n"
        "Candidate services: a, b, c\n"
        "Which service is the root cause (the service where the fault was injected)?\n"
        'Answer with {"ranking": [first, second, third]}: the three most likely root-cause '
        "services from the candidate list, most likely first, no duplicates."
    )


def _trace():
    return Trace("trace", (Span("span", None, "trace", "a", "GET /", None, 0, 10, None),))


@pytest.mark.parametrize("task", ["T1", "T2"])
def test_tiny_window_prompts_differ_only_in_serialized_traces(task):
    window = [_trace()]
    texts = [SERIALIZERS[fmt](window) for fmt in ("F0", "F1")]
    builder = build_t1_prompt if task == "T1" else lambda t, n: build_t2_prompt(t, n, CANDIDATES)
    prompts = [builder(text, len(window)) for text in texts]
    assert texts[0] != texts[1]
    assert prompts[0].replace(texts[0], "TRACE") == prompts[1].replace(texts[1], "TRACE")


def test_candidates_include_ineligible_traces_and_are_sorted_unique():
    trace = _trace()
    orphan = Trace("orphan", (replace(trace.spans[0], service="z", parent_id="missing"),))
    no_root = Trace("no-root", (replace(trace.spans[0], service="b", parent_id="absent"),))
    assert orphan.root is None and no_root.root is None
    assert case_candidates([orphan, trace, no_root, trace]) == ["a", "b", "z"]
    assert case_candidates([]) == []


def test_schemas_are_exact_and_fresh():
    assert t1_schema() == {
        "type": "object",
        "properties": {"phase": {"type": "string", "enum": ["pre", "post"]}},
        "required": ["phase"],
        "additionalProperties": False,
    }
    assert t2_schema(CANDIDATES) == {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": CANDIDATES},
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }
    t1_schema()["properties"]["phase"]["enum"].clear()
    t2_schema(CANDIDATES)["properties"]["ranking"]["items"]["enum"].clear()
    assert t1_schema()["properties"]["phase"]["enum"] == ["pre", "post"]
    assert CANDIDATES == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "task, content, status, value",
    [
        ("T1", ' {"phase": "pre"} \n', "ok", "pre"),
        ("T1", '{"phase": "post"}', "ok", "post"),
        ("T2", '{"ranking": ["b", "a", "c"]}', "ok", ["b", "a", "c"]),
        ("T1", "garbage", "invalid_json", None),
        ("T1", 'prefix {"phase": "pre"}', "invalid_json", None),
        ("T1", '{"phase": "pre",}', "invalid_json", None),
        ("T1", "{}", "schema_violation", None),
        ("T1", '{"phase": "pre", "extra": 1}', "schema_violation", None),
        ("T1", '{"phase": "before"}', "schema_violation", None),
        ("T1", '{"phase": ["pre"]}', "schema_violation", None),
        ("T1", "null", "schema_violation", None),
        ("T1", "[]", "schema_violation", None),
        ("T2", "{}", "schema_violation", None),
        ("T2", '{"ranking": ["a", "b", "c"], "extra": 1}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "b"]}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "b", "c", "d"]}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "a", "b"]}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "b", "unknown"]}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "b", 3]}', "schema_violation", None),
        ("T2", '{"ranking": ["a", "b", {}]}', "schema_violation", None),
        ("T2", '{"ranking": "a,b,c"}', "schema_violation", None),
    ],
)
def test_strict_parsing(task, content, status, value):
    parsed = parse_answer(task, content, CANDIDATES)
    assert (parsed.status, parsed.value, parsed.fenced) == (status, value, False)


@pytest.mark.parametrize("language", ["", "json"])
@pytest.mark.parametrize(
    "task, answer", [("T1", {"phase": "pre"}), ("T2", {"ranking": ["a", "b", "c"]})]
)
def test_strip_one_whole_fence_and_count_it(language, task, answer):
    parsed = parse_answer(task, f" \n```{language}\n{json.dumps(answer)}\n``` \n", CANDIDATES)
    assert parsed.status == "ok" and parsed.fenced
    assert parsed.value == next(iter(answer.values()))


@pytest.mark.parametrize(
    "content, fenced",
    [
        ("```json\ninvalid\n```", True),
        ("```json\n{}\n```", True),
        ('```json\n{"phase":"pre"}\n```\nextra', False),
        ('```\n```json\n{"phase":"pre"}\n```\n```', False),
        ('```\n{"phase":"pre"}\n```\n```\n{}\n```', False),
    ],
)
def test_fences_do_not_enable_additional_repair(content, fenced):
    parsed = parse_answer("T1", content, [])
    assert parsed.status != "ok" and parsed.fenced == fenced


def _record(task="T1", **overrides):
    return {
        "model": "m",
        "format": "F0",
        "task": task,
        "system": "ob",
        "status": "ok",
        "label": "pre" if task == "T1" else "a",
        "informative": None if task == "T1" else True,
        "content": '{"phase":"pre"}' if task == "T1" else '{"ranking":["a","b","c"]}',
        **({"candidates": CANDIDATES} if task == "T2" else {}),
        **overrides,
    }


@pytest.mark.parametrize("status", ["truncated", "thinking_leak", "invalid_json"])
@pytest.mark.parametrize("task", ["T1", "T2"])
def test_any_failed_call_scores_zero_even_with_correct_content(status, task):
    score = score_record(_record(task, status=status))
    assert score["status"] == status
    assert score["parse_status"] == "ok"
    assert score["correct"] == 0 if task == "T1" else (score["hit_at_1"] == score["hit_at_3"] == 0)


def test_hand_computed_t1_metrics_count_invalid_answers_as_wrong():
    records = [
        _record(label="pre", content='{"phase":"pre"}'),  # correct pre
        _record(label="pre", content='{"phase":"post"}'),  # false positive
        _record(label="post", content='```json\n{"phase":"post"}\n```'),  # true positive
        _record(label="post", content='{"phase":"pre"}'),  # false negative
        _record(label="post", content="invalid"),  # invalid false negative
        _record(label="pre", content='{"phase":"pre"}', status="truncated"),
        _record(label="post", content='{"phase":"post"}', status="thinking_leak"),
    ]
    scored = [score_record(record) for record in records]
    result = metrics(scored)
    assert len(result) == 2
    row = result[result["system"].isna()].iloc[0]
    assert row["n"] == 7
    assert row["accuracy"] == pytest.approx(2 / 7)
    assert row["precision_post"] == 1 / 2
    assert row["recall_post"] == 1 / 4
    assert row["f1_post"] == pytest.approx(1 / 3)
    assert row["post_rate"] == pytest.approx(2 / 7)
    assert row["invalid_rate"] == row["fence_rate"] == pytest.approx(1 / 7)
    assert row["n_truncated"] == row["n_thinking_leak"] == 1
    assert row["truncated_rate"] == row["thinking_leak_rate"] == pytest.approx(1 / 7)


def test_hand_computed_t2_metrics_chance_and_informative_subset():
    records = [
        _record("T2", candidates=["a", "b", "c"], system="ob"),  # top 1 + 3
        _record("T2", content='{"ranking":["b","a","c"]}', system="ob"),  # top 3
        _record("T2", content='{"ranking":["b","c","d"]}', informative=False, system="tt"),
        _record("T2", content='{"ranking":["a","a","b"]}', system="tt"),  # invalid
        _record("T2", status="truncated", informative=False, system="tt"),
    ]
    scored = [score_record(record) for record in records]
    assert [(r["hit_at_1"], r["hit_at_3"]) for r in scored] == [
        (1, 1),
        (0, 1),
        (0, 0),
        (0, 0),
        (0, 0),
    ]
    result = metrics(scored)
    assert len(result) == 3
    row = result[result["system"].isna()].iloc[0]
    assert row["n"] == 5 and row["n_informative"] == 3
    assert row["acc_at_1"] == 1 / 5 and row["acc_at_3"] == 2 / 5
    assert row["chance_at_1"] == pytest.approx((1 / 3 + 4 / 4) / 5)
    assert row["chance_at_3"] == pytest.approx((1 + 4 * 3 / 4) / 5)
    assert row["acc_at_1_informative"] == 1 / 3
    assert row["acc_at_3_informative"] == 2 / 3
    assert result[result["system"] == "ob"].iloc[0]["acc_at_1"] == 1 / 2
    assert result[result["system"] == "tt"].iloc[0]["invalid_rate"] == 1 / 3


def test_empty_metrics_and_undefined_denominators():
    assert metrics([]).empty and "accuracy" in metrics([]).columns
    row = metrics([score_record(_record(content="invalid"))]).iloc[0]
    assert row["accuracy"] == row["precision_post"] == row["recall_post"] == row["f1_post"] == 0
    row = metrics([score_record(_record("T2", informative=False))]).iloc[0]
    assert row["n_informative"] == 0
    assert pd.isna(row["acc_at_1_informative"]) and pd.isna(row["acc_at_3_informative"])


def test_groups_do_not_mix_models_formats_or_tasks():
    records = [
        score_record(_record(task, model=model, format=fmt))
        for task in ("T1", "T2")
        for model in ("m1", "m2")
        for fmt in ("F0", "F1")
    ]
    result = metrics(records)
    assert len(result) == 16 and (result["n"] == 1).all()
