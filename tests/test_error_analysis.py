"""Spec 010 on synthetic records and traces; the real-data hash check is a dry run on the Mac."""

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from traceformatbench.analysis import errors
from traceformatbench.data.trace import Span, Trace

PILOT = {"model": "m1", "format": "F0", "case": "ob_a", "window_index": 0}


def _config(tmp_path=None, **overrides):
    values = dict(
        seed=20260930,
        scores=str(tmp_path / "scores.jsonl") if tmp_path else "unused",
        task_config="unused",
        models=["m1", "m2"],
        cell_sizes={"F0": {"ob": 3, "tt": 2}, "F1": {"ob": 2, "tt": 3}},
        habitual={"ob": "hab_ob", "tt": "hab_tt"},
        excluded=[PILOT],
        rater2_size=4,
        reasons=["caller_blame", "longest_root", "weak_signal", "unclear"],
        codebook="unused",
        q2={"model": "m2", "format": "F1", "reference": "F0"},
        output_dir=str(tmp_path / "out") if tmp_path else "unused",
    )
    return errors.ErrorAnalysisConfig(**{**values, **overrides})


def _record(model, fmt, system, case, index, hit=0, informative=True, task="T2", **extra):
    return {
        "case": case,
        "system": system,
        "root_cause_service": "db",
        "task": task,
        "window_index": index,
        "informative": informative,
        "model": model,
        "format": fmt,
        "hit_at_1": hit,
        "answer": ["api", "web", "db"],
        "content": '{"ranking": ["api", "web", "db"]}',
        "parse_status": "ok",
        "prompt_sha256": "abc",
        **extra,
    }


def _records():
    records = []
    for model in ("m1", "m2"):
        for fmt in ("F0", "F1"):
            for system in ("ob", "tt"):
                for c in "abcde":
                    for index in range(3):
                        records.append(_record(model, fmt, system, f"{system}_{c}", index))
                records.append(_record(model, fmt, system, f"{system}_a", 3, hit=1))
                records.append(_record(model, fmt, system, f"{system}_b", 4, informative=False))
                records.append(_record(model, fmt, system, f"{system}_c", 5, task="T1"))
    return records


def test_cell_sizes_split_and_eligibility():
    items = errors.sample_items(_records(), _config())
    assert len(items) == 20
    for model in ("m1", "m2"):
        for fmt, sizes in {"F0": {"ob": 3, "tt": 2}, "F1": {"ob": 2, "tt": 3}}.items():
            for system, n in sizes.items():
                cell = [
                    i
                    for i in items
                    if (i["model"], i["format"], i["system"]) == (model, fmt, system)
                ]
                assert len(cell) == n
                assert len({i["case"] for i in cell}) == n  # no case twice when avoidable
    keys = [(i["model"], i["format"], i["case"], i["window_index"]) for i in items]
    assert len(keys) == len(set(keys))
    assert ("m1", "F0", "ob_a", 0) not in keys
    assert all(i["task"] == "T2" and i["informative"] is True and i["hit_at_1"] == 0 for i in items)


def test_same_seed_same_sample_and_ids_new_seed_changes_sample():
    first = errors.assign_item_ids(errors.sample_items(_records(), _config()), 1)
    again = errors.assign_item_ids(errors.sample_items(list(reversed(_records())), _config()), 1)
    assert first == again

    def keys(items):
        return [(i["case"], i["window_index"]) for i in items]

    base = errors.sample_items(_records(), _config())
    assert keys(errors.sample_items(_records(), _config(seed=1))) != keys(base)
    assert [i["item_id"] for i in first] == [f"E{n:02d}" for n in range(1, 21)]


def test_short_cell_takes_all_and_warns(capsys):
    records = [r for r in _records() if not (r["model"] == "m1" and r["case"] in {"ob_b", "ob_c"})]
    records = [
        r
        for r in records
        if not (r["model"] == "m1" and r["system"] == "ob" and r["window_index"] > 0)
    ]
    items = errors.sample_items(records, _config())
    cell = [i for i in items if (i["model"], i["format"], i["system"]) == ("m1", "F0", "ob")]
    assert len(cell) == 2  # ob_d, ob_e; ob_a window 0 is the excluded pilot item
    assert "WARNING: m1 F0 ob: only 2 eligible" in capsys.readouterr().out


def test_tampered_prompt_hash_stops():
    item = {**_record("m1", "F0", "ob", "ob_a", 1), "item_id": "E07"}
    errors.check_prompt_hash(item, "abc")
    with pytest.raises(ValueError, match="E07"):
        errors.check_prompt_hash(item, "tampered")


def _span(span_id, service, start, duration, parent=None, trace="t1"):
    return Span(span_id, parent, trace, service, "op", None, start, duration, None)


def _traces():
    # t1: web (root, 100) -> api (80) -> db (root cause, 50). t2: an unrelated, longer root (300).
    t1 = Trace(
        "t1",
        (
            _span("w", "web", 0, 100_000),
            _span("a", "api", 10_000, 80_000, "w"),
            _span("d", "db", 20_000, 50_000, "a"),
        ),
    )
    t2 = Trace("t2", (_span("b", "batch", 0, 300_000, trace="t2"),))
    return [t1, t2]


@pytest.mark.parametrize(
    ("top1", "ancestor", "longest"),
    [("api", True, False), ("web", True, False), ("batch", False, True), ("cache", False, False)],
)
def test_automatic_flags_on_hand_built_traces(top1, ancestor, longest):
    record = _record("m1", "F0", "ob", "ob_a", 1)
    flags = errors.automatic_flags(record, [top1, "x", "db"], _traces(), habitual="api")
    assert flags["top1_is_ancestor"] is ancestor
    assert flags["top1_owns_longest_root"] is longest
    assert flags["habitual"] is (top1 == "api")
    assert flags["contract"] is False
    assert flags["true_rank"] == 3
    assert flags["root_span_count"] == 1
    assert flags["root_max_self_ms"] == 50.0


def test_contract_item_uses_first_listed_service():
    record = _record(
        "m1",
        "F0",
        "ob",
        "ob_a",
        1,
        answer=None,
        parse_status="schema_violation",
        content='{"ranking": ["db", "db", "db"]}',
    )
    ranking = errors.raw_ranking(record)
    assert ranking == ["db", "db", "db"]
    flags = errors.automatic_flags(record, ranking, _traces(), habitual="db")
    assert flags["contract"] is True
    assert flags["habitual"] is False  # top-1 is the root cause
    assert flags["top1_is_ancestor"] is False
    assert flags["true_rank"] == 1
    no_ranking = {**record, "content": "not json"}
    assert errors.raw_ranking(no_ranking) is None
    assert errors.automatic_flags(no_ranking, None, _traces(), "db")["habitual"] is False


def test_root_cause_without_span_is_an_error():
    record = {**_record("m1", "F0", "ob", "ob_a", 1), "root_cause_service": "ghost"}
    with pytest.raises(ValueError, match="ghost"):
        errors.automatic_flags(record, ["api"], _traces(), "api")


def test_cohen_kappa_hand_computed():
    # p_o = 3/4; p_e = 0.5 * 0.25 + 0.5 * 0.75 = 0.5; kappa = 0.25 / 0.5.
    assert errors.cohen_kappa(["x", "x", "y", "y"], ["x", "y", "y", "y"]) == pytest.approx(0.5)
    assert errors.cohen_kappa(["x", "y"], ["x", "y"]) == pytest.approx(1.0)
    assert math.isnan(errors.cohen_kappa(["x", "x"], ["x", "x"]))


def test_q2_counts():
    records = [
        _record("m2", "F1", "ob", "c", 0, hit=1),
        _record("m2", "F0", "ob", "c", 0, hit=0),
        _record("m2", "F1", "ob", "c", 1, hit=0),
        _record("m2", "F0", "ob", "c", 1, hit=1),
        _record("m2", "F1", "ob", "c", 2, hit=1),
        _record("m2", "F0", "ob", "c", 2, hit=1),
        _record("m2", "F1", "ob", "c", 3, hit=1),
        _record("m2", "F0", "ob", "c", 3, hit=0),
        _record("m2", "F1", "ob", "c", 4, hit=1, informative=False),
        _record("m2", "F0", "ob", "c", 4, hit=0, informative=False),
        _record("m1", "F1", "ob", "c", 0, hit=1),
        _record("m1", "F0", "ob", "c", 0, hit=0),
    ]
    assert errors.q2_counts(records, "m2", "F1", "F0") == {
        "F1_right_F0_wrong": 2,
        "F0_right_F1_wrong": 1,
        "n_windows": 4,
    }


def test_constant_answer_counts_each_informative_t2_window_once():
    habitual = {"ob": "hab_ob", "tt": "hab_tt"}
    records = [
        # One window, three model x format records: counted once.
        _record("m1", "F0", "ob", "ob_a", 0, label="hab_ob"),
        _record("m1", "F1", "ob", "ob_a", 0, label="hab_ob"),
        _record("m2", "F0", "ob", "ob_a", 0, label="hab_ob"),
        _record("m1", "F0", "ob", "ob_a", 1, label="db"),
        _record("m1", "F0", "ob", "ob_b", 0, label="db"),
        _record("m1", "F0", "tt", "tt_a", 0, label="hab_tt"),
        # Left out: another system's habitual service, not informative, T1.
        _record("m1", "F0", "tt", "tt_a", 1, label="hab_ob"),
        _record("m1", "F0", "tt", "tt_b", 0, label="hab_tt", informative=False),
        _record("m1", "F0", "tt", "tt_b", 1, label="post", informative=None, task="T1"),
    ]
    assert errors.constant_answer_counts(records, habitual) == {
        "ob": {"hits": 1, "windows": 3},
        "tt": {"hits": 1, "windows": 2},
        "all": {"hits": 2, "windows": 5},
    }


def test_constant_answer_counts_rejects_inconsistent_or_empty_input():
    habitual = {"ob": "hab_ob", "tt": "hab_tt"}
    with pytest.raises(ValueError, match="different systems or labels"):
        errors.constant_answer_counts(
            [
                _record("m1", "F0", "ob", "ob_a", 0, label="hab_ob"),
                _record("m2", "F0", "ob", "ob_a", 0, label="db"),
            ],
            habitual,
        )
    with pytest.raises(ValueError, match="No habitual service"):
        errors.constant_answer_counts([_record("m1", "F0", "xx", "xx_a", 0, label="db")], habitual)
    with pytest.raises(ValueError, match="No informative T2 windows"):
        errors.constant_answer_counts(
            [_record("m1", "F0", "ob", "ob_a", 0, label="db", informative=False)], habitual
        )


def test_constant_answer_script_prints_counts(tmp_path, monkeypatch, capsys):
    path = Path(__file__).parents[1] / "scripts" / "constant_answer.py"
    spec = importlib.util.spec_from_file_location("constant_answer", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = tmp_path / "error_analysis.yaml"
    config.write_text(yaml.safe_dump(dataclasses.asdict(_config(tmp_path))), encoding="utf-8")
    records = [
        _record("m1", "F0", "ob", "ob_a", 0, label="hab_ob"),
        _record("m1", "F0", "ob", "ob_a", 1, label="db"),
        _record("m1", "F0", "ob", "ob_b", 0, label="db"),
    ]
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", [str(path), str(config)])
    script.main()
    assert capsys.readouterr().out.splitlines()[1:] == [
        "  ob, always hab_ob: 1 of 3 = 0.333",
        "  tt, always hab_tt: 0 of 0",
        "  all: 1 of 3 = 0.333",
    ]


def _write_sample(config, tmp_path):
    items = errors.assign_item_ids(errors.sample_items(_records(), config), config.seed)
    for item in items:
        ranking = errors.raw_ranking(item)
        item.update(raw_ranking=ranking, **errors.automatic_flags(item, ranking, _traces(), "api"))
    ids = [i["item_id"] for i in items]
    rater2 = errors.rater2_item_ids(ids, config.seed, config.rater2_size)
    texts = {i["item_id"]: errors.item_markdown(i, "sys", "user", "f2") for i in items}
    return errors.output_files(items, rater2, texts, "# Codebook\n")


def test_outputs_byte_identical_and_hide_the_model(tmp_path):
    config = _config(tmp_path)
    files = _write_sample(config, tmp_path)
    assert files == _write_sample(config, tmp_path)
    assert "m1" not in files["labels.csv"] and "m2" not in files["labels.csv"]
    assert not any(
        "m1" in text or "m2" in text for name, text in files.items() if name.startswith("items/")
    )
    assert len(files["rater2_items.txt"].split()) == 4
    assert all('"content"' not in line for line in files["sample.jsonl"].splitlines())


def test_summary_skips_kappa_when_unlabeled_and_rejects_unknown_reasons(tmp_path):
    config = _config(tmp_path)
    errors.write_files(_write_sample(config, tmp_path), config.output_dir)
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in _records()), encoding="utf-8"
    )
    lines = errors.write_summary(config)
    assert any("Rater 2: skipped" in line for line in lines)
    assert any(line.startswith("Q2 (m2") for line in lines)
    labels = tmp_path / "out" / "labels.csv"
    labels.write_text(
        labels.read_text(encoding="utf-8").replace(",,\n", ",bogus,\n", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bogus"):
        errors.write_summary(config)
