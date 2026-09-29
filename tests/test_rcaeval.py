import pandas as pd

from traceformatbench.data.rcaeval import (
    find_traces_file,
    load_traces,
    summarize_traces,
    trace_quality,
)


def _spans() -> pd.DataFrame:
    # Two traces: t1 has 3 spans, t2 has 1 span. Synthetic, not real RCAEval data.
    return pd.DataFrame(
        {
            "traceID": ["t1", "t1", "t1", "t2"],
            "spanID": ["a", "b", "c", "d"],
            "parentSpanID": [None, "a", "a", None],
            "serviceName": ["frontend", "cartservice", "frontend", "adservice"],
            "duration": [1200, 300, 150, 90],
            "startTime": [1_000_000_000, 1_000_000_100, 1_000_000_200, 2_000_000_000],
            "statusCode": [0, 0, None, 2],
        }
    )


def test_summary_counts():
    summary = summarize_traces(_spans())
    assert summary["n_spans"] == 4
    assert summary["n_traces"] == 2
    assert summary["spans_per_trace"]["min"] == 1
    assert summary["spans_per_trace"]["max"] == 3
    assert summary["services"] == ["adservice", "cartservice", "frontend"]


def test_summary_larger_trace_has_more_chars():
    chars = summarize_traces(_spans())["json_chars_per_trace"]
    assert chars["max"] > chars["min"] > 0


def test_summary_without_trace_id_column():
    summary = summarize_traces(_spans().drop(columns=["traceID"]))
    assert "n_traces" not in summary
    assert "note" in summary


def test_load_traces_prefers_parquet(tmp_path):
    _spans().to_parquet(tmp_path / "traces.parquet")
    _spans().head(1).to_csv(tmp_path / "traces.csv", index=False)
    assert find_traces_file(tmp_path).name == "traces.parquet"
    assert len(load_traces(tmp_path)) == 4


def test_load_traces_missing_returns_none(tmp_path):
    assert find_traces_file(tmp_path) is None
    assert load_traces(tmp_path) is None


def test_quality_clean_traces():
    q = trace_quality(_spans(), inject_time_s=1500, root_cause_service="cartservice")
    assert q["traces_by_root_count"] == {"0": 0, "1": 2, ">1": 0}
    assert q["traces_with_orphan_spans"] == 0
    assert q["traces_start_before_inject"] == 1
    assert q["traces_start_at_or_after_inject"] == 1
    assert q["traces_touching_root_cause_service"] == 1
    assert q["status_codes"]["<NA>"] == 1


def test_quality_missing_root_is_orphan():
    spans = _spans()
    spans = spans[spans["spanID"] != "a"]  # drop t1's root; b and c become orphans
    q = trace_quality(spans, inject_time_s=1500)
    assert q["traces_by_root_count"]["0"] == 1
    assert q["traces_with_orphan_spans"] == 1
    assert "traces_touching_root_cause_service" not in q
