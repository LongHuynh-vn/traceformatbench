# 001 Trace assembly and traces-only case download

Status: DRAFT written by Claude. The student reviews and edits this file before any agent implements it.

## Goal
Turn one RCAEval case into a list of in-memory trace trees, and download only the files the
benchmark needs for all traced cases. This is the input layer for every serializer (F0-F5).

## Known facts about the data (from `scripts/inspect_case.py` on re2ob_checkoutservice_cpu_1)
- One row per span. Columns: time, traceID, spanID, serviceName, methodName, operationName,
  parentSpanID, startTimeMillis, startTime, duration, statusCode.
- `startTime` is in microseconds (`startTime // 1000 == startTimeMillis`). `duration` is assumed
  to be microseconds too; confirm on one Train Ticket case before relying on it.
- Root span: `parentSpanID` is missing. Almost every trace has exactly one root; a few have none
  (their spans point to a parent that is not in the data).
- `statusCode` is 0 or missing in this case. Do not assume error codes exist.
- Traced cases: RE2-OB, RE2-TT, RE3-OB, RE3-TT (`has_traces` in `cases.parquet`).

## Inputs
- `case_dir`: path to a downloaded case directory containing `traces.parquet` and `inject_time.txt`.
- For download: a list of case names and a destination directory.

## Outputs
1. `download_traced_cases(cases: list[str], dest) -> list[Path]` in `data/rcaeval.py`:
   downloads only `traces.parquet` and `inject_time.txt` per case (not metrics or logs).
   Skips cases already present. Idempotent.
2. `data/trace.py` with frozen dataclasses:
   - `Span`: span_id, parent_id (None for root), trace_id, service, operation, method,
     start_us (int), duration_us (int), status_code (int | None).
   - `Trace`: trace_id, spans (sorted by start_us, then span_id), `root` (Span | None),
     `children(span_id) -> list[Span]`, `orphan_span_ids` (spans whose parent is not in the trace),
     `start_us`, `end_us`, `services` (sorted unique).
   - `load_case_traces(case_dir) -> list[Trace]`, sorted by (start_us, trace_id) so order is deterministic.

## Edge cases
- Trace with no root span: `root` is None, orphans listed, trace still returned.
- Trace with more than one root: keep all spans; `root` is the earliest root; record the others
  as extra roots (field `extra_root_ids`).
- Empty string parentSpanID is treated the same as missing.
- Missing statusCode becomes None, never 0.
- A case without `traces.parquet` raises a clear error naming the case.

## Tests that must pass
- Synthetic fixture: one normal trace, one trace with missing root, one trace with two roots.
- Children of the root are returned in start order.
- Order of `load_case_traces` output is stable across two calls.
- `download_traced_cases` is tested with the download function mocked (no network in CI).

## Out of scope
- Case sampling, labels, train/test splits (student decides after the label discussion).
- Any serializer.
