# 002 Serializers F0 (raw rows) and F1 (minified)

Status: DRAFT written by Claude. The student reviews and edits this file before any agent implements it.
Decisions behind it: see "Decisions log" in `03_IDEA_A_TRACEFORMATBENCH.md` (2026-09-29).

## Goal
Turn a window (a group of traces) into text for the LLM prompt. F0 is the control condition:
the rows exactly as RCAEval ships them. F1 is the cheapest compression: drop fields that are
derivable or repeated, nothing else.

## Known facts (verified on re2ob_checkoutservice_cpu_1 and re2tt_ts-auth-service_cpu_1)
- Original column order: time, traceID, spanID, serviceName, methodName, operationName,
  parentSpanID, startTimeMillis, startTime, duration, statusCode.
- `time` equals the UTC "HH:MM" of `startTime` for 100% of rows in both cases.
- `startTimeMillis == startTime // 1000` for every row.
So all 11 columns can be rebuilt exactly from `Span` (spec 001).

## Inputs
- `window: Sequence[Trace]` (from `traceformatbench.data.trace`), in the order given by the caller.
  Spans inside each trace keep the `Trace.spans` order.

## Outputs
New package module `src/traceformatbench/serializers/` with:
1. `serialize_f0(window) -> str`
   - One flat JSON array. One object per span, all traces concatenated in window order.
   - Keys and key order exactly as the original columns (list above).
   - `time` = UTC "HH:MM" of start_us; `startTimeMillis` = start_us // 1000;
     `startTime` = start_us; `duration` = duration_us.
   - Missing values (parentSpanID of the root, methodName, statusCode) are JSON `null`.
   - Encoding: `json.dumps(records, ensure_ascii=False)` with default separators, no indent.
2. `serialize_f1(window) -> str`
   - JSON of the form `[{"traceID": ..., "spans": [...]}, ...]`: traceID written once per trace.
   - Each span keeps only: spanID, parentSpanID, serviceName, operationName, methodName,
     startTime, duration, statusCode (drops `time` and `startTimeMillis`, which are derivable).
   - Keys with `null` values are omitted (root has no parentSpanID key; missing method/status omitted).
   - Encoding: `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`.
3. `SERIALIZERS: dict[str, Callable[[Sequence[Trace]], str]] = {"F0": serialize_f0, "F1": serialize_f1}`
   so runners pick formats by ID from configs.

## Edge cases
- Empty window: raise `ValueError`.
- Trace without root: serialized like any other trace, no special marking.
- Unicode in operation names: kept as is (no \u escapes).

## Tests that must pass
- F0 round trip: for a synthetic `traces.parquet` fixture, `json.loads(serialize_f0(load_case_traces(...)))`
  equals the original rows (same keys, same order of keys, same values, nulls as None),
  with rows ordered as `Trace.spans`.
- F1 golden test on a tiny window (2 traces, 4 spans): exact expected string.
- F1 omits null keys; root span has no `parentSpanID` key.
- Determinism: calling each serializer twice on the same window gives byte-identical strings.
- Window order is respected: reversing the window reverses trace order in the output.
- `SERIALIZERS` contains exactly "F0" and "F1".

## Out of scope
- Window sampling (how traces are chosen, window length, size cap): separate spec.
- Token counting (depends on each model's tokenizer; measured by runners).
- F2-F5.

## Decisions
- F1 is pure field dropping and keeps absolute `startTime`. Relative timestamps are a separate
  transformation, deferred to F2 or an ablation, so an F1-vs-F0 difference is attributable to
  field dropping alone.
