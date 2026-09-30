# 006 Serializers F2-F5 (span tree, heaviest path, per-service aggregates, top-k spans)

Status: DRAFT written by Claude from the student's answers (2026-09-30). The student reviews and edits this
file before any agent implements it. Items marked [default] were not asked; change them here if needed.

## Goal
Add the four structured formats from plan 03 so the main grid can compare F0-F5 on the same windows.
Each format is a deterministic function `Sequence[Trace] -> str`, registered in `SERIALIZERS`.
The prompts (spec 005) do not change: every format explains itself in its first line.

## Known facts (verified 2026-09-30 on re2ob_checkoutservice_cpu_1 and re2tt_ts-auth-service_cpu_1, eligible traces)
- Sibling spans never overlap in time (0 parents with overlapping children in both cases).
- Children can end after their parent: 1 parent in OB, 496 of 20,956 parents in TT (async calls).
- `statusCode` is 0 or missing everywhere; `methodName` exists only in OB and repeats the end of `operationName`.
- Within a 5-trace T2 window, most (service, operation) pairs in TT occur about once, so a within-window
  z-score is not computable for most spans.
- 26% of OB spans last < 50 us; 1% < 5 us. TT spans are all >= 100 us.
- Size check with a throwaway prototype of this spec, first-line legends not included (they add about
  100-230 characters) (20 random post windows per case, characters, median):
  OB F1 17,783 / F2 6,404 / F3 700 / F4 159 / F5 681; TT F1 6,160 / F2 2,404 / F3 1,401 / F4 201 / F5 538.
  Every F2-F5 window was smaller than its F1 window, so the F1-based runtime budget (spec 004) stays an upper bound.

## Shared definitions (module `src/traceformatbench/serializers/structure.py`)
1. `self_time_us(trace, span) -> int`: `span.duration_us` minus the length of the UNION of its direct
   children's intervals `[child.start_us, child.start_us + child.duration_us)`, each clipped to the parent's
   interval `[span.start_us, span.start_us + span.duration_us)`; never below 0. Clipping handles async
   children that end after the parent; the union handles overlap if it ever occurs.
2. `fmt_ms(us: int | float) -> str`: `f"{us / 1000:.2f}"` (milliseconds, 2 decimals) [default].
3. Trace numbering: `Trace i` where i is the 1-based position in the window. No traceID or spanID is printed.
4. Tree order: children in `Trace.children()` order (start time, then span ID). Every root span
   (`parent_id is None`) starts a tree at depth 1, roots in `Trace.spans` order (normally one root).
5. Indentation: 2 spaces per depth level (depth 1 = 2 spaces).
6. Offset: `at=+{fmt_ms(span.start_us - trace.start_us)}ms` (relative to the trace's earliest span).
7. Output: lines joined with `"\n"`, no trailing newline. Empty window: raise `ValueError`.

## Formats (module `src/traceformatbench/serializers/structured.py`)

### F2 span tree
First line: `# Span tree per trace. dur = span duration, at = start offset from the trace start, in ms.`
Per trace: header `Trace {i} ({len(trace.spans)} spans)`, then every span as
`{indent}{service} {operation} dur={fmt_ms(duration_us)}ms at=+{offset}ms`.
Drops spanID, parentSpanID (indentation replaces them), methodName, statusCode, absolute timestamps.

### F3 heaviest path (revised 2026-09-30 after review, see Decisions 2)
First line: `# Heaviest path per trace: from the root, always into the child span with the longest duration. self = span duration minus time covered by its child spans, in ms.`
Per trace: header as F2, then the path starting at `trace.root`; at each step go to the child with the
largest `duration_us`, ties broken by later end (`start_us + duration_us`), then by `Trace.children()` order.
Each hop: `{indent}{service} {operation} dur={fmt_ms(duration_us)}ms self={fmt_ms(self_time_us)}ms`,
indentation grows by one level per hop. Then `  (+{N} other spans)` where N = spans of the trace not on the
path (extra roots included); omit this line when N = 0.

### F4 per-service aggregates
First line: `# Per-service statistics over all spans of the 5 traces in this window. self = span duration minus time covered by its child spans, in ms.`
(the number is `len(window)`). Second line (column header): `service spans mean_self_ms max_self_ms`.
Then one line per service present in the window, sorted by service name (A-Z, not by any statistic):
`{service} {span_count} {fmt_ms(mean self_time_us)} {fmt_ms(max self_time_us)}`, single spaces.
No error rate (statusCode unusable), no percentiles (too few spans per service in a 5-trace window).

### F5 top-k spans by self-time
Parameter `k = 5` per window [default], constant `F5_K` in the module.
First line: `# The 5 spans with the highest self-time in this window (marked *) and their ancestors. dur = span duration, at = start offset from the trace start, self = span duration minus time covered by its child spans, in ms.`
(the number is `min(k, total spans)`).
Selection: rank all spans of the window by `(-self_time_us, trace position, index in Trace.spans)` and take the
first k. Keep each selected span plus all its ancestors up to the root.
Rendering: only traces that contain a selected span, in window order, with the header `Trace {i} (...)` as F2
(the original position i, so skipped traces are visible as gaps). Kept spans in F2 tree order and F2 line format;
selected spans append ` self={fmt_ms(self_time_us)}ms *`.
Scoring uses only spans inside the window (no pre-injection reference), like every other format.

## Registry and runners
- `SERIALIZERS` gains "F2", "F3", "F4", "F5" (F0 and F1 unchanged, byte-identical output).
- `runners/tasks.py` and `analysis/runtime_probe.py` currently reject formats outside {"F0", "F1"}:
  validate against `set(SERIALIZERS)` instead.
- Small helper `scripts/show_formats.py <case_dir> [--phase post] [--index 0]`: prints F0-F5 for one window of
  consecutive eligible traces (window config defaults) and each format's character count. For review only.

## Edge cases
- Empty window: `ValueError` (all formats).
- Span without children: self-time = duration.
- Child ending after its parent: clipped (see self-time); child starting before its parent: clipped too.
- Several roots: F2/F5 render each root's tree; F3 follows `trace.root` only and counts the rest in "+N".
- k larger than the number of spans in the window: select all spans.
- Unicode in names: kept as is.

## Tests that must pass (`tests/test_structured_serializers.py`)
- `self_time_us`: no children; two sequential children; child ending after the parent (clipped);
  overlapping children (union, not sum); result never negative.
- Golden tests: one tiny hand-built window (2 traces, about 4 spans each, one async child) with the exact
  expected string for each of F2, F3, F4, F5, written by hand from this spec.
- F3: a long early child beats a short trailing child; equal duration -> later end; full tie -> children order.
- F4: services sorted A-Z; mean and max computed on self-time, not duration.
- F5: exactly min(k, total spans) lines end with ` *`; ancestors of selected spans present; traces without a
  selected span absent; ties resolved by trace position then span index.
- Determinism: each serializer twice on the same window -> byte-identical.
- `SERIALIZERS` keys are exactly F0-F5; existing F0/F1 tests unchanged and passing.
- Runner and runtime-probe configs accept F2-F5 and still reject unknown IDs.

## Out of scope
- Prompt changes, token counting (runners measure tokens), running the grid or a probe with the new formats.
- Non-LLM baselines (analysis spec), token-budget sweep, attribute ablations.
- Any z-score or pre-injection reference data.

## Decisions (student, 2026-09-30)
1. F2 line = service, operation, duration and start offset from the trace start (relative timestamps, the
   transformation that spec 002 deferred from F1).
2. F3 = heaviest path: from the root, always into the longest child (plan 03: "longest-latency path");
   hops show duration and self-time; off-path spans are only counted.
   History: the first draft used "latest-ending child" and called it the true critical path. Review found this
   wrong: siblings here run sequentially, so the latest-ending child is just the last call, and the true critical
   path would include every sequential child (F3 would become F2 without offsets). On post-injection traces with
   more than one span, the latest-ending path missed the heaviest non-root span (by self-time) in 68% (OB) / 49%
   (TT) of traces; the longest-child path missed it in 6% / 15%. The student switched to the longest child.
3. F4 = per service over the window: span count, mean and max self-time. Self-time instead of total duration so
   the calling service does not absorb the latency of everything below it; no p95 on so few spans.
4. F5 = top-k spans by self-time plus ancestors, window-internal only. A z-score against pre-injection data was
   rejected: it would give F5 information the other formats do not get and confound the comparison.

## Open points for the student (before or during review)
- [default] k = 5, 2-decimal milliseconds, one self-describing first line per format. Change here if needed.
- For the analysis spec: F4/F5 make "service with the highest self-time" easy to read off. A matching non-LLM
  heuristic baseline (highest total or max self-time per service) would show whether the LLM adds anything
  beyond it; plan 03 already lists a latency heuristic baseline.
