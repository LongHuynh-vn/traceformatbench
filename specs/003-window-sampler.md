# 003 Window sampler and size-cap statistics

Status: APPROVED 2026-09-29. Open questions decided by the student (see "Decisions").
Numbers marked (probe) come from the exploratory token probe (`results/token_probe/`) and an ad-hoc
analysis of two cases. This spec makes them reproducible from committed code.

## Goal
Build the exact, fixed set of trace windows that every later experiment uses, so all formats and
models see identical inputs. Also produce the statistics needed to report how the size cap changes
the data (selection bias).

## Background (probe)
- Characters per token are stable: F0 ~2.0-2.1 (min 1.95), F1 ~2.2-2.7. gemma4:12b gives ~3% more
  tokens than qwen3.5:9b on the same text.
- OB traces: ~3.2-4.1K F0 tokens each. At num_ctx 32,768, F0 fits ~5 traces, not 10.
- TT is bimodal: ~54% of traces are <= ~8.4K F0 characters, the rest are ~64K characters or more.
- A per-trace cap of 10,000 F0 characters keeps ~89% of OB traces and ~54% of TT traces, and in
  re2tt_ts-auth-service_cpu_1 it raises the share of kept post-injection traces that pass through the
  root-cause service from ~30% to ~56%. The cap is not neutral and must be reported.
- With 5 random post-injection OB traces, about half of windows contain no trace through the
  root-cause service.
- Ollama silently truncates prompts longer than num_ctx (see token probe). Windows must fit by design.

## Definitions
- Trace size = `len(serialize_f0([trace]))` (F0 characters). F0 is the largest format, so a window that
  fits in F0 fits in every format.
- Eligible trace: size <= `max_trace_chars`, has a root span, and has no orphan spans.
- Phase of a trace, using its start time and the case's `inject_time`:
  - pre: start < inject_time - guard_seconds
  - post: start >= inject_time + guard_seconds
  - traces inside the guard band belong to no phase and are never sampled.
- Window: `window_size` consecutive eligible traces of one phase, ordered by start time, beginning at a
  random position within that phase (a short time slice, not a random scatter).
- A T2 window is `informative` if at least one of its traces includes the root-cause service.

## Inputs
- `configs/windows_v1.yaml`, for example:
  ```yaml
  seed: 20260929
  data_dir: data/raw/rcaeval
  datasets: [RE2-OB, RE2-TT, RE3-OB, RE3-TT]
  repetitions: [1]            # 1 case per (fault, service) pair: 78 cases; null = all 240
  window_size: 5
  max_trace_chars: 10000
  guard_seconds: 60
  t1_windows_per_phase: 3     # T1: this many pre AND this many post windows per case (balanced)
  t2_windows: 3               # T2: this many post windows per case
  output_windows: data/windows/windows_v1.jsonl
  output_stats: results/window_stats/windows_v1.csv
  ```
- Case index `cases.parquet` (via `load_case_index`), filtered to `has_traces` and the config filters.

## Outputs
1. `src/traceformatbench/data/windows.py`
   - `trace_size_chars(trace) -> int`
   - `phase_of(trace, inject_time_s, guard_seconds) -> Literal["pre", "post"] | None`
   - `eligible(trace, max_trace_chars) -> bool`
   - `sample_windows(traces, phase, window_size, n, rng) -> list[list[Trace]]`: non-overlapping
     windows when possible; if the phase has fewer than `window_size` eligible traces, return fewer
     windows (never pad, never reuse traces within one window).
   - `build_case_windows(case_meta, traces, config, rng) -> tuple[list[dict], dict]`: window records
     plus one stats row.
2. `scripts/build_windows.py configs/windows_v1.yaml`
   - Downloads missing cases with `download_traced_cases`, processes one case at a time (memory:
     a TT case needs ~1.8 GB), writes both outputs, prints a short summary.
   - One RNG per case: `random.Random(f"{seed}:{case}")`, so adding or removing a case does not change
     the windows of other cases.
3. Window record (one JSON line):
   `case, dataset, system, fault, root_cause_service, task ("T1"|"T2"), phase ("pre"|"post"),
   label ("pre"|"post" for T1; root_cause_service for T2), window_index, trace_ids (ordered),
   n_spans, f0_chars, informative (T2 only, else null), config_name`.
4. Stats row per case (CSV):
   `case, dataset, n_traces, n_eligible, share_eligible, n_eligible_pre, n_eligible_post,
   share_post_via_root_cause_all, share_post_via_root_cause_eligible, n_t1_windows, n_t2_windows,
   n_t2_informative`.
   These columns regenerate the probe's cap-bias numbers for every case.

## Edge cases
- Phase with fewer than `window_size` eligible traces: fewer windows for that case; recorded in stats.
- T1 balance: if one phase yields fewer windows than the other, keep only the matched number from
  both phases so T1 labels stay 50/50 per case.
- Case without traces or missing download: raise with the case name (do not skip silently).
- Determinism: same config => byte-identical `windows_v1.jsonl`.

## Tests that must pass
- Phase assignment including the guard band (boundary values exactly at inject_time +/- guard).
- Eligibility: oversize trace, trace without root, trace with orphans are excluded.
- `sample_windows`: consecutive traces, correct length, no duplicates, deterministic for a seed,
  returns fewer windows when the phase is short.
- T1 balance rule.
- `informative` flag true/false on synthetic windows.
- Per-case RNG: removing one case from the config leaves the other cases' windows unchanged.
- Stats row values on a small synthetic case (hand-computed expected numbers).
- No network in tests (download mocked).

## Out of scope
- Prompts, tasks, scoring, model runs (next specs).
- Choosing final values of `t1_windows_per_phase` / `t2_windows`: the student fixes them in
  PREREGISTRATION.md after a runtime estimate.

## Decisions (student, 2026-09-29)
1. Case set: repetition 1 only (`repetitions: [1]`, 78 cases: RE2-OB 30, RE2-TT 30, RE3-OB 10, RE3-TT 8).
2. Guard band: keep `guard_seconds: 60`.
3. Windows per case: keep `t1_windows_per_phase: 3` and `t2_windows: 3` as provisional config values.
   Final counts are chosen after a small runtime test and fixed in PREREGISTRATION.md.
   Update 2026-09-29: chosen from the runtime probe (spec 004): `t1_windows_per_phase: 2`, `t2_windows: 3`.
4. T2: primary analysis on all T2 windows, plus a pre-registered secondary analysis on informative
   windows only. This spec only needs to produce the `informative` flag; the analysis itself is out of scope.

## Amendment (student, 2026-09-30, after building windows_v1 for all 78 cases)
5. T2 only for answerable cases: a case gets T2 windows only if its root-cause service appears in the
   case's traces (the same set as the T2 candidate list, `case_candidates`). Otherwise the case keeps its
   T1 windows and gets no T2 windows; the stats row records `t2_answerable`.
   Reason: in 4 RE3-OB cases (adservice f3/f4/f5, cartservice f1) the root-cause service never appears in
   any trace, so every T2 answer would be wrong regardless of format (20 windows, 240 main-grid calls).
   T2 sampling is the last use of the per-case RNG, so T1 windows of every case are unchanged.
   Test: `test_no_t2_when_root_cause_is_not_among_the_case_services`.
