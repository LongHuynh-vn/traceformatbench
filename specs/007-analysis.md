# 007 Analysis: bootstrap CIs, paired format comparisons, self-time baseline, Pareto

Status: DRAFT written by Claude from the student's answers (2026-09-30). The student reviews and edits this
file before any agent implements it. Items marked [default] were not asked; change them here if needed.
Hypotheses (H1-H3) are NOT part of this spec: the student writes them in PREREGISTRATION.md.
Revised 2026-09-30 after an independent review (a second Claude agent); changes are listed under "Review changes".

## Goal
One command turns scored model calls (`scores.jsonl` from `scripts/score_tasks.py`) plus the heuristic baseline
into the tables and the plot that answer the research question, following `06_METHODOLOGY_AND_EVAL_STANDARDS.md`:
case-level bootstrap 95% CIs, paired bootstrap on differences against the F0 control, a non-LLM baseline,
and accuracy versus tokens.

## Known facts (windows_v1, verified 2026-09-30)
- T2: 370 windows in 74 cases; informative 193 windows in 54 cases
  (RE2-OB 89 in 24 cases, RE2-TT 73 in 21, RE3-OB 5 in 1, RE3-TT 26 in 8). Informative windows per case: 1 to 5.
- T1: 154 windows (77 pre / 77 post) in 77 cases.
- Scored records already carry: case, dataset, system, task, window_index, informative, label, model, format,
  status, parse_status, prompt_eval_count, total_s, correct (T1) or hit_at_1 / hit_at_3 / chance_at_1 / chance_at_3 (T2).
- Every failed call (call status or parse status not ok) is already scored as wrong (spec 005).
- Call status is one of "ok", "truncated", "thinking_leak" (spec 004 `call_status`).
- `window_index` is NOT unique within (case, task) for T1: each case has a pre window 0 and a post window 0.
  Window key used everywhere in this spec: `(case, task, label, window_index)` (verified unique on windows_v1).
- `dataset` values are "RE2-OB", "RE2-TT", "RE3-OB", "RE3-TT" (system is already part of the name).

## Analysis sets
- `T2-informative` (PRIMARY): T2 windows with `informative == true`.
- `T2-all` (secondary): all T2 windows.
- `T1-all` (secondary): all T1 windows.
Primary metric: T2 Acc@1 on `T2-informative`. Secondary: T2 Acc@3 (both T2 sets), T2 Acc@1 on `T2-all`,
T1 accuracy, plus descriptive rates (invalid_rate, truncated_rate, T1 post_rate).

## Estimation
1. Point estimate: pooled mean over the windows of the set (every window weight 1; a case with 5 windows
   counts 5 times as much as a case with 1).
2. Bootstrap unit: CASES, stratified by the `dataset` field. Resampling population of a set = the cases with at least one
   window in that set. In each iteration, within each stratum draw as many cases as the stratum has, with
   replacement; a drawn case brings all its windows in the set (a case drawn twice counts twice). The statistic of
   the iteration is the pooled mean over the drawn windows.
3. Small strata [default]: a stratum with fewer than 5 cases in the set is merged into the stratum of the other
   dataset of the same system (RE3-OB into RE2-OB, RE3-TT into RE2-TT). In `T2-informative`, RE3-OB (1 case)
   is merged into RE2-OB; RE3-TT (8 cases) stays separate. Merging is decided per analysis set in one pass and
   never crosses systems; if a merged stratum still has fewer than 5 cases it stays merged and a warning is
   printed. `strata.csv` lists, per set, the original and the merged strata with their case counts.
4. Iterations and interval: B = 10,000, 95% percentile interval (2.5th and 97.5th percentiles,
   `np.percentile(..., method="linear")`), seed from the config.
5. Shared resamples: for each analysis set, draw the B resamples ONCE and reuse
   them for every model, format and the baseline. This makes every comparison paired by construction and the
   output independent of the order in which conditions are processed.
   RNG: `np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{set_name}".encode()).digest()[:8], "big"))`.
   Never Python's built-in `hash()` (randomized per process).
6. Paired comparison (format X vs control F0, same model, same task, same set): in each iteration compute
   stat(X) - stat(F0) on the same drawn cases; point estimate = difference of the pooled means; 95% percentile CI.
   Verdict: "clear difference" if and only if ci_low > 0 or ci_high < 0 (strict; a bound exactly at 0 is
   "no clear difference"), otherwise "no clear difference". No other wording.
   Also report the discordant counts: windows where only X is correct, windows where only F0 is correct.
   Before comparing, check that X and F0 have exactly the same window keys; a mismatch raises an error naming
   the missing keys. Key uniqueness is checked per (model, format, task) before any computation.

## Comparisons
- MAIN (student, 2026-09-30): F1, F2, F3, F4, F5 each vs F0, per model, on T2 Acc@1 / `T2-informative`
  (5 comparisons per model, 10 in total). All are reported, whatever the result.
- Secondary (same method): the same five comparisons for T2 Acc@3, for `T2-all`, and for T1 accuracy;
  each LLM condition vs the self-time baseline on T2 Acc@1 (both T2 sets).
- No correction for multiple comparisons; the report states the number of main comparisons as a limitation.

## Non-LLM baseline: total self-time (module `src/traceformatbench/analysis/baselines.py`)
- Score of a candidate service in a window = sum of `self_time_us` over that service's spans in the window's
  traces, calling the SAME function the F4/F5 serializers use (`serializers.structure.self_time_us`);
  candidates absent from the window score 0.
- Ranking shown in `answer`: sort candidates by (-score, service name), first 3.
- Scoring with ties (review change): ties must not decide hits through alphabetical order. For Acc@k, if the
  top-k cut falls inside a group of tied scores, the label (when in that group) gets expected credit
  = (number of tied slots left inside the top k) / (size of the tied group), i.e. the expected hit of a random
  tie-break. So `hit_at_1` / `hit_at_3` of baseline records may be fractions. Report the share of windows with a
  tie at the cut (`tie_rate`).
- Candidates: exactly the T2 candidate list of spec 005 (`case_candidates` of all the case's traces).
- `scripts/run_baseline.py configs/windows_v1.yaml` writes `results/baselines/selftime_v1.jsonl` with one scored
  record per T2 window in the same schema as `scores.jsonl` (model = "heuristic-selftime", format = "-",
  status = parse_status = "ok", hit_at_1, hit_at_3, chance_at_1, chance_at_3, informative, ...).
  Cases are loaded one at a time, as in `build_windows`.
- ORDER RULE [default]: the baseline is implemented and tested now, but its accuracy on windows_v1 is first
  computed AFTER PREREGISTRATION.md is committed, together with the main grid, so its result cannot shape
  the hypotheses. Tests use synthetic windows only.

## Tokens and efficiency
- Tokens per call: `prompt_eval_count` from every call whose status is not "truncated" (parse failures and
  thinking leaks still processed the full prompt; truncated calls report a wrong count, spec 004).
  Per condition and set: mean prompt tokens per window, with its bootstrap CI on the same resamples.
- Reliability flag `tokens_reliable` = false for a condition with any truncated call or any call with
  `prompt_eval_cached_count > 0`; such points are marked on the plot. The main grid must use a num_ctx large
  enough that no call truncates (largest window is ~42.6k F0 characters).
- Token counts come from each model's own tokenizer, so they are compared only WITHIN a model, never across models.
- Accuracy per 1K tokens (plan 03): Acc@1 / (mean prompt tokens / 1000), numerator and denominator over the same
  windows of the same set; descriptive, no CI.
- Pareto plot `pareto_t2_informative.png`: one panel per model (no cross-model frontier), x = mean prompt tokens, y = Acc@1 with 95% CI
  bars, one point per format, Pareto-optimal formats (no other format has fewer tokens AND higher Acc@1) marked,
  baseline and chance (mean chance_at_1) as horizontal lines. Needs `matplotlib` added with `uv add matplotlib`.
- Chance: `summary.csv` also has a "chance" row per T2 set (mean chance_at_1 / chance_at_3 with CI on the same
  resamples), since the number of candidates differs by case (7 in RE3-OB, more in TT).
- Wall time: mean `total_s` per call per condition, descriptive (compute cost; local models cost no money).

## Inputs, outputs, command
- Config `configs/analysis_v1.yaml`: seed, n_boot (10000), ci (0.95), control ("F0"), min_stratum_cases (5),
  scores path(s), baseline path, output dir.
- Command: `uv run python scripts/analyze.py configs/analysis_v1.yaml`
- Outputs in the output dir:
  - `summary.csv`: one row per (model, format, task, set, metric) with n_windows, n_cases, estimate, ci_low,
    ci_high, plus mean_prompt_tokens (+ CI), acc_per_1k_tokens, mean_total_s, invalid_rate, truncated_rate.
  - `comparisons.csv`: one row per comparison with is_main, estimate_diff, ci_low, ci_high, verdict.
  - `strata.csv`: strata and case counts per set after merging.
  - `pareto_t2_informative.png`.
- Modules: `analysis/bootstrap.py` (resample drawing, pooled statistic, paired difference; vectorized with numpy:
  per-case sums and counts times the resample count matrix), `analysis/report.py` (tables, plot).

## Edge cases
- A condition with no windows in a set: row with n_windows 0 and NaN estimate, no error.
- Missing or duplicate window keys for a condition: raise with the keys.
- A set with fewer than 2 cases in total: estimate reported, CI NaN, verdict "not enough cases".
- Records whose `informative` is None are never in a T2 set; T1 records never in a T2 set and vice versa.

## Tests that must pass
- Bootstrap: all-correct data -> estimate 1 and CI [1, 1]; identical X and F0 -> diff 0 and CI [0, 0].
- Vectorized bootstrap equals a naive loop implementation on a 6-case toy set with the same resample indices.
- T1 pre and post windows with the same window_index in one case are two different keys (no collision).
- A CI bound exactly at 0 gives "no clear difference".
- Cross-process determinism: two runs in separate processes give byte-identical `summary.csv` and `comparisons.csv`.
- Token rule: parse-invalid calls count for tokens, truncated calls do not; `tokens_reliable` flips on one
  truncated or cached call.
- Baseline tie scoring: hand-built ties at the top-1 and top-3 cut give the expected fractional credit.
- LLM-vs-baseline pairing works with the baseline's format "-".
- Stratification: every resample draws exactly n_s cases from each stratum; small-stratum merge rule
  (1-case RE3-OB merged into RE2-OB, 8-case RE3-TT kept).
- Case weighting: a hand-built example with fixed resample indices gives the hand-computed pooled mean
  (a case drawn twice counts twice).
- Shared resamples: results do not change when conditions are passed in a different order; determinism for a seed.
- Paired check: mismatched window keys raise.
- Verdict strings: CI excluding 0 on either side -> "clear difference"; CI containing 0 -> "no clear difference".
- Baseline: golden ranking on a hand-built window (sum over spans of a service, absent candidates score 0,
  A-Z tie-break); record schema matches `scores.jsonl` fields used by the analysis.
- Pareto: dominated formats not marked (hand-built points).
- Smoke: `analyze.py` runs end to end on `results/tasks_smoke/scores.jsonl` with a tiny n_boot (numbers not checked).

## Out of scope
- Running the main grid, running the baseline on windows_v1 (see ORDER RULE), writing PREREGISTRATION.md.
- Error analysis (failure sampling and taxonomy): separate spec after the main grid.
- T1P, token-budget sweep, attribute ablations, `make reproduce` wrapper.

## Interpretation notes (for the report)
- F4 and F5 hand the model self-time numbers directly; "F4/F5 vs baseline" therefore mainly tests whether the
  model reads the maximum of a table it was given. Say so when reporting.
- 10 main comparisons at 95% without correction: about one false "clear difference" is expected by chance
  if nothing differs. Stated in threats to validity.

## Review changes (independent review agent, 2026-09-30)
Window key includes `label` (T1 collision); stratum = `dataset` field, merge edge cases; SHA-256 seeding;
strict verdict at 0 and discordant counts; baseline uses the serializers' self-time and expected credit on ties;
token counts from all non-truncated calls, within-model only, reliability flag; chance rows; extra tests.

## Decisions (student, 2026-09-30)
1. Primary T2 set = informative windows (193); all 370 T2 windows = pre-registered secondary.
   This reverses the 2026-09-29 choice (all windows primary); reason: 20 answerable cases have no informative
   window, where no format can see the fault, so those windows only dilute format differences.
   PREREGISTRATION.md must record this change and its reason.
2. Bootstrap = cases, stratified by dataset, 10,000 iterations, 95% percentile CI.
3. Main comparisons = F1-F5 vs F0 per model, paired bootstrap, 95%, "clear difference" only if the CI excludes 0.
4. Non-LLM baseline = total self-time per service.
