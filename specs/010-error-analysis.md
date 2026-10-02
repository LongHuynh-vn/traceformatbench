# 010 Error analysis: sample wrong T2 answers, flag them automatically, label the reason by hand

Status: READY FOR IMPLEMENTATION (2026-10-02). Design agreed by the student on 2026-10-02 after the open-reading pilot
(`Progress and Plan/Research/error_pilot/`, outside the repo). The student owns the codebook, every label and every
conclusion; code only samples, computes automatic flags and prepares the labeling material.

This analysis is EXPLORATORY: it is not part of the pre-registration (PREREGISTRATION.md section 14 lists "the error
analysis of plan 03 (not pre-registered here)"). To keep it honest anyway, `CODEBOOK.md` and the sample are committed
BEFORE any item is labeled, labels are reported as counts, and nothing here changes registered scores or verdicts.

## Goal
Describe HOW the models fail on the primary task (T2 Acc@1, informative windows): a short table of failure reasons per
model and format, plus 2-3 concrete examples for the write-up (`06_METHODOLOGY_AND_EVAL_STANDARDS.md` section 5).

## Questions
- Q1 (main): which failure reasons occur, and how often, per model (and per format)?
- Q2 (light, counts only): on how many informative windows is gemma F2 right where F0 is wrong, and the reverse?
  Answered by a count in the summary script; no extra labeling.
- Q3 (F4 OB vs TT) is NOT part of this spec; a small targeted round may follow later.

## Known facts (main grid, results on main at 1b3c406, counted 2026-10-02 from results/main_grid/scores.jsonl)
- T2 informative: 193 windows / 54 cases per model x format; 77-115 wrong rank-1 answers per model x format cell,
  about 1,170 in total. Invalid answers (`parse_status == "schema_violation"`): qwen F1 13, F2 19, F3 16, F5 27; gemma F5 4.
- Scored records carry case, dataset, system, fault, root_cause_service, window_index, label, model, format,
  candidates, answer (list), content, hit_at_1, hit_at_3, parse_status, prompt_sha256, prompt_eval_count.
  Prompts are not stored; they are rebuilt from `data/windows/windows_v1.jsonl` with the same code as the grid
  (`runners/tasks.py`: `load_task_windows`, `build_t2_prompt`, `SYSTEM_PROMPT`, `prompt_sha256`).
- Pilot finding (exploratory): models very often rank a fixed "habitual" service first: `currencyservice` on OB,
  `ts-auth-service` on TT. These are also the most frequent true root causes (OB 35/94, TT 49/99 informative windows).

## Design (agreed)
1. Unit = one wrong T2 answer on an informative window (`task == "T2"`, `informative == true`, `hit_at_1 == 0`).
2. Two layers per item: AUTOMATIC flags computed by script (section "Automatic flags"), and ONE MANUAL reason chosen by
   the student from `CODEBOOK.md`. The script never assigns a manual reason.
3. Sample: 60 items, 30 per model, 5 per model x format cell. Within a cell: F0, F2, F4 take 3 OB + 2 TT, F1, F3, F5
   take 2 OB + 3 TT (so each model has 15 OB + 15 TT). Inside a cell no case is used twice when avoidable.
   Draw with `random.Random(f"{seed}:{model}:{format}:{system}")`, seed 20260930, from the eligible records sorted by
   (case, window_index). Excluded: the 10 pilot items (listed in the config). Drawn once and committed before labeling.
4. The labeler sees, per item: the prompt rebuilt in the format the model saw, plus the F2 view of the same window for
   easier reading, the true root cause, the model's answer and the automatic flags. The model name is hidden in the
   labeling sheet (kept in a separate key file); the format is visible because the prompt shows it.
5. Reliability: a second rater labels 12 items (20%, drawn with the same seed) blind to the first rater's labels.
   Report raw agreement and Cohen's kappa on those 12.

## Automatic flags (computed per item from the rebuilt window, not from the text)
- `contract`: `parse_status != "ok"` (invalid answer, e.g. one service repeated).
- `habitual`: top-1 service == the habitual service of the system (`currencyservice` for ob, `ts-auth-service` for tt;
  names fixed in the config) and top-1 != root cause. Always false for contract items without a top-1.
- `true_rank`: 1-3 if the root cause is in the answer, else 0.
- `top1_is_ancestor`: top-1 service owns a span that is an ancestor of a root-cause-owned span in the same trace.
- `top1_owns_longest_root`: top-1 service owns the longest root span (by duration) among the window's traces.
- `root_span_count`, `root_max_self_ms`: spans owned by the root-cause service in the window and their largest
  self-time (same self-time definition as `serializers/structure.py` `self_time_us`).

## CODEBOOK.md (v1, written from the pilot; the student may edit the wording before the commit)
One manual reason per item:
- `caller_blame`: top-1 is a caller (ancestor) of the root cause, and its long duration is mostly time spent inside
  that child path.
- `longest_root`: top-1 owns the longest top-level request in the window, and that length is not explained by the
  root-cause path.
- `weak_signal`: the root-cause service is present but shows nothing unusual in the window (e.g. only short spans).
- `unclear`: none fits; write a one-line note.
Contract items whose first listed service is the true root cause (`contract` True and `true_rank` 1): use `unclear`
with the note `correct service, invalid answer only`. They failed only the answer contract, not the diagnosis.
Every item also gets a free-text `note` (one short sentence). `habitual` and `contract` are flags, not reasons.

## Inputs
- `results/main_grid/scores.jsonl`, `data/windows/windows_v1.jsonl`, raw RCAEval data under `data/raw/rcaeval/`.
- `configs/error_analysis_v1.yaml` (new): seed 20260930, per-cell sizes (as in Design 3), habitual services,
  excluded pilot items, second-rater size 12, output dir `results/error_analysis_v1`.
  Pilot items to exclude, all format F0, keyed (model, case, window_index):
  qwen3.5:9b: (re2ob_productcatalogservice_socket_1, 1), (re3tt_ts-route-service_f3_1, 4),
  (re2ob_recommendationservice_disk_1, 3), (re2tt_ts-travel-service_cpu_1, 2), (re2ob_recommendationservice_mem_1, 0);
  gemma4:12b: (re2ob_productcatalogservice_mem_1, 3), (re2tt_ts-order-service_cpu_1, 2),
  (re2ob_recommendationservice_delay_1, 3), (re2tt_ts-order-service_loss_1, 2), (re2ob_productcatalogservice_socket_1, 0).

## Outputs
`scripts/sample_errors.py configs/error_analysis_v1.yaml` writes to `results/error_analysis_v1/`:
- `sample.jsonl`: one record per item: `item_id` (E01-E60, in a seeded random order so model/format are not
  grouped), the scored-record keys except `content`, plus the automatic flags.
- `labels.csv`: item_id, system, format, true root cause, answer, automatic flags, empty `reason` and `note` columns
  (no model column).
- `key.csv`: item_id, model.
- `rater2_items.txt` + `labels_rater2.csv`: the 12 second-rater items, same columns as labels.csv, empty labels.
- `items/<item_id>.md`: flags, answer, true root cause, the F2 view, then the full prompt the model saw in a
  `<details>` block.
- `CODEBOOK.md`: the codebook above.
`scripts/summarize_errors.py configs/error_analysis_v1.yaml` (run after labeling) prints and writes `summary.csv`:
counts of reason x model x format, counts of the automatic flags, raw agreement + Cohen's kappa on the rater-2 items
(skipped with a message if labels are empty), and the Q2 counts (gemma informative windows: F2 right & F0 wrong,
F0 right & F2 wrong) from scores.jsonl.

## Edge cases
- Rebuilt prompt hash differs from `prompt_sha256` -> stop with an error naming the item (never label a prompt the model
  did not see).
- Contract items: the answer may repeat a service; flags use the first item when present.
- A cell with fewer eligible records than requested -> take all, print a warning with the count.
- The root-cause service owning no span in an informative window -> error (should not happen).
- `summarize_errors.py`: unknown reason in labels.csv -> error listing the allowed reasons.

## Tests that must pass
- Same config and inputs -> byte-identical outputs; changing the seed changes the sample.
- Cell sizes and OB/TT split as in Design 3; no item twice; no pilot item; every item is T2, informative, hit_at_1 0.
- Prompt hash check passes for every sampled item on real data (dry run on the Mac) and fails on a tampered fixture.
- Automatic flags correct on a hand-built 3-service trace (caller -> callee -> root cause, plus a longer unrelated root).
- Cohen's kappa on a small hand-computed example; Q2 counts on a fixture.
- `ruff check`, `ruff format --check`, full `pytest` green.

## Out of scope
- Any change to registered scores, PREREGISTRATION, analysis_v1 or main-grid outputs.
- New model calls; automatic (LLM) labeling of reasons; Q3.

## Implementation choices (2026-10-02, simplest reading where the spec was open)
- Prompts are rebuilt with `TaskConfig` from `configs/main_grid.yaml`, narrowed to the sampled cases, task T2 and the
  sampled formats plus F2, through `load_task_windows` (which also checks the trace IDs against windows_v1.jsonl).
  Each case's traces are loaded a second time for the flags, because `load_task_windows` does not keep them.
- Contract items: the registered `answer` is null, so the ranking the model wrote is parsed from `content` (same fence
  rule as the scorer) and stored as `raw_ranking`; that list is the answer shown to the labeler and the source of the
  first item. `answer` itself is kept unchanged in `sample.jsonl`.
- `top1_is_ancestor` follows `parent_id` links inside one trace and is false when top-1 is the root cause itself
  (possible only for contract items), so a root-cause span nested under another root-cause span does not count.
- `top1_owns_longest_root`: a trace's root span is `Trace.root` (its first parentless span); on a tie for the longest
  duration, the flag is true if top-1 owns any of the tied roots.
- No repeated case in a cell: the cell's eligible records are shuffled with the cell RNG, the first records with new
  cases are taken, and only if that gives too few the rest is filled from the same shuffled order.
- Item order: items sorted by (model, format, system, case, window_index), shuffled with
  `random.Random(f"{seed}:item_order")`, numbered E01-E60. Rater-2 items: `random.Random(f"{seed}:rater2")` samples
  12 of the sorted item IDs.
- The codebook text lives in `configs/error_analysis_v1_codebook.md` and is copied to `CODEBOOK.md`; edit the config
  copy, then re-run the sampler. Allowed reasons are listed in the config (`reasons`).
- `root_max_self_ms` is self-time in microseconds divided by 1000, unrounded. In the CSVs, flags are written as
  `True`/`False` and the answer as `a | b | c`.
- `summarize_errors.py` computes agreement and kappa only when all rater-2 items have a reason in both label files
  (kappa is NaN if expected agreement is 1). `summary.csv` is long format (section, model, format, name, value):
  reasons and flag counts per model x format, `true_rank` counts, reliability and Q2.
