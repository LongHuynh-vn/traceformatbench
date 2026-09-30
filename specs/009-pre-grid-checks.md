# 009 Pre-grid fixes: truncation-band re-count, grid completeness check, multiplicity wording

Status: IMPLEMENTED (2026-09-30, branch `feat/pre-grid-checks`). Drafted with Claude from PREREGISTRATION.md
section 14 (K1, K2, K3, K7). The rules come from the pre-registration; this spec only implements them. No main-grid
call is made by anything here.

## 1. K1: re-count calls flagged only by the num_ctx/2 band (runners/tasks.py)
Problem: `call_status` marks a call `truncated` when `prompt_eval_count` is within 16 tokens of `num_ctx // 2`
(16,368-16,400 for num_ctx 32768), even if nothing was truncated. Rough estimate from the real v1 prompt sizes: a few
qwen3.5:9b F0 prompts land in or near this band, so without the fix F0 could lose correct answers.
Rule (PREREGISTRATION section 6): when a main-grid call is marked `truncated` ONLY because of the band (count below
num_ctx and no other reason), send the same request once more with num_ctx = num_ctx * 3 // 4 (24576). If the second
`prompt_eval_count` equals the first, the call was not truncated: store status `ok` and score the ORIGINAL answer.
Otherwise keep `truncated`. The answer of the re-count request is never used.
Record fields added to every call record: `recount_num_ctx` (int or null), `recount_prompt_eval_count` (int or null),
`status_before_recount` (string or null). Calls outside the band never trigger a second request.
Resume: a record that already has a completed re-count is not re-counted again.

## 2. K2 + K3: `scripts/check_grid.py configs/main_grid.yaml results/main_grid/calls.jsonl`
Read-only check, exit code 0 only if all pass, printing one line per check:
- exactly one record per planned call of the config's call list (built as in `--dry-run`, no Ollama); report the
  counts of missing, extra and duplicate keys (total must be 6,288 for configs/main_grid.yaml);
- for each model: exactly one `model_digest` and one `ollama_version` across its records (print them);
- counts per status (`ok`, `truncated`, `thinking_leak`) and the number of re-counted calls.
`scripts/analyze.py` is not changed; the pre-registration requires this check to pass before scoring.

## 3. K7: wording
- `analysis/report.py` and `specs/007-analysis.md`: "about one false clear difference" -> "about 0.5 false clear
  differences are expected among 10 comparisons at 95% (at least one with about 40% probability if independent)".

## Tests that must pass
- Band hit with equal re-count -> status `ok`, original answer kept and scored; band hit with a different re-count ->
  `truncated`; count >= num_ctx -> `truncated` without a re-count; a normal call -> no second request (fake client).
- Resume does not repeat a completed re-count.
- `check_grid.py`: passes on a complete synthetic grid; fails on a missing, an extra and a duplicate key; fails on two
  digests or two Ollama versions for one model.
- Report text test updated for the new wording.

## Implementation choices beyond the literal text (reviewed 2026-09-30)
1. A band-hit call whose re-count matches keeps `thinking_leak` (not `ok`) when its original answer leaked thinking,
   so it still scores 0 as PREREGISTRATION section 6 requires. The thinking check is shared as `leaked_thinking()` in
   `analysis/runtime_probe.py`.
2. The prompt-length preflight also runs against num_ctx 24576 before a re-count request. The largest main-grid
   prompt (43,165 characters, about 22,136 + 256 tokens by the preflight estimate) passes it.
3. `check_grid.py` fails if any record has a status other than `ok`, `truncated` or `thinking_leak`.
4. `check_grid.py` stops with "No calls file at ..." when the calls file does not exist.
5. A record is written only after its re-count finished; a run interrupted between the two requests has no record,
   so resume sends both requests again.

## Out of scope
- Starting the grid, changing prompts, windows, scoring rules or the analysis.
