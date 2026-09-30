# 008 Main-grid readiness: per-case task availability, dry run, main-grid config

Status: DRAFT written by Claude (2026-09-30) after a review found that the task runner cannot run the 78-case grid.
The student reviews this file before any agent implements it. No model calls are made by anything in this spec.

## Problem (verified in `runners/tasks.py`, 2026-09-30)
1. `load_task_windows` raises "Root-cause service not in candidates" for any case whose root cause is not among the
   case's trace services, whatever the task. The 4 RE3-OB cases without T2 (spec 003 amendment) keep T1 windows, so a
   main grid with tasks [T1, T2] over all 78 cases stops before the first call.
2. It raises "produced no windows for a requested task" when a case lacks windows for one task.
   `re2tt_ts-train-service_socket_1` has no T1 windows (only 4 eligible pre traces), so the grid stops again.
3. There is no main-grid config and no way to check the call list without calling Ollama.

## Changes
1. Per-case task availability comes from the windows themselves (`build_case_windows`, which already applies the
   spec 003 rules):
   - Root cause not among the candidates: allowed when the case has no T2 windows (then only its T1 windows run);
     still an error if such a case has T2 windows (cannot happen with the current window builder).
   - Fewer than 3 candidates: an error only if the case has T2 windows.
   - A case without windows for a requested task: that task is skipped for that case (no error). A case with no
     windows at all for any requested task is still an error.
   - The check against the committed `data/windows/windows_v1.jsonl` (`check_saved_windows`) stays as is.
2. `scripts/run_tasks.py --dry-run`: build the windows and the full call list exactly as a real run would, make no
   Ollama request and write no output file, and print: number of calls per (model, task, format), total calls, cases
   skipped per task with the reason, and the largest prompt in characters.
3. `configs/main_grid.yaml`: seed 20260930, prompt_version v1, models [qwen3.5:9b, gemma4:12b] (one model at a time,
   in this order), num_ctx 32768, num_predict 256, temperature 0, windows_config configs/windows_v1.yaml,
   cases = all 78 cases of windows_v1 in sorted order, formats [F0, F1, F2, F3, F4, F5], tasks [T1, T2],
   output results/main_grid/calls.jsonl. A header comment says: run only after PREREGISTRATION.md is committed.

## Expected dry-run result (windows_v1)
- T1: 154 windows x 6 formats x 2 models = 1,848 calls; T2: 370 x 6 x 2 = 4,440; total 6,288.
- Skipped: T2 for re3ob_adservice_f3_1, re3ob_adservice_f4_1, re3ob_adservice_f5_1, re3ob_cartservice_f1_1
  (root cause not in traces); T1 for re2tt_ts-train-service_socket_1 (no T1 windows).

## Tests that must pass
- A case whose root cause is absent from its traces runs T1 and skips T2; a case with T2 windows and an absent root
  cause still raises (use a monkeypatched window builder).
- A case without T1 windows runs its T2 windows; a case with no windows for any requested task raises.
- Fewer than 3 candidates raises only when the case has T2 windows.
- `--dry-run` makes zero client calls, writes no file, and prints the per-(model, task, format) counts.
- `configs/main_grid.yaml` loads, lists exactly the 78 windows_v1 cases in sorted order, and passes `TaskConfig`.
- Existing runner tests updated only where their expectation is the old global error.

## Out of scope
- Running the main grid (waits for PREREGISTRATION.md), prompt or scoring changes, baseline runs.
