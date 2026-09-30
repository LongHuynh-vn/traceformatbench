# Runtime probe F0-F5 (2026-09-30): exploratory, not a benchmark result

Config: `configs/runtime_probe_f0_f5.yaml` (spec 004 probe with all six formats). Placeholder prompt; answers were never
parsed or scored. Cases: re2ob_currencyservice_cpu_1 (largest windows), re2tt_ts-auth-service_disk_1,
re3tt_ts-auth-service_f4_1, re3ob_adservice_f5_1; 23 windows x 6 formats x 2 models + 2 warm-ups = 278 calls.

## Findings
- 278/278 calls `ok`, done_reason `stop`, no cached prompt tokens. Largest prompt: 20,622 tokens, so num_ctx 32768 is
  enough for the largest windows_v1 window.
- Median seconds per call: qwen3.5:9b F0 17.6, F1 9.4, F2 4.6, F3 2.7, F4 1.3, F5 1.7;
  gemma4:12b F0 30.7, F1 16.6, F2 8.0, F3 4.8, F4 1.9, F5 2.7.
- `raw calls`: `calls.jsonl` holds the full prompts (serialized RCAEval traces) and is git-ignored;
  `calls_no_prompts.jsonl` is the same records without `system_prompt` and `user_prompt`.

## Main-grid estimate for windows_v1 (`grid_estimate_windows_v1.csv`)
- Method: per model, a linear fit of `total_s` on `prompt_eval_count` over the 276 measured calls; token counts of all
  524 windows x 6 formats estimated from the characters of the REAL v1 task prompts (spec 005, incl. T2 candidate
  lists) divided by the probe's median characters per token for that model and format.
- Result: about 22 hours for 6,288 calls (qwen3.5:9b ~7.8 h, gemma4:12b ~14.2 h), within the 48-hour budget.
  Estimated largest prompt ~21.5k tokens (F0), under num_ctx 32768.
- `estimate.csv` is the probe script's older F1-bound estimate (F2-F5 timed as F1) and overstates the grid.
