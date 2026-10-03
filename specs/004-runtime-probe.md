# 004 Runtime probe

Status: APPROVED 2026-09-29. The student accepted both recommendations (see "Decisions on open questions").
Exploratory measurement, not a benchmark result. Its only job is to choose the number of windows per
case (`t1_windows_per_phase`, `t2_windows`) before PREREGISTRATION.md.

## Goal
Measure how long one model call takes on this laptop for real spec-003 windows, per model and format,
and turn that into a runtime estimate for the main grid under different window counts.

## Decisions already made (student, 2026-09-29)
- Thinking is OFF for the main grid (direct JSON answer). A thinking ablation is parked in IDEAS_BACKLOG.md.
- Compute budget for the main grid: about 48 hours on this laptop (resumable, split over several runs).
- This probe runs before the task spec, with a placeholder instruction. Real prompts come in spec 005.

## Background
- Main grid size: 78 cases x (2 * t1_windows_per_phase + t2_windows) windows x 6 formats (F0-F5) x 2 models.
  With 3/3 that is 8,424 calls: ~47 h at 20 s per call, ~94 h at 40 s per call.
- Only F0 and F1 exist so far. F2-F5 are compressions of F1, so F1 time is used as their upper bound.
- Windows are at most ~17.7K tokens (spec 003 review), well under num_ctx 32,768.
- Ollama silently truncates prompts over num_ctx (token probe); every call must be checked for it.
- Ollama caches a shared prompt prefix; cached tokens make prefill look faster than it will be.

## Inputs
- `configs/runtime_probe.yaml`, for example:
  ```yaml
  seed: 20260929
  ollama_url: http://localhost:11434
  models: [qwen3.5:9b, gemma4:12b]   # run one model at a time, in this order
  num_ctx: 32768
  num_predict: 256
  temperature: 0
  windows_config: configs/windows_v1.yaml
  cases: [re2ob_checkoutservice_cpu_1, re2tt_ts-auth-service_cpu_1]   # already downloaded
  formats: [F0, F1]
  output: results/runtime_probe/calls.jsonl
  ```
- Windows: all windows that `build_case_windows` produces for these cases with `windows_v1.yaml`
  (9 per case), so the probe times exactly the inputs the benchmark will use. No download.

## Placeholder prompt
- Chat request with one system message and one user message.
- System (fixed, English): "You analyze distributed traces from a microservice system. Reply with one
  JSON object only." User: the serialized window, then the line
  `Answer as JSON: {"phase": "pre" or "post", "service": "<service name>"}`.
- The same placeholder for T1 and T2 windows. Answers are stored for debugging only and never scored.

## Outputs
1. `src/traceformatbench/runners/ollama.py`: new method
   `chat(model, system, user, num_ctx, num_predict, temperature, seed, think) -> ChatResult`
   using `/api/chat` with `stream: false`. `ChatResult` holds: `content`, `thinking` (empty string if
   none), `done_reason`, `prompt_eval_count`, `prompt_eval_cached_count` (0 if absent), `eval_count`,
   `load_duration_ns`, `prompt_eval_duration_ns`, `eval_duration_ns`, `total_duration_ns`.
   Plus `model_capabilities(model) -> list[str]` from `/api/show`.
   `think` is sent only when the model lists the `thinking` capability; otherwise it is omitted.
2. `src/traceformatbench/analysis/runtime_probe.py`: pure functions for building the call list,
   the truncation check (reuse `near_truncation_point` from `token_probe`), summarizing, and estimating.
3. `scripts/runtime_probe.py configs/runtime_probe.yaml`
   - Per model: one warm-up call on the first window, with a short prefix added to its user message so
     it shares no cached prefix with any measured call (recorded with `warmup: true`, excluded from the
     summary), then all windows x formats. Models run one after the other, never interleaved.
   - Streams one JSON line per call and resumes after interruption (same pattern as `measure_tokens.py`).
   - Prints progress per call, then writes `summary.csv` and `estimate.csv` next to `calls.jsonl`.
4. Call record (one JSON line): `case, dataset, model, model_digest, ollama_version, format, task,
   window_index, f0_chars, prompt_chars, num_ctx, num_predict, think_sent, warmup, prompt_eval_count,
   prompt_eval_cached_count, eval_count, done_reason, load_s, prompt_eval_s, eval_s, total_s, wall_s,
   thinking_chars, content, status` where `status` is `ok`, `truncated` or `thinking_leak`.
5. `summary.csv`: per (model, format, system OB/TT): n, median and max of call time
   (`total_s - load_s`, so a model reload after a resume does not inflate it), median prompt tokens,
   median eval_count, median prefill tokens/s, median decode tokens/s, share of `done_reason == "length"`.
6. `estimate.csv`: main-grid hours for every (t1_windows_per_phase, t2_windows) in {1, 2, 3, 4} x {1, 2, 3, 4}:
   `hours_f1_bound = sum over models and systems of n_cases(system) * (2*t1 + t2) * (median_F0 + 5 * median_F1) / 3600`
   (F1 time as the bound for F2-F5) and the conservative
   `hours_f0_bound = sum over models and systems of n_cases(system) * (2*t1 + t2) * 6 * median_F0 / 3600`,
   with n_cases OB = 40 (RE2-OB 30 + RE3-OB 10) and TT = 38 (RE2-TT 30 + RE3-TT 8).
   Plus `n_calls`, `fits_48h_f1_bound` and `fits_48h_f0_bound`.

## Edge cases
- `prompt_eval_count` near num_ctx // 2: status `truncated`, keep the record, exclude it from the summary,
  print a warning (should never happen with these windows).
- Non-empty `thinking` or `<think>` in `content`: status `thinking_leak`, excluded from the summary.
- `prompt_eval_cached_count > 0` on a non-warm-up call: keep it, but report the count in the summary.
- Model not pulled or Ollama not running: stop with a clear message before the first call.

## Tests that must pass
- `chat()` request payload: `think: false` sent only when the model has the thinking capability;
  `num_ctx`, `num_predict`, `temperature`, `seed` always sent explicitly (HTTP mocked, no real server).
- `ChatResult` parsing from a recorded `/api/chat` response, including a missing `prompt_eval_cached_count`.
- Truncation and thinking-leak status on synthetic records.
- Summary excludes warm-up, truncated and thinking-leak records.
- Estimate formula on hand-computed numbers (one model, one system, t1 = 1, t2 = 1).
- Resume: records already in the output file are not called again.
- No network and no Ollama in tests.

## Out of scope
- Real task prompts, output JSON schema, parsing answers, scoring (spec 005).
- Formats F2-F5 and the token-budget sweep.
- Downloading more cases or running the full 78-case window build.
- Choosing the final window counts: the student decides from `estimate.csv` and writes them in
  PREREGISTRATION.md.

## Decisions on open questions (student, 2026-09-29)
1. Keep `num_predict: 256`.
2. Keep the placeholder prompt as written; it does not fix the wording of the real T1/T2 prompts.

## Outcome (2026-09-29)
- 74/74 calls `ok`: no truncation, no thinking leak, no cached prefix, every `done_reason` was `stop`.
  Largest prompt: 16,511 tokens. Client overhead (wall minus total) at most 0.02 s.
- Prefill dominates call time: qwen3.5:9b ~380-390 tokens/s, gemma4:12b ~220-240 tokens/s;
  outputs are ~20 tokens (decode ~40 and ~25 tokens/s).
- Median call time (s), F0 / F1: qwen OB 31.9 / 17.3, qwen TT 19.1 / 9.7, gemma OB 59.0 / 30.6, gemma TT 34.3 / 17.4.
- Main-grid estimate with F1 time as the bound for F2-F5: 3/3 windows = ~51.0 h (over budget);
  2/3 = ~39.7 h; 2/4 and 3/2 = ~45.3 h. With F0 time for all six formats, only up to 5 windows per case fit in 48 h.
- Decision (student): `t1_windows_per_phase: 2`, `t2_windows: 3` (6,552 calls, ~8 h margin in the budget).
- Note for spec 005: all 37 gemma4:12b answers were wrapped in a ```json code fence, qwen3.5:9b answers were bare JSON.
  The answer parser or Ollama structured output (`format`) must handle this.
- Raw calls with full prompts stay local (git-ignored); `calls_no_prompts.jsonl` is committed.
