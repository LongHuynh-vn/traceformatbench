# 005 Task prompts, output schemas, scoring, and task runner

Status: APPROVED 2026-09-30. Drafted by Claude from the student's decisions below; the student accepted the
prompt wording and the defaults for all open questions (see "Decisions on open questions").

## Goal
Define the fixed T1 and T2 prompts, the JSON output schema for each task, how answers are parsed and
scored, and a resumable runner that calls the models on windows. The same runner will later run the main
grid; in this spec it is only used for a smoke run on the two downloaded cases.

## Decisions already made (student, 2026-09-30)
1. T2 answer is a ranked list of exactly 3 services, most likely first. Acc@1 = first item is the root
   cause; Acc@3 = the root cause is among the three. One call gives both metrics.
2. T2 candidates: every service that appears in any span of the case's traces (all traces, before any
   eligibility filter), sorted A-Z, given in the prompt. Chance level per window: Acc@1 = 1/k, Acc@3 = 3/k
   (k = number of candidates; about 7 for OB and 27 for TT on the two downloaded cases).
3. Output is constrained with Ollama structured output: a JSON schema passed in the `format` field of
   `/api/chat`, the same schema for both models. T2 service names are an `enum` of the case's candidates.
   A strict parser still validates every answer, so a server that ignores the schema is caught.
4. T1 primary metric: accuracy (chance 0.5, the design is balanced: 2 pre + 2 post windows per case).
   Secondary: precision, recall and F1 with `post` as the positive class, and the share of `post` answers.
- From earlier specs: zero-shot, thinking OFF, temperature 0, fixed seed, `num_ctx` explicit, windows from
  `configs/windows_v1.yaml`. Case-level bootstrap CIs follow 06_METHODOLOGY but are out of scope here.

## Prompts (fixed, English, version `v1`)
Identical for every format: only the serialized window changes. The prompt does not name or describe the
format. Order: context line, traces, question, answer instruction (the long input comes before the question).

System (both tasks):
```
You analyze distributed traces from a microservice system. Reply with one JSON object only.
```

T1 user message (`{n}` = traces in the window, `{traces}` = serialized window):
```
Below are {n} traces recorded during one time window of a microservice system.

<traces>
{traces}
</traces>

At some point, a fault was injected into one service of this system. Were these traces recorded before
or after the fault injection?
Answer with {"phase": "pre"} if before, or {"phase": "post"} if after.
```

T2 user message (`{candidates}` = candidate services, comma-separated, A-Z):
```
Below are {n} traces recorded after a fault was injected into exactly one service of a microservice system.

<traces>
{traces}
</traces>

Candidate services: {candidates}
Which service is the root cause (the service where the fault was injected)?
Answer with {"ranking": [first, second, third]}: the three most likely root-cause services from the
candidate list, most likely first, no duplicates.
```
Line breaks inside a sentence above are only for readability of this spec; in code each sentence is one line.

## Output schemas (sent as Ollama `format`)
- T1: `{"type": "object", "properties": {"phase": {"type": "string", "enum": ["pre", "post"]}},
  "required": ["phase"], "additionalProperties": false}`
- T2: `{"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "string",
  "enum": <candidates>}, "minItems": 3, "maxItems": 3}}, "required": ["ranking"], "additionalProperties": false}`

## Inputs
- `configs/tasks_smoke.yaml`, for example:
  ```yaml
  seed: 20260930
  prompt_version: v1
  ollama_url: http://localhost:11434
  models: [qwen3.5:9b, gemma4:12b]   # one model at a time, in this order
  num_ctx: 32768
  num_predict: 256
  temperature: 0
  windows_config: configs/windows_v1.yaml
  cases: [re2ob_checkoutservice_cpu_1, re2tt_ts-auth-service_cpu_1]
  formats: [F0, F1]
  tasks: [T1, T2]
  output: results/tasks_smoke/calls.jsonl
  ```
- Windows come from `build_case_windows` (same function and per-case seed as `windows_v1.jsonl`), as in the
  runtime probe. If `data/windows/windows_v1.jsonl` exists, the runner checks that each case's trace IDs match
  it and stops on any mismatch.

## Outputs
1. `src/traceformatbench/tasks/prompts.py`: `PROMPT_VERSION = "v1"`, `SYSTEM_PROMPT`,
   `build_t1_prompt(traces_text, n) -> str`, `build_t2_prompt(traces_text, n, candidates) -> str`,
   `case_candidates(traces) -> list[str]` (sorted union of span service names).
2. `src/traceformatbench/tasks/schemas.py`: `t1_schema()`, `t2_schema(candidates)` returning the dicts above.
3. `src/traceformatbench/tasks/scoring.py`:
   - `parse_answer(task, content, candidates) -> ParsedAnswer` with `status` in
     `ok | invalid_json | schema_violation` and the parsed value (`phase` or `ranking`) when ok.
     Parsing: strip whitespace; if the whole content is wrapped in one Markdown code fence, remove it (count
     how often this happens); `json.loads`; then validate exactly against the schema (T1 phase in {pre, post};
     T2 a list of exactly 3 strings, each in the candidates). No other repair.
   - `score_record(call_record) -> dict`: T1 `correct`; T2 `hit_at_1`, `hit_at_3`, `n_candidates`,
     `chance_at_1`, `chance_at_3`. Any status other than ok scores as wrong (0).
   - `metrics(scored_records) -> DataFrame` per (model, format, task) and per (model, format, task, system):
     `n`, `invalid_rate`, `fence_rate`; T1 `accuracy`, `precision_post`, `recall_post`, `f1_post`,
     `post_rate`; T2 `acc_at_1`, `acc_at_3`, mean `chance_at_1`, `chance_at_3`, and the same two accuracies on
     informative windows only (`informative == True`) with their `n`.
4. `src/traceformatbench/runners/ollama.py`: `chat()` gets an optional `output_schema: dict | None = None`,
   sent as the payload key `format` when not None. (Named `output_schema` to avoid a clash with the trace
   formats F0-F5.)
5. `scripts/run_tasks.py <config.yaml>`: for each model (never interleaved): all cases x tasks x windows x
   formats. Before each call, the prompt-length check from the runtime probe; after it, `call_status`
   (`ok | truncated | thinking_leak`). Streams one JSON line per call and resumes after interruption.
   Call record: `case, dataset, system, fault, root_cause_service, task, window_index, informative, label,
   model, model_digest, ollama_version, format, prompt_version, seed, num_ctx, num_predict, temperature,
   think_sent, candidates (T2 only), prompt_sha256, prompt_chars, prompt_eval_count,
   prompt_eval_cached_count, eval_count, done_reason, total_s, load_s, content, status`.
   The record does NOT contain the prompt text (it holds raw traces); prompts can be rebuilt from the
   window, format and prompt version, and `prompt_sha256` proves which one was sent. So `calls.jsonl` can be
   committed.
6. `scripts/score_tasks.py <calls.jsonl>`: writes `scores.jsonl` and `metrics.csv` next to it. Scoring is
   separate from calling, so answers can be re-scored without new model calls.

## Edge cases
- Root-cause service not in the case's candidates (name mismatch or service never traced): stop before the
  first call and list those cases. Do not silently drop them.
- Fewer than 3 candidates: stop with a clear message (T2 needs 3).
- A T2 ranking with duplicates or an unknown service: `schema_violation`, scored wrong.
- `truncated` or `thinking_leak` calls: kept in `calls.jsonl`, scored as wrong in `metrics.csv`, and counted
  in their own columns so they are visible.
- Model not pulled or Ollama not running: stop before the first call.

## Tests that must pass
- Prompt builders on a tiny window: exact expected strings, identical except `{traces}` across F0 and F1.
- `case_candidates`: sorted, unique, includes services seen only in ineligible traces.
- Schemas: exact dicts; T2 enum equals the candidates.
- `chat()` payload contains `format` only when `output_schema` is given (HTTP mocked).
- `parse_answer`: bare JSON, fenced JSON, invalid JSON, missing key, extra key, wrong phase value,
  ranking of length 2 and 4, duplicate, unknown service.
- `score_record` and `metrics` on hand-computed records (T1 accuracy, precision/recall/F1 of post,
  post_rate; T2 Acc@1, Acc@3, chance, informative subset; invalid answers count as wrong).
- Resume: records already in the output are not called again.
- No network and no Ollama in tests.

## Smoke run (after merge, exploratory, not a result)
- `configs/tasks_smoke.yaml` on the two downloaded cases: 2 cases x 7 windows x 2 formats x 2 models = 56 calls
  (about 20-25 min from the runtime probe timings).
- Only checks: every status is ok, `invalid_rate` and `fence_rate` per model (expected ~0 with structured
  output), answer token counts. Accuracy on 2 cases means nothing and must not be quoted.

## Out of scope
- Formats F2-F5 (next spec) and the token-budget sweep.
- Bootstrap CIs, paired comparisons, plots (analysis spec).
- Running the main grid, building windows for all 78 cases.

## Decisions on open questions (student, 2026-09-30; all defaults accepted)
1. Prompt wording above: accept as is, or change? [accept]
2. Should a T2 ranking with a duplicate be scored as invalid, or de-duplicated and scored? [invalid]
3. Strip one Markdown code fence before parsing, or treat any fence as invalid? [strip, and report fence_rate]

## Implementation notes (review, 2026-09-30)
- Undefined T1 precision/recall/F1 (e.g. a model that never answers `post`) are reported as 0.0.
- The fence strip only removes a fence with the content on its own lines; a one-line fence counts as `invalid_json`.
