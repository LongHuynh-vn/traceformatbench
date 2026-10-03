# 005b Paired T1 task (T1P) and a small pilot

Status: APPROVED 2026-09-30. The student approved adding T1P and running the pilot below.
T1P is added next to T1. Whether T1P replaces T1 in the main grid is decided by the student after the pilot and
written in PREREGISTRATION.md.

## Why
In the spec 005 smoke run every T1 answer was constant per model (qwen3.5:9b always `pre`, gemma4:12b always
`post`, 16/16 each, both formats, both cases). A constant answer scores exactly 0.5 on the balanced design, so T1
may not separate formats. A single window gives the model no reference for "normal"; a pair gives one.

## Task definition (T1P)
- Input: one pre-injection window and one post-injection window of the same case, shown as window A and window B.
- Question: which window was recorded after the fault injection? Label: `A` or `B`.
- Pairs: pair i = (T1 pre window i, T1 post window i) from `build_case_windows` with `configs/windows_v1.yaml`
  (so 2 pairs per case, built from the same windows as T1; no new sampling).
- Order is counterbalanced per case: even pair index -> A = pre, B = post (label `B`);
  odd pair index -> A = post, B = pre (label `A`). With 2 pairs per case, a model that always answers the same
  letter scores exactly 0.5.

## Prompt (fixed, English, task `T1P`, prompt version `v1`)
System: the same `SYSTEM_PROMPT` as spec 005.
User (`{n}` = traces per window, `{a}`/`{b}` = serialized windows in the same format):
```
Below are two windows of {n} traces each, recorded in the same microservice system.
At some point, a fault was injected into one service of this system. One window was recorded before the fault
injection and the other after it.

<window_a>
{a}
</window_a>

<window_b>
{b}
</window_b>

Which window was recorded after the fault injection?
Answer with {"after": "A"} or {"after": "B"}.
```
Line breaks inside a sentence are only for readability here; in code each sentence is one line.

## Output schema
`{"type": "object", "properties": {"after": {"type": "string", "enum": ["A", "B"]}}, "required": ["after"],
"additionalProperties": false}`

## Changes
1. `tasks/prompts.py`: `build_t1p_prompt(a_text, b_text, n) -> str`.
2. `tasks/schemas.py`: `t1p_schema()`.
3. `tasks/scoring.py`: `parse_answer` accepts task `T1P` (same strict rules, key `after`, value in {A, B});
   `score_record` sets `correct`; `metrics` for T1P reports `n`, `invalid_rate`, `fence_rate`, truncated and
   thinking-leak counts, `accuracy`, and `a_rate` (share of valid answers equal to `A`). Add `a_rate` to the columns.
4. `runners/tasks.py`: `TaskConfig.tasks` may contain `T1P`; the call list builds one call per (model, case, pair,
   format). Call record as in spec 005, with `task = "T1P"`, `window_index` = pair index, `label` = `A`/`B`,
   extra field `a_phase` (`pre` or `post`), `informative` = null. No prompt text in the record (hash only).
5. `configs/t1p_pilot.yaml`: like `tasks_smoke.yaml`, but `tasks: [T1P]`, `num_ctx: 65536`,
   `output: results/t1p_pilot/calls.jsonl`.
   Reason: a pair doubles the prompt. On the downloaded OB case the largest F0 pair is ~67.7K characters,
   ~35K tokens at 1.95 characters per token, over 32,768. The existing prompt-length check must still pass.
6. T1 and T2 behavior, prompts and records are unchanged.

## Edge cases
- A case with fewer than 2 pre or 2 post T1 windows: build only the complete pairs; if a case has no pair, stop
  with a clear message (as for missing tasks in spec 005).
- Pre and post windows of a pair must share `case`, `window_index` and format; assert this.

## Tests that must pass
- `build_t1p_prompt` exact string; A/B texts placed correctly.
- Pair construction: pair 0 has A = pre (label B), pair 1 has A = post (label A); trace IDs come from the T1 windows.
- `t1p_schema` exact dict; `parse_answer` for T1P: valid A/B, lowercase `a` (schema_violation), extra key, fenced JSON.
- `metrics` on hand-computed T1P records: accuracy, a_rate, invalid answers count as wrong.
- A config with `tasks: [T1P]` builds 2 calls per (model, case, format) on a synthetic case.
- Existing 239 tests still pass. No network and no Ollama in tests.

## Pilot (after merge, exploratory, not a result)
- `configs/t1p_pilot.yaml` on the two downloaded cases: 2 cases x 2 pairs x 2 formats x 2 models = 16 calls
  (roughly 15-25 min; pairs are twice as long as T1 windows).
- Memory note: `num_ctx 65536` needs more KV-cache memory than 32,768. Watch `ollama ps` during the first call;
  if a model no longer fits fully on the GPU (Ollama shows a CPU/GPU split) or calls get much slower, stop and
  report before continuing.
- Only look at: status, invalid/fence rate, and the answer distribution (`a_rate`, whether answers vary across
  pairs). Accuracy on 4 pairs per model means nothing and must not be quoted.
- Decision rule (student): answers still constant per model -> keep T1 as is (or drop it to secondary);
  answers vary -> consider T1P as the T1 definition in PREREGISTRATION.md.

## Revision 1 (2026-09-30, student-approved): align window times in T1P
Problem found in review before the pilot: F0 and F1 keep absolute timestamps (F0 `time` HH:MM, `startTimeMillis`,
`startTime`; F1 `startTime`). In a pair, the window with the later timestamps is always the post window, so T1P could
be solved by comparing clock times without reading any fault signal.

Rule (T1P only): before serializing a pair, shift every span of each window by a constant so that the earliest span
start of BOTH windows equals `anchor = min(earliest start of pre window, earliest start of post window)`.
- Within a window, relative start offsets, durations, parent links, span order and all other fields are unchanged.
- The serializers are not changed; they receive shifted copies of the traces.
- T1 and T2 inputs, prompts and records are unchanged (their serialized texts must be byte-identical to before).
- Suggested interface: `align_pair_times(pre: Sequence[Trace], post: Sequence[Trace]) -> tuple[list[Trace], list[Trace]]`
  in `src/traceformatbench/tasks/pairs.py`; T1P texts are serialized from the aligned traces (the runner needs the
  window traces, not only the pre-serialized texts).
- Call record: add `time_aligned: true` for T1P.

Tests that must pass (in addition to the ones above):
- `align_pair_times` on synthetic windows: both outputs have the same earliest start (= anchor); every span's offset
  from its window's earliest start and every duration are unchanged; the input traces are not mutated.
- For a synthetic pair, the T1P prompt in F0 and F1 has the same minimum `startTime` in window A and window B.
- T1 and T2 prompts for a synthetic case are identical before and after this change.

## Out of scope
- Changing T1 or T2, formats F2-F5, the main grid, PREREGISTRATION.md.

## Pilot outcome (2026-09-30, exploratory, not a result)
- 16/16 calls `ok`, every `done_reason` was `stop`, invalid 0, fence 0. `ollama ps` showed qwen3.5:9b at 100% GPU with
  context 65536 (6.8 GB). Largest prompt 32,655 tokens (gemma, OB F0 pair). Call time without load: qwen 23-90 s,
  gemma 41-170 s.
- Answer distribution: gemma4:12b answered `B` in 8/8 pairs; qwen3.5:9b answered `B` in 7/8 (one `A`). The constant
  label answer of T1 (pre/post) became a constant position answer (last window shown).
- Per the decision rule above, answers are still (almost) constant per model, so T1P does not fix T1. For comparison,
  T2 rankings in the spec 005 smoke run did change across windows (and sometimes across formats).
- Decision (student, 2026-09-30): T2 becomes the primary task; T1 (single window) stays as a secondary task with
  1 window per phase; T2 goes from 3 to 5 windows per case (`configs/windows_v1.yaml`: 1/5, still 7 windows per case,
  same compute budget). T1P is not used in the main grid; its code stays for reproducibility. To be written in
  PREREGISTRATION.md.
