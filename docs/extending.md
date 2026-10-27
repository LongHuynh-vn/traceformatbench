# Extending TraceFormatBench

The original experiment compares F0–F5 on fixed windows with two local models. Its ten primary comparisons,
40 exploratory secondary format comparisons and 24 exploratory baseline comparisons are specified in
[`PREREGISTRATION.md`](../PREREGISTRATION.md), section 9. Adding a format, model or backend creates additional
conditions; it does not extend the original registration retrospectively.

Start with a written specification in `specs/` for the proposed change. The research owner defines the information
retained or discarded, tasks, labels, prompts, metrics, baselines and analysis. Keep original configs, windows,
calls, scores, baseline records and analysis outputs intact. Use a new YAML configuration and output directory for
each extension, such as `configs/my_extension.yaml` and `results/my_extension/`. Any changed prompt or scoring rule
needs its own versioned contract. Preserve the original prompt builders and scorer for rechecking saved results.

The controlled input is the same window and call setup, not identical information in every representation: F1/F2
omit fields, and F3–F5 select or aggregate spans. An extension must state these content changes explicitly.

## Recheck the stored experiment

Run from the repository root with the locked environment:

```bash
uv sync --locked
uv run python scripts/verify_results.py configs/verify_v1.yaml
```

The verifier writes only to a temporary directory. Its five comparisons are:

| Checked output | Expected size | Comparison |
|---|---:|---|
| `results/main_grid/scores.jsonl` | 6,288 records | Exact records and order after rescoring saved calls |
| `results/main_grid/metrics.csv` | 72 rows | Recomputed metric table |
| `results/analysis_v1/summary.csv` | 68 rows | Recomputed summary table |
| `results/analysis_v1/comparisons.csv` | 74 rows | Recomputed paired-comparison table |
| `results/analysis_v1/strata.csv` | 12 rows | Recomputed resampling-strata table |

Table text must match exactly; numeric cells use the tolerances in
[`configs/verify_v1.yaml`](../configs/verify_v1.yaml). The analysis uses the committed baseline records as inputs.
Passing these checks establishes consistency of the stored scores and four tables. It does not reconstruct the
raw dataset or windows, regenerate the baseline from traces, rerun inference, authenticate calls, verify expected
model builds or runtime versions, or check manuscript text and figures. Prompt hashes and termination reasons are
also outside this verifier's checks. See [`analysis/verify.py`](../src/traceformatbench/analysis/verify.py).

Two separate commands have narrower additional purposes:

```bash
uv run python scripts/check_grid.py configs/main_grid.yaml results/main_grid/calls.jsonl
uv run python scripts/plot_paper_figure.py configs/paper_figure_v1.yaml
```

The grid check needs local RCAEval traces. It reconstructs planned calls and compares their full resume keys,
including prompt hashes and generation settings; it also checks for missing, extra or duplicate keys, a single
recorded digest/version per model, and recognized statuses. It does not compare builds against an expected digest,
authenticate their recorded values, or verify the raw files' source revision and checksums. The figure command
renders the paper figure from the committed analysis tables; it is separate from the five saved-result checks.

Rebuilding byte-identical windows needs identical source bytes, configuration and software behavior. The current
download paths in [`data/rcaeval.py`](../src/traceformatbench/data/rcaeval.py) do not pass a pinned revision, and the
trace downloader skips files already present without validating their provenance. The same YAML does not establish
raw-to-window reconstruction. A successful cached installation or local rerun also does not establish installation
from a fresh downloaded artifact on another machine.

## Run a small offline interface example

This synthetic fixture exercises the current serializer, prompt, schema and parser interfaces. It does not load
RCAEval, contact a model, write outputs, or claim a benchmark result. The candidate list normally comes from all
traces in a case before window filtering; here the synthetic case contains one trace.

```bash
uv run python - <<'PY'
import json

from traceformatbench.data.trace import Span, Trace
from traceformatbench.runners.tasks import prompt_sha256
from traceformatbench.serializers import SERIALIZERS
from traceformatbench.tasks.prompts import SYSTEM_PROMPT, build_t2_prompt, case_candidates
from traceformatbench.tasks.schemas import t2_schema
from traceformatbench.tasks.scoring import parse_answer

trace = Trace("demo", (
    Span("root", None, "demo", "gateway", "/order", None, 0, 1000, None),
    Span("b", "root", "demo", "billing", "/pay", None, 100, 200, None),
    Span("c", "root", "demo", "catalog", "/item", None, 400, 100, None),
))
window = [trace]
text = SERIALIZERS["F1"](window)
assert isinstance(text, str)
assert text.encode("utf-8") == SERIALIZERS["F1"](window).encode("utf-8")
assert [s["spanID"] for s in json.loads(text)[0]["spans"]] == ["root", "b", "c"]
candidates = case_candidates(window)
user = build_t2_prompt(text, len(window), candidates)
schema = t2_schema(candidates)
assert schema["properties"]["ranking"]["items"]["enum"] == candidates
assert len(prompt_sha256(SYSTEM_PROMPT, user)) == 64
assert parse_answer("T2", '{"ranking":["billing","catalog","gateway"]}', candidates).status == "ok"
assert parse_answer("T2", '{"ranking":["billing","billing","billing"]}', candidates).status == "schema_violation"
print("Offline example passed: F1 -> v1 prompt/schema -> strict parser.")
PY
```

The last assertion matters: the transmitted T2 schema constrains three candidate strings but does not contain
`uniqueItems`. The prompt and parser require three distinct services. Duplicate rankings remain invalid and score
zero under the original rules; do not silently repair them or replace the original schema.

## Add a serializer

The extension point is [`serializers/__init__.py`](../src/traceformatbench/serializers/__init__.py):
`SERIALIZERS` maps a format ID to a `Callable[[Sequence[Trace]], str]`. Implement a function such as
`serialize_f6(window: Sequence[Trace]) -> str` in a new serializer module, import it there, and add
`"F6": serialize_f6` to the registry after its specification is approved. Then select `F6` in the extension YAML's
`formats` list. `TaskConfig` validates IDs against this registry, and `serialize_windows()` uses it to serialize the
trace IDs already chosen for each window. There is no separate serializer plugin loader.

[`Trace` and `Span`](../src/traceformatbench/data/trace.py) store times and durations in microseconds. A `Trace`
orders spans by `(start_us, span_id)`; the loader orders traces by `(start_us, trace_id)`. Specify how the format
uses incoming window order, sorting, ties, rounding, null fields, Unicode and line endings. F0/F1 preserve incoming
window order; the structured formats have their own specified ordering. Do not assume every new format must have
the same information-preservation or ordering invariants. If it uses self-time, the existing
[`self_time_us()`](../src/traceformatbench/serializers/structure.py) subtracts the union of clipped direct-child
intervals; it does not subtract every descendant duration independently.

Use [`tests/test_serializers.py`](../tests/test_serializers.py) and
[`tests/test_structured_serializers.py`](../tests/test_structured_serializers.py) as examples. Tests for an added
format should cover exact expected strings on a small fixture, repeatable UTF-8 bytes, specified ordering and tie
breaks, empty-window rejection, missing/extra roots and orphans where relevant, Unicode, rounding, and input
non-mutation. Test both retained information and invariance to fields the specification says are discarded.
Keep all F0–F5 expected outputs intact. The current structured-serializer registry test asserts exactly F0–F5,
and a task-config rejection test uses `F6` as an unknown ID: update those inventories for an approved addition
while retaining a genuinely unknown-ID rejection test.

Include F0 alongside the added format when using the current paired analysis: it requires matching window keys
for each condition and its control. Give scoring and analysis separate output paths: `score_tasks.py` writes
`scores.jsonl` and `metrics.csv` beside the supplied calls file; `AnalysisConfig` reads its `scores`, `baseline`
and `output_dir` fields. Its `is_main` column is a programmatic selection rule, not proof that an added comparison
was registered. Do not merge extension rows into the original tables or present them as original primary results.

## Configure another Ollama model

There is no fixed model-name registry. Copy [`configs/tasks_smoke.yaml`](../configs/tasks_smoke.yaml) to a new
configuration under `configs/` and set `models` to the exact locally pulled tag and `output` to a new calls path.
Set `num_ctx` explicitly to a supported value and retain explicit `num_predict`, `temperature`, `seed`,
`prompt_version`, `ollama_url`, `windows_config`, `cases`, `formats` and `tasks`. `TaskConfig` currently accepts
`T1`, `T1P` and `T2`, requires the current prompt version (`v1`) and temperature zero, and rejects duplicate or
empty selection lists. A tag is not an immutable build: the runner records the local model digest and Ollama
version, but does not pin them to expected values or include them in its resume key. Use a fresh output after any
model-build or runtime change and check the recorded identity before combining runs.

For a new configuration named `configs/my_model_smoke.yaml`, this command makes no model requests and writes no
calls file, but needs its configured local traces and case index:

```bash
uv run python scripts/run_tasks.py configs/my_model_smoke.yaml --dry-run
```

This is a planning check, not an inference smoke test. `check_prompt_length()` currently estimates input tokens
as `ceil(prompt_chars / 1.95)` and rejects a prompt when that estimate plus `num_predict` reaches `num_ctx`.
The factor was measured for the original setup; it is not a tokenizer guarantee for a new model or representation.
Before authorizing inference, establish the new model's actual tokenization/context behavior with an approved
probe or exact chat-token counting. `OllamaClient.count_prompt_tokens()` counts raw text without a chat template
and generates one token; it is neither offline nor an exact count of the final chat request.

The current live runner sends `truncate=false`, the fixed system/user messages and output schema, and explicitly
sets context, output limit, temperature and seed. It sends `think=false` only when the model advertises thinking
support, recording `think_sent=null` otherwise. Returned counts at or above the context limit, or within the
legacy half-context truncation band, trigger status checks; eligible band cases are recounted at three quarters
of the configured context. The original answer remains the scored answer. This Ollama-specific protocol must be
validated for added conditions rather than assumed to transfer to another runtime.

The supported local machine has 24 GB unified memory. The runner processes one model at a time and unloads each
before the next. Check that both weights and the configured context fit; identify a need for a GPU, paid API or
more memory before running. A passing dry run does not establish resource fit, schema support or valid answers.

Token comparisons remain within a model. In [`analysis/report.py`](../src/traceformatbench/analysis/report.py),
`tokens_reliable=false` if any record in the entire `(model, format, task)` condition is truncated or has cached
prompt tokens, even outside the selected analysis set. Cached calls stay in token means; truncated calls are
excluded, while parse failures and thinking leaks otherwise remain eligible. The Pareto `!` is this condition-wide
flag. All original informative LLM rows are flagged and all saved call statuses are `ok`: the flag is not evidence
that the original calls were truncated. Record and explain token-count/cache semantics for every new model.

## Implement another backend

Changing `models` or `ollama_url` does not add another backend. The current CLI constructs `OllamaClient` directly,
`TaskConfig` has no backend selector, and there is no generic backend registry. A backend extension needs an
approved adapter/configuration and runner integration, with mocked transport tests before any paid or local run.
The current task runner calls these methods:

| Current method | Required responsibility |
|---|---|
| `version()` | Identify the runtime serving the calls |
| `model_digest(model)` | Identify the exact available model build |
| `model_capabilities(model)` | Discover supported behavior before timing calls |
| `chat(model, system, user, num_ctx, num_predict, temperature, seed, think, output_schema)` | Preserve the requested messages/settings/schema and normalize the response |
| `unload_model(model)` | Release resources after the model batch, including failure paths |

`chat()` returns [`ChatResult`](../src/traceformatbench/runners/ollama.py): content, thinking text, termination
reason, prompt and cached-prompt counts, generated-token count, and load/prompt-evaluation/generation/total timings
in nanoseconds. An added backend must define unavailable fields and unit conversions explicitly; do not invent
zero counts, digests, seed support or timings to satisfy this interface. The existing dataclass and analysis may
need a versioned extension to represent unavailable values. Probes additionally use
`count_prompt_tokens(model, text, num_ctx, seed) -> PromptTokenCount`; matching this protocol requires documented
tokenizer identity, chat-template treatment, cache semantics and overflow detection, not just a similarly named
token count.

Preserve exact ordered messages and hash them with `prompt_sha256()` (UTF-8 canonical JSON of system then user).
Reserve output space, reject over-context requests before generation using a valid backend-specific method, and
record any truncation or reasoning output. Do not copy Ollama's half-context heuristic or recount protocol without
evidence that it applies. Map provider termination reasons, transport failures and schema failures explicitly.
Currently transport errors stop the CLI without a successful call record; call `status` is `ok`, `truncated` or
`thinking_leak`, while the scorer separately supplies `parse_status` (`ok`, `invalid_json`, `schema_violation`).
`done_reason` is recorded but is not itself checked by `call_status()`. A new backend must specify retries,
partial-response handling and finish-reason mappings rather than silently treating every response as successful.

The complete current T1/T2 raw-record contract is defined by `build_call_list()` and `make_record()` in
[`runners/tasks.py`](../src/traceformatbench/runners/tasks.py), with an exact-field test in
[`tests/test_task_runner.py`](../tests/test_task_runner.py):

| Fields | Purpose |
|---|---|
| `case`, `dataset`, `system`, `fault`, `root_cause_service`, `task`, `window_index`, `informative`, `label` | Window/task identity and scoring metadata |
| `model`, `model_digest`, `ollama_version`, `format` | Model, runtime and representation provenance |
| `prompt_version`, `seed`, `num_ctx`, `num_predict`, `temperature`, `think_sent`, `prompt_sha256`, `prompt_chars` | Exact call setup and prompt identity |
| `prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `done_reason`, `total_s`, `load_s`, `content`, `status` | Original response, token accounting, termination and duration |
| `recount_num_ctx`, `recount_prompt_eval_count`, `status_before_recount` | Recount evidence, or null when absent |
| `candidates` | Additional field for T2 |
| `a_phase`, `time_aligned` | Additional fields for T1P |

Raw task records omit prompts, output schemas and raw traces; retain their reconstructible specification. They
also omit the separate thinking text and prompt-evaluation/generation timings held by `ChatResult`. A new backend
must add truthful backend-specific provenance instead of populating `ollama_version` with another provider's
version. Any schema extension needs corresponding config, scoring, analysis, verification and compatibility tests.

`record_key()` resumes by case, task, label, window index, model, format, prompt version, seed, context, output
limit, temperature and prompt hash. The CLI appends and flushes each completed record; it can discard an incomplete
last JSONL line on resume but rejects interior corruption. Define how backend identity and changed model builds
affect an extension's resume keys so a resumed run cannot silently mix incompatible calls. Test configuration
validation, request payloads, response normalization, errors, context overflow, cache counts, schema violations,
provenance and resume behavior with fake clients; the existing client and task-runner tests provide examples.

For an implementation, run its targeted tests followed by the repository gates:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
