# Pre-registration

Status: ready to commit. Every section was checked against specs 001-008, the code, the configs, the committed windows
and the committed results on 2026-09-30. Commit BEFORE the main grid and before the baseline is run on windows_v1.
After that commit, changes go only in the "Deviations" section. Known issues found during the check are listed in
section 14. LOG.md, STATUS.md, plans 03/06, the novelty check and `related_work.md` are in `../Progress and Plan/`,
outside this repository.

Date committed: the date of the commit tagged `prereg-v1` (a file cannot contain its own commit hash).
Timestamp procedure: (1) commit this file on a branch and squash-merge it into `main` through a pull request; (2) create
the annotated tag `prereg-v1` on the resulting `main` commit (the squash merge gives a new hash, so the tag is made
after the merge) and push the tag to https://github.com/LongHuynh-vn/traceformatbench; (3) register a frozen copy of
this file on OSF Registries (open-ended template), stating the commit hash, before the first main-grid or baseline call,
because the repository is private, `main` is not protected and a git date is set by the committer; (4) state the hash,
the tag and the OSF URL in the first commit that adds main-grid results.

## 1. Research question and positioning
Research question:

> When a small local LLM must name the root-cause service from a window of five distributed traces, does the way the
> window is written into the prompt change how often it ranks that service first? And does it matter whether a format
> only changes the layout of the spans or selects and aggregates them?

The question is tested on RCAEval with two models of similar size. Accuracy is reported next to prompt tokens.
Classifying a window as before or after the fault is a secondary task.

This question is narrower than the one in `README.md` and plan 03 ("... and how does this depend on model size and
token budget?"). The grid has two models of similar size and no token-budget sweep, so it cannot answer either part
(section 13). The question above is the registered one (section 14, K10).

Positioning:

> Earlier studies measure how the data format changes LLM accuracy and token counts for generic structured data (the
> TOON benchmarks; "Notation Matters"). TELLER measures how compressing GPU execution traces of LLM inference affects
> root-cause diagnosis, using a learned trace encoding and a trained model. GALA and GALA+ run LLM agent workflows for
> root-cause analysis on RCAEval, the dataset used here, with traces reduced to service-dependency context; neither
> compares alternative span serializations under a fixed prompt. This study uses one fixed zero-shot prompt and the
> same five-trace windows for every format. It compares a control of RCAEval span-row JSON (not raw OTLP) with two
> formats that keep every span but change the layout and three formats that select or aggregate spans using trace
> structure and timing, on two small local models, and reports each model's own prompt-token counts next to accuracy.
> A search on 2026-09-30 found no study of this comparison; this is a search result, not proof that none exists.

Rules for the report: no claim to be the first study of trace or telemetry representation for LLMs; input-token budgets
are not part of this study; fewer tokens are not equated with lower operational cost (novelty check, points 1-6).

References, checked on 2026-09-30 against arXiv and GitHub:
- RCAEval: L. Pham, H. Zhang, H. Ha, F. Salim, X. Zhang. WWW Companion '25, pp. 777-780, DOI 10.1145/3701716.3715290,
  arXiv:2412.17015 (v5).
- TELLER: R. Xu, J. Li, P. Chen, Z. Xie. "TELLER: Non-intrusive Cross-Layer Root-Cause Analysis for LLM Inference",
  arXiv:2608.01975, ASE 2026.
- GALA: Y. Tian, Y. Liu, Z. Chong, Z. Huang, H.-A. Jacobsen. "GALA: Can Graph-Augmented Large Language Model Agentic
  Workflows Elevate Root Cause Analysis?", arXiv:2508.12472 (2025).
- GALA+: Y. Tian, Y. Liu, Z. Chong, Z. Huang, Y. Li, H.-A. Jacobsen. "GALA: Graph-Augmented LLM Agents for Root Cause
  Analysis and Incident Response in Microservices", arXiv:2608.08968, ASE '26. This is a separate paper, not a "v2" of
  2508.12472 (section 14, K11).
- TOON: J. Schopplich, https://github.com/toon-format/toon (no peer-reviewed paper by the author). The IEEE Access TOON
  evaluation (DOI 10.1109/ACCESS.2026.3707409) listed in the novelty check could not be verified and is not cited
  (section 14, K12).
- Notation Matters: L. Kutschka, B. Geiger, arXiv:2605.29676 (2026).

## 2. Hypotheses
The content of the hypotheses was decided on 2026-09-30 (LOG.md). Unit for H1a and H1b: one (format, model) cell,
metric T2 Acc@1 on informative windows, paired difference against F0 of the same model, 95% case-bootstrap CI
(section 9).
- H1a (confirmatory): the evidence-reducing formats F3, F4 and F5 each reach a higher T2 Acc@1 than the control F0.
  - A cell is supported if the CI of (format minus F0) lies entirely above 0, not supported if the CI includes 0, and
    contradicted if the CI lies entirely below 0.
  - H1a as a whole is supported only if all 6 cells (F3, F4, F5 x 2 models) are supported. Because every one of the 6
    CIs must lie above 0, the chance of wrongly calling H1a supported is no larger than for one cell, so this claim
    needs no multiplicity correction.
  - If 1 to 5 cells are supported, H1a is reported as "partly supported: k of 6 cells" and the cells are named; if no
    cell is supported, as "not supported". Contradicted cells are always named. Patterns per format or per model are
    read from the same 6 cells and are descriptive.
- H1b (confirmatory): the re-encoding formats F1 and F2 (every span kept, layout changed) show no clear difference
  from F0.
  - A cell counts against H1b if the CI of (F1 or F2 minus F0) excludes 0, in either direction; the direction is
    reported. H1b as a whole holds if all 4 cells (F1, F2 x 2 models) show no clear difference; otherwise the cells
    with a clear difference are named.
  - A CI that includes 0 is "no clear difference", which is not evidence that the formats are equivalent. H1b is not
    tested as equivalence: no smallest effect size of interest is set, because no external threshold exists for this
    task, and with 54 cases a CI narrow enough to show equivalence within a few percentage points is unlikely. For
    every H1b cell the report gives both CI bounds: the largest drop and the largest gain relative to F0 that the data
    do not rule out.
  - F2 is not purely a change of syntax: it drops span IDs, absolute times, `methodName` and `statusCode` (section 5).
    Plan 03 H1 predicted that the span tree beats raw JSON; H1b now predicts no clear difference. That is a deliberate
    change.
- H2 (exploratory, no test): for each of F1-F5, the sign of (format minus F0) on the primary metric is the same for
  qwen3.5:9b and gemma4:12b. Read off `comparisons.csv`; no threshold, no cross-model statistic.
- H3 (descriptive only, no test): T2 Acc@1 versus mean prompt tokens per format, one panel per model (Pareto plot,
  section 9).
- Not tested here: plan 03 H2 (small open models are more sensitive to format than large API models; no API model is
  in this grid) and plan 03 H3 (token-budget threshold; the truncation sweep is future work). See section 13.

## 3. Data and sampling
- Dataset: RCAEval (Pham et al., "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry
  Data", WWW Companion 2025, arXiv:2412.17015), Hugging Face `phamquiluan/RCAEval`. License: MIT per the RCAEval
  GitHub README and the Hugging Face page; the Zenodo record (DOI 10.5281/zenodo.14590730) lists CC-BY-4.0.
  RCAEval is existing public data; the student did not collect it.
- Dataset revision: Hugging Face commit `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`, recorded in the local download
  metadata (`data/raw/rcaeval/.cache/huggingface/download/**/*.metadata`, git-ignored) for all 160 downloaded files,
  including `cases.parquet` and all 78 `traces.parquet`. The loader does not pin this revision (section 14, K6). The
  committed windows file lists the trace IDs of every window and is the source of truth.
- Subsets with traces: RE2-OB, RE2-TT, RE3-OB, RE3-TT (the RCAEval README: traces exist only for Online Boutique and
  Train Ticket); repetition 1 only (one case per fault x service): 78 cases (RE2-OB 30, RE2-TT 30, RE3-OB 10,
  RE3-TT 8). Used per case: the traces, and from the case index `cases.parquet` the injection time, the root-cause
  service, the fault and the dataset. The window builder downloads only `traces.parquet` and `inject_time.txt` per
  case, and no code reads metrics or logs. The early inspection script `scripts/inspect_case.py` downloaded full case
  folders: metrics for re2ob_checkoutservice_cpu_1 and re2tt_ts-auth-service_cpu_1, and logs for the first. These
  files are unused.
- Trace start = earliest span start (microseconds); `inject_time` is in Unix seconds.
- Phases: a trace is "pre" if it starts more than 60 s before `inject_time`, "post" if it starts at least 60 s after;
  traces inside the 60 s guard band are not used.
- Eligible traces: a root span, no orphan spans, at most 10,000 characters in F0 (the per-trace size cap).
- Windows: 5 consecutive eligible traces of one phase ("consecutive" among the eligible traces sorted by start time
  and trace ID; ineligible traces in between are skipped). Per case: 1 T1 pre window, 1 T1 post window (balanced),
  and 5 T2 post windows. Per-case RNG seeded with `"20260929:<case>"` (config `configs/windows_v1.yaml`).
- Overlap: the T2 windows of a case never share traces. A T1 post window is drawn separately and shares traces with a
  T2 window in 4 cases (different tasks and labels).
- T2 only for answerable cases: a case gets T2 windows only if its root-cause service appears in its traces
  (spec 003 amendment). Answerability is decided on all traces of the case, before the eligibility filter, so a case
  can be answerable although no eligible trace passes through the root cause. Excluded from T2: re3ob_adservice_f3_1,
  re3ob_adservice_f4_1, re3ob_adservice_f5_1, re3ob_cartservice_f1_1. Background: RCAEval traces cover only part of the
  services.
- Result, fixed in the committed `data/windows/windows_v1.jsonl`: 524 windows; T1 154 (77 pre, 77 post; no T1 for
  re2tt_ts-train-service_socket_1, which has only 4 eligible pre traces); T2 370 in 74 cases, of which 193 in 54 cases
  are informative (at least one trace of the window passes through the root-cause service).
- Informative T2 windows (cases) per dataset: RE2-OB 89 (24), RE2-TT 73 (21), RE3-OB 5 (1), RE3-TT 26 (8).
  Informative windows per case range from 1 to 5.
- Known effects of the size cap (from `results/window_stats/windows_v1.csv`): median share of eligible traces per
  case is 0.89 (RE2-OB), 0.54 (RE2-TT), 0.88 (RE3-OB) and 0.70 (RE3-TT). Of the 74 T2 cases, 20 have no informative
  window (RE2-TT 9, RE2-OB 6, RE3-OB 5, RE3-TT 0): re2ob_checkoutservice_{delay,loss,socket}_1,
  re2ob_emailservice_{cpu,loss,socket}_1, re2tt_ts-order-service_{delay,mem}_1, re2tt_ts-route-service_disk_1,
  re2tt_ts-train-service_{cpu,delay,disk,loss,mem,socket}_1, re3ob_emailservice_f{1,2,3,4,5}_1. With the 4 cases
  without T2, 24 of 78 cases add nothing to the primary metric. For the 6 ts-train-service cases, about 40% of all
  post traces pass through the root-cause service but 0% of the eligible ones: their root-cause traces are far above
  the cap (spec 003).

## 4. Tasks and labels
- Prompts: version v1 (`src/traceformatbench/tasks/prompts.py`), zero-shot, one system and one user message,
  identical across formats except for the serialized window; the prompt does not describe the format. System prompt:
  "You analyze distributed traces from a microservice system. Reply with one JSON object only."
- T1 (secondary): is the window from before or after the fault injection? Label = phase.
  Output `{"phase": "pre"|"post"}`. User message (verbatim, `{n}` = 5):
  ```
  Below are {n} traces recorded during one time window of a microservice system.

  <traces>
  {traces}
  </traces>

  At some point, a fault was injected into one service of this system. Were these traces recorded before or after the fault injection?
  Answer with {"phase": "pre"} if before, or {"phase": "post"} if after.
  ```
- T2 (primary): rank the three most likely root-cause services. Candidates = every service seen in any span of any trace
  of the case (all traces, including the 60 s guard band, before the eligibility filter), sorted A-Z; the same list for
  every window and format of a case. The list can name services that are absent from the window. Label = RCAEval
  root-cause service. Output `{"ranking": [s1, s2, s3]}`. User message (verbatim):
  ```
  Below are {n} traces recorded after a fault was injected into exactly one service of a microservice system.

  <traces>
  {traces}
  </traces>

  Candidate services: {candidates, comma-separated, A-Z}
  Which service is the root cause (the service where the fault was injected)?
  Answer with {"ranking": [first, second, third]}: the three most likely root-cause services from the candidate list, most likely first, no duplicates.
  ```
- Structured output: the task's JSON schema is sent in Ollama's `format` field with every call (`tasks/schemas.py`):
  T1 enum `pre`/`post`; T2 an array of exactly 3 items from the case's candidates; no extra keys.
- Scoring (spec 005, `tasks/scoring.py`): exactly one whole surrounding Markdown code fence is stripped, with no other
  repair. The answer must be a JSON object whose only key is `phase` (T1) or `ranking` (T2); T1 value `pre` or `post`;
  T2 value a list of 3 distinct strings, each exactly equal (case-sensitive) to a candidate. Anything else is
  `invalid_json` or `schema_violation`. A call whose status is not `ok` (truncated, thinking leak) or whose answer is
  invalid scores 0 on every metric and stays in the denominator. T1 correct = answer equals the label. T2: Acc@1 = first
  entry equals the label; Acc@3 = label among the 3 entries. Acc@3 is a top-3 hit, not RCAEval's Avg@k, so the numbers
  are not comparable with published RCAEval or GALA tables.
- History: in the smoke run each model gave one constant T1 answer (qwen always "pre", gemma always "post"). A paired
  T1 variant (T1P, spec 005b) was piloted and gave almost position-constant answers (window B in 8 of 8 gemma4:12b
  calls and 7 of 8 qwen3.5:9b calls), so it is not used; T2 became primary.
  Both runs are described in section 11.

## 5. Formats (serializers, `src/traceformatbench/serializers/`)
- F0 (CONTROL): RCAEval span-row JSON (not raw OTLP). All 11 RCAEval columns in the original order (`time`,
  `traceID`, `spanID`, `serviceName`, `methodName`, `operationName`, `parentSpanID`, `startTimeMillis`, `startTime`,
  `duration`, `statusCode`), nulls kept, one flat JSON array with default separators. The rows are rebuilt from the
  assembled spans (`time` and `startTimeMillis` re-derived; spec 002 found them identical to the shipped values on 2
  checked cases); traces in window order, spans in start-time order.
- F1: minified JSON, spans grouped by trace (`traceID` written once per trace); derivable fields `time` and
  `startTimeMillis` and null-valued keys dropped; absolute timestamps kept.
- F2: indented span tree per trace; each line: service, operation, duration, start offset from the trace start (ms).
  Drops span IDs (structure shown by indentation), absolute times, `methodName` and `statusCode`.
- F3: heaviest path per trace: from the root, always into the child with the longest duration (ties: later end, then
  first child in start order); hops with duration and self-time; other spans only counted. (The spec 006 draft used
  the latest-ending child; changed during review of spec 006, before the serializer was merged and before any model
  call used F3, because with sequential calls that is only the last call.)
- F4: per-service statistics over the window: span count, mean and max self-time; services sorted A-Z. No operation
  names, no trace structure.
- F5: the 5 spans with the highest self-time in the window (ties: earlier trace, then earlier span) plus all their
  ancestors, as a tree with start offsets.
- Self-time = span duration minus the union of its children's intervals clipped to the span, never below 0.
  Milliseconds with 2 decimals. F2-F5 start with one self-describing `#` line (F4 adds a column-header line); F0 and F1
  have none. In F2, F3 and F5 each trace starts with `Trace i (n spans)`; no trace or span IDs are printed.
- Evidence kept: F1 re-encodes the same spans; F2 re-encodes the spans minus IDs, absolute times, `methodName` and
  `statusCode` (`statusCode` is 0 or missing in the checked data, spec 006); F3-F5 select or aggregate spans.
  F3-F5 use no ground truth. F0 and F1 show absolute clock times; F2-F5 do not.
- Differences from plan 03 (all logged decisions): F0 is RCAEval span rows, not OTLP JSON; F4 has no error rate and no
  p50/p95 (status codes unusable); F5 ranks spans by self-time, not by a latency z-score.

## 6. Models and inference settings
- qwen3.5:9b, Ollama digest 6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7; architecture qwen35,
  9.7B parameters, quantization Q4_K_M, Apache-2.0.
- gemma4:12b, Ollama digest 4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c; architecture gemma4,
  11.9B parameters, quantization Q4_K_M, Apache-2.0.
- Both from `ollama show` on 2026-09-30. Both were pulled on 2026-09-29 (manifest file dates 05:11 and 05:15 local
  time). Both have a trained context of 262,144 tokens; the grid caps it at num_ctx 32768.
- Ollama 0.35.0 (runtime probe, 2026-09-30). The Ollama version and model digest are stored in every call record.
  The code does not enforce "no upgrade", so it is a procedure: Ollama is not upgraded and no model is re-pulled during
  the grid, and before each resume the student checks `ollama --version` and `ollama list` against this file. Before
  the analysis, every model must have one digest and one Ollama version across all its records (section 14, K3). (The
  smoke run, the T1P pilot, the first runtime probe and the token probe ran on Ollama 0.34.4; only the F0-F5 runtime
  probe ran on 0.35.0.)
- Request: one non-streamed `/api/chat` call (system + user message) with the task's JSON schema in `format` and
  `truncate: false`. Set explicitly: num_ctx 32768, num_predict 256, temperature 0, seed 20260930. The largest main-grid
  prompt is 43,165 characters (dry run), at most about 22,136 tokens by the preflight ratio below (plus 256 output
  tokens); the largest prompt measured in the F0-F5 runtime probe was 20,622 tokens, with no truncation.
- Thinking is disabled. Both models think by default, so `think: false` is sent explicitly and recorded as
  `think_sent` in every call. A call that still returns thinking text gets status `thinking_leak` and scores 0.
- Not overridden, so taken from each model's Ollama defaults: qwen3.5:9b top_k 20, top_p 0.95, presence_penalty 1.5;
  gemma4:12b top_k 64, top_p 0.95. At temperature 0, top_k and top_p have no effect. Whether presence_penalty still
  changes greedy output was not checked.
- One call per window x format x model, no repeats. Temperature 0 with a fixed seed does not guarantee identical
  output across hardware or Ollama versions, so the CIs cover case sampling only, not run-to-run variation.
- Truncation checks on every call: (a) before sending, the prompt length is estimated (characters / 1.95 plus
  num_predict) and the run stops if it could reach num_ctx; (b) `truncate: false` is sent; (c) a returned prompt token
  count at or above num_ctx sets status `truncated` (scored 0, left out of token means).
- The num_ctx/2 band: the code also sets `truncated` when the returned count is within 16 tokens of num_ctx // 2
  (16,368-16,400), the silent-truncation signature of older Ollama. A correct, untruncated prompt of that length
  would also be marked `truncated` and scored 0; a few F0 calls may fall in this band (estimate only). Rule: a call
  that is marked `truncated` only because of this band is re-counted by sending the same request once more with
  num_ctx 24576 (num_ctx x 3/4, the check used in the token probe, spec 004), where the band lies elsewhere. If the
  re-counted prompt tokens equal the first count, the call was not truncated and its original answer is scored
  normally; otherwise it stays `truncated`. The answer of the re-count request is never used. This check is added
  before the grid starts (section 14, K1).
- Cached prompt tokens (`prompt_eval_cached_count`) are recorded for every call. They do not stop or drop a call. The
  analysis sets `tokens_reliable = false` for a (model, format, task) condition if any of its calls was truncated or
  had cached prompt tokens.
- Order (fixed, not randomized): all qwen3.5:9b calls, then all gemma4:12b; within a model by case (sorted), task (T1
  then T2), window, format F0 to F5. No warm-up call. The first call of each model, and the first call after each
  resume, include the model load time (`total_s` includes `load_s`, which is also recorded).
- Config: `configs/main_grid.yaml` (spec 008): 6,288 calls (T1 1,848; T2 4,440) = (154 T1 + 370 T2 windows) x 6
  formats x 2 models. The call list was checked with `scripts/run_tasks.py configs/main_grid.yaml --dry-run` (no
  model call).
- Hardware: MacBook Pro M4 Pro, 24 GB unified memory; one model loaded at a time.

## 7. Baselines
- Non-LLM heuristic (spec 007, `analysis/baselines.py`): for each T2 window, score every candidate of the case by
  the total self-time of its spans in the 5 window traces (0 if absent; same self-time function as F4/F5) and rank by
  score. Ties get expected credit: a tied label earns the share of its tie group's slots that fall inside the top k.
  Deterministic, no LLM, no serialized text. Computed on windows_v1 only after this file is committed
  (`results/baselines/selftime_v1.jsonl`).
- Chance (T2): 1/n and 3/n per window (n = number of candidates), averaged over windows.
- T1 reference: chance 0.5 (77 pre / 77 post windows); a constant answer also scores 0.5. No T1 heuristic is run.
- Dropped from plan 03: the latency z-score and error-flag baselines (status codes unusable); replaced by the
  self-time baseline.

## 8. Metrics
- Primary: T2 Acc@1 on informative windows (193).
- Secondary: T2 Acc@3 (informative and all), T2 Acc@1 on all 370 T2 windows, T1 accuracy.
- Descriptive: invalid rate (share of calls whose answer is not `ok` after parsing), truncation rate, T1 post rate
  (valid "post" answers / all T1 calls), mean prompt tokens per window (`prompt_eval_count` of the whole request,
  including system prompt, instructions and candidates; each model's own tokenizer, compared only within a model;
  truncated calls excluded), accuracy per 1K tokens (T2 only: Acc@1 / (mean prompt tokens / 1000) over the same
  non-truncated windows; no CI), mean seconds per call (includes model load time, section 6), baseline tie rate.
  Also reported as a check: the number of thinking leaks per condition (`n_thinking_leak` in `metrics.csv` from
  `scripts/score_tasks.py`). A call that stops at num_predict (`done_reason` "length", recorded per call) almost
  surely returns incomplete JSON, so it is inside the invalid rate; it is not counted separately (section 14, K4).
- Cost: mean prompt tokens and mean seconds per call (`summary.csv`). Output tokens (`eval_count`) are recorded per
  call but not summarized. Money cost is zero (local models only).
- Change on 2026-09-30 before any main-grid run: primary set changed from all T2 windows to informative windows,
  because in non-informative windows no format contains the fault and those windows only dilute format differences.

## 9. Analysis plan (spec 007, `scripts/analyze.py configs/analysis_v1.yaml`)
- Pipeline: `scripts/run_tasks.py configs/main_grid.yaml` (calls.jsonl) -> `scripts/score_tasks.py
  results/main_grid/calls.jsonl` (scores.jsonl, metrics.csv) -> `scripts/run_baseline.py configs/windows_v1.yaml`
  -> `scripts/analyze.py configs/analysis_v1.yaml` (summary.csv, comparisons.csv, strata.csv, Pareto plot).
- The analysis is run only on the complete grid (section 12). The code refuses duplicate window keys, a window key
  present in one (model, format, task) condition but missing in another, and a format whose windows differ from F0 of
  the same model. It does not check the records against `configs/main_grid.yaml`: a missing model, a missing non-F0
  format, or a T1 window missing everywhere would not be caught. So before the analysis `calls.jsonl` must hold
  exactly 6,288 records (T1 1,848, T2 4,440), one per planned call (section 14, K2).
- Point estimate: pooled mean over windows, so a case weighs by its number of windows in the set. CIs: case bootstrap
  (unit = case; a drawn case brings all its windows of the set) stratified by dataset; a stratum with fewer than 5
  cases is merged with the other dataset of the same system. For the primary set this merges RE3-OB (1 case) into
  RE2-OB, giving strata RE2-OB + RE3-OB (25 cases), RE2-TT (21), RE3-TT (8). 10,000 resamples, 95% percentile
  interval, seed 20260930 (one derived seed per analysis set); the same resamples are shared by all conditions of a
  set, so comparisons are paired.
- Confirmatory family: F1-F5 vs F0, per model, on T2 Acc@1 / informative (10 comparisons). "Clear difference" if and
  only if the 95% CI of the difference excludes 0; otherwise "no clear difference". Decision rules per hypothesis:
  section 2. No multiplicity correction: if no format differs from F0, about 0.5 false clear differences are expected
  among the 10 (5% each), and the chance of at least one is about 40% if the comparisons were independent. This is
  reported as a limitation. The whole-hypothesis rule for H1a (all 6 cells) needs no correction (section 2). No Holm or
  other adjustment is applied: the code gives CIs, not p-values, and a stricter level would only make "no clear
  difference" for H1b more likely.
- Secondary comparisons (exploratory, same paired bootstrap): F1-F5 vs F0 per model for T2 Acc@3 (informative and
  all), T2 Acc@1 on all windows, and T1 accuracy (40 comparisons); every model x format condition, including F0, vs
  the self-time baseline on T2 Acc@1 (informative and all; 24 comparisons; the baseline uses expected credit for ties).
  No comparison between models. All comparison rows are reported.
- H2: sign of each format-vs-F0 estimate, compared across models by reading `comparisons.csv`. H3: Pareto plot per
  model; a format is dominated only if another format of the same model has strictly fewer mean prompt tokens and
  strictly higher Acc@1.
- Planned robustness views, reported next to the primary result and never instead of it: the primary metric on all
  370 T2 windows, Acc@3, and accuracy per system (OB and TT, from `metrics.csv`, no CIs). If a robustness view
  disagrees with the primary result, both are reported. Breakdowns by dataset or fault type are exploratory.
- Sensitivity: no minimum detectable effect is given. A power calculation would rest on assumptions (F0 accuracy, how
  often two formats disagree on a window, and how strongly the windows of one case are correlated) that cannot be
  checked without results. Instead, the report gives the half-width of every 95% CI as the achieved precision; a
  difference smaller than it cannot be told apart from 0 in this grid. The effective sample size is about 54 cases,
  not 193 windows.
- Interpretation note: F4/F5 hand the model self-time numbers, so F4/F5 vs the baseline mainly tests reading a table.

## 10. Budget
- Local only, no paid API. Estimated main-grid time ~22 h (qwen ~7.8 h, gemma ~14.2 h), from the F0-F5 runtime probe
  of 2026-09-30 (`results/runtime_probe_f0_f5/grid_estimate_windows_v1.csv`, method in `NOTES.md` there; the script
  that wrote the CSV is not committed, section 14, K8). This replaces the older 39.7 h estimate in spec 004, which
  timed F2-F5 as F1.
- Compute cap 48 h. The run is resumable. If 48 h pass before the grid is complete, the run continues; no partial grid
  is analyzed and the grid is not reduced without a Deviations entry.

## 11. Data access and prior knowledge
- RCAEval is public data collected by others. Before this registration the student:
  - (a) read the RCAEval paper, README and GitHub issues (notes in `related_work.md`);
  - (b) downloaded the traces of all 78 cases and built windows_v1, seeing window counts, size-cap statistics and
    informative-window counts (section 3), and inspected 2 full case folders with `scripts/inspect_case.py`
    (section 3);
  - (c) ran a scored smoke run (`results/tasks_smoke/`, Ollama 0.34.4) on 2 cases, re2ob_checkoutservice_cpu_1 and
    re2tt_ts-auth-service_cpu_1, with F0 and F1, T1 and T2, both models;
  - (d) ran a scored T1P pilot (`results/t1p_pilot/`, Ollama 0.34.4, num_ctx 65536) on the same 2 cases;
  - (e) ran two runtime probes with a placeholder prompt asking for a phase and a service, whose answers are stored
    but were never parsed or scored. The first is `results/runtime_probe/` (Ollama 0.34.4, F0/F1, the 2 smoke-run
    cases, older window design). The second is `results/runtime_probe_f0_f5/` (Ollama 0.35.0, F0-F5, the 23 windows_v1
    windows of re2ob_currencyservice_cpu_1, re2tt_ts-auth-service_disk_1, re3tt_ts-auth-service_f4_1 and
    re3ob_adservice_f5_1);
  - (f) ran a token probe (`results/token_probe/`, Ollama 0.34.4, F0/F1, the 2 smoke-run cases) that sent serialized
    text in raw mode with num_predict 1, so it has no answers.
- Scored model accuracy has been computed only for the 2 smoke-run cases and formats F0/F1. The placeholder answers of
  the F0-F5 runtime probe cover 4 main-grid cases; no script scored them and no accuracy was computed from them. No
  self-time baseline result on windows_v1 exists.
- The smoke run and the T1P pilot used the older window design (2 T1 windows per phase and 3 T2 windows per case).
  None of the 28 smoke-run prompts equals a main-grid prompt. Only one smoke window has the same traces as a windows_v1
  window (re2ob_checkoutservice_cpu_1: smoke T1 post window 0 = windows_v1 T2 window 0), and that window is not
  informative, so it is not in the primary set.
- Design changes made after (c) and (d): T2 became primary, T1P was dropped, the windows became 1 T1 per phase and 5 T2
  per case, and the primary set became informative windows. The F3 change (section 5) came before any F3 model call.
- All 6 cases touched by the smoke run, the pilot and the probes stay in the main grid, because excluding them now
  would be a choice made after seeing their outputs. The 2 smoke-run cases contribute 6 of the 193 primary windows. No
  analysis without these cases is planned.

## 12. Completion, failed calls and exclusions
- The grid is complete when every window x format x model call in `configs/main_grid.yaml` (6,288) has exactly one
  recorded call. No result is looked at before then; there is no interim analysis and no early stopping.
- A call that returns is recorded once and never re-sampled: truncated calls, thinking leaks, invalid JSON and schema
  violations are kept and score 0 (section 4). Nothing is re-run with another seed, prompt or setting because a
  result looks bad. Two exceptions: an incomplete last line from an interrupted write is dropped on resume and that
  call is sent again; and the re-count check for the num_ctx/2 band (section 6) sends a second request whose answer
  is never used.
- Before the analysis: `calls.jsonl` holds exactly 6,288 records, one per planned call (section 9), and every model
  has one digest and one Ollama version across all its records (section 6). If either check fails, the analysis does
  not start and the failure is recorded under Deviations.
- A call that fails at transport level (connection or HTTP error, 600 s client timeout) produces no record and stops
  the run; on resume the same call is sent again with the same prompt, seed and settings. There is no automatic retry.
  The 600 s timeout is a client default in `runners/ollama.py`, not a config value.
- Resume matches calls by case, task, label, window, model, format, prompt version, prompt hash, seed and inference
  settings. The key does not include the model digest or Ollama version, so the no-upgrade procedure and the
  pre-analysis check in section 6 protect a resumed run (section 14, K3).
- The only exclusions are those fixed in section 3: the 60 s guard band, the eligibility rules (a root span, no orphan
  spans, the per-trace size cap), the T1 balance rule (T1 windows only if both phases yield one), the answerability
  rule for T2 and the informative-window rule for the primary set. No case, window or model output is excluded after
  results are seen, and latency is never used to exclude a call.

## 13. Scope and known limitations
- Not in this pre-registration (future or exploratory work; anything run later is labeled exploratory): the
  token-budget truncation sweep, API or larger models, attribute ablation, thinking mode, format hints in the prompt,
  TOON or Markdown-KV formats, an OTLP-shaped F0, and the error analysis of plan 03 (not pre-registered here).
- Limitations known before the run:
  - Small sample: 193 informative windows in 54 cases; RE3-OB adds 1 case to the primary set.
  - Selection by the size cap: the cap keeps shorter traces and removes every pre- and post-phase trace through the
    root cause in the 6 ts-train-service cases.
  - Partial trace coverage: the RCAEval traces cover only some services; 4 cases have no T2 and 20 have no informative
    window.
  - One prompt (v1), zero-shot, no format hint, so prompt sensitivity is not measured.
  - One call per cell, so run-to-run variation is not measured.
  - Possible training-data contamination: RCAEval has been public at least since December 2024 (arXiv:2412.17015).
  - Qwen's default presence_penalty 1.5 is not overridden; whether it changes greedy output was not checked.
  - Two models of similar size from two families only.
  - No multiplicity correction (section 9).
  - Constant T1 answers were seen in the smoke run.
  - F0 and F1 show absolute clock times and F2-F5 do not; this matters for T1 and is confounded with the format.

## 14. Known issues found before registration
Found while checking this file against the code on 2026-09-30. Fixing code or documents below changes no rule in this
file. K1 must be fixed before the grid starts; K2 and K3 are checks before the analysis; the rest do not affect
scored results.
- K1 (code, before the grid): `call_status` (`analysis/runtime_probe.py`, used by `runners/tasks.py`) marks a call
  `truncated` when its prompt token count is within 16 of num_ctx // 2, even if nothing was truncated. The re-count
  rule in section 6 is added before the first main-grid call.
- K2 (code, before the analysis): `analysis/report.py` does not compare the records with `configs/main_grid.yaml`, so a
  missing model, a missing non-F0 format or a T1 window missing everywhere passes. Covered by the 6,288-record check
  in section 12.
- K3 (code, before the analysis): the resume key (`record_key` in `runners/tasks.py`) leaves out `model_digest` and
  `ollama_version`, the resume never compares them with the current ones, and `report.py` groups records by model
  tag. After a re-pull or an Ollama upgrade, a resumed run would mix builds without warning. Covered by the
  procedure and the one-digest-one-version check in sections 6 and 12.
- K4 (code): `scripts/analyze.py` outputs no thinking-leak count, no count of `done_reason` "length" and no output
  tokens. The thinking-leak count comes from `metrics.csv` (`scripts/score_tasks.py`); the other two are recorded per
  call but not summarized (section 8).
- K5 (code): the 600 s request timeout is a default in `runners/ollama.py`, not a config value (AGENTS.md rule 4), and
  `scripts/run_tasks.py` reports a timeout as "Cannot reach Ollama", which is misleading.
- K6 (code): `data/rcaeval.py` downloads from the Hugging Face main branch without `revision=`. The revision actually
  used is recorded in section 3.
- K7 (code and spec): `analysis/report.py` prints "about one false clear difference" for 10 comparisons, and spec 007
  (line 152) says the same; the correct expected number is 0.5 (section 9).
- K8 (code): `results/runtime_probe_f0_f5/grid_estimate_windows_v1.csv` (the ~22 h estimate) was written by a script
  that is not committed. It is used only for the budget, not for any result.
- K9 (spec): spec 004 still records the superseded window decision (2 T1 windows per phase, 3 T2 windows, 6,552
  calls) and the 39.7 h estimate; `configs/windows_v1.yaml` and section 10 hold the current values.
- K10 (README): `README.md` lines 5-6 and 9-10 promise results across model sizes and token budgets and name anomaly
  detection. This grid tests neither size nor budget, and T1 is before/after classification of a window. The
  research question in section 1 is the registered one; the README is aligned with it after this commit.
- K11 (research notes): `related_work.md` lines 416 and 666 call arXiv 2608.08968 "GALA v2". It is a separate paper
  (GALA+, ASE '26); section 1 cites it correctly.
- K12 (research notes): the IEEE Access TOON evaluation (DOI 10.1109/ACCESS.2026.3707409) in the novelty check could
  not be found at its source; it is not cited.

## Deviations
Deviation policy: after the commit named at the top, any change to a rule in this file is added here with the date,
what changed, why, and whether any model result had been seen at that time. Results that depend on a deviation are
labeled as such in the report. Code fixes that change no scored output are listed but not counted as deviations.

None yet.
