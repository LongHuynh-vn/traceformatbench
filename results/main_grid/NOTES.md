# Main grid (2026-09-30 to 2026-10-02): registered results

Sections 1-6 are facts copied from the committed outputs and checked against the files. Section 7 is the interpretation.

Pre-registration: `PREREGISTRATION.md` at tag `prereg-v1` (commit d24b1067c97bdbafe9342736983108431a33e1dc),
registered on OSF before the first main-grid call: https://osf.io/brha2

## 1. Run
- Config `configs/main_grid.yaml`, branch `exp/main-grid`, code at 1820449 (PR #16). Started 2026-09-30 ~22:40 (Linz).
- 6,288 / 6,288 calls (per model: T1 924, T2 2,220), 6,288 unique keys, every status `ok`, every done_reason `stop`.
- qwen3.5:9b: digest 6488c96fa5fa..., 7.80 h model time, 2 truncation-band re-counts (both cleared).
  gemma4:12b: digest 4eb23ef187e2..., 12.88 h model time, 1 re-count (cleared). Ollama 0.35.0 for both, num_ctx 32768.
- Stopped with Ctrl+C and resumed twice (at 4,014 and 6,030 calls); no duplicate records.
- `scripts/check_grid.py configs/main_grid.yaml results/main_grid/calls.jsonl`: all PASS.
- Pipeline (section 12): `score_tasks.py` -> `run_baseline.py configs/windows_v1.yaml` -> `analyze.py
  configs/analysis_v1.yaml`. Outputs: `results/main_grid/{calls,scores}.jsonl`, `metrics.csv`,
  `results/baselines/selftime_v1.jsonl`, `results/analysis_v1/{summary,comparisons,strata}.csv`,
  `pareto_t2_informative.png`. `calls.jsonl` holds no prompts (only hashes and lengths).
- Strata: in the informative set RE3-OB has 1 case and was merged into RE2-OB by the registered merge rule
  (`strata.csv`).

## 2. Primary result: T2 Acc@1, informative windows (193 windows, 54 cases)
Stratified case bootstrap, 10,000 resamples, 95% percentile CI.

| | qwen3.5:9b | gemma4:12b |
|---|---|---|
| F0 | 0.487 [0.360, 0.604] | 0.503 [0.380, 0.615] |
| F1 | 0.446 [0.330, 0.560] | 0.528 [0.416, 0.634] |
| F2 | 0.456 [0.348, 0.564] | 0.601 [0.485, 0.707] |
| F3 | 0.440 [0.325, 0.554] | 0.534 [0.420, 0.640] |
| F4 | 0.404 [0.282, 0.522] | 0.461 [0.330, 0.584] |
| F5 | 0.518 [0.412, 0.624] | 0.591 [0.463, 0.709] |
| Self-time baseline | 0.290 [0.175, 0.404] | (same) |
| Chance | 0.089 | (same) |

Main comparisons (10), paired, difference vs F0 [95% CI], CI half-width in brackets after it:

| Format | qwen3.5:9b | gemma4:12b |
|---|---|---|
| F1 | -0.041 [-0.108, 0.027] (0.068) | +0.026 [-0.016, 0.076] (0.046) |
| F2 | -0.031 [-0.119, 0.060] (0.090) | **+0.098 [0.021, 0.185] (0.082), clear difference** |
| F3 | -0.047 [-0.153, 0.065] (0.109) | +0.031 [-0.046, 0.112] (0.079) |
| F4 | -0.083 [-0.192, 0.030] (0.111) | -0.041 [-0.145, 0.064] (0.104) |
| F5 | +0.031 [-0.094, 0.161] (0.128) | +0.088 [-0.005, 0.193] (0.099) |

- 1 clear difference out of 10. With no correction, about 0.5 false clear differences are expected among 10 if no
  format differs from F0 (PREREGISTRATION section 9).
- Independent check: Acc@1 per condition recomputed from `scores.jsonl`, and an own stratified case bootstrap
  (4,000 resamples) gave gemma F2-F0 +0.098 [0.025, 0.183], gemma F5-F0 +0.088 [-0.005, 0.191], qwen F4-F0 -0.083
  [-0.193, 0.028]; consistent with `analyze.py`.

## 3. Hypotheses: what the registered rules give
- H1a (F3, F4, F5 each clearly better than F0, both models; 6 cells): 0 of 6 cells is a clear improvement.
- H1b (F1, F2 no clear difference from F0; 4 cells): 3 of 4 cells show no clear difference; gemma F2 is a clear
  improvement.
- H2 (exploratory, sign agreement across models): signs agree for 2 of 5 formats (F4 both negative, F5 both positive);
  F1, F2, F3 have opposite signs (qwen negative, gemma positive).
- H3 (descriptive, cost): see section 5.

## 4. Secondary and robustness views (registered, reported next to the primary)
- Every LLM condition, F0 included, is clearly better than the self-time baseline on T2 Acc@1, on informative and on
  all windows (24 of 24 comparisons).
- All 370 T2 windows (74 cases), Acc@1 vs F0: gemma F2 +0.051 [0.011, 0.097] clear; the other 9 not clear.
- Acc@3 vs F0, informative: gemma F2 +0.057 [0.015, 0.102] clear; the other 9 not clear.
  Acc@3 vs F0, all windows: gemma F2 +0.027 [0.003, 0.054] clear; qwen F2 -0.076 [-0.127, -0.024] and qwen F4 -0.059
  [-0.103, -0.014] clearly worse; the other 7 not clear.
- Per system, T2 Acc@1 on informative windows (`metrics.csv`, no CIs; OB 94 windows, TT 99 windows):

| | qwen OB | qwen TT | gemma OB | gemma TT |
|---|---|---|---|---|
| F0 | 0.447 | 0.525 | 0.468 | 0.535 |
| F1 | 0.468 | 0.424 | 0.489 | 0.566 |
| F2 | 0.521 | 0.394 | 0.628 | 0.576 |
| F3 | 0.468 | 0.414 | 0.511 | 0.556 |
| F4 | 0.223 | 0.576 | 0.330 | 0.586 |
| F5 | 0.564 | 0.475 | 0.628 | 0.556 |

  F4 is the format most dependent on the system for both models (OB well below F0, TT above F0).
- Invalid T2 answers (all 370 windows): qwen F0 0%, F1 6.2%, F2 8.4%, F3 7.3%, F4 0%, F5 13.0% (mostly TT: up to
  22.1% for F5); gemma 0% except F3 0.3% and F5 1.4%. Registered rule: invalid = wrong.
- T1 (secondary): accuracy 0.474-0.539 for every condition, no clear difference vs F0. qwen answers "pre" for every
  window in F0, F1, F3, F4, F5 (accuracy exactly 0.500); gemma answers "post" for 57-94% of windows depending on format.

## 5. Cost (H3, T2 informative, mean per call)
- Prompt tokens: F0 9,390 (qwen) / 9,631 (gemma); F4 331 / 369; F5 552 / 588; i.e. F4/F5 use 16-28x fewer prompt
  tokens than F0.
- Time: F0 27.7 s (qwen) / 45.9 s (gemma); F4 1.8 s / 3.4 s; F5 2.3 s / 3.5 s; i.e. 12-15x faster than F0.
- `tokens_reliable` is False in every row because some calls had small prefix-cache hits (at most 53 tokens, mostly
  the shared system prompt). Evidence that `prompt_eval_count` still counts the whole prompt: a band re-count after a
  model reload (no cache) gave 16,396, equal to the original count of 16,396 that had 21 cached tokens. The token means
  are therefore reported, with this note.
- Pareto plot: `results/analysis_v1/pareto_t2_informative.png`.

## 6. Exploratory (NOT registered; labeled as such wherever used)
- Every qwen invalid T2 answer on informative windows in F1, F2 and F3 (13, 19 and 16) lists one service three times,
  and that service is the correct one. In F5, 26 of the 27 invalid answers are such triples and 24 of 27 have the
  correct service first. The registered rule scores all of these 0, so the primary result above is unchanged.
- If the first item of such an answer were scored (not the registered rule), qwen Acc@1 on informative windows would be
  F1 0.513, F2 0.554, F3 0.523, F5 0.642 vs F0 0.487. Gemma has 4 invalid answers on informative windows (all F5),
  3 of them triples, 2 with the correct service first (first-item Acc@1 0.601 vs registered 0.591).

## 7. Interpretation

This section follows the [registered hypothesis rules](../../PREREGISTRATION.md#2-hypotheses) and the committed
[paired comparisons](../analysis_v1/comparisons.csv). Explanations of the observed patterns below are tentative.

### 7.1. Hypothesis verdicts

**H1a is not supported under the registered rule.** None of the six F3/F4/F5-by-model comparisons shows a clear
improvement over F0, and none is contradicted by a CI entirely below zero. This leaves the predicted consistent
advantage of evidence reduction unestablished. It does not show that these formats never help: the intervals admit
both gains and losses, and F5 has a positive point estimate for both models.

**H1b does not hold as a whole under its registered rule.** Three of its four cells show no clear difference, but
gemma F2 improves over F0 by 9.8 percentage points, with a 95% CI from 2.1 to 18.5 points. This is the one cell that
counts against H1b. The other three intervals include zero and retain the uncertainty shown in section 2; they do
not establish equivalence. Equivalence requires a specified range of differences considered negligible and an
appropriate analysis, neither of which was registered here. See
[Lakens (2017), *Equivalence Tests: A Practical Primer*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/).

The exploratory H2 sign comparison also gives a mixed picture: only F4 and F5 have matching signs across models.
This grid therefore does not establish a format advantage that transfers consistently across the two models.

### 7.2. How much weight to give the gemma F2 result

Gemma F2 meets the stated criterion for a clear difference, but it is the only such result among ten unadjusted
primary comparisons. Individual 95% intervals do not guarantee 95% simultaneous coverage for the full family; this
distinction is explained in the [NIST discussion of multiple-comparison intervals](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).

The registration's expectation of about 0.5 false clear differences applies under an all-null scenario at the nominal
error rate. It is not a 50% probability that this particular finding is false. Likewise, the approximately 40% chance
of at least one false clear difference assumes independent comparisons. These comparisons reuse cases and F0
answers, so that probability is an illustration, not an estimated error probability for this grid.

The positive gemma F2 comparisons on all windows and on Acc@3 are consistent with the primary finding. They are
registered exploratory views of overlapping observations, however, rather than independent replications. The
defensible claim is a positive F2-vs-F0 result for gemma in this setup, with the multiplicity caveat. It does not
establish that trees are universally superior, and F2 also removes fields, so it does not isolate indentation as the
cause of the difference.

### 7.3. What the F4 OB/TT split might mean

F4 is below F0 on Online Boutique and above it on Train Ticket for both models (section 4). Pooling the systems
therefore hides an important descriptive pattern. The per-system estimates have no CIs or interaction test, so the
observed reversal does not establish a reliable difference in the format effect between systems.

F4 supplies service-level counts and mean/max self-time while discarding individual operation names, parent-child
structure and the arrangement of spans within traces
([format specification](../../specs/006-f2-f5-serializers.md)). A possible explanation is that the retained timing
summary is more useful for the sampled TT cases, while information lost during aggregation matters more for the
sampled OB cases. This is an untested explanation: differences in faults, service coverage and sampled windows
could also contribute. The current results do not identify which discarded information caused any error.

Both models produce valid F4 rankings throughout the grid, so invalid-answer penalties do not explain this split.
The practical implication is to inspect system-specific accuracy alongside token savings before treating F4 as a
general replacement for span-level input.

### 7.4. Duplicate rankings: answer validity and label agreement

The qwen pattern shows that strict ranking validity contributes to the recorded errors. On informative windows,
all invalid F1/F2/F3 answers repeat the correct service three times; most invalid F5 answers are also triples
(section 6). All invalid F1/F2/F3 answers match the label in their first position, as do 24 of the 27 invalid F5
answers. They nevertheless fail the requested three-distinct-services answer contract. Registered Acc@1 therefore
measures first-rank correctness together with whole-answer validity.

There is a concrete distinction between the supplied schema and the scoring rule. The
[T2 schema](../../src/traceformatbench/tasks/schemas.py) specifies three candidate strings but omits `uniqueItems`.
Uniqueness is required by the [prompt](../../src/traceformatbench/tasks/prompts.py) and checked by the
[parser](../../src/traceformatbench/tasks/scoring.py). JSON Schema treats array uniqueness as a separate constraint
([official documentation](https://json-schema.org/understanding-json-schema/reference/array#uniqueness)). A repeated
candidate can consequently satisfy the transmitted schema and still receive `schema_violation` from the task parser.
The pattern alone is not evidence that the runtime violated the schema it received.

The first-item rescoring in section 6 is useful as an exploratory diagnostic of this distinction. It cannot replace
the registered scores or change the H1 verdicts. Matching the label in an invalid answer also does not prove sound
causal reasoning: these outputs do not reveal why the service was selected or why it was repeated. The evidence
supports a contribution from answer-contract failures, without establishing a reasoning-versus-decoding mechanism.

### 7.5. Token savings and the baseline

F4/F5 use substantially fewer prompt tokens and have shorter measured calls than F0 (section 5). That is a useful
descriptive result even though H1a is not supported. However, a CI that includes zero does not establish preserved
accuracy, so these reductions cannot be described as free compression or equal accuracy at lower cost.

Every model/format condition beats the registered self-time heuristic on T2 Acc@1 under the reported comparison
rule. This supports a benefit over that baseline in this dataset, without establishing general causal reasoning. F4/F5 already
expose engineered self-time information to the model, which narrows what their baseline comparison demonstrates.
The token counts remain subject to the cache-reliability qualification in section 5, and the timings describe this
local runtime and hardware. There is no token-budget sweep or measurement of deployment-wide monetary cost.

### 7.6. Limits on the conclusions

These limits follow the [registered scope](../../PREREGISTRATION.md#13-scope-and-known-limitations), sampling rules
and observed answer patterns.

- **Sample size and dependence.** The primary evidence comes from 54 cases, not 193 independent windows or thousands
  of independent model calls. Case resampling accounts for windows belonging to the same case, but the small case
  count limits precision. Cases with more informative windows contribute more to the pooled estimate; RE3-OB
  contributes only one primary case and is merged with RE2-OB for resampling.
- **Selection and coverage.** The primary set requires the root-cause service to appear in a window. Partial trace
  coverage and the F0 character cap remove cases or traces from this set, including every root-cause trace in the six
  ts-train-service cases. Results concern this selected subset of the
  [RCAEval benchmark](https://arxiv.org/abs/2412.17015), not arbitrary production incidents. The all-window analysis
  broadens the view within the sampled grid but does not undo its sampling restrictions.
- **Representation changes several things.** F2 removes fields as well as changing layout; F3-F5 select or aggregate
  evidence. F0/F1 retain absolute times while F2-F5 do not. These comparisons evaluate complete representations and
  cannot attribute their differences solely to syntax or token count. F0 is RCAEval span-row JSON, not raw OTLP.
- **Model and prompt scope.** There are two similarly sized local models, one zero-shot prompt per task and one
  scored answer per combination of window, format and model. Prompt sensitivity, run-to-run variation, larger/API
  models and inference with thinking enabled were not measured.
  Qwen's default presence penalty was not independently assessed, and possible benchmark contamination remains
  unresolved. The observed model patterns are descriptive, without a formal cross-model comparison.
- **Task scope.** T2 is service ranking among a supplied candidate list after a known injected fault. Its registered
  score includes answer validity and does not establish a model's reasoning process. T1's near-chance accuracy and
  strong answer biases give little evidence of useful before/after discrimination with this prompt. This does not
  establish that fault detection from traces is impossible. The study's Acc@3 is also not RCAEval's Avg@k.
- **Uncertainty and follow-up.** The unadjusted comparison family, broad intervals and descriptive subgroup results
  limit strong recommendations. Any follow-up that changes scoring, prompts, schemas or sampling needs a new
  student-written spec and must be identified separately from these registered results.
