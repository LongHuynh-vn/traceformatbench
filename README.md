# TraceFormatBench

When a small local LLM has to find the faulty service in a set of distributed traces, does the way the traces are
written into the prompt matter? TraceFormatBench compares six serializations of the same trace windows, from raw span
JSON to compact per-service summaries, and reports accuracy next to prompt tokens.

The main experiment was registered before the first main-grid call: [`PREREGISTRATION.md`](PREREGISTRATION.md) at tag
`prereg-v1`, with a frozen copy on OSF (https://osf.io/brha2). Registered results and exploratory follow-ups are kept
apart below.

## Research question
> When a small local LLM must name the root-cause service from a window of five distributed traces, does the way the
> window is written into the prompt change how often it ranks that service first? And does it matter whether a format
> only changes the layout of the spans or selects and aggregates them?

(Quoted from [`PREREGISTRATION.md`](PREREGISTRATION.md), section 1.)

## Data
[RCAEval](https://github.com/phamquiluan/RCAEval) (Pham et al., WWW Companion 2025, arXiv:2412.17015), MIT License,
the RE2 and RE3 suites for the two systems that include traces:

- **OB**: Online Boutique.
- **TT**: Train Ticket.

78 fault cases (one repetition per fault and service) are cut into fixed windows of five consecutive traces. The
windows are committed in [`data/windows/windows_v1.jsonl`](data/windows/windows_v1.jsonl) (524 windows: 154 for T1,
370 for T2). The primary analysis uses the 193 T2 windows from 54 cases in which at least one trace passes through the
root-cause service ("informative" windows). Raw data is downloaded on demand and never committed.
Sampling rules: [`PREREGISTRATION.md`](PREREGISTRATION.md), section 3, and [spec 003](specs/003-window-sampler.md).

## Tasks
- **T2 (primary)**: given a window recorded after a fault was injected and the list of the case's services, rank the
  three most likely root-cause services. Metric: Acc@1 (first entry is the injected service). An invalid answer
  scores 0.
- **T1 (secondary)**: was the window recorded before or after the fault injection?

One fixed zero-shot prompt per task, identical across formats except for the serialized window
([spec 005](specs/005-task-prompts-scoring.md)).

## Formats
| ID | Format | What changes |
|---|---|---|
| F0 | RCAEval span-row JSON (control; not raw OTLP) | nothing: all 11 columns, nulls kept |
| F1 | minified JSON grouped by trace | drops derivable and null fields; every span kept |
| F2 | indented span tree | every span kept as service, operation, duration, start offset; IDs and absolute times dropped |
| F3 | heaviest path per trace | from the root, always into the longest child; other spans only counted |
| F4 | per-service aggregates | span count, mean and max self-time per service; no operations, no structure |
| F5 | top-5 spans by self-time | the 5 spans with the highest self-time plus their ancestors, as a tree |

F1 and F2 change the layout; F3 to F5 select or aggregate spans. Definitions: [spec 002](specs/002-f0-f1-serializers.md),
[spec 006](specs/006-f2-f5-serializers.md). To print all six formats for one window:
`uv run python scripts/show_formats.py <case_dir>`.

## Models and setup
- `qwen3.5:9b` and `gemma4:12b` (both Q4_K_M) through Ollama 0.35.0, run locally on a MacBook Pro M4 Pro (24 GB).
- Temperature 0, seed 20260930, `num_ctx` 32768, thinking disabled, JSON schema sent as structured output.
- One call per window, format and model: 6,288 calls in total, all with status `ok`.
- No paid API.

Details: [`PREREGISTRATION.md`](PREREGISTRATION.md), section 6, and [`configs/main_grid.yaml`](configs/main_grid.yaml).

## Main results (pre-registered)
T2 Acc@1 on the 193 informative windows (54 cases). 95% CIs come from a case bootstrap stratified by dataset, 10,000
resamples. "Difference vs F0" is paired, from the same resamples. Source:
[`results/analysis_v1/summary.csv`](results/analysis_v1/summary.csv) and
[`results/analysis_v1/comparisons.csv`](results/analysis_v1/comparisons.csv).

| Format | qwen3.5:9b Acc@1 | difference vs F0 | gemma4:12b Acc@1 | difference vs F0 |
|---|---|---|---|---|
| F0 | 0.487 [0.360, 0.604] | | 0.503 [0.380, 0.615] | |
| F1 | 0.446 [0.330, 0.560] | -0.041 [-0.108, 0.027] | 0.528 [0.416, 0.634] | +0.026 [-0.016, 0.076] |
| F2 | 0.456 [0.348, 0.564] | -0.031 [-0.119, 0.060] | 0.601 [0.485, 0.707] | **+0.098 [0.021, 0.185]** |
| F3 | 0.440 [0.325, 0.554] | -0.047 [-0.153, 0.065] | 0.534 [0.420, 0.640] | +0.031 [-0.046, 0.112] |
| F4 | 0.404 [0.282, 0.522] | -0.083 [-0.192, 0.030] | 0.461 [0.330, 0.584] | -0.041 [-0.145, 0.064] |
| F5 | 0.518 [0.412, 0.624] | +0.031 [-0.094, 0.161] | 0.591 [0.463, 0.709] | +0.088 [-0.005, 0.193] |

Self-time baseline (no LLM): 0.290 [0.175, 0.404]. Chance: 0.089.

Mean prompt tokens per T2 call on the same windows (same source file):

| | F0 | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|---|
| qwen3.5:9b | 9,390 | 5,042 | 1,872 | 749 | 331 | 552 |
| gemma4:12b | 9,631 | 5,257 | 1,946 | 817 | 369 | 588 |

What the registered rules give ([`results/main_grid/NOTES.md`](results/main_grid/NOTES.md), sections 2 to 5):

- **1 of 10** format-vs-F0 comparisons is a clear difference: gemma F2, +9.8 points. With 10 unadjusted comparisons,
  about 0.5 false clear differences are expected if no format differs from F0.
- **H1a** (F3, F4, F5 beat F0) is not supported: 0 of 6 cells.
- **H1b** (F1, F2 show no clear difference from F0) does not hold as a whole: gemma F2 is the exception.
- **Every** model and format beats the self-time baseline (24 of 24 comparisons).
- **F4 and F5** use 16 to 28 times fewer prompt tokens than F0, with no clear loss or gain in Acc@1.
- **T1** stays near chance for every condition (accuracy 0.474 to 0.539).

A "no clear difference" result means the CI includes 0. It does not show that two formats are equivalent. Per-system
results (OB vs TT) are in [`results/main_grid/metrics.csv`](results/main_grid/metrics.csv). The Pareto plot of
accuracy vs tokens is [`results/analysis_v1/pareto_t2_informative.png`](results/analysis_v1/pareto_t2_informative.png).

## Error analysis (exploratory, not pre-registered)
60 wrong T2 answers (5 per model and format) were labeled by hand with a fixed codebook; a second rater labeled 12 of
them blind (12 of 12 agreement). Full notes: [`results/error_analysis_v1/NOTES.md`](results/error_analysis_v1/NOTES.md),
design in [spec 010](specs/010-error-analysis.md). Nothing in this section changes the registered results above.

- On TT, most sampled errors name `ts-auth-service`, which owns the longest request in the window
  (`longest_root`, 23 of 30 TT items).
- On OB, the true root-cause service is usually present, but its own spans are short (18 of 30 OB items).
- Both models often put the same service first (`currencyservice` on OB, `ts-auth-service` on TT). These are also the
  most frequent true root causes, so a constant answer would score 0.435 Acc@1, close to F0. This baseline was not
  registered.

## Limitations
- **Small sample**: 54 cases, not 193 independent windows. The CIs are wide.
- **Narrow setting**: two local models of similar size, one prompt, one call per cell. Prompt sensitivity, run-to-run
  variation and larger or API models were not measured.
- **Formats change more than syntax**: F2 also drops fields, and F3 to F5 select or aggregate evidence, so a
  difference cannot be attributed to layout or token count alone.
- **Selection**: a per-trace size cap and partial trace coverage in RCAEval remove some cases from the primary set.
  Results describe this subset of RCAEval, not arbitrary production incidents.
- **No multiplicity correction** across the 10 main comparisons.
- **Possible contamination**: RCAEval has been public at least since December 2024.
- **T2 is not RCAEval's metric**: Acc@3 here is a top-3 hit, not Avg@k, so the numbers are not comparable with
  published RCAEval tables.

Full list: [`PREREGISTRATION.md`](PREREGISTRATION.md), section 13, and
[`results/main_grid/NOTES.md`](results/main_grid/NOTES.md), section 7.6.

## Reproduce
Requires [uv](https://docs.astral.sh/uv/). The analysis steps need no model and run from the committed outputs;
step 2 also reads the RCAEval traces, which step 1 downloads.

```
uv sync
uv run pytest

# 1. Build the fixed windows (downloads the needed RCAEval traces)
uv run python scripts/build_windows.py configs/windows_v1.yaml

# 2. Score saved calls, run the self-time baseline, run the analysis
uv run python scripts/score_tasks.py results/main_grid/calls.jsonl
uv run python scripts/run_baseline.py configs/windows_v1.yaml
uv run python scripts/analyze.py configs/analysis_v1.yaml

# Optional: check the saved calls against the planned grid
uv run python scripts/check_grid.py configs/main_grid.yaml results/main_grid/calls.jsonl
```

To check the committed results without the raw traces or a model:
`uv run python scripts/verify_results.py configs/verify_v1.yaml`. It recomputes the main-grid scores, metrics and
analysis tables from the committed calls and baseline records in a temporary directory and compares them with the
committed files.

Re-running the model calls needs Ollama 0.35.0 with `qwen3.5:9b` and `gemma4:12b` pulled. The original run took
7.80 h of model time for qwen3.5:9b and 12.88 h for gemma4:12b on the hardware above:

```
uv run python scripts/run_tasks.py configs/main_grid.yaml --dry-run   # list the calls, no model requests
uv run python scripts/run_tasks.py configs/main_grid.yaml
```

The exploratory error analysis: `uv run python scripts/summarize_errors.py configs/error_analysis_v1.yaml`. The
constant-answer reference (0.435): `uv run python scripts/constant_answer.py configs/error_analysis_v1.yaml`. The
model times (7.80 h, 12.88 h): `uv run python scripts/model_time.py configs/model_time_v1.yaml`.

## Repository layout
```
src/traceformatbench/   data loaders, serializers, tasks, runners, analysis
configs/                one YAML per experiment
data/windows/           the fixed trace windows
results/                raw model outputs (JSONL), summaries (CSV) and notes
scripts/                entry points
specs/                  design specs, written before implementation
tests/
```

## How this was built
Built with coding agents. The research question, experiment protocol, metrics and analysis were designed by hand;
agents helped implement components, and every change was reviewed.

## License
MIT, see [`LICENSE`](LICENSE). Raw RCAEval files are not included. The repository does include data derived from
RCAEval: trace IDs and labels in [`data/windows/windows_v1.jsonl`](data/windows/windows_v1.jsonl) and serialized
trace excerpts for 60 items across 50 windows in [`results/error_analysis_v1/items/`](results/error_analysis_v1/items/). RCAEval
(Pham et al., WWW Companion 2025) is MIT-licensed per its GitHub README; its Zenodo record
(DOI 10.5281/zenodo.14590730) lists CC BY 4.0.
