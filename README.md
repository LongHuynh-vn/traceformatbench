# TraceFormatBench

**Status: work in progress. No results yet.**

When an LLM has to reason over distributed traces, how should the traces be serialized?
TraceFormatBench measures accuracy per token across serialization formats, model sizes, and token budgets.

## Research question
Which trace serialization gives the best accuracy per token on anomaly detection and fault
localization, and how does this depend on model size and token budget?

Hypotheses and the analysis plan are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any run.

## Data
[RCAEval](https://github.com/phamquiluan/RCAEval) (MIT License), RE2 and RE3 suites, Online Boutique and
Train Ticket (the systems that include traces). Raw data is downloaded on demand and never committed.

## Quick start
Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
uv run python scripts/inspect_case.py      # download one case and describe its traces
```

## Repository layout
```
src/traceformatbench/   data loaders, serializers, tasks, runners, analysis
configs/                one YAML per experiment
results/                raw model outputs (JSONL) and summaries (CSV)
scripts/                entry points
specs/                  design specs, written before implementation
tests/
```

## Results
Not available yet. Every number that appears here will be reproducible from this repository with one command.

## How this was built
Built with coding agents. I designed the research question, experiment protocol, metrics and analysis;
agents helped implement components, and I reviewed every change.
