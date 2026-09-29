# AGENTS.md

Rules for every coding agent working in this repository (Claude Code, Codex, OpenCode, or any other).

## What this repo is
TraceFormatBench: a benchmark of how distributed traces (OpenTelemetry-style spans) should be serialized
for LLMs, measured as accuracy per token across formats, models, and token budgets.

## Ownership
The student owns the research question, hypotheses, task definitions, labels, metrics, baselines,
and every conclusion. Agents implement from written specs; they do not decide research design.

| Student (must own) | Agent (may do) |
|---|---|
| Research question, hypotheses | Boilerplate, scaffolding |
| Task definitions, labels, metrics | Serializers from a written spec in `specs/` |
| Baseline choice, confound controls | Runners, retries, logging |
| Reading results, conclusions | Plotting, tables, CI config |
| Final wording of README, report, PRs | Drafts for the student to rewrite |

If a request touches the left column and no spec exists in `specs/`, stop and ask for one.

## Hard rules
1. English everywhere: code, comments, docs, commit messages.
2. Never invent numbers, results, citations, or links. Numbers in `README.md` or `REPORT.md` must come
   from committed code run on committed raw outputs.
3. Do not edit `PREREGISTRATION.md` except in its "Deviations" section, and only when asked.
4. No hard-coded experiment parameters in scripts. Every run is defined by a YAML file in `configs/`.
5. Fixed seeds, fixed prompts, exact model IDs recorded in every raw output line.
6. Ollama runs: always set `num_ctx` explicitly and log it. Check prompt length against `num_ctx`
   before sending; never let the server truncate input silently.
7. Hardware: MacBook Pro M4 Pro, 24 GB unified memory. Run one local model at a time. Flag anything
   that needs a GPU, a paid API, or more memory before doing it.
8. Do not commit raw datasets (`data/raw/` is git-ignored).
9. Keep changes small. One feature per branch, with tests.

## Layout
```
src/traceformatbench/
  data/          # dataset loaders and case sampling
  serializers/   # trace -> text formats (F0-F5), each unit-tested
  tasks/         # prompts, output schemas, scoring
  runners/       # model adapters (ollama, api)
  analysis/      # bootstrap CIs, tables, plots
configs/         # one YAML per experiment
results/         # raw outputs (JSONL) and summaries (CSV)
scripts/         # one-off entry points
specs/           # student-written specs, one per feature
tests/
```

## Commands
```
uv sync                 # install pinned dependencies
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Definition of done for a change
- Tests pass and cover the new behavior, including edge cases listed in the spec.
- `ruff check` is clean.
- A short explanation of what changed and why, so the student can defend every line.
