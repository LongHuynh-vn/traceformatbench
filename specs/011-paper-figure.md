# 011 Paper forest figure

Implements the written A2 section 3.3 figure plan and G3 acceptance checks. This is a
presentation of saved results, with no new inference, resampling, or research design.

## Inputs and validation

- Read `comparisons.csv` and `summary.csv` from the configured analysis directory.
- Require exactly ten `is_main` rows: qwen3.5:9b then gemma4:12b, each F1–F5 against
  its own F0, T2 / T2-informative / acc_at_1, 193 windows and 54 cases.
- Reject missing/duplicate/unexpected primary rows, malformed booleans, non-finite
  estimates/endpoints, reversed intervals, estimates outside their intervals,
  inconsistent counts, and unsupported verdicts (including `not enough cases`).
  `clear difference` must mean exactly that the interval excludes zero.
- Join one summary row per model/format/task/set/metric, including each model's F0.
  Require matching counts and finite positive token means. Compute F0/format means
  within each model. Preserve both source reliability flags: append `!` if either
  operand is unreliable. Do not reinterpret the registered flag.

## Presentation

- One 3.487 × 2.05 inch panel; all text 8 pt. No title, footnote, or legend inside.
- Pin the bundled STIX General TrueType font by filename and SHA-256; no font
  fallback. Embed TrueType fonts with `pdf.fonttype: 42`, never Type 3.
- qwen circles followed by gemma squares, F1–F5 order, with a small model gap.
  Markers 4.5 pt, filled only for a clear difference; 1 pt capped intervals.
- Dashed grey zero line, 0.6 pt. Axis limits [−0.21, 0.21], ticks at
  −0.2, −0.1, 0, 0.1, 0.2; label `Acc@1 difference vs. F0 (95% CI)`.
- Right column: `fewer prompt tokens than F0`; ratios use one decimal below 10,
  integers otherwise, followed by `×` and the reliability flag when needed.
- Shape, fill and row labels remain sufficient in grayscale. All paths, selected
  conditions, counts, geometry and style are defined by `configs/paper_figure_v1.yaml`.
  Reset to Matplotlib's built-in defaults before applying that explicit style.

## Reproduction and acceptance

`uv run --locked python scripts/plot_paper_figure.py configs/paper_figure_v1.yaml`
writes PDF, PNG and JSON provenance. The manuscript copies the generated PDF
byte-for-byte. Metadata excludes dates and identity; provenance records input,
config, font and output hashes plus rendering-library versions. Identical bytes
are checked for repeated renders in the same locked environment; cross-platform
byte identity is not promised. Figure generation is separate from `verify_results.py`.

Tests cover every plotted coordinate, fill, marker, ratio and flag, invalid inputs,
font size, bounds/overlap, PDF dimensions/fonts/metadata, repeatability and hostile
Matplotlib settings. Inspect the final-size figure and grayscale rendering.

The manuscript caption supplies 193 windows/54 cases, paired differences, pooled
window-level estimates with case resampling, stratification and the RE3-OB into
RE2-OB small-stratum merge, 10,000 resamples, no multiplicity correction, marker
meanings and the sole clear gemma F2 comparison. It explains that `!` is the
condition-wide cached/truncated-call reliability flag: cached calls remain in token
means, truncated calls are excluded, and the flag is not proof of truncation.
