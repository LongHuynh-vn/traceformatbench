# Error analysis v1 (spec 010): notes

EXPLORATORY: not part of the pre-registration (`PREREGISTRATION.md` section 14). Sections 1-6 are facts
copied from the committed files (checked against `labels.csv`, `summary.csv` and
`results/main_grid/scores.jsonl` on 2026-10-02). Rater 1 owns the labels and every conclusion. Nothing here changes registered scores or verdicts.

## 1. What was done
- Unit: one wrong T2 rank-1 answer on an informative window. 60 items, 30 per model, 5 per model x format cell,
  15 OB + 15 TT per model, seed 20260930, 10 pilot items excluded ([spec 010](../../specs/010-error-analysis.md)).
- Codebook v1 ([CODEBOOK.md](CODEBOOK.md)) and the sample were committed in #18 before any item was labeled.
- Rater 1 labeled all 60 items; rater 2 labeled 12 items, blind to rater 1's labels (confirmed by rater 2, 2026-10-02). Labels were
  made blind to the model; the automatic flags were visible in the label files.
- Reasons: `caller_blame`, `longest_root`, `weak_signal`, `unclear`. Automatic flags are not reasons.

## 2. Reasons (rater 1, 60 items)

| Reason | qwen3.5:9b | gemma4:12b | Total |
|---|---|---|---|
| longest_root | 11 | 14 | 25 |
| weak_signal | 11 | 9 | 20 |
| unclear | 7 | 4 | 11 |
| caller_blame | 1 | 3 | 4 |

- By system: TT longest_root 23, unclear 4, weak_signal 2, caller_blame 1; OB weak_signal 18, unclear 7,
  caller_blame 3, longest_root 2.
- Per model x format counts are in [summary.csv](summary.csv). Each cell has 5 items, so the cells are not
  compared with each other here.

## 3. Reasons against the automatic flags
- All 25 `longest_root` items have `top1_owns_longest_root` True; the flag is True on 28 items in total.
- All 4 `caller_blame` items have `top1_is_ancestor` True; the flag is True on 7 items in total.
- `weak_signal` items have `root_max_self_ms` from 0.013 to 4.57 ms; the other 40 items range from 0.016 to 248.956 ms.
- `unclear` (11): 2 are the contract items with the correct service first (E03, E18), labeled with the codebook rule
  `correct service, invalid answer only`. Of the other 9, 8 have `habitual` True: the model ranked
  `currencyservice` (OB) or `ts-auth-service` (TT) first, while the notes describe visible signal on the true
  root-cause service (E05, E06, E15, E19, E20, E24, E52, E60). The ninth (E58) is not habitual.
- `habitual` is True on 43 of 60 items (qwen 23, gemma 20). `true_rank` 0 on 29 items (true root cause not in the
  top 3), 1 on 2, 2 on 17, 3 on 12.

## 4. Agreement
- Rater 2, 12 items (E01 E03 E05 E12 E19 E21 E31 E39 E40 E43 E49 E60): raw agreement 12/12, Cohen's kappa 1.000.
- Limits: 12 items; both raters saw the automatic flags, and several reasons are close to a flag by definition
  (section 3), so the agreement partly reflects the flags.

## 5. The habitual answer (exploratory, post hoc)
Counted from `results/main_grid/scores.jsonl`, T2 informative windows (193: OB 94, TT 99).
- The habitual services are also the most frequent true root causes: OB `currencyservice` 35 of 94, TT
  `ts-auth-service` 49 of 99.
- A constant answer (always `currencyservice` on OB, always `ts-auth-service` on TT) would score 84/193 = 0.435
  Acc@1, against registered F0 0.487 (qwen) and 0.503 (gemma). This baseline was not registered.
- F0 Acc@1 split by whether the true root cause is the habitual service:

| Model, format | Acc@1 | Truth habitual (84) | Truth other (109) | First listed = habitual |
|---|---|---|---|---|
| qwen F0 | 0.487 | 0.94 | 0.14 | 156/193 |
| gemma F0 | 0.503 | 0.92 | 0.18 | 146/193 |
| gemma F2 | 0.601 | 0.95 | 0.33 | 150/193 |

Counted ad hoc (no committed script yet); across all 12 model x format cells, truth-other Acc@1 ranges 0.14 to 0.40.

## 6. Q2 (counts only)
gemma4:12b, informative windows (n = 193): F2 right and F0 wrong on 26, F0 right and F2 wrong on 7. This recounts
the main-grid gemma F2 result; it is not new evidence.

## 7. Interpretation (rater 1)

Numbers below come from sections
2-6 or were counted post hoc from `results/main_grid/scores.jsonl` and `results/baselines/selftime_v1.jsonl`
(counts checked twice, independently); the explanations are tentative and all of section 7 is exploratory.

### 7.1. Two failure patterns, split by system
In this sample, wrong answers on Train Ticket mostly name `ts-auth-service`, which owns the longest top-level request
in the window although that request does not explain the fault (`longest_root`, 23 of 30 TT items). On Online
Boutique, the true root-cause service is usually present but its own spans are short (`weak_signal`, 18 of 30 OB
items).

A simple "pick the slowest service" rule does not explain both. The self-time baseline (`results/baselines/selftime_v1.jsonl`) puts
`ts-auth-service` first in 95 of 99 TT windows, so on TT "pick the slowest" and the habitual answer (7.2) cannot be
told apart. On OB the baseline puts `frontendservice` first in 83 of 94 windows, yet at F0 the models name it first in
only 1 (qwen) and 4 (gemma) of 94; they name `currencyservice` instead. Only at F4, which keeps per-service timing
summaries and drops the span structure, do the models move to `frontendservice` (49 and 34 of 94), and OB Acc@1
falls (qwen 42 to 21, gemma 44 to 31 of 94). This fits the F4 OB/TT split in the main-grid notes (section 7.3 there).

The sample takes 5 items from every model x format cell (77-115 wrong answers per cell), so the label counts describe
the 60 items; they are not rates over all wrong answers.

### 7.2. The habitual answer
Both models often put the same service first: `currencyservice` on OB and `ts-auth-service` on TT. Across all 1,164
wrong T2 answers on informative windows, that service is listed first in 735 (qwen 366 of 627, gemma 369 of 537).
These services are also the most frequent true root causes in the primary set, so a constant answer would score
0.435, close to F0 (0.487 qwen, 0.503 gemma). F0 is right on 0.94/0.92 of windows whose truth is the habitual service
and 0.14/0.18 of the others; on TT windows whose truth is not `ts-auth-service`, both models are right on 4 of 50.
Part of the measured accuracy may therefore come from naming a common root cause rather than from reading the trace.

The habitual services are the most frequent truth only after the registered informative filter: 84 of the 85 T2
windows of the 17 cases with a habitual root cause are informative, against 109 of 285 for the other 57 cases. One
possible reason is that these services appear in almost every request, so their windows nearly always qualify. This
is a property of the evaluation set, not a flaw in the models, and it is worth stating next to the primary result.
The constant baseline was not registered and is a post hoc reference point.

### 7.3. What this adds to the gemma F2 result
The 26 windows where gemma is right with F2 and wrong with F0 are mostly OB (21) and network faults (delay 9, loss 6,
socket 5). In 24 of them the true service was already in F0's top 3, and F0's first answer was the habitual service in
only 6. Habitual-first answers barely change (146 of 193 at F0, 150 at F2). So F2 does not seem to help by moving
the model off the habitual answer; it more often promotes a service the model already ranked second or third. On OB
windows whose truth is not habitual, gemma goes from 5 to 16 of 23 on network faults with F2, but only from 11 to 12
of 36 on resource faults (cpu, mem, disk). This was found after the fact, and the 5 labeled F2 items per model cannot
test why tree structure would matter more for network faults.

### 7.4. What a format change can and cannot fix
`weak_signal` (20 items in 18 windows; 18 of the items on OB) covers most fault types, mostly cpu, disk and mem. Eight of the
18 windows are wrong in all 12 model x format cells, so some faults may leave too little in the root-cause service's
own spans for any of these formats. In at least one item the fault is visible elsewhere: in E57 (loss on
`currencyservice`) the frontend's client calls to CurrencyService take 3.0-3.6 s (GetSupportedCurrencies) and about 0.6 s (Convert),
while the currency server spans take at most 0.31 ms. The evidence can sit on the caller's side of a call, so "weak signal" describes the root-cause service's
spans, not the whole window. Whether a format that pairs client and server spans would help is a question for a
later round, not something this analysis shows.

### 7.5. Limits
60 items from 1,164 wrong answers, stratified rather than proportional; one rater for all items and a second rater
on 12; both raters saw the automatic flags, and two reasons match a flag by definition; the codebook was written after
an open-reading pilot on 10 other items. The baseline, fault-type and filter counts in 7.1-7.4 were not planned.

### 7.6. Examples for the write-up (pick 2-3)
- E01 (TT, F3, `longest_root`): truth `ts-route-service`, answer starts with `ts-auth-service`.
- E05 (OB, F0, `unclear`): long recommendation spans are visible, yet `currencyservice` is named first.
- E57 (OB, F1, `weak_signal`): the currency server spans are tiny, but the frontend's calls to it take up to 3.6 s.
- E04 (TT, F1, `caller_blame`): `ts-food-service` named first; its time is mostly in the call to the travel branch.

## 8. Deviations and notes on the process
- Labels were typed directly into the CSVs (not through a tool).
- Notes: at the first summary run 17 of 60 items had a note; rater 1 completed the other 43 on 2026-10-02
  (reasons unchanged), as the codebook asks for one note per item.
