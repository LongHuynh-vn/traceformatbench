# Codebook v1: reasons for a wrong T2 answer (spec 010)

Written from the open-reading pilot. EXPLORATORY: not part of the pre-registration.
Committed together with the sample, before any item is labeled.

Choose ONE reason per item and write a one-sentence `note` for every item.

- `caller_blame`: top-1 is a caller (ancestor) of the root cause, and its long duration is mostly time spent inside
  that child path.
- `longest_root`: top-1 owns the longest top-level request in the window, and that length is not explained by the
  root-cause path.
- `weak_signal`: the root-cause service is present but shows nothing unusual in the window (e.g. only short spans).
- `unclear`: none fits; write a one-line note.

Contract items whose first listed service is the true root cause (`contract` True and `true_rank` 1): use `unclear`
with the note `correct service, invalid answer only`. They failed only the answer contract, not the diagnosis.

`habitual` and `contract` are automatic flags, not reasons. Do not use the flags as labels; read the item first.
