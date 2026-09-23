# Both useful-warning definitions, side by side (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document. Everything
below is the three existing `train_val_v2` checkpoints
(`risk_gru_k1p0_v2_s1234/s1235/s1236.pt`) re-scored on `data/features/train_val_v2`
only, under both definitions from
[`../design/useful_warning_definition.md`](../design/useful_warning_definition.md).
**No operating point is selected here.** No hyperparameter was tuned. No
checkpoint was retrained.

## What changed and what didn't

**Changed:** `evaluate()` now always computes, alongside the legacy numbers,
an episode-based `useful_warning_rate_v2` that credits a clip if *any*
confirmed alert episode's active interval intersects `[alert_t, event_t]`, not
only the clip's first crossing — plus two lead figures and a three-part
prematurity axis. See [`../design/useful_warning_definition.md`](../design/useful_warning_definition.md)
for the full justification, written and committed before any of this
document's numbers existed.

**Did not change:** `useful_warning_rate`, `n_useful`, `n_too_early`,
`false_alarm_rate`, `mean_lead_s` — computed by the exact same code as before,
bit for bit (locked in by
`tests/test_train_risk_head.py::test_legacy_numbers_reproduce_committed_checkpoint_bit_for_bit`).
`train()`'s epoch-selection logic reads only these keys and is unaffected. The
AP-at-cutoffs figures (the actual Nexar competition metric) are untouched —
nothing in this document's method reaches them at all.

## A correction found while producing this document

The first implementation of `find_alert_episodes()` credited an episode at the
frame its `confirm`-length debounce *completed*, not the episode's own first
frame — a literal reading of the task specification's wording. That produced a
real defect: on 6 of the 45 grid cells below, the corrected definition credited
**fewer** clips as useful than the legacy one (worst case 32/103 → 24/103),
because a long `confirm` window pushed the debounce-completion frame of an
early-starting run past `event_t`. That is the opposite of what this
definition exists to fix, and it broke the "v2 is always ≥ legacy" guarantee.

Asked rather than decided silently: the fix credits an episode at its own
**first frame**, matching `first_alert_time`'s existing, already-documented
convention ("the model actually knew at i-need+1, and charging it the debounce
delay would understate its lead time"). Verified across the full grid below:
**0 of 45 cells have `useful_warning_rate_v2 < useful_warning_rate`,** for all
three checkpoints. The design doc and the episode function's docstring are
updated with this correction recorded, not silently absorbed.

## Both-definitions grid, all three seeds

### Seed 1234

| thr | confirm | useful (legacy) | n | useful_v2 | n_v2 | FA | lead (legacy, s) | lead vs event (v2, s) | lead vs alert (v2, s) |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.330 | 34 | 0.942 | 97 | 0.667 | 2.70 | 2.47 | −0.90 |
| 0.60 | 3 | 0.252 | 26 | 0.796 | 82 | 0.490 | 2.52 | 2.18 | −0.58 |
| 0.70 | 3 | 0.282 | 29 | 0.699 | 72 | 0.304 | 2.13 | 1.75 | −0.10 |
| 0.80 | 3 | 0.311 | 32 | 0.515 | 53 | 0.147 | 1.62 | 1.31 | 0.33 |
| 0.90 | 3 | 0.194 | 20 | 0.252 | 26 | 0.029 | 1.14 | 0.90 | 0.75 |
| 0.50 | 5 | 0.291 | 30 | 0.903 | 93 | 0.627 | 2.75 | 2.56 | −0.97 |
| 0.60 | 5 | 0.252 | 26 | 0.786 | 81 | 0.461 | 2.49 | 2.16 | −0.56 |
| 0.70 | 5 | 0.301 | 31 | 0.689 | 71 | 0.275 | 1.95 | 1.76 | −0.11 |
| 0.80 | 5 | 0.291 | 30 | 0.495 | 51 | 0.088 | 1.56 | 1.28 | 0.33 |
| 0.90 | 5 | 0.136 | 14 | 0.194 | 20 | 0.029 | 1.14 | 0.96 | 0.67 |
| 0.50 | 8 | 0.291 | 30 | 0.903 | 93 | 0.569 | 2.69 | 2.55 | −0.96 |
| 0.60 | 8 | 0.233 | 24 | 0.748 | 77 | 0.373 | 2.48 | 2.23 | −0.60 |
| **0.70** | **8** | 0.243 | 25 | 0.612 | 63 | 0.196 | 1.95 | 1.83 | −0.20 |
| **0.80** | **8** | **0.252** | 26 | **0.447** | 46 | **0.088** | 1.64 | 1.39 | 0.27 |
| 0.90 | 8 | 0.087 | 9 | 0.136 | 14 | 0.020 | 1.30 | 1.17 | 0.37 |
| **0.60** | **5** | **0.252** | 26 | **0.786** | 81 | **0.461** | 2.49 | 2.16 | −0.56 |

(thr 0.60/confirm 5 duplicated on the last row for visibility — it's the same
row as above, marked as one of the two reference points from Step 7.)

Prematurity axis, seed 1234:

| thr | confirm | frac_premature | median_premature_s | n_gross_premature |
|---|---|---|---|---|
| 0.50 | 3 | 0.612 | 1.88 | 1 |
| 0.60 | 3 | 0.544 | 1.25 | 1 |
| 0.70 | 3 | 0.417 | 1.00 | 0 |
| 0.80 | 3 | 0.223 | 0.83 | 0 |
| 0.90 | 3 | 0.087 | 0.58 | 0 |
| 0.50 | 5 | 0.612 | 1.83 | 1 |
| 0.60 | 5 | 0.534 | 1.33 | 1 |
| 0.70 | 5 | 0.398 | 0.93 | 0 |
| 0.80 | 5 | 0.214 | 0.81 | 0 |
| 0.90 | 5 | 0.078 | 0.55 | 0 |
| 0.50 | 8 | 0.612 | 1.73 | 1 |
| 0.60 | 8 | 0.515 | 1.17 | 1 |
| 0.70 | 8 | 0.379 | 0.90 | 0 |
| **0.80** | **8** | **0.204** | 0.78 | 0 |
| 0.90 | 8 | 0.058 | 0.55 | 0 |

### Seed 1235

| thr | confirm | useful (legacy) | n | useful_v2 | n_v2 | FA | lead (legacy, s) | lead vs event (v2, s) | lead vs alert (v2, s) |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.340 | 35 | 0.816 | 84 | 0.480 | 1.99 | 1.59 | −0.00 |
| 0.60 | 3 | 0.359 | 37 | 0.728 | 75 | 0.363 | 1.67 | 1.29 | 0.35 |
| 0.70 | 3 | 0.350 | 36 | 0.573 | 59 | 0.206 | 1.49 | 1.25 | 0.38 |
| 0.80 | 3 | 0.330 | 34 | 0.447 | 46 | 0.108 | 1.17 | 1.00 | 0.62 |
| 0.90 | 3 | 0.272 | 28 | 0.301 | 31 | 0.039 | 0.97 | 0.76 | 0.87 |
| 0.50 | 5 | 0.330 | 34 | 0.757 | 78 | 0.402 | 1.95 | 1.66 | −0.07 |
| 0.60 | 5 | 0.350 | 36 | 0.709 | 73 | 0.265 | 1.62 | 1.31 | 0.31 |
| 0.70 | 5 | 0.320 | 33 | 0.524 | 54 | 0.196 | 1.44 | 1.25 | 0.37 |
| 0.80 | 5 | 0.301 | 31 | 0.417 | 43 | 0.098 | 1.17 | 1.01 | 0.63 |
| 0.90 | 5 | 0.223 | 23 | 0.243 | 25 | 0.029 | 1.05 | 0.84 | 0.87 |
| 0.50 | 8 | 0.301 | 31 | 0.728 | 75 | 0.314 | 1.91 | 1.67 | −0.06 |
| **0.60** | **8** | 0.330 | 34 | 0.621 | 64 | 0.206 | 1.57 | 1.35 | 0.28 |
| 0.70 | 8 | 0.301 | 31 | 0.495 | 51 | 0.137 | 1.37 | 1.27 | 0.28 |
| **0.80** | **8** | **0.311** | 32 | **0.398** | 41 | **0.078** | 1.15 | 1.00 | 0.65 |
| 0.90 | 8 | 0.175 | 18 | 0.194 | 20 | 0.020 | 1.02 | 0.90 | 0.75 |

Prematurity axis, seed 1235:

| thr | confirm | frac_premature | median_premature_s | n_gross_premature |
|---|---|---|---|---|
| 0.50 | 3 | 0.485 | 0.91 | 1 |
| 0.60 | 3 | 0.379 | 0.69 | 0 |
| 0.70 | 3 | 0.243 | 0.63 | 0 |
| 0.80 | 3 | 0.126 | 0.40 | 0 |
| 0.90 | 3 | 0.049 | 0.53 | 0 |
| 0.50 | 5 | 0.437 | 0.90 | 0 |
| 0.60 | 5 | 0.369 | 0.65 | 0 |
| 0.70 | 5 | 0.223 | 0.59 | 0 |
| 0.80 | 5 | 0.126 | 0.40 | 0 |
| 0.90 | 5 | 0.039 | 1.05 | 0 |
| 0.50 | 8 | 0.437 | 0.89 | 0 |
| 0.60 | 8 | 0.311 | 0.65 | 0 |
| 0.70 | 8 | 0.204 | 0.58 | 0 |
| **0.80** | **8** | **0.107** | 0.40 | 0 |
| 0.90 | 8 | 0.029 | 0.36 | 0 |

### Seed 1236

| thr | confirm | useful (legacy) | n | useful_v2 | n_v2 | FA | lead (legacy, s) | lead vs event (v2, s) | lead vs alert (v2, s) |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.340 | 35 | 0.592 | 61 | 0.402 | 1.72 | 1.40 | 0.22 |
| 0.60 | 3 | 0.223 | 23 | 0.379 | 39 | 0.196 | 1.32 | 1.08 | 0.53 |
| 0.70 | 3 | 0.146 | 15 | 0.184 | 19 | 0.069 | 0.96 | 0.88 | 0.74 |
| 0.80 | 3 | 0.029 | 3 | 0.029 | 3 | 0.010 | 0.62 | 0.62 | 1.68 |
| 0.90 | 3 | 0.000 | 0 | 0.000 | 0 | 0.000 | n/a | n/a | n/a |
| 0.50 | 5 | 0.311 | 32 | 0.553 | 57 | 0.343 | 1.49 | 1.42 | 0.18 |
| 0.60 | 5 | 0.223 | 23 | 0.359 | 37 | 0.157 | 1.24 | 1.09 | 0.53 |
| 0.70 | 5 | 0.107 | 11 | 0.136 | 14 | 0.039 | 0.93 | 0.93 | 0.79 |
| 0.80 | 5 | 0.029 | 3 | 0.029 | 3 | 0.010 | 0.62 | 0.62 | 1.68 |
| 0.90 | 5 | 0.000 | 0 | 0.000 | 0 | 0.000 | n/a | n/a | n/a |
| 0.50 | 8 | 0.291 | 30 | 0.515 | 53 | 0.284 | 1.50 | 1.46 | 0.13 |
| **0.60** | **5** | **0.223** | 23 | **0.359** | 37 | **0.157** | 1.24 | 1.09 | 0.53 |
| 0.70 | 8 | 0.068 | 7 | 0.097 | 10 | 0.029 | 1.14 | 1.14 | 0.51 |
| **0.80** | **8** | **0.010** | 1 | **0.010** | 1 | **0.010** | 0.65 | 0.65 | 0.45 |
| 0.90 | 8 | 0.000 | 0 | 0.000 | 0 | 0.000 | n/a | n/a | n/a |

(Two reference-point rows bolded and duplicated for visibility from the full
grid above them.)

Prematurity axis, seed 1236:

| thr | confirm | frac_premature | median_premature_s | n_gross_premature |
|---|---|---|---|---|
| 0.50 | 3 | 0.282 | 0.79 | 1 |
| 0.60 | 3 | 0.165 | 0.53 | 0 |
| 0.70 | 3 | 0.039 | 0.15 | 0 |
| 0.80 | 3 | 0.000 | n/a | 0 |
| 0.90 | 3 | 0.000 | n/a | 0 |
| 0.50 | 5 | 0.252 | 0.77 | 0 |
| **0.60** | **5** | 0.136 | 0.49 | 0 |
| 0.70 | 5 | 0.029 | 0.07 | 0 |
| 0.80 | 5 | 0.000 | n/a | 0 |
| 0.90 | 5 | 0.000 | n/a | 0 |
| 0.50 | 8 | 0.223 | 0.78 | 0 |
| 0.60 | 8 | 0.117 | 0.49 | 0 |
| 0.70 | 8 | 0.029 | 0.07 | 0 |
| **0.80** | **8** | **0.000** | n/a | 0 |
| 0.90 | 8 | 0.000 | n/a | 0 |

### Reading the grid

At the loosest operating points (thr 0.50, confirm 3–5) the corrected
definition credits **80–94%** of positives on seeds 1234/1235 — most of what
looked like a near-total failure under the legacy rule was continuous coverage
starting slightly early, exactly the clip-1013 pattern. That gap narrows
sharply as threshold rises: by thr 0.90 the two definitions nearly agree
(0.19–0.30 vs 0.19–0.25), because a stricter threshold leaves less room for a
run to straddle the debounce boundary in the first place. **`false_alarm_rate`
is identical in both definitions at every cell** — it was never touched.

`n_gross_premature` is almost always 0 or 1 out of 103 positives, concentrated
at the loosest thresholds. This is the opposite of the old threshold engine's
failure mode (median 14.4 s premature, firing from frame 0): what the corrected
definition is crediting is not indiscriminate early firing, it is a model that
starts warning a fraction of a second to ~2 s ahead of the annotated moment and
then holds the warning through the actionable window — visibly distinct from
gross prematurity on its own dedicated axis, as designed.

## Onset-to-event lead: does the ~3 s estimate hold?

`early_firing_diagnosis.md` estimated onset-to-event lead informally as
~1.5 s (onset before `alert_t`) + ~1.6 s (`alert_t` before `event_t`) ≈ 3 s,
inside the 2–6 s target. That arithmetic combined two different subpopulations
at different operating points (median offsets of `too_early` clips only), not
a single population's actual onset-to-event distribution. Checked directly:
for every positive that fires at all, `onset_t` = the earliest confirmed
episode anywhere in the clip (not restricted to episodes intersecting the
window), and `lead = event_t − onset_t`.

| seed | op point | n fired / 103 | min | Q1 | median | Q3 | max | mean | fraction in [2,6] s |
|---|---|---|---|---|---|---|---|---|---|
| 1234 | thr 0.80/c8 | 48 | −0.13 | 0.90 | 1.44 | 2.08 | 4.97 | 1.60 | 13/48 = 27.1% |
| 1234 | thr 0.60/c5 | 84 | −0.40 | 1.18 | **2.24** | 3.43 | 6.60 | 2.40 | 41/84 = 48.8% |
| 1235 | thr 0.80/c8 | 45 | −0.20 | 0.57 | 0.90 | 1.47 | 4.37 | 1.09 | 4/45 = 8.9% |
| 1235 | thr 0.60/c5 | 76 | −0.40 | 0.74 | 1.40 | 2.08 | 4.50 | 1.57 | 24/76 = 31.6% |
| 1236 | thr 0.80/c8 | 1 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0/1 |
| 1236 | thr 0.60/c5 | 42 | −0.31 | 0.26 | 0.87 | 1.60 | 4.47 | 1.07 | 6/42 = 14.3% |

**Verdict: partially supports, does not confirm the specific ~3 s figure.**
The best case (seed 1234, thr 0.60/confirm 5) gets a median of 2.24 s — inside
the target range, and its third quartile (3.43 s) sits comfortably within
2–6 s. But that is the best of six seed/operating-point combinations shown, and
even there only 48.8% of fired clips land in the target window; at the
stricter reference point (thr 0.80/confirm 8) the median drops to 1.44 s,
below the 2 s floor. The spread is wide (roughly 5 s tip-to-tip at every
setting shown) and the minimum is frequently negative — some onsets occur
*after* `event_t` is nominally reached relative to a differently-scaled clip,
which is possible when a clip's own window is short. **The 2–6 s target is
reachable for a meaningful minority to near-half of positives depending on
operating point, not for a typical or median positive across the board.**
Overstating this as "the target is met" would not be supported by this
distribution; understating it as "unreachable" would also be wrong, since a
real and substantial fraction clears it. Report both quartile figures, not
the mean alone, whenever this number is quoted again.

## What this document does not do

No operating point is chosen here — the two points repeated in the tables
above are the same reference points used for diagnosis in
`early_firing_diagnosis.md`, not a selection. No checkpoint is retrained, no
hyperparameter tuned, and `data/features/eval_v2` remains unread.
