# Re-partitioned splits and clean retrain (2026-09-23)

The splits have been redrawn. This document records why, what the new partition
looks like, and what three retrained seeds produce on it.

**The new eval split has not been scored.** No number in this document comes
from `data/features/eval_v2`. The operating point is also **not selected** — the
project's decision rule had no survivors, and per instruction no gate was
dropped unilaterally. Both are pending a decision.

## 1. Why the partition had to be redrawn

[`label_window_audit.md`](label_window_audit.md) traced the disjoint
alert-to-event window to its source. `data/nexar/labels.xlsx` — the hand-built
355-row table `prepare_nexar.py` consumed for the original download — has its
positive rows **sorted by window descending and truncated at the 65 longest
clips in the dataset**. `dev` and `eval` were drawn from that subset, so they
absorbed all 65. The cut is 4 ms wide: 65th largest window 2.971 s, 66th
2.967 s.

The consequence was that **the held-out split was the most anticipatable slice
of the data by construction**, and the training pool had nothing above 2.967 s.
Raw `train.csv` is continuous over 0.033–4.466 s and raw → index → `.npz` agree
exactly on all 1,485 extracted clips, so no layer of the pipeline was at fault —
only the partition. Only the partition is changed here.

A second consequence, which is what makes this more than a fairness issue:
`alert_weights()` zeroes every frame outside `[alert_t, event_t]`, so the
objective can never request more lead than a clip's own window. With no
training clip above 2.967 s, the 2–6 s lead target was never learnable. The
lead-time "gap" this project has been reporting was a partition artifact.

## 2. The pool, and the untimed-positive question

Working from the **1,485 clips that have extracted features** (745 pos / 740
neg). The 15 `dev` clips have no `.npz` and are **out of scope for this task**;
they keep their v1 membership in the new freeze and were not touched.

The premise that `train_risk` holds 685 positives of which only 582 are "timed"
does not hold. **Every positive in the pool is timed** — zero are missing
`event_time_s`, zero are missing `alert_time_s`:

```
positives missing event_time_s: 0
positives missing alert_time_s: 0
positives fully timed         : 745
negatives with any time       : 0   (expected: a negative has no event)
```

The 685-vs-582 figure was my own bookkeeping from the Step 4 carve: 582
(`train_risk`) + 103 (`train_val`) = 685, plus 60 in `eval` = 745 in the pool,
plus 5 `dev` positives = the 750 in the raw table. There is no untimed subset,
so the contingency for where untimed positives should go is moot and no clips
were set aside. `repartition_splits.py` still refuses to run if any positive
lacks a window, so the assumption is enforced rather than remembered.

For the record, since it was asked: `evaluate()` skips a positive with no
`event_time_s`/`alert_time_s` entirely — it is excluded from `n_timed`
(`train_risk_head.py:119-121`) and from every AP cutoff
(`train_risk_head.py:133-136`), so such a clip would contribute nothing to any
reported metric. It would still reach the loss, which tracks them as
`n_untimed`. None exist, so none of this fires.

## 3. The new partition

`scripts/repartition_splits.py`, seed 1234. Positives are binned into **deciles
of window length** (equal-count, 74–75 clips per bin) and each split is
apportioned proportionally from every bin by largest remainder. Deciles were
chosen over quartiles because they still leave exactly 6 eval positives per bin
— fine enough to control the shape, coarse enough that no bin is thin.
Negatives carry no window and are stratified by label alone.

Per-bin rounding was tried first and is wrong: it overshoots each split's cap,
and patching the overshoot afterwards takes the excess from whichever bin was
processed last — the top decile, i.e. precisely the clips this repartition
exists to spread. That produced 2 eval and 4 train_val positives in decile 10
against 6 and 10 elsewhere. The committed version apportions globally.

### Window distribution, positives only

| split | n | mean | median | min | max |
|---|---|---|---|---|---|
| pool | 745 | 1.589 | 1.433 | 0.033 | 4.466 |
| train_core_v2 | 582 | 1.596 | 1.433 | 0.033 | 4.466 |
| train_val_v2 | 103 | 1.553 | 1.369 | 0.134 | 4.100 |
| eval_v2 | 60 | 1.587 | 1.435 | 0.200 | 4.085 |

### Per-decile counts

| decile | window range (s) | pool | train_core | train_val | eval |
|---|---|---|---|---|---|
| 1 | 0.03–0.65 | 74 | 58 | 10 | 6 |
| 2 | 0.67–0.91 | 75 | 58 | 11 | 6 |
| 3 | 0.91–1.07 | 74 | 58 | 10 | 6 |
| 4 | 1.07–1.23 | 75 | 58 | 11 | 6 |
| 5 | 1.23–1.42 | 74 | 58 | 10 | 6 |
| 6 | 1.43–1.63 | 75 | 58 | 11 | 6 |
| 7 | 1.63–1.90 | 74 | 58 | 10 | 6 |
| 8 | 1.90–2.27 | 75 | 59 | 10 | 6 |
| 9 | 2.27–2.82 | 74 | 58 | 10 | 6 |
| 10 | 2.82–4.47 | 75 | 59 | 10 | 6 |

### Old vs new

| | v1 (superseded) | v2 |
|---|---|---|
| train window mean | 1.41 s | 1.596 s |
| train window max | **2.967 s** | **4.466 s** |
| eval window mean | 3.51 s | 1.587 s |
| eval window min | **2.971 s** | **0.200 s** |
| ranges overlap? | **no — disjoint at 2.97 s** | **yes, all three span the pool** |

**The disjointness is gone.** All three splits span essentially the full range
and their means sit within 0.04 s of the pool's.

### Sizes and integrity

`train_core_v2` 1,160 (582 pos / 578 neg), `train_val_v2` 205 (103/102),
`eval_v2` 120 (60/60). Verified: pairwise disjoint, union equals the 1,485-clip
pool, 112-column features throughout, and the v2 freeze's eval membership
matches the directory exactly.

Nothing was deleted. `data/features/train_all/`, `data/features/eval/` and
`eval/split_freeze.json` are all on disk unchanged; the new splits are copies.
No features were re-extracted — `event_time_s` is unchanged, so every existing
feature window is still correctly placed.

### The guard

`eval/split_freeze_v2.json` is the current freeze and
`a3ps/common/splits.py::DEFAULT_FREEZE_PATH` points at it, so `train_risk_head.py`
and `sweep_operating_point.py` now refuse the new eval split.

Consulting v1 *as well* was implemented and then reverted. v2 moved 97 of v1's
120 eval clips into `train_core_v2`, so v1 membership no longer means "held
out"; guarding on it would refuse legitimate training data. Guarding on v2
alone still catches the stale directories on their own merits, because
`data/features/train_all` holds 108 clips that v2 freezes as eval and
`data/features/eval` holds 12. Verified directly:

| directory | guard |
|---|---|
| `train_core_v2` | allowed |
| `train_val_v2` | allowed |
| `eval_v2` | **refused** |
| `train_all` | **refused** (108 v2-eval clips) |
| `eval` (v1 dir) | **refused** (12 v2-eval clips) |

`train_all` being refused matters: under v2 it is no longer a legal training
directory, because 108 of the new eval clips are inside it.

## 4. Retrain: three seeds

kappa 1.0, pre-alert-weight 0.5, pos-weight 1.0, fa-target 0.20, epochs 30,
patience 8, batch-size 8, hidden 64, layers 1. Trained on `train_core_v2`,
validated on `train_val_v2`. Validation numbers, at the default scoring point
(threshold 0.5 / confirm 3) the training loop uses for selection:

| seed | best epoch | useful | FA | mean lead | mean AP | runtime |
|---|---|---|---|---|---|---|
| 1234 | 4 | 0.330 | 0.667 | 2.70 s | 0.630 | 82 s |
| 1235 | 13 | 0.340 | 0.480 | 1.99 s | 0.590 | 138 s |
| 1236 | 1 | 0.340 | 0.402 | 1.72 s | 0.501 | 61 s |

Best epoch again lands anywhere from 1 to 13 and mean AP spans 0.501–0.630 — a
0.13-wide spread, wider than the 0.06 seen under v1. A single run remains a
draw, not a point estimate.

Mean lead on validation has roughly doubled versus the v1 carve (1.34–1.68 s
there, 1.72–2.70 s here). That is the expected mechanical consequence of
validation now containing clips whose windows allow a longer warning, not
evidence of a better model.

## 5. The decision rule has no survivors — operating point NOT selected

Sweep of the seed-1234 checkpoint on `train_val_v2` only (seed 1234 because it
is the config's seed, fixed before any of these numbers existed — not because it
scored best). Full grid:

| threshold | confirm | useful | too early | FA | mean lead |
|---|---|---|---|---|---|
| 0.50 | 3 | 0.330 | 63 | 0.667 | 2.70 |
| 0.60 | 3 | 0.252 | 56 | 0.490 | 2.52 |
| 0.70 | 3 | 0.282 | 43 | 0.304 | 2.13 |
| 0.80 | 3 | 0.311 | 23 | 0.147 | 1.62 |
| 0.90 | 3 | 0.194 | 9 | 0.029 | 1.14 |
| 0.50 | 5 | 0.291 | 63 | 0.627 | 2.75 |
| 0.60 | 5 | 0.252 | 55 | 0.461 | 2.49 |
| 0.70 | 5 | 0.301 | 41 | 0.275 | 1.95 |
| 0.80 | 5 | 0.291 | 22 | 0.088 | 1.56 |
| 0.90 | 5 | 0.136 | 8 | 0.029 | 1.14 |
| 0.50 | 8 | 0.291 | 63 | 0.569 | 2.69 |
| 0.60 | 8 | 0.233 | 53 | 0.373 | 2.48 |
| 0.70 | 8 | 0.243 | 39 | 0.196 | 1.95 |
| 0.80 | 8 | 0.252 | 21 | 0.088 | 1.64 |
| 0.90 | 8 | 0.087 | 6 | 0.020 | 1.30 |

`KAPPA_RETRAIN.md` §5 as written: FA ≤ 0.20, **useful-warning ≥ 0.70**, lead
≥ 1.0 s, then maximise mean lead.

**Zero rows survive.** The maximum useful-warning anywhere in the grid is
**0.330**, less than half the 0.70 gate. Fixing the partition did not rescue the
rule — it was never close.

No gate was dropped. For reference, applying only FA ≤ 0.20 and lead ≥ 1.0 s and
maximising lead would pick **threshold 0.80 / confirm 8** (useful 0.252, FA
0.088, lead 1.64 s) — but that is a deviation from the documented rule, and it
is recorded here as an illustration, **not as a selection**.

The `too early` column is worth noting on its own: at the operating points with
any lead to speak of, 40–60 of 103 positives fire *before* the dataset's own
`time_of_alert` and earn no credit. The model is firing early and being
penalised for it, which is a different failure from firing late.

## 6. What the loss is actually asking for

`scripts/compare_kappa.py:43` hardcodes `MEAN_ALERT_LEAD_S = 3.49`, commented
"measured over the 65 local positives" — the v1 `dev`/`eval` population, the
only one available when it was written. `train_risk_head.py:536` hardcodes the
same 3.49. That constant is what the "requested lead" column of
`eval/kappa_comparison.md` was computed from.

| | mean window | `expected_lead_time()` at kappa 1.0 |
|---|---|---|
| committed constant (v1 dev/eval) | 3.49 s | **1.459 s** |
| new `train_core_v2` | **1.596 s** | **0.667 s** |
| old broken train pool | 1.41 s | 0.589 s |

**The correct value of `MEAN_ALERT_LEAD_S` for the pool actually trained on is
1.60 s, not 3.49 s**, and the loss is asking for **0.667 s** of lead at kappa
1.0, not the documented 1.46 s. Across kappas on the new pool: 0.5 → 0.732 s,
1.0 → 0.667 s, 2.0 → 0.548 s, 3.0 → 0.448 s.

Per instruction the constant has **not** been changed; that is the next task,
along with the per-clip lead dump.

Note the repartition barely moved this number (1.41 → 1.60 s mean window),
because the pool's own mean was always ~1.59 s. Redrawing the split fixed *who
gets the long-window clips*; it did not create more of them. The dataset has 63
positives with a window over 3 s out of 750, and that is the ceiling on what any
partition can offer.

## 7. Every previous eval number is now incomparable

**Every number this project has ever reported on a held-out split — the old
threshold engine's 0.050 useful / 0.767 FA, the selection-biased GRU's 0.750 /
0.167, and the clean-protocol GRU's 0.633 / 0.200 alike — was measured on the v1
eval split, which was the 60 most anticipatable positives in the dataset. None
of them is comparable to anything produced after this change.**

This is not only a population difference. 97 of v1's 120 eval clips are now in
`train_core_v2`, so the models trained in §4 have seen most of the old test set.
Any comparison between a v1 eval number and a v2 eval number is meaningless in
both directions, and the v1 numbers cannot be rehabilitated by rescoring.

The v1 artifacts are retained, not deleted: `eval/split_freeze.json`,
`data/features/eval/`, `data/features/train_all/`, every existing checkpoint and
every `eval/*.md` from earlier rounds. They are the record of what was measured,
under a partition now known to be unrepresentative.

## 8. State, and what is pending

Done: pool established, partition redrawn and frozen, guards rewired, three
seeds trained, operating-point grid produced on validation, requested-lead
figures corrected.

Pending a decision:
1. **The operating point.** The documented rule has no survivors (§5). Dropping
   the useful-warning gate, lowering it, or changing the rule is a call to be
   made deliberately, not by whoever runs the sweep next.
2. **The eval read.** `data/features/eval_v2` has not been scored and will not
   be until the operating point is fixed.
3. **`MEAN_ALERT_LEAD_S`** and the 2–6 s lead target in `docs/design/metrics.md`,
   both still stated against the v1 eval population (§6).
