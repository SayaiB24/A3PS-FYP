# Clean train/val protocol — the honest held-out number (2026-09-23)

The headline this project has been quoting — useful-warning 0.750, false-alarm
0.167, mean lead 1.67 s, mean AP 0.680 — was **selected** on the frozen eval
split, not measured against it. [`eval_leakage_audit.md`](eval_leakage_audit.md)
establishes that from the checkpoints and scripts themselves: every GRU
checkpoint used `--val-features data/features/eval` for early stopping and
best-epoch selection, and the operating point, the kappa choice and the
capacity call were all made on eval-scored numbers.

This document reports what the same pipeline produces when selection is moved
off the test set. **The numbers got worse. That is the correct outcome**, and
nothing here was tuned to recover them.

## What changed

1. **A validation split now exists.** `scripts/carve_train_val.py` carves 205
   clips (103 pos / 102 neg) out of `train_all` at frac 0.15, seed 1234,
   stratified by label, leaving 1,160 for training. Each clip is its own source
   video (1,500 clips, 1,500 distinct paths), so a clip-level split is already a
   video-level one — no near-duplicate frames straddle it. The features were
   copied from the existing `train_all/` `.npz` files, so nothing was
   re-extracted and `train_all/` is untouched.
2. **`train_val` is frozen** in `eval/split_freeze.json` alongside `dev` and
   `eval`, so it inherits the same membership/label drift protection. A
   validation set that silently swaps clips with the training set is the same
   failure the eval freeze exists to prevent, one layer in.
3. **The guard gap is closed.** `--allow-leakage` only ever compared
   *directories*, so `--features train_all --val-features eval` passed it.
   `train_risk_head.py` and `sweep_operating_point.py` now consult the freeze
   and refuse any clip frozen as `eval` unless `--final-eval-report` is passed.
   `sweep_operating_point.py` also lost its `--features data/features/eval`
   default, which is how the operating point came to be chosen on the test set
   without anyone passing a flag.
4. **Selection happened only on `train_val`**; `data/features/eval` was read
   once per pre-registered question, after the checkpoint and operating point
   were already fixed.

## Side-by-side

| | old — **selection-biased, superseded — retained for the record** | new — clean protocol |
|---|---|---|
| checkpoint | `risk_gru_k1p0.pt` (best epoch 8) | `risk_gru_k1p0_clean.pt` (best epoch 6) |
| trained on | `train_all`, 1,365 clips | `train_core`, 1,160 clips |
| early stopping / best epoch chosen on | **`data/features/eval`** | `data/features/train_val` |
| operating point chosen on | **`data/features/eval`** | `data/features/train_val` |
| operating point | thr 0.60 / confirm 8 | thr 0.70 / confirm 8 |
| useful-warning | **0.750** (45/60) | **0.633** (38/60) |
| false-alarm | **0.167** | **0.200** |
| mean lead | 1.67 s | 1.62 s |
| mean AP | 0.680 | 0.661 |
| too-early alerts | 1/60 | 1/60 |

Both columns are scored on the same 120-clip frozen eval split. The old column
is kept because deleting a number you have already quoted is worse than
labelling it; it is not a result and must not be requoted as one.

## Are the targets still met?

**No for detection, marginally yes for false alarms.**

- **Useful-warning ≥ 0.75: not met.** 0.633 under the clean protocol, against a
  0.750 that was partly selection. The gap is 0.117, or seven clips.
- **False-alarm ≤ 0.20: met, exactly at the boundary.** 0.200 is at the target,
  not comfortably under it, and it is the *first* eval read of this checkpoint —
  there is no headroom and no second draw.
- **Mean lead 2–6 s: not met**, as before (1.62 s). This target was already
  missed under the biased protocol, so nothing changed here.

## Seed spread

Three seeds, identical config, each scored at its own `train_val`-selected
operating point. These are validation numbers, not eval numbers.

| seed | best epoch | operating point | useful | FA | mean lead | mean AP |
|---|---|---|---|---|---|---|
| 1234 | 6 | thr 0.70 / c8 | 0.369 | 0.196 | 1.34 s | 0.596 |
| 1235 | 1 | thr 0.60 / c8 | 0.252 | 0.176 | 1.68 s | 0.538 |
| 1236 | 12 | thr 0.70 / c5 | 0.369 | 0.196 | 1.41 s | 0.579 |

Best epoch lands anywhere from 1 to 12 and mean AP spans 0.538–0.596. A single
run of this config is not a point estimate; it is a draw from a distribution
roughly 0.06 mean-AP wide. Seed 1234 is reported above because it is the
headline config's seed, chosen before any of these numbers existed — not
because it is the best of the three.

## Kappa, re-selected on train_val

Kappa 1.0 was itself chosen on eval, so carrying it forward would leave residual
bias. All four kappas were retrained on `train_core`, validated on `train_val`,
and swept on `train_val`.

| kappa | best epoch | train_val pick | useful | FA | mean lead | mean AP |
|---|---|---|---|---|---|---|
| 0.5 | 6 | thr 0.70 / c8 | 0.359 | 0.167 | **1.351 s** | 0.592 |
| 1.0 | 6 | thr 0.70 / c8 | 0.369 | 0.196 | 1.344 s | 0.596 |
| 2.0 | 7 | thr 0.80 / c5 | 0.388 | 0.186 | 1.265 s | 0.590 |
| 3.0 | 5 | thr 0.80 / c5 | 0.398 | 0.176 | 1.220 s | 0.584 |

**Kappa 0.5 wins by 0.007 s of mean lead.** That margin is far inside the seed
spread above and should not be read as a real preference between 0.5 and 1.0.

Its single eval read, at the same fixed thr 0.70 / confirm 8:

| | useful | FA | mean lead | mean AP |
|---|---|---|---|---|
| kappa 1.0 (headline config) | 0.633 (38/60) | 0.200 | 1.62 s | 0.661 |
| kappa 0.5 (val-selected) | 0.583 (35/60) | 0.183 | 1.70 s | 0.658 |

The val-selected kappa scores *worse* on useful-warning than the config it
replaced. Selecting kappa on 205 validation clips by a 0.007 s margin did not
transfer, which is what a margin that size should be expected to do.

### The decision rule could not be applied as written

The project's rule (`docs/handoff/KAPPA_RETRAIN.md` §5) gates on FA ≤ 0.20,
useful-warning ≥ 0.70 and lead ≥ 1.0 s, then maximises mean lead. **On
`train_val`, no row of any kappa reaches useful-warning 0.70** — the maximum
anywhere in the 4 kappas × 15 operating points grid is 0.418. The full rule has
zero survivors.

The rule was not adjusted to fit. The useful-warning gate was dropped as
unreachable and the remaining gates applied unchanged (FA ≤ 0.20, lead ≥ 1.0 s,
maximise mean lead, tie-break mean AP), and that deviation is recorded here
rather than hidden in a threshold choice. Every pick above was still made on
`train_val` alone.

## Two things that are not bias, and are not separated here

**The training set shrank from 1,365 to 1,160 clips** — 15% less data — because
the validation split had to come from somewhere. Part of the drop from 0.750 to
0.633 is therefore less training data rather than bias removal. **No attempt is
made here to separate the two**, and it would take a further run on the full
1,365 with selection still on `train_val` (impossible without overlapping the
two) or a nested scheme to do it properly.

**`train_val` and `eval` are not exchangeable.** This turned up while checking
why validation useful-warning (~0.37) sits so far below eval's (0.633):

| split | timed positives | event − alert gap: mean | median | min | max |
|---|---|---|---|---|---|
| `train_risk` (train_core) | 582 | 1.41 s | 1.33 s | 0.03 | 2.97 |
| `train_val` | 103 | 1.50 s | 1.50 s | 0.03 | 2.97 |
| **`eval`** | 60 | **3.51 s** | **3.48 s** | **2.97** | 4.47 |
| `dev` | 5 | 3.27 s | 3.26 s | 3.10 | 3.43 |

The ranges are **disjoint** — every training positive has an alert-to-event
window under 2.97 s and every eval positive has one over it. A useful warning
must land inside that window, so eval's positives are systematically easier to
warn about than any training clip, and the two splits cannot be compared
directly. This also means part of the original 0.750 was never selection bias
at all: it was eval's wider anticipation windows.

`prepare_nexar.py::assign_splits` shuffles at random and cannot produce this, so
the likely cause is that `dev`/`eval` were frozen on 2026-08-18 from the 355-row
`index2.csv` under a different alert-time convention than the full 1,500-clip
Kaggle labels the training pool came from. **This is unresolved and out of scope
here** — this task was not to touch the eval split — but it undercuts both the
old and the new headline and should be the next thing investigated. Until it is,
treat `train_val` as a weak proxy for `eval`, and treat every eval number in
this project, old or new, as measured on an easier population than the one the
model trains on.

## What has *not* been redone

The four feature-ablation checkpoints (`ego`, `corridor`, `collision_prob`,
`ttc`) and the hidden-128 capacity run were selected the same biased way —
eval-validated training, eval-scored comparison. **Their absolute numbers carry
the same caveat as the old column above and have not been redone here.** Their
*relative* comparison is less affected, since every arm was biased the same way
by the same mechanism, so the ranking between arms is more trustworthy than any
individual figure. Redoing them under the clean protocol is a later step.

## Reproducing

See [`../handoff/REPRODUCE_BY_HAND.md`](../handoff/REPRODUCE_BY_HAND.md) §4–5,
updated to the clean recipe. The superseded command is marked there as
not-to-be-reproduced rather than deleted.

Artifacts: `eval/operating_point_sweep_k*_clean_val.md` (selection, on
`train_val`), `eval/final_eval_read_k1p0_clean.md` and
`eval/final_eval_read_k0p5_clean.md` (the two eval reads),
`eval/risk_gru_history_k*_clean*.json` (training curves).
