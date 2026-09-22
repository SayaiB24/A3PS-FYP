# Eval-set leakage audit (2026-09-22, read-only)

Concern: the documented retraining command uses `--val-features
data/features/eval` — the frozen 120-clip held-out split
(`eval/split_freeze.json`) — for early stopping and best-epoch selection.
If so, the headline 0.750 useful-warning / 0.167 false-alarm was partly
*selected* on the test set, not just reported on it, which would
contradict "never tune on eval." This document establishes what actually
happened from evidence (checkpoint `extra` dict, saved args, logs, docs,
git history, the scripts themselves), not from what the docs claim. No
retraining, re-extraction, or re-sweeping was run to produce this audit.

## 1. How was `risk_gru_k1p0.pt` trained? Which directory was validation?

**`data/features/eval` — the frozen held-out split itself.**

Evidence:
- `docs/handoff/REPRODUCE_BY_HAND.md:177-183`, the only documented
  reproduction command for this checkpoint:
  ```
  python scripts/train_risk_head.py `
      --features data/features/train_all --val-features data/features/eval `
      --kappa 1.0 --pre-alert-weight 0.5 --pos-weight 1.0 --fa-target 0.20 `
      --epochs 30 --patience 8 --batch-size 8 --seed 1234 `
      --out notebooks/models/risk_gru_k1p0.pt `
      --history-json eval/risk_gru_history_k1p0.json
  ```
- The checkpoint's own saved `extra` dict (read directly from
  `notebooks/models/risk_gru_k1p0.pt`, not from a doc) confirms this
  matches what actually happened:
  `{'best_epoch': 8, 'val': {'n_clips': 120, 'n_timed': 60, ...
  'mean_AP': 0.6797...}, 'leakage_warning': False}`. `n_clips: 120` is the
  exact size of the frozen eval split (`eval/split_freeze.json`: 60 pos +
  60 neg = 120), not a size consistent with any other directory in
  `data/features/` (`train_all` is 1,365 clips).
- `scripts/train_risk_head.py:253-351` (`train()`): every epoch calls
  `evaluate(model, val_clips, ...)` and uses the result to decide whether
  this epoch is the new best (`on_improve` saves those weights to disk)
  and whether `patience` has run out (early stop). `val_clips` is loaded
  from whatever `--val-features` points at
  (`scripts/train_risk_head.py:429`, `load_clips(args.val_features)`) —
  here, `data/features/eval`.
- `leakage_warning: False` in the checkpoint is **not** a clean bill of
  health for this concern — see §5: that flag only fires when
  `--features` and `--val-features` are the *same directory*
  (`train_risk_head.py:426-435`). Here they are different directories
  (`train_all` vs `eval`), so the guard is silent even though `eval` is
  still the frozen test set.

**Answer: yes, the frozen eval split was used directly as the validation
set for early stopping and best-epoch selection.**

## 2. The operating point 0.60/confirm 8 — which clips?

**`data/features/eval` again.**

- `scripts/sweep_operating_point.py:52-53`: `--features` defaults to
  `data/features/eval`, described in its own help text as "the held-out
  split."
- `eval/operating_point_sweep_k1p0.md:1-4`: "checkpoint:
  `risk_gru_k1p0.pt`... scored on: `data/features/eval` — 120 clips, 60
  positive." The 0.60/confirm-8 row is read directly from this sweep.
- `eval/kappa_comparison.md:3`: "scored on `data/features/eval` — 120
  clips, 60 positive (frozen held-out split)" — the same file that picks
  kappa 1.0 *and* the threshold/confirm operating point together ("kappa
  1.0 wins, at threshold 0.60/confirm 8").

**Answer: yes, the eval set.**

## 3. Kappa choice and "hidden 64 beats hidden 128×2" — which clips?

Both on `data/features/eval` as well.

- Kappa: `eval/kappa_comparison.md:3` (quoted above) — the four kappa
  values (0.5/1.0/2.0/3.0) are each swept on `data/features/eval`, and the
  decision rule ("maximise mean lead among rows passing all gates") is
  applied to those eval-scored rows directly.
- Capacity: `eval/operating_point_sweep_h128.md:1-5`: "checkpoint:
  `risk_gru_cap_h128.pt` (best epoch 21)... scored on: `data/features/eval`
  — 120 clips, 60 positive." Compared against the committed model's own
  eval-scored row in `docs/status/ablations.md`'s capacity table. The
  hidden-128 checkpoint's `best_epoch: 21` selection also came from a
  `--val-features data/features/eval` training run, per the same
  `train()` logic as §1 (no separate training command exists for this
  checkpoint that points elsewhere; `train_log_cap_h128.txt` has no header
  giving a different `--val-features` value than the standard command
  template).

**Answer: yes, the eval set, for both.**

## 4. Is there a validation split carved out of `train_all`? Does `splits.py` support one?

**No, on both counts.**

- `a3ps/common/splits.py:33`: `FROZEN_SPLITS = ("dev", "eval")`. There is
  a third, small frozen split called `dev` (`eval/split_freeze.json`:
  `{"dev": {"pos": 5, "neg": 10}}`, 15 clips total) — but:
  - No `data/features/dev/` directory exists on disk (`data/features/`
    contains only `cache_eval`, `eval`, `train_all`, `train_neg`,
    `train_pos` — confirmed by listing the directory directly).
  - `dev` is wired into the **legacy** threshold-engine tooling only
    (`scripts/eval_anticipation.py:879` and `scripts/run_ablations.py:178`,
    both `--split` choices for the old pipeline), not into
    `train_risk_head.py`, which has no `--split` argument and no
    awareness of `splits.py` at all — it just loads whatever directory
    `--features`/`--val-features` point to.
  - `train_traj` is the only split `splits.py`'s own docstring
    (`splits.py:16-18`) says is allowed to grow; `dev` and `eval` are
    frozen membership lists, not a mechanism for producing a *third*,
    fresh validation carve-out from `train_all`.
- Nothing in `train_risk_head.py` splits `--features` internally either —
  `load_clips(args.features)` (the training set) and
  `load_clips(args.val_features)` (validation) are two independently
  supplied directories; whatever the caller passes as `--val-features` is
  used whole, with no held-back fraction of `--features` ever spun off
  automatically.

**Answer: no clean validation split exists in the code today; `dev` is a
different, tiny, disk-absent-for-GRU-features split used only by the old
pipeline.**

## 5. Do the drift guards catch this, or only same-split-membership drift?

**Only membership/label drift — not eval-as-val-features.**

Two separate guard mechanisms exist, and both are read-only checks on
*something else*:

- `a3ps/common/splits.py::verify_freeze()` (`splits.py:201-235`): checks
  that a live clip index still agrees with `eval/split_freeze.json` on
  split membership and label per clip, and flags a clip that would
  "silently enlarge the held-out set." This guards against the held-out
  set's *contents drifting* across runs (the exact failure mode
  `splits.py`'s module docstring describes: re-running `prepare_nexar.py`
  after growing the pool re-shuffles and silently redraws the split). It
  says nothing about what directory a training run points `--val-features`
  at.
- `train_risk_head.py`'s `--allow-leakage` guard (`train_risk_head.py:
  426-435, 459-460, 500`): refuses to run when `--features` and
  `--val-features` resolve to the **identical directory**, unless
  `--allow-leakage` is passed, and then stamps `leakage_warning: true` in
  the checkpoint. This guards against training and validating on the
  exact same clips (a pure plumbing-check scenario, per its own printed
  message: "This is a PLUMBING CHECK, not a real result"). It does **not**
  check whether `--val-features` is the frozen eval split — `train_all`
  and `eval` are different directories, so this guard is silent for
  every run described in §1-3, and the checkpoint duly records
  `leakage_warning: False`.

**Answer: the guards only catch (a) eval-split membership drift across
data-prep runs, and (b) train==val identical-directory runs. Neither one
flags "validated on the frozen eval split while training," which is
exactly what happened here.**

## 6. Every decision that touched the eval set, classified

| decision | what touched eval | classification |
|---|---|---|
| Best-epoch / early-stopping for every GRU checkpoint (`risk_gru_k1p0.pt`, all four ablation checkpoints, `risk_gru_cap_h128.pt`) | `evaluate()` on `data/features/eval` every epoch, drives which weights get saved and when training stops | **(a) model selection** |
| Kappa choice (1.0 of 0.5/1.0/2.0/3.0) | each kappa's checkpoint swept on `data/features/eval`; kappa picked by the eval-scored rows | **(b) hyperparameter selection** |
| Operating point (threshold 0.60, confirm 8) | swept on `data/features/eval`; picked by maximising eval-scored mean lead among eval-scored gate-passing rows | **(b) hyperparameter/operating-point selection** |
| Capacity check (hidden 64 vs hidden 128×2) | both checkpoints' `best_epoch` selected via eval-validated training; compared via eval-scored sweeps | **(a) model selection** (for `best_epoch`) **and (b)** (for the capacity go/no-go call) |
| Post-fix feature ablations (ego/corridor/collision_prob/ttc) | same as capacity: eval-validated training, eval-scored comparison against 0.680 | **(a)** and **(b)** |
| Headline numbers reported (0.750 useful-warning, 0.167 FA, mean AP 0.680, mean lead 1.67s) | final read of already-selected checkpoint/operating-point on eval | **(c) pure reporting** — but only for *this last step*; everything upstream of it in this table is (a)/(b), which is what makes the final report's independence from eval false in practice, whatever this row's own classification is. |

Net: **every decision that shaped the final model and its operating
point — which epoch to keep, which kappa, which capacity, which feature
groups, which threshold/confirm — passed through eval-scored numbers.**
Nothing in this pipeline reports a number on eval that wasn't already
used to pick something.

## 7. Sketch of a clean protocol

**Validation split to carve from `train_all`.** `train_all` is 1,365
clips (685 pos / 680 neg, per `REPRODUCE_BY_HAND.md:21`), roughly balanced.
A held-out validation slice of **10-15%** (≈135-200 clips, stratified to
keep the ~50/50 pos/neg ratio) is in line with the eval split's own size
(120 clips) and leaves ≈1,165-1,230 clips for training — comparable to
what other projects use for a dataset this size, and large enough that
early-stopping/kappa/operating-point decisions aren't themselves noisy
from a tiny val set. If clip-to-source-video metadata exists (worth
checking `data/nexar/index.csv` / `prepare_nexar.py`'s output columns,
not confirmed in this read-only pass), the split should keep all clips
from one source video on the same side, the same way `dev`/`eval` are
frozen at the clip-id level today — otherwise near-duplicate frames from
one video could leak signal across the split.

**Scripts needing a flag or change:**
- `scripts/train_risk_head.py`: no change needed to the training loop
  itself — it already accepts any `--val-features` directory. The change
  is *procedural*: stop pointing `--val-features` at `data/features/eval`
  in the documented command, and point it at a new
  `data/features/train_val` (or similar) directory instead.
- A new split needs a name and a freeze entry. Either extend
  `a3ps/common/splits.py`'s `FROZEN_SPLITS` with a fourth split (e.g.
  `train_val`) so it gets the same membership/label drift protection
  `dev`/`eval` already have, or carve it deterministically inside
  `train_risk_head.py`/a small new script from `train_all` with a fixed
  seed and accept it isn't drift-protected the way `dev`/`eval` are (worse
  — recommend extending `splits.py` instead, since the whole point of that
  module is exactly this kind of silent-redraw risk).
- `scripts/extract_features.py` needs no change if the new split's clips
  are just a subset of clips already extracted into `train_all` — the
  split can be done at the clip-id/file level after extraction, no
  re-extraction required.
- `scripts/sweep_operating_point.py` and the kappa-comparison workflow
  (`docs/handoff/KAPPA_RETRAIN.md`) need their `--features` default (or
  documented command) changed from `data/features/eval` to the new
  validation directory for every *selection* step; `data/features/eval`
  should only be touched once, at the very end, to produce the number
  that gets reported.
- Add an explicit check (extending the existing `--allow-leakage` guard
  in `train_risk_head.py`, or a new one) that refuses `--val-features`
  pointing at a directory recorded as `eval` in `split_freeze.json`,
  unless a new `--i-am-doing-final-reporting-only` style flag is passed —
  closing exactly the gap §5 identifies.

**Rough time, CPU:**
- Carving the split: minutes (file listing + stratified sample + writing
  a freeze-style JSON), no retraining.
- One clean retrain of the headline config on the new
  `train_all_minus_val` / `train_val` split: same order as today's
  ~4 min/run (per `docs/status/ablations.md`'s own estimate for a single
  ablation arm) — likely 5-10 min given training-set size barely changes.
- Re-running the kappa sweep (4 values) and the operating-point sweep on
  the new validation directory: each retrain ~5-10 min → **4 kappa
  retrains ≈ 20-40 min**, plus sweep scoring itself is seconds per
  checkpoint (it's a forward pass + threshold grid, not training).
  Capacity (1 more retrain) and the four ablations (already have a
  rerun script, `scripts/rerun_ablation_eval.py`, that would need
  pointing at the new val directory) add roughly another 30-50 min
  combined.
- **Total: well under half a day of CPU time**, dominated by the kappa
  sweep's four retrains, not by any single step. The one true eval-split
  read at the end (final reported numbers) is seconds.

---

## Summary

| Step 3 question | Answer |
|---|---|
| 1. Trained with eval as validation? | **Yes** — checkpoint's own `extra` dict (`n_clips: 120`) and the documented reproduction command both confirm `--val-features data/features/eval`. |
| 2. Operating point (0.60/8) chosen on eval? | **Yes** — `sweep_operating_point.py` default and every operating-point doc score on `data/features/eval`. |
| 3. Kappa and capacity choices made on eval? | **Yes**, both. |
| 4. Clean val split carved from `train_all`, supported by `splits.py`? | **No** — only `dev` (15 clips, unused by the GRU path, no extracted features) and `eval` are frozen splits; nothing carves a third validation split from `train_all`. |
| 5. Do drift guards prevent this? | **No** — `verify_freeze()` only catches split-membership/label drift; `train_risk_head.py`'s `--allow-leakage` guard only catches train==val identical-directory runs, not "val is the frozen eval set." |

No retraining, re-extraction, or re-sweeping was performed to write this
document.
