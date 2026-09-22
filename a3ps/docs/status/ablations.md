# Learned head — feature ablations and capacity (2026-09-19)

Two experiments that had run but had no write-up: four feature-group ablations
(`eval/train_log_abl_*.txt`) and a larger-capacity run (`eval/train_log_cap_h128.txt`).

> **⚠️ The four ablation results in the table directly below are INVALID — do
> not quote them.**
> `scripts/train_risk_head.py` zeroed the ablated columns on the *training* clips
> only. With a separate `--val-features` directory the validation clips are a
> different list and were left intact, so each model trained on ablated inputs and
> was scored on unablated ones. That is a train/eval shift, not a measurement of
> what the missing features contribute. The most visible symptom is `abl_ttc`:
> train loss fell normally (0.73 → 0.37) while validation mean AP stayed at 0.466,
> below chance, at every epoch.
>
> **Fixed** 2026-09-19 (validation is now ablated too) and guarded by
> `tests/test_train_risk_head.py::test_ablation_zeroes_val_features_too`, which
> fails on the old code. **The ablations have been re-run post-fix** — see
> "Post-fix rerun" below for the numbers to actually use.

## Ablations — what was run, and why the numbers cannot be used

Each run zeroes a named feature group (input width and parameter count unchanged,
34,465 parameters), same kappa 1.0, seed 1234 and data as the committed model.

| run | columns zeroed | mean AP reported | status |
|---|---|---|---|
| `abl_ego` | `ego_tx, ego_ty, ego_log_scale, ego_rot` (4) | 0.688 | invalid |
| `abl_corridor` | corridor distance / in-corridor (13) | 0.666 | invalid |
| `abl_collprob` | Phase III collision probability (6) | 0.664 | invalid |
| `abl_ttc` | looming TTC: `tau_area`, `tau_h`, `ttc_pred` (12) | 0.465 | invalid |

The AP column is kept only so nobody rediscovers these logs and quotes them. The
committed model's mean AP is 0.680 (full features).

Two further reasons the logged rows were never comparable with the headline even
before the bug: they are scored at the default threshold 0.5 (FA 0.42–0.48,
versus 0.167 at the committed operating point), and `abl_ttc`'s "best epoch" was
chosen by FA-aware selection from a run where every epoch was useless — an
FA-compliant epoch outranks a non-compliant one however low its useful-warning
rate, so it kept a model with useful-warning 0.017.

**How they were redone** (already done, 2026-09-19, using the command below —
about 4 min each, CPU, on the training features merged into one directory as in
`handoff/REPRODUCE_BY_HAND.md` §4):

```powershell
foreach ($g in "ego","corridor","collision_prob","ttc") {
  python scripts/train_risk_head.py `
      --features data/features/train_all --val-features data/features/eval `
      --kappa 1.0 --pre-alert-weight 0.5 --pos-weight 1.0 --fa-target 0.20 `
      --epochs 30 --patience 8 --batch-size 8 --seed 1234 --ablate $g `
      --out notebooks/models/risk_gru_abl_$g.pt `
      --history-json eval/risk_gru_history_abl_$g.json *>&1 `
      | Tee-Object eval/train_log_abl_$g.txt
}
```

Each retrained checkpoint was then swept to the committed operating point
(0.60 / 8) with `scripts/rerun_ablation_eval.py`, scoring on `data/features/eval`
(the frozen split, untouched) with the same columns zeroed at eval time as at
train time for that arm. Output: `eval/ablation_rerun_2026-09-22/`.

## Post-fix rerun — threshold 0.60 / confirm 8

Scored on the frozen eval split (`eval/split_freeze.json`, 120 clips), same
committed operating point as the headline model. Committed full-feature model
for reference (`eval/operating_point_sweep_k1p0.md`): useful-warning 0.750
(45/60), false-alarm 0.167, mean lead 1.67 s, mean AP 0.680.

| group | columns zeroed | useful-warning | false-alarm | mean lead (s) | mean AP | Δ mean AP vs 0.680 |
|---|---|---|---|---|---|---|
| `ego` | 4 | 0.683 (41/60) | 0.150 | 1.68 | 0.689 | +0.009 |
| `corridor` | 13 | 0.583 (35/60) | 0.100 | 1.35 | 0.654 | −0.026 |
| `collision_prob` | 6 | 0.717 (43/60) | 0.267 | 1.65 | 0.659 | −0.021 |
| `ttc` | 12 | 0.833 (50/60) | 0.383 | 1.95 | 0.650 | −0.030 |

**Caveats — read before quoting these:**

- **One seed, one checkpoint per arm.** Every Δ mean AP here (+0.009 to
  −0.030) falls within the ±0.022 noise level this same document assigns to
  the capacity experiment below (one seed, one checkpoint, picked by
  validation, on the same 120 clips). At that noise level these numbers show
  that **no single feature group is individually essential** — removing any
  one of them does not collapse the model — but they do **not** establish a
  ranked feature-importance ordering; a real ranking would need 2-3 more
  seeds per group before the differences could be trusted as distinguishing
  the arms from one another.
- **Useful-warning at a fixed threshold is not like-for-like across arms.**
  Threshold 0.60/confirm 8 was tuned for the full-feature model's probability
  calibration. Each ablated model has its own calibration, so its
  useful-warning/false-alarm numbers at that same fixed threshold reflect
  where its scores happen to fall relative to a threshold picked for a
  *different* model, not a controlled comparison of detection ability at a
  matched operating point. Mean AP (threshold-free, ranks by peak probability)
  is the more comparable number across arms; useful-warning/false-alarm at
  0.60/8 is included for reference against the headline operating point, not
  as a ranking signal.

## Capacity — `--hidden 128 --layers 2` (192,353 parameters)

The run was interrupted at epoch 26 of 30 (no final summary block; early stopping
had not fired). Its saved checkpoint is epoch 21. Swept with
`scripts/sweep_operating_point.py` → `eval/operating_point_sweep_h128.md`.

| model | mean AP | best row with FA ≤ 0.20 and useful ≥ 0.70 | useful | FA | mean lead |
|---|---|---|---|---|---|
| committed, hidden 64 (34,465 params) | 0.680 | thr 0.60 / confirm 8 | 0.750 | 0.167 | **1.67 s** |
| hidden 128 × 2 layers (192,353 params) | **0.702** | thr 0.70 / confirm 5 (FA exactly 0.200) | 0.750 | 0.200 | 1.59 s |

**Reading it.** Roughly 5.6× the parameters raised mean AP by +0.022 but did **not**
improve lead time at an acceptable false-alarm rate (1.59 s vs 1.67 s), so it does
not close the lead-time gap and does not replace the committed model. The +0.022
is one seed, one checkpoint picked by validation, on 120 clips — within the
epoch-to-epoch noise the training logs show (val useful-warning swings by ±0.2
between adjacent epochs), so treat it as "capacity helps little if at all", not as
a measured gain.

The high lead times in the raw log (2.1–3.5 s) come with FA 0.4–0.9: they reflect
firing on weak evidence, not anticipation. Lead time is only meaningful at a
compliant FA.

**Status of the lead-time target:** still open. Kappa (loss weighting) and capacity
have now both failed to move it past ~1.7 s at FA ≤ 0.20. The remaining
candidates are a longer feature window (needs GPU re-extraction) and better
features — see `../handoff/FUTURE_WORK.md`. "This feature set supports about
1.7 s of warning at an acceptable false-alarm rate" is a defensible finding to
report.

Not done: a clean completed re-run of the capacity model, and `--hidden 256`.
