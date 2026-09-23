# Step 10.6: ensemble + calibration (2026-09-23)

`data/features/eval_v2` was **not read**. Everything below is scored on
`data/features/train_val_v2` (205 clips, 103 positive), v2 definition, the same
threshold {0.5..0.9} x confirm {3,5,8} grid as every prior sweep. Nothing was
selected against eval_v2 and no operating point is committed to here.

## Result in one paragraph

**Neither the ensemble nor calibration beats the seed-1234 baseline by any
amount that means anything.** Best gated cell per candidate: baseline 0.612,
ensemble 0.631 (+0.019 = 2 clips), calibrated ensemble 0.592 (-0.019). Paired
bootstrap CIs on the difference straddle zero, and the seed-to-seed sd is
0.10-0.12. **Recommendation: carry the plain seed-1234 baseline (thr 0.70 /
confirm 8) into Step 11.** It is still a near-miss against the 0.70 floor (gap
0.088), unchanged.

## 1. The comparison table (the required one)

"Own best operating point" = highest `useful_warning_rate_v2` among grid cells
passing FA <= 0.20, lead_vs_event_v2 >= 1.0 s and n_gross_premature == 0 (the
same reduction as `operating_point_selection.md` "How close it got"). The 0.70
floor is *not* applied, since nothing meets it.

| candidate | best cell | useful_v2 | FA | lead_vs_event_v2 | mean AP | n_gross_premature | paired diff vs baseline (95% CI) |
|---|---|---|---|---|---|---|---|
| **Baseline: seed 1234 alone** | thr 0.70 / c8 | **0.612** | 0.196 | 1.83 s | 0.630 | 0 | - |
| Ensemble (5 seeds, uncalibrated) | thr 0.70 / c5 | 0.631 | 0.196 | 1.39 s | 0.611 | 0 | +0.019 [-0.039, +0.078] |
| Ensemble, temperature-calibrated (primary calibrated candidate) | thr 0.70 / c8 | 0.592 | 0.176 | 1.53 s | 0.611 | 0 | -0.019 [-0.068, +0.029] |
| *(secondary)* mean of individually calibrated members | thr 0.70 / c5 | 0.631 | 0.196 | 1.38 s | 0.613 | 0 | +0.019 [-0.039, +0.078] |
| *(check)* ensemble, temperature fit 5-fold cross-validated | thr 0.70 / c8 | 0.602 | 0.186 | 1.53 s | 0.614 | 0 | -0.010 [-0.049, +0.029] |

Reading it honestly:

- The ensemble's "win" is 2 positive clips out of 103, with lead time 0.44 s
  *shorter* and mean AP 0.019 *lower* than the baseline. That is a wash, not an
  improvement.
- Taking the max over 15 cells on the same set everything was tuned on is
  optimistic for every row, the baseline included. The comparison is fair; the
  absolute numbers are upper-biased.
- 0.612 -> 0.631 is about one fifth of the seed-to-seed sd.
- Calibration is fit in-sample on train_val_v2 (as specified). The CV row shows
  that costs 0.010 of apparent useful_v2 and does not change the conclusion.
- The optimal threshold did **not** change: 0.70 in every row. Only the confirm
  count moved (5 vs 8).

## 2. The calibration hypothesis, tested directly

`instability_and_checkpoint_audit.md` proposed that the wide useful_v2 spread at
a shared cell is a probability-scale mismatch. Temperature scaling (`z -> z/T`,
one scalar per model, fit by minimising the anticipation loss the head was
trained with; monotone, so mean AP is invariant) tests that. Fitted T:

| model | 1234 | 1235 | 1236 | 1237 | 1238 (valid) | ensemble |
|---|---|---|---|---|---|---|
| T | 0.751 | 1.265 | 1.066 | 1.356 | 1.141 | 0.857 |

At the shared cell thr 0.70 / c8 (useful_v2 / FA):

| seed | uncalibrated | calibrated |
|---|---|---|
| 1234 | 0.612 / 0.196 | 0.670 / 0.275 |
| 1235 | 0.495 / 0.137 | 0.485 / 0.108 |
| 1236 | 0.864 / 0.431 | 0.854 / 0.431 |
| 1237 | 0.388 / 0.078 | 0.252 / 0.049 |
| 1238 (valid) | 0.612 / 0.196 | 0.592 / 0.176 |
| **sd across seeds** | **0.159** | **0.200** |

**The hypothesis is not supported as the main explanation.** Temperature scaling
did not narrow the spread (sd rose 0.159 -> 0.200). It mostly moved each model
along its own useful/FA trade-off (1234: +0.058 useful but FA 0.196 -> 0.275)
rather than lining the models up. The fitted Ts all sit within 0.75-1.36, a mild
scale difference, and seed 1236 still fires at FA 0.43 at this cell after
calibration. The seeds seem to differ in where in time and on which clips they
fire, not just in global scale, so one temperature cannot reconcile them. Platt
scaling (adds a bias) was not tried; a bias shifts the same threshold axis, so I
would not expect it to change this, but that is an expectation, not a result.

## 3. Recommendation

Stay with **seed 1234 at thr 0.70 / confirm 8**. Nothing here beats it by more
than noise; the ensemble gives up 0.44 s of lead and 0.019 of AP for two clips;
the calibrated ensemble is numerically below the baseline. This is a clean
negative result for the writeup: "5-seed probability averaging and temperature
scaling did not improve on the best single seed on train_val_v2 (useful_v2
0.631 / 0.592 vs 0.612, both within paired-bootstrap noise)."

For the writeup rather than as a reason to switch: the ensemble is smoother
across cells (thr 0.70/c8 gives useful 0.553 at FA 0.127, where the baseline
needs thr 0.80/c5 for 0.495 at FA 0.088) and would not hinge on one seed. That
is a robustness argument, not a performance one, and costs 5x inference. Whether
it matters for Step 11 is your call; the numbers do not push toward it.

## 4. Deviations and things to know

- **Seed 1238 could not be "retrained past" epoch 1 by simply re-running it.**
  The corrected check only *flags* a degenerate pick, it does not reselect, and
  training is bit-for-bit deterministic, so an identical rerun would pick the
  same epoch-1 checkpoint. I added an **opt-in** `--skip-near-chance-epochs`
  flag to `scripts/train_risk_head.py` (11 lines, default off, existing
  selection unchanged; 23/23 tests in `tests/test_train_risk_head.py` pass)
  that makes near-chance epochs ineligible. Same hyperparameters and seed as the
  original 1238 run, saved to a **new** file:
  `notebooks/models/risk_gru_k1p0_v2_selfix_s1238_valid.pt` (best epoch 7, mean
  AP 0.615, `degenerate_checkpoint: False`); history in
  `eval/risk_gru_history_k1p0_v2_selfix_s1238_valid.json`. That selection rule
  is my judgment call. This member's own useful_v2 at thr 0.70/c8 is 0.612 (FA
  0.196), the same as seed 1234.
- The other four members are the existing checkpoints, unmodified.
- The "primary calibrated candidate" (temperature fit on the ensemble's own
  logit) was designated in the script docstring before results were seen; the
  member-averaged and CV variants are secondary.
- All members and the calibration fit share train_val_v2, which was also the
  early-stopping/selection set for every member, so members are not independent
  of the scoring set. This favours all candidates equally.

## 5. Nothing lost, nothing overwritten

- `git diff --name-status HEAD` lists exactly one changed tracked file:
  `scripts/train_risk_head.py` (+11 lines, opt-in flag). Every other tracked
  file, including all 8 `risk_gru_k1p0_v2_selfix*.pt` checkpoints, every
  `eval/*` artifact and every `docs/status/*` doc, is identical to HEAD.
- **The seed-1234 baseline checkpoint
  `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt` is on disk, tracked and
  unmodified.** The original (degenerate) `..._s1238.pt` is also untouched.
- `scripts/sweep_operating_point.py` was not touched. New files:
  `scripts/sweep_ensemble.py`, `eval/ensemble_calibration/`
  (`ensemble_calibration_grids.{md,json}`, `comparison.json`), the `_valid`
  checkpoint and history, and this doc. Nothing is committed yet.

## 6. Full grids

The complete per-cell grids (ensemble, calibrated ensemble, member-averaged,
CV-calibrated, and every single seed, each with the delta against seed 1234
cell by cell) are in
[eval/ensemble_calibration/ensemble_calibration_grids.md](../../eval/ensemble_calibration/ensemble_calibration_grids.md)
(machine-readable: `ensemble_calibration_grids.json`). Both required grids are
reproduced below.

### Ensemble (uncalibrated)

| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature | n_gross_premature | mean AP |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.854 | 88 | 0.529 | 2.16 | 0.524 | 0 | 0.611 |
| 0.60 | 3 | 0.709 | 73 | 0.373 | 1.84 | 0.447 | 0 | 0.611 |
| 0.70 | 3 | 0.631 | 65 | 0.255 | 1.40 | 0.311 | 0 | 0.611 |
| 0.80 | 3 | 0.476 | 49 | 0.127 | 1.10 | 0.155 | 0 | 0.611 |
| 0.90 | 3 | 0.243 | 25 | 0.029 | 0.81 | 0.068 | 0 | 0.611 |
| 0.50 | 5 | 0.835 | 86 | 0.500 | 2.17 | 0.524 | 0 | 0.611 |
| 0.60 | 5 | 0.699 | 72 | 0.333 | 1.86 | 0.427 | 0 | 0.611 |
| 0.70 | 5 | 0.631 | 65 | 0.196 | 1.39 | 0.301 | 0 | 0.611 |
| 0.80 | 5 | 0.427 | 44 | 0.108 | 1.10 | 0.126 | 0 | 0.611 |
| 0.90 | 5 | 0.214 | 22 | 0.020 | 0.87 | 0.049 | 0 | 0.611 |
| 0.50 | 8 | 0.806 | 83 | 0.412 | 2.21 | 0.524 | 0 | 0.611 |
| 0.60 | 8 | 0.680 | 70 | 0.284 | 1.86 | 0.417 | 0 | 0.611 |
| 0.70 | 8 | 0.553 | 57 | 0.127 | 1.47 | 0.282 | 0 | 0.611 |
| 0.80 | 8 | 0.408 | 42 | 0.088 | 1.14 | 0.117 | 0 | 0.611 |
| 0.90 | 8 | 0.155 | 16 | 0.010 | 0.96 | 0.029 | 0 | 0.611 |

### Ensemble (temperature-calibrated, T = 0.857)

| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature | n_gross_premature | mean AP |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.854 | 88 | 0.529 | 2.16 | 0.524 | 0 | 0.611 |
| 0.60 | 3 | 0.738 | 76 | 0.402 | 1.82 | 0.476 | 0 | 0.611 |
| 0.70 | 3 | 0.650 | 67 | 0.275 | 1.47 | 0.359 | 0 | 0.611 |
| 0.80 | 3 | 0.563 | 58 | 0.167 | 1.15 | 0.214 | 0 | 0.611 |
| 0.90 | 3 | 0.320 | 33 | 0.049 | 0.88 | 0.078 | 0 | 0.611 |
| 0.50 | 5 | 0.835 | 86 | 0.500 | 2.17 | 0.524 | 0 | 0.611 |
| 0.60 | 5 | 0.718 | 74 | 0.333 | 1.86 | 0.466 | 0 | 0.611 |
| 0.70 | 5 | 0.641 | 66 | 0.235 | 1.48 | 0.340 | 0 | 0.611 |
| 0.80 | 5 | 0.515 | 53 | 0.127 | 1.22 | 0.214 | 0 | 0.611 |
| 0.90 | 5 | 0.291 | 30 | 0.039 | 0.92 | 0.078 | 0 | 0.611 |
| 0.50 | 8 | 0.806 | 83 | 0.412 | 2.21 | 0.524 | 0 | 0.611 |
| 0.60 | 8 | 0.689 | 71 | 0.284 | 1.87 | 0.447 | 0 | 0.611 |
| 0.70 | 8 | 0.592 | 61 | 0.176 | 1.53 | 0.320 | 0 | 0.611 |
| 0.80 | 8 | 0.476 | 49 | 0.108 | 1.28 | 0.184 | 0 | 0.611 |
| 0.90 | 8 | 0.243 | 25 | 0.029 | 0.96 | 0.068 | 0 | 0.611 |

### Baseline for reference (seed 1234 alone, unchanged from `operating_point_selection.md`)

Re-scored by the new script and reproduced exactly (0.70/c8: useful_v2 0.612,
FA 0.196, lead 1.83, 0 gross premature), which validates the harness.
