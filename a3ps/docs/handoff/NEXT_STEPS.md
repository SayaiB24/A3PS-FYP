# Next steps — what to do from here

Ordered by priority. Each step states what it costs, what machine it needs, what
"done" looks like, and how to tell it went wrong.

> **Rewritten 2026-09-29.** The previous version of this file described the
> superseded v1 state (`risk_gru_k1p0.pt`, 0.750 at FA 0.167, selected on the
> test set) and its §1 training command validated on `data/features/eval`,
> the exact leak `../status/eval_leakage_audit.md` documents. That content is in
> git history. For the week before judging, [`JUDGES_PREP_PLAN.md`](JUDGES_PREP_PLAN.md)
> Part 3 is the day-by-day plan; this file is the order of work after it.

## Where the project stands

| quantity | value |
|---|---|
| checkpoint | `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt` (34,465 params, kappa 1.0, `pre_alert_weight` 0.5, seed 1234) |
| training / selection data | `train_core_v2` 1,160 clips (582 / 578), `train_val_v2` 205 (103 / 102) |
| held-out | `eval_v2`, 120 clips (60 / 60), frozen in `eval/split_freeze_v2.json` |
| operating point | threshold 0.70, confirm 8 (chosen on `train_val_v2`) |
| useful-warning | **0.650** (39/60, v2 episode-based) / 0.317 (19/60, legacy) |
| false-alarm | **0.167** (10/60) — target ≤ 0.20 met as a point estimate (95% CI ≈ 0.09–0.28) |
| mean lead | 1.51 s (target 2–6 s — **not met**) |
| mean AP | 0.646 (0.735 / 0.642 / 0.561 @ 0.5 / 1.0 / 1.5 s) |
| gross-premature | 0 |
| `eval_v2` reads | **one spent**, at most one more, ever |

Full story and error bars: [`../REPORT_evaluation_audit.md`](../REPORT_evaluation_audit.md).

## What is already done

| area | state |
|---|---|
| Split freeze + drift guards, held-out refusal in train/sweep | ✅ |
| Feature extraction, 1,485 clips, 10 Hz, ego motion on all | ✅ |
| v2 partition (window-stratified), fixed selector, v2 metric | ✅ |
| Single `eval_v2` read | ✅ 0.650 / 0.167 |
| Kappa sweep | ✅ no effect on AP (flat within 0.004) |
| Ensemble + temperature calibration | ✅ no gain beyond noise |
| Capacity (h128, h256) | ✅ no gain beyond noise |
| Seq2Seq-LSTM forecaster end to end | ✅ tie on coverage, 0.56 s less lead; Kalman stays |
| GPU latency re-timing | ✅ 61.9 ms/frame perception+tracking on RTX 3050 Laptop |
| `dashboard_v2` | ✅ committed; manual QA open |
| Feature ablations under the clean protocol | ⬜ `JUDGES_PREP_PLAN.md` §3.2 |
| Simple baselines (single feature, logistic regression) | ⬜ `JUDGES_PREP_PLAN.md` §3.2 |
| **Lead time ≥ 2 s** | ⬜ **open — step 1 below** |

---

## 1. Lead time: change the objective, not the model  ⭐ top priority

**Mean lead is 1.51 s against a 2–6 s target.** Five levers that act on the model
(kappa, ensemble, calibration, capacity, forecaster) left it where it is, and the
reason is structural: `alert_weights()` gives the loss **zero positive signal
outside `[alert_t, event_t]`**, and that window averages ~1.6 s on this dataset
(`REPORT_evaluation_audit.md` §5.3). The model is never asked to warn earlier.

Next candidate: an objective that rewards warning before `alert_t` (a decaying
positive ramp before it, or event-time-keyed exponential weighting). Details and
risks in [`FUTURE_WORK.md`](FUTURE_WORK.md) Tier 1.

**Protocol, unchanged:** train on `train_core_v2`, select on `train_val_v2`,
compare with `scripts/rescore_candidates_both_defs.py` (both definitions, same
selection rule). Never compare against the `useful` column of an
`operating_point_sweep_*.md`, which is the legacy definition.

**Done when:** a candidate reaches v2 lead ≥ 2 s on `train_val_v2` at FA ≤ 0.20
with zero gross-premature and `frac_premature` no worse than the baseline's, and
holds up across at least three seeds. **Went wrong if:** lead rises but
`frac_premature` or `n_gross_premature` rises with it. That is the old threshold
engine's failure mode coming back.

---

## 2. Get an external number: Kaggle submission

**~9 h GPU extraction + ~2 h.** The only way to get a new held-out number without
spending the last `eval_v2` read, and the only number nobody could accuse us of
mis-scoring. It also answers "how do you compare with published work?" See
[`FUTURE_WORK.md`](FUTURE_WORK.md) Tier 1.

---

## 3. Decide how to spend the last `eval_v2` read

**Minutes to decide; do it in writing, before running anything.** Two legitimate
uses, pick at most one:

- a genuinely better single config from step 1, locked on `train_val_v2` first, or
- a multi-seed report of the **same locked recipe** (mean ± sd over 5 seeds). That
  replaces a single draw with a distribution, which the paper promises.

`final_eval_read.py` refuses to overwrite its own output. Don't work around it.

---

## 4. Make the old-vs-new comparison strict

**~1 h GPU + minutes CPU.** The threshold engine has only been scored on the
superseded split, on full clips at 30 Hz. Re-run it on the same 13 s windows as the
learned head. On `train_val_v2` this costs no read budget. Until then the paper's
"No comparable baseline yet" limitation stands, and `eval/phase4_comparison.md`
(0.050 → 0.750) **must not be quoted**: it is on the v1 split with the
selection-biased model.

---

## 5. Optional improvements

Everything else is in [`FUTURE_WORK.md`](FUTURE_WORK.md), tiered: per-actor head,
feature attribution, seed-variance reduction, real-time path, ReID, BEV
calibration, cross-dataset evaluation, CI.

---

## 6. Housekeeping

- **Do not re-run `prepare_nexar.py` without `--freeze`.** It re-draws the held-out split and silently invalidates every number.
- **Do not delete `data/nexar/videos/`.** It holds the held-out positives.
- **`data/features/`** is gitignored. Back it up. `features_full.zip` in the repo root (2026-08-19) holds all 1,485 Kalman-feature clips in the v1 directory layout (`train_pos`, `train_neg`, `eval`). The v2 directories are the same files regrouped per `eval/split_freeze_v2.json` (no re-extraction; see `../status/repartition_results.md` §3). It does **not** include the `_seq2seq` features (~10 GPU-hours to regenerate).
- **Demo `raw.mp4` files are gitignored.** Re-run `build_dashboard_demo.py` after a fresh clone, **with** `--checkpoint notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt --features data/features/train_val_v2 --threshold 0.70 --confirm 8`. The script's defaults still point at the superseded v1 model and pool (`JUDGES_PREP_PLAN.md` §1.12).
- **Both split freezes (`eval/split_freeze.json`, `eval/split_freeze_v2.json`) are committed and must stay that way.**
- **`REPRODUCE_BY_HAND.md` stops at the v1 clean protocol.** Bringing it up to the v2 partition (repartition → train with the fixed selector → one-shot read) is owed; the banner at its top points to the status docs that hold the v2 commands meanwhile.

---

## Where things live

| what | path |
|---|---|
| current result, full story | [`../REPORT_evaluation_audit.md`](../REPORT_evaluation_audit.md) |
| the held-out read | [`../status/final_eval_read.md`](../status/final_eval_read.md), `eval/final_eval_read_v2_s1234.json` |
| what was tried to improve it | [`../status/headline_number_attempts.md`](../status/headline_number_attempts.md), `eval/candidates_both_definitions.md` |
| trained checkpoint | `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt` |
| training history | `eval/risk_gru_history_k1p0_v2_selfix_s1234.json` |
| operating-point grid (v2 definition) | `figures/fig_operating_point_data.json` |
| clean demo clips | 621, 206, 630, 932, 1426, 1564 ([`../status/instability_and_checkpoint_audit.md`](../status/instability_and_checkpoint_audit.md) §3) |
| problems solved, for the thesis | [`CHALLENGES.md`](CHALLENGES.md) |
| optional future work | [`FUTURE_WORK.md`](FUTURE_WORK.md) |
