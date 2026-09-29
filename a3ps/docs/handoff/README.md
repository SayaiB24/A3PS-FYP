# Handoff — read this first

Written for continuing A3PS **by hand, without an assistant**. Everything needed
to reproduce the committed results, extend them, and write them up honestly.

For anything outside this folder, [`../INDEX.md`](../INDEX.md) is the master
index — it also flags which older docs describe the superseded threshold system,
which matters before quoting any number.

## Start here

| doc | read it when |
|---|---|
| **[JUDGES_PREP_PLAN.md](JUDGES_PREP_PLAN.md)** | The week before judging: day-by-day plan, demo clips, Q&A prep. |
| **[NEXT_STEPS.md](NEXT_STEPS.md)** | You want to know what to do after that. Priority-ordered, with cost and "done when" for each step. |
| **[../REPORT_evaluation_audit.md](../REPORT_evaluation_audit.md)** | You need the current result and why it can be trusted, in one document. |
| **[REPRODUCE_BY_HAND.md](REPRODUCE_BY_HAND.md)** | You want to re-run a result. ⚠️ Covers the v1 partition up to the clean protocol; its banner points to the v2 steps. |
| **[CHALLENGES.md](CHALLENGES.md)** | You're writing the thesis, or wondering why something is built the way it is. Problems with symptom → diagnosis → fix → what generalises. |
| **[FUTURE_WORK.md](FUTURE_WORK.md)** | You have time for optional improvements. Tiered by value per effort. |
| [KAPPA_RETRAIN.md](KAPPA_RETRAIN.md) | History only. The kappa sweep is done and did not move mean AP. |

## Where the project stands

| quantity | value |
|---|---|
| checkpoint | `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt` (34,465 params, kappa 1.0, seed 1234) |
| training data | `train_core_v2`, 1,160 clips (582 pos / 578 neg); selection on `train_val_v2`, 205 clips |
| held-out eval | `eval_v2`, 120 clips (60/60), **frozen** — `eval/split_freeze_v2.json`, read once |
| **operating point** | **threshold 0.70, confirm 8** |
| useful-warning | **0.650** (39/60, v2 episode-based) / 0.317 (19/60, legacy first-crossing) |
| false-alarm | **0.167** (10/60) — target ≤ 0.20 met as a point estimate |
| mean lead | 1.51 s (target 2–6 s — **not met**) |
| mean AP | 0.646 (0.735 / 0.642 / 0.561 @ 0.5 / 1.0 / 1.5 s) |

The earlier 0.750 headline was selected on the test set and is superseded; see
the report above before quoting any number from an older doc.

## The things most likely to trip you up

1. **Never re-run `prepare_nexar.py` without `--freeze`.** Split assignment
   depends on pool size, so it silently re-draws the held-out set and invalidates
   every number. ([CHALLENGES.md](CHALLENGES.md) §1)

2. **`eval_v2` has at most one read left, ever.** Select on `train_val_v2`
   only. Reading the test set repeatedly across candidates is how the old 0.750
   came about.

3. **`threshold` and `confirm` are free; `kappa` and `pre_alert_weight` are not.**
   The first two are applied after the model runs, so sweep them on an existing
   checkpoint in seconds. ([CHALLENGES.md](CHALLENGES.md) §14)

4. **Two useful-warning definitions exist; don't mix them.** The headline uses v2
   (episode-based). `sweep_operating_point.py` prints legacy only. Compare runs
   with `scripts/rescore_candidates_both_defs.py`.

5. **Quote the pairing, not the best single number.** A useful-warning rate is
   meaningless without the false-alarm rate it was measured at: 0.942 is
   available on this checkpoint at FA 0.667 on validation. The honest result is
   **0.650 at FA 0.167**.

## Launching the dashboard

```powershell
python scripts/serve_dashboard_v2.py --port 8010      # results + replay dashboard
```

Open <http://localhost:8010>. **Demo only these clips: 621, 206, 630, 932, 1426,
1564** (all `train_val_v2`, never trained on). The other "Phase IV" clips in the
dropdown (488, 1004, 690, 1085, 1261, 1054, 1042) were in the model's training
set ([`../status/instability_and_checkpoint_audit.md`](../status/instability_and_checkpoint_audit.md) §3).

If the demo clips are missing (fresh clone; `raw.mp4` is gitignored), rebuild
them with explicit flags. The script's defaults still point at the superseded
model and pool:

```powershell
python scripts/build_dashboard_demo.py `
    --checkpoint notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt `
    --features data/features/train_val_v2 --threshold 0.70 --confirm 8 `
    --clips auto --n 6
python scripts/update_manifest.py --require-video
```

The legacy dashboard (`python scripts/serve_dashboard.py --port 8000`) still
works and shows the learned head against the old threshold system on the same
clip.
