# Handoff — read this first

Written for continuing A3PS **by hand, without an assistant**. Everything needed
to reproduce the committed results, extend them, and write them up honestly.

## Start here

| doc | read it when |
|---|---|
| **[NEXT_STEPS.md](NEXT_STEPS.md)** | You want to know what to do next. Priority-ordered, with cost and "done when" for each step. |
| **[REPRODUCE_BY_HAND.md](REPRODUCE_BY_HAND.md)** | You want to re-run any result. End-to-end: raw clips → features → training → operating point → dashboard, with every command and config value. |
| **[KAPPA_RETRAIN.md](KAPPA_RETRAIN.md)** | You're ready to do the top-priority improvement (fixing lead time). Standalone runbook — reasoning, commands, decision rule, failure modes. |
| **[CHALLENGES.md](CHALLENGES.md)** | You're writing the thesis, or wondering why something is built the way it is. 19 problems with symptom → diagnosis → fix → what generalises. |
| **[FUTURE_WORK.md](FUTURE_WORK.md)** | You have time for optional improvements. Tiered by value per effort. |

## Where the project stands

| quantity | value |
|---|---|
| checkpoint | `notebooks/models/risk_gru_v1.pt` (34,465 params, best epoch 5) |
| training data | 1,365 clips (685 pos / 680 neg), real ego motion on all |
| held-out eval | 120 clips (60/60), **frozen** — `eval/split_freeze.json` |
| **operating point** | **threshold 0.70, confirm 5** |
| useful-warning | 0.717 (43/60) |
| false-alarm | **0.167** ✅ (target ≤ 0.20) |
| mean lead | 1.58 s (target 2–6 s — **the known gap**) |
| mean AP | 0.693 (0.795 / 0.708 / 0.575 @ 500 / 1000 / 1500 ms) |

## The three things most likely to trip you up

1. **Never re-run `prepare_nexar.py` without `--freeze`.** Split assignment
   depends on pool size, so it silently re-draws the held-out set and invalidates
   every number. `python scripts/freeze_split.py verify --index data/nexar/index.csv`
   must print `OK` before you trust anything. ([CHALLENGES.md](CHALLENGES.md) §1)

2. **`threshold` and `confirm` are free; `kappa` and `pre_alert_weight` are not.**
   The first two are applied after the model runs, so sweep them on an existing
   checkpoint in seconds. Only retrain when you've exhausted them.
   ([CHALLENGES.md](CHALLENGES.md) §14)

3. **Quote the pairing, not the best single number.** useful-warning 0.833 was
   measured at FA 0.583, which is unusable. The honest result is **0.717 at FA
   0.167**. False-alarm rate validates everything else.

## Launching the dashboard

```powershell
python scripts/build_dashboard_demo.py --clips auto --n 6   # if demo clips are missing
python scripts/serve_dashboard.py --port 8000
```

Open <http://localhost:8000>. Demo clips (621, 488, 1004, 690, 1085, 1261) are
listed first. The **MODEL COMPARISON** panel shows the learned head against the
old threshold system on the same clip, with Nexar's `time_of_alert` and
`time_of_event` marked — so the improvement is visible, not asserted. Details in
[REPRODUCE_BY_HAND.md](REPRODUCE_BY_HAND.md) §6.
