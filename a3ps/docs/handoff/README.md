# Handoff — read this first

Written for continuing A3PS **by hand, without an assistant**. Everything needed
to reproduce the committed results, extend them, and write them up honestly.

For anything outside this folder, [`../INDEX.md`](../INDEX.md) is the master
index — it also flags which older docs describe the superseded threshold system,
which matters before quoting any number.

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
| checkpoint | `notebooks/models/risk_gru_k1p0.pt` (34,465 params, kappa 1.0, best epoch 8) |
| training data | 1,365 clips (685 pos / 680 neg), real ego motion on all |
| held-out eval | 120 clips (60/60), **frozen** — `eval/split_freeze.json` |
| **operating point** | **threshold 0.60, confirm 8** |
| useful-warning | 0.750 (45/60) |
| false-alarm | **0.167** ✅ (target ≤ 0.20) |
| mean lead | 1.67 s (target 2–6 s — **the known gap**) |
| vs old system | useful 0.050 → **0.750**, FA 0.767 → **0.167**, AP 0.546 → **0.680** |
| mean AP | 0.680 (0.785 / 0.697 / 0.557 @ 500 / 1000 / 1500 ms) |

## The three things most likely to trip you up

1. **Never re-run `prepare_nexar.py` without `--freeze`.** Split assignment
   depends on pool size, so it silently re-draws the held-out set and invalidates
   every number. `python scripts/freeze_split.py verify --index data/nexar/index.csv`
   must print `OK` before you trust anything. ([CHALLENGES.md](CHALLENGES.md) §1)

2. **`threshold` and `confirm` are free; `kappa` and `pre_alert_weight` are not.**
   The first two are applied after the model runs, so sweep them on an existing
   checkpoint in seconds. Only retrain when you've exhausted them.
   ([CHALLENGES.md](CHALLENGES.md) §14)

3. **Quote the pairing, not the best single number.** A useful-warning rate is
   meaningless without the false-alarm rate it was measured at — 0.833 is
   available on this checkpoint at FA 0.300, which is unusable. The honest result
   is **0.750 at FA 0.167**. False-alarm rate validates everything else.

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
