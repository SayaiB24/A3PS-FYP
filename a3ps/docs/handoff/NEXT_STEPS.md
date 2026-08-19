# Next steps — what to do from here

Ordered by priority. Each step states what it costs, what machine it needs, what
"done" looks like, and how to tell it went wrong.

**Current state (all committed):** the learned risk head is trained on 1,365
clips and validated on the frozen 120-clip eval split, with a chosen operating
point and a dashboard that shows the improvement against the old threshold
system. Reproduce any of it from
[`REPRODUCE_BY_HAND.md`](REPRODUCE_BY_HAND.md).

| where you are | value |
|---|---|
| checkpoint | `notebooks/models/risk_gru_v1.pt` (best epoch 5) |
| operating point | threshold 0.70, confirm 5 |
| useful-warning | 0.717 (43/60) |
| false-alarm | 0.167 ✅ (target ≤ 0.20) |
| mean lead | 1.58 s (target 2–6 s — **the gap**) |
| mean AP | 0.693 (0.795 / 0.708 / 0.575 @ 500/1000/1500 ms) |

## What is already done

Nothing below needs redoing. Listed so you can tell at a glance what is settled
and what is not; the per-item detail is in
[`../status/TODO.md`](../status/TODO.md).

| area | state |
|---|---|
| Split freeze + drift guards | ✅ done — `eval/split_freeze.json`, 135 clips pinned |
| Dataset indexed (1,500 labelled clips) | ✅ done — 685 pos / 680 neg train, 60+60 frozen eval |
| Feature extraction (the only GPU step) | ✅ done — 1,485 clips, 10 Hz, 0 failures, ego motion on all |
| Learned head trained + checkpointed | ✅ done — 34,465 params, best epoch 5 |
| Operating point chosen | ✅ done — 0.70/5, FA target met |
| Metric bugs fixed (AP ties, invented window) | ✅ done — corrected mean AP 0.546; keyed to `alert_time_s` |
| Nexar cutoff APs (500/1000/1500 ms) | ✅ done |
| Dashboard old-vs-new comparison | ✅ done — 6 demo clips, overlays, grouped dropdown |
| Handoff documentation | ✅ done — this folder |
| **Lead time ≥ 2 s** | ⬜ **open — step 1 below** |
| Paper tables against the learned head | ⬜ open — step 2 |
| Batched training / FA-aware selection | ⬜ open — step 4, deliberately deferred |

---

## 1. Kappa retrain — fix lead time  ⭐ top priority

**~1 h 5 min · CPU · no re-extraction.**

Lead time (1.58 s) is the only target the model misses, and the cause is
diagnosed: `kappa=3.0` asks the loss for a warning only ~0.98 s before impact, so
it trained a detector rather than an anticipator.

Full standalone runbook: **[`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md)** — reasoning,
the kappa-vs-lead table, exact commands, what must stay fixed, the decision rule,
and the failure modes.

**Done when:** you have a winner chosen by the documented rule (highest mean lead
subject to FA ≤ 0.20 and useful ≥ 0.70), or a written finding that no kappa beats
1.58 s at compliant FA — which is itself a legitimate result, not a failure.

---

## 2. Regenerate the paper tables against the learned head

**~30 min · CPU.**

`eval/anticipation.md` and the other paper artefacts still describe the **old
threshold system**. They need to reflect the learned head before any of it goes
in the thesis.

```powershell
# re-score the frozen eval split (uses cached clips; no GPU)
python scripts/eval_anticipation.py --index eval/nexar_index_gpu.csv `
    --split eval --clips-dir eval/anticipation

# collect everything into the paper summary
python scripts/collect_paper_stats.py --out eval/paper_stats_summary.md
```

**Watch for:** `eval_anticipation.py` refuses to run if the split has drifted
from the freeze. That guard is correct — fix the index, do not bypass it with
`--allow-split-drift` unless you intend non-comparable numbers.

**Note the honest framing** the learned head requires: it is scored on 13 s
windows at 10 Hz, whereas the old system was scored on full clips at 30 Hz. Those
are not the same observation conditions. State it, or re-run the old system on
the same windows for a strict comparison.

**Done when:** every table you intend to publish names which system produced it
and under what observation window.

---

## 3. Report the numbers honestly — what to claim and what not to

**No compute; a writing step, but the most important one.**

**Defensible claims:**
- The learned head meets the false-alarm target (0.167) at 0.717 useful-warning on a frozen held-out split, keyed to Nexar's own `time_of_alert`.
- On the demo clips, the old threshold system fires 18–24 s premature with 5–30 events per clip; the learned head fires inside the actionable window. This is a real recorded contrast, not a re-simulation.
- Positives include near-misses by dataset definition, so the task is risky-event vs ordinary-driving.

**Do not claim:**
- That lead time meets the 2–6 s target. It does not (1.58 s). Say so.
- That useful-warning is 0.833. That was at threshold 0.5 where FA is 0.583 — unusable. The honest pairing is 0.717 @ FA 0.167.
- Any latency figure without naming the machine. CPU and GPU differ ~20–40× here.
- Anything from a run whose FA is out of range. FA validates the rest; see [`../design/metrics.md`](../design/metrics.md).

---

## 4. Fix the batching inefficiency

**~1–2 h to implement and verify · CPU.**

`--batch-size` does not batch the compute: `[model(c["X"]) for c in batch]` runs
one forward per clip, so effective batch size is 1 and training gets ~1.85×
parallelism on 8 cores. Full description and the fix approach in
[`REPRODUCE_BY_HAND.md`](REPRODUCE_BY_HAND.md) §8.

Worth doing **before** any large hyperparameter search, since ~5× compounds
across every run. Not urgent if you only plan the kappa sweep.

**Critical:** verify the batched loss matches the unbatched one on a fixed seed
before trusting any number. A padding-mask bug would change the objective
silently rather than crash.

---

## 5. Optional improvements

Only after 1–3. Details in [`FUTURE_WORK.md`](FUTURE_WORK.md).

| idea | cost | why |
|---|---|---|
| More capacity (`--hidden 128 --layers 2`) | ~1 h | 34k params may be underfitting; AP 0.693 is the ceiling to move |
| Longer feature window (20 s) | GPU re-extraction ~4 h | more run-up context for earlier warnings |
| Kaggle submission on the 1,344 unlabelled test clips | ~2 h | external unbiased AP for the thesis |
| ReID appearance arm | ~1 day | deferred from v1; needs `onnxruntime`, changes association |
| Per-actor temporal head | ~2 days | current head pools over actors, so it cannot attribute risk to one |

---

## 6. Housekeeping

- **Do not re-run `prepare_nexar.py` without `--freeze`.** It re-draws the held-out split and silently invalidates every number.
- **Do not delete `data/nexar/videos/`** — it holds all 60 held-out eval positives.
- **`data/features/`** is gitignored. Back it up: it is ~32 MB and cost ~3.5 h of GPU to produce. `features_full.zip` in the repo root is that backup.
- **Demo `raw.mp4` files are gitignored** — re-run `build_dashboard_demo.py` after a fresh clone to recreate them.
- **The eval split freeze (`eval/split_freeze.json`) is committed and must stay that way.** It is what makes every number in this project comparable to every other.

---

## Where things live

| what | path |
|---|---|
| how to reproduce everything | [`REPRODUCE_BY_HAND.md`](REPRODUCE_BY_HAND.md) |
| kappa sweep runbook | [`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md) |
| problems solved, for the thesis | [`CHALLENGES.md`](CHALLENGES.md) |
| optional future work | [`FUTURE_WORK.md`](FUTURE_WORK.md) |
| metric definitions and how to read them | [`../design/metrics.md`](../design/metrics.md) |
| trained checkpoint | `notebooks/models/risk_gru_v1.pt` |
| training log / curve | `eval/train_log.txt`, `eval/risk_gru_history.json` |
| operating-point grid | `eval/operating_point_sweep.md` |
| dashboard comparison data | `dashboard/clips/<id>/risk.json` |
