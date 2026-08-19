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
| checkpoint | `notebooks/models/risk_gru_k1p0.pt` (kappa 1.0, best epoch 8) |
| operating point | threshold 0.60, confirm 8 |
| useful-warning | 0.750 (45/60) |
| false-alarm | 0.167 ✅ (target ≤ 0.20) |
| mean lead | 1.67 s (target 2–6 s — **the gap**) |
| mean AP | 0.680 (0.795 / 0.708 / 0.575 @ 500/1000/1500 ms) |

## What is already done

Nothing below needs redoing. Per-item detail in
[`../status/TODO.md`](../status/TODO.md).

| area | state |
|---|---|
| Split freeze + drift guards | ✅ done |
| Dataset indexed (1,500 labelled clips) | ✅ done — 685 pos / 680 neg train, 60+60 frozen eval |
| Feature extraction (the only GPU step) | ✅ done — 1,485 clips, 10 Hz, 0 failures, ego motion on all |
| Learned head trained | ✅ done — kappa 1.0, best epoch 8 |
| Operating point chosen | ✅ done — 0.60/8, FA target met |
| **Kappa sweep** | ✅ **done — did NOT fix lead time** (see below) |
| **Batched forward pass** | ✅ done — 6.14× faster, gradients verified |
| **FA-aware checkpoint selection** | ✅ done |
| Metric bugs fixed (AP ties, invented window) | ✅ done |
| Paper tables regenerated | ✅ done — incl. `eval/phase4_comparison.md` |
| Dashboard old-vs-new comparison | ✅ done — 6 demo clips |
| Handoff documentation | ✅ done |
| **Lead time ≥ 2 s** | ⬜ **still open — step 1 below** |

---

## 1. Close the lead-time gap  ⭐ top priority

**Mean lead is 1.67 s against a 2–6 s target.** This is the only unmet target.

**Kappa is exhausted as a lever.** The sweep
([`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md), `eval/kappa_comparison.md`) tried
kappa ∈ {0.5, 1.0, 2.0, 3.0}: lead moved only 1.59–1.67 s and **mean AP stayed
flat within 0.004**. A parameter that changes only *when* the model fires leaving
discrimination unmoved means the ceiling is capacity or features, not the loss.

Next candidates, cheapest first — training is now ~4 min per run:

1. **More capacity** — `--hidden 128 --layers 2`, then `--hidden 256`. ~15 min for both. 34,465 params against 1,365 clips may simply be underfitting, and mean AP 0.680 is the number to move. Watch for overfitting: the best epoch already lands around 8 of 30.
2. **Longer feature window** — features are 13 s at 10 Hz, so lead time is mechanically bounded by how much run-up the model can see. `--window-s 20` needs a full GPU re-extraction (~4 h) and makes results incomparable with everything above unless the eval split is re-extracted too.
3. **Better features** — see [`FUTURE_WORK.md`](FUTURE_WORK.md).

```powershell
python scripts/train_risk_head.py `
    --features data/features/train_all --val-features data/features/eval `
    --kappa 1.0 --hidden 128 --layers 2 --fa-target 0.20 `
    --epochs 40 --patience 8 --batch-size 8 --seed 1234 `
    --out notebooks/models/risk_gru_h128.pt `
    --history-json eval/risk_gru_history_h128.json *>&1 `
    | Tee-Object eval/train_log_h128.txt

python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_h128.pt --features data/features/eval `
    --thresholds 0.5,0.6,0.7,0.8 --confirms 3,5,8 `
    --out eval/operating_point_sweep_h128.md
```

**Done when:** either a checkpoint reaches lead ≥ 2 s at FA ≤ 0.20 and useful
≥ 0.70, or you have shown that capacity does not move mean AP either — which
would make "this feature set supports ~1.7 s of warning" the honest finding.

---

## 2. Write the numbers up

**No compute; the most important step.**

**Quote:**
- useful-warning **0.750 at FA 0.167** — always paired. The useful-warning rate alone is meaningless: 0.833 is available on the same checkpoint at FA 0.300.
- **The old-vs-new comparison**, which is the real headline (`eval/phase4_comparison.md`): useful-warning 0.050 → 0.750, false alarms 0.767 → 0.167, mean AP 0.546 → 0.680.
- mean AP 0.680 as the threshold-free measure of discrimination.

**State plainly:**
- Lead time (1.67 s) does not meet the 2–6 s target, and the kappa sweep showed the loss weighting is not the cause.
- The old system's *higher* raw detection rate (0.917 vs 0.750) reflects that it alerted on 46 of 60 negatives, not better detection.
- The two systems are scored under different observation conditions (13 s @ 10 Hz vs full clips @ 30 Hz) — see step 3.

**Do not:**
- Quote any latency figure without naming the machine (CPU/GPU differ ~20–40× here).
- Quote numbers from `../status/results.md` or `../design/explanation.md` — both pre-pivot. See [`../INDEX.md`](../INDEX.md) §4.

---

## 3. Make the old-vs-new comparison strict

**~1 h GPU + minutes CPU.** Currently the learned head is scored on 13 s windows
at 10 Hz while the threshold system is scored on full ~40 s clips at 30 Hz. The
useful-warning and AP gains are large enough to survive that mismatch, but the
lead-time comparison is not defensible across it.

To fix: re-run the threshold pipeline restricted to the same 13 s windows and
score both on identical observations. Until then, quote the useful-warning and AP
improvements rather than the lead-time difference.

---

## 4. Optional improvements

Only after 1–3. Details in [`FUTURE_WORK.md`](FUTURE_WORK.md). Capacity and the
longer window are listed under step 1 instead, since they target the open gap.

| idea | cost | why |
|---|---|---|
| Kaggle submission on the 1,344 unlabelled test clips | ~2 h | external unbiased AP — the only number here nobody could accuse us of mis-scoring |
| ReID appearance arm | ~1 day | deferred from v1; needs `onnxruntime`, changes association |
| Per-actor temporal head | ~2 days | current head pools over actors, so it cannot attribute risk to one — the dashboard's actor colour is scene-level for this reason |
| BEV per-clip calibration | ~1 day | metric distances instead of pixel stand-ins; interpretability, no headline metric |
| User study | weeks | or state the absence as an explicit limitation |

---

## 5. Housekeeping

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
| trained checkpoint | `notebooks/models/risk_gru_k1p0.pt` (kappa 1.0) |
| training log / curve | `eval/train_log_k1p0.txt`, `eval/risk_gru_history_k1p0.json` |
| operating-point grid | `eval/operating_point_sweep_k1p0.md` |
| kappa sweep results | `eval/kappa_comparison.md` |
| old-vs-new headline | `eval/phase4_comparison.md` |
| dashboard comparison data | `dashboard/clips/<id>/risk.json` |
