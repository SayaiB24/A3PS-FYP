# Kappa retrain runbook — fixing lead time

> ## ✅ DONE (2026-08-19) — and it did not fix lead time
>
> Ran kappa ∈ {0.5, 1.0, 2.0, 3.0}, swept each checkpoint's operating point, and
> applied the §5 decision rule. Results in `eval/kappa_comparison.md`.
>
> | kappa | requested lead | best passing row | useful | FA | mean lead | mean AP |
> |---|---|---|---|---|---|---|
> | 0.5 | 1.60 s | *none passes* | — | — | — | 0.636 |
> | **1.0** | 1.46 s | **thr 0.60 / confirm 8** | **0.750** | **0.167** | **1.67 s** | 0.680 |
> | 2.0 | 1.20 s | thr 0.60 / confirm 5 | 0.767 | 0.183 | 1.59 s | 0.678 |
> | 3.0 | 0.98 s | thr 0.60 / confirm 5 | 0.783 | 0.183 | 1.63 s | 0.676 |
>
> **kappa 1.0 won on the rule** and is the committed model
> (`notebooks/models/risk_gru_k1p0.pt`). Against the previous kappa 3.0 model at
> identical FA it is a modest real gain: useful 0.717 → 0.750, lead 1.58 → 1.67 s.
>
> **But the hypothesis was largely wrong.** Lead time moved only 1.59–1.67 s
> across the whole sweep, nowhere near the 2–6 s target, and **mean AP stayed flat
> within 0.004**. A parameter that changes only *when* the model fires leaving
> discrimination unmoved is the signature of a **capacity/feature ceiling, not a
> loss-weighting problem** — exactly the outcome §5 said to report rather than
> bury. Lower kappa at 0.5 made things worse: no operating point passed the gates
> at all.
>
> **Do not re-run this sweep.** The next levers are capacity
> (`--hidden 128 --layers 2`), a longer feature window, or better features — see
> [`FUTURE_WORK.md`](FUTURE_WORK.md) Tier 1. The rest of this file is kept as the
> method: it is the template for any future loss-parameter sweep, and §6's failure
> modes still apply.

Standalone: everything needed to run the whole sweep and pick a winner is in
this file plus the repo. No assistant required.

- **Cost:** ~4 min per kappa value now that the forward pass is batched (was ~21 min).
- **No GPU, no re-extraction.** Reuses the existing `.npz` features unchanged.
- **Safe to interrupt:** each run saves its best checkpoint the moment it appears.

---

## 1. Why — kappa=3.0 trained a detector, not an anticipator

The anticipation loss weights each timestep by how close it is to the event,
using `kappa` as the sharpness of that weighting. A large kappa concentrates all
the weight immediately before impact, so the model is rewarded for firing *late*.
`expected_lead_time()` in [`a3ps/risk/anticipation_loss.py`](../../a3ps/risk/anticipation_loss.py)
reports what lead the loss is actually asking for, and its own docstring warns:

> *"At kappa=3 over a 3.5 s window it is ~1.0 s, so a large kappa quietly turns
> an anticipation objective into a detection one."*

Measured over the dataset's mean 3.49 s alert-to-event window:

| kappa | loss asks for a warning | note |
|---|---|---|
| 0.5 | **1.60 s** before impact | |
| 1.0 | **1.46 s** | |
| 2.0 | **1.20 s** | |
| **3.0** | **0.98 s** | **what we trained — barely a second** |
| 5.0 | 0.67 s | |
| 8.0 | 0.44 s | |

Regenerate that table yourself any time:

```powershell
python -c "
import sys; sys.path.insert(0,'.')
from a3ps.risk.anticipation_loss import expected_lead_time
for k in (0.25,0.5,1.0,1.5,2.0,3.0,5.0):
    print(f'kappa={k:5.2f} -> asks for ~{expected_lead_time(0.0, 3.49, k):.2f}s before impact')
"
```

This matters because lead time is the one target the committed model misses:

| target | committed model (thr 0.70 / confirm 5) | verdict |
|---|---|---|
| false-alarm ≤ 0.20 | 0.167 | ✅ met |
| detection ≥ 0.75 | 0.717 | marginally short |
| **lead 2–6 s** | **1.58 s** | **✗ short — this is what kappa addresses** |
| mean AP | 0.693 | ceiling; threshold cannot change it |

**Lower kappa → the loss asks for an earlier warning → longer lead.** That is
the hypothesis. It is not guaranteed: an earlier-firing model may also produce
more false alarms, which is exactly what the sweep is for.

Note this is *not* the same lever as `pre_alert_weight`. That penalises firing
before `time_of_alert`, and the operating-point sweep already showed premature
alarms were a threshold artefact (they fell 5 → 1 just by raising the threshold),
so `pre_alert_weight` is aimed at an already-solved problem. **Sweep kappa, not
`pre_alert_weight`.**

---

## 2. What to change, and what must stay fixed

`kappa` is passed on the command line — **do not edit any file.** The default
lives at `DEFAULT_KAPPA` in
[`a3ps/risk/anticipation_loss.py`](../../a3ps/risk/anticipation_loss.py); leave it
alone so the committed baseline stays reproducible, and override per run with
`--kappa`.

**Sweep:** `kappa ∈ {0.5, 1.0, 2.0}` (plus the existing 3.0 baseline = 4 points).

**Must stay identical to the baseline**, or runs are not comparable:

| parameter | value | why it must not move |
|---|---|---|
| `--features` | `data/features/train_all` | same 1,365 clips |
| `--val-features` | `data/features/eval` | the frozen held-out split |
| `--pre-alert-weight` | `0.5` | isolate kappa as the only change |
| `--pos-weight` | `1.0` | changes class balance otherwise |
| `--epochs` / `--patience` | `30` / `8` | same budget and stopping rule |
| `--batch-size` | `8` | affects loss aggregation |
| `--lr` / `--weight-decay` | `1e-3` / `1e-4` | |
| `--hidden` / `--layers` / `--dropout` | `64` / `1` / `0.1` | same capacity |
| **`--seed`** | **`1234`** | **critical — different seeds move useful-warning by several points at this dataset size, which would swamp the kappa effect** |
| feature rate | 10 Hz (already baked into the `.npz`) | do not re-extract |

Before starting, confirm the split has not drifted — otherwise the comparison is
against a different eval set:

```powershell
python scripts/freeze_split.py verify --index data/nexar/index.csv
```

Must print `OK: ... matches the 135 pinned clips`.

---

## 3. Run the sweep (~1 h 5 min)

Three runs, each to its own checkpoint, log, and history so nothing overwrites
the baseline. Run them **sequentially** — parallel runs contend for CPU and the
per-epoch timings become meaningless.

```powershell
cd D:\Coding\FYP\A3PS_MAIN\a3ps

foreach ($k in 0.5, 1.0, 2.0) {
  $tag = "k$($k -replace '\.','p')"        # k0p5, k1p0, k2p0
  Write-Host "=== kappa $k -> $tag ==="
  python scripts/train_risk_head.py `
      --features data/features/train_all --val-features data/features/eval `
      --kappa $k --pre-alert-weight 0.5 --pos-weight 1.0 `
      --epochs 30 --patience 8 --batch-size 8 --seed 1234 `
      --out "notebooks/models/risk_gru_$tag.pt" `
      --history-json "eval/risk_gru_history_$tag.json" `
      *>&1 | Tee-Object "eval/train_log_$tag.txt"
}
```

Where output lands:

| file | contents |
|---|---|
| `notebooks/models/risk_gru_k0p5.pt` etc. | best checkpoint per kappa |
| `eval/train_log_k0p5.txt` etc. | full per-epoch log |
| `eval/risk_gru_history_k0p5.json` etc. | machine-readable curve |

Watch a run live in another terminal:

```powershell
Get-Content eval\train_log_k0p5.txt -Wait
```

**Confirm early stopping fired correctly.** Each log should end with either:

```
early stop: no improvement for 8 epoch(s) (best was epoch N at 0.XXX). Stopping at epoch M/30.
```

or reach epoch 30 having improved recently. Then a `best epoch N` block. Also
check the header line records the kappa you intended:

```
loss: kappa=0.5 pre_alert_weight=0.5 -> asks for a warning ~1.60s before impact on a mean-lead clip
```

**If that line says kappa=3.0, the flag did not take — stop and fix it**, or you
will sweep three identical runs.

---

## 4. Evaluate each checkpoint

Training reports metrics at the default threshold 0.5, which is *not* the
committed operating point. Sweep each checkpoint properly:

```powershell
foreach ($tag in "k0p5","k1p0","k2p0") {
  python scripts/sweep_operating_point.py `
      --checkpoint "notebooks/models/risk_gru_$tag.pt" `
      --features data/features/eval `
      --thresholds 0.5,0.6,0.7,0.8,0.9 --confirms 3,5,8 `
      --out "eval/operating_point_sweep_$tag.md"
}
```

Also re-sweep the baseline for a like-for-like comparison if you have not kept
`eval/operating_point_sweep.md`:

```powershell
python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_v1.pt --features data/features/eval `
    --thresholds 0.5,0.6,0.7,0.8,0.9 --confirms 3,5,8 `
    --out eval/operating_point_sweep_k3p0.md
```

Then build the comparison table. For each kappa, take **the best row that meets
FA ≤ 0.20** from its sweep file:

| kappa | requested lead | best FA-compliant row (thr/confirm) | useful | FA | **mean lead** | mean AP |
|---|---|---|---|---|---|---|
| 3.0 (baseline) | 0.98 s | 0.70 / 5 | 0.717 | 0.167 | 1.58 s | 0.693 |
| 2.0 | 1.20 s | | | | | |
| 1.0 | 1.46 s | | | | | |
| 0.5 | 1.60 s | | | | | |

---

## 5. Decision rule — pick the winner

Apply in order. Do not skip to lead time.

1. **Hard gate: FA ≤ 0.20.** Discard any kappa with no compliant row at all.
2. **Hard gate: detection (useful-warning) ≥ 0.70.** Below that the model misses
   too much to be worth a longer warning. (0.75 is the stated target; 0.70 is the
   floor for still being interesting.)
3. **Maximise mean lead time** among survivors. This is the point of the sweep.
4. **Tie-break on mean AP.** Higher = better underlying discrimination, which is
   the more durable property.
5. **Sanity floor: reject any lead < 1.0 s** regardless of other numbers — there
   is no time to react.

Winner = highest mean lead subject to FA ≤ 0.20 and useful ≥ 0.70.

**If no kappa beats the baseline's 1.58 s lead at compliant FA**, that is a
genuine and reportable finding: lead time is limited by the model's
discrimination (mean AP ≈ 0.69), not by the loss weighting, and the next lever
is features or capacity — not kappa. Write it up that way rather than burying it.

Once chosen, update: the operating point in
[`REPRODUCE_BY_HAND.md`](REPRODUCE_BY_HAND.md) §0–1, `DEFAULT_THRESHOLD`/
`DEFAULT_CONFIRM` in [`scripts/build_dashboard_demo.py`](../../scripts/build_dashboard_demo.py)
if it changed, then rebuild the dashboard:

```powershell
python scripts/build_dashboard_demo.py --clips auto --n 6 `
    --checkpoint notebooks/models/risk_gru_<winner>.pt `
    --threshold <thr> --confirm <confirm>
```

---

## 6. Failure modes to watch for

**Overfitting / best epoch very early.** The baseline peaked at epoch 5 of 30.
If a kappa run's best epoch is 1–2, it is likely not learning the objective at
all rather than learning it fast — check the loss is actually decreasing across
epochs in the log. Validation useful-warning oscillating by ±0.2 between epochs
is normal at 120 val clips; judge by the best, never the last.

**Too-early alarms creeping back.** Lower kappa explicitly asks for earlier
warnings, so `n_too_early` (positives fired before `time_of_alert`) will rise.
Watch the `early` count in the sweep table, not just FA. If a kappa reaches good
lead time mainly by firing before there is anything to see, it has gamed the
metric — those are excluded from useful-warning by design, so a rising `early`
count alongside a *falling* useful rate is the signature. This is the one
scenario where raising `--pre-alert-weight` (try 1.0) genuinely becomes the right
follow-up.

**AP ceiling not moving.** `mean AP` is threshold-free discrimination. Expect it
to stay near 0.69 across all kappa values — kappa changes *when* the model fires,
not how well it separates risky from ordinary driving. If AP is flat everywhere,
that confirms the ceiling is a feature/capacity limit, and further loss tuning
has little left to give. Do not read flat AP as the sweep having failed; it is
information.

**Lead time rising while FA explodes.** The expected trade: firing earlier means
firing on weaker evidence. If every kappa below 2.0 pushes FA above 0.20 at every
threshold, the honest conclusion is that this model cannot deliver a 2 s+ warning
at an acceptable false-alarm rate, and the writeup should say so.

**Silently sweeping identical runs.** Always check the `loss: kappa=...` header
line in each log matches the intended value (§3).

---

## 7. If kappa alone is not enough

In rough order of expected value:

1. **Fix the batching inefficiency** (see [`REPRODUCE_BY_HAND.md`](REPRODUCE_BY_HAND.md) §8) — ~5× faster training makes every subsequent sweep cheap. Do this before any large hyperparameter search.
2. **More capacity** — `--hidden 128 --layers 2`. 34k params is very small; the AP ceiling may simply be underfitting.
3. **Longer window** — features are 13 s at 10 Hz. Re-extracting at `--window-s 20` gives more run-up context, at the cost of a full GPU re-extraction pass.
4. **Better features** — the ReID appearance arm, or richer ego motion with detection masking. See [`FUTURE_WORK.md`](FUTURE_WORK.md).
