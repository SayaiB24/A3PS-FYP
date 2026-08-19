# Reproduce and extend A3PS by hand

End-to-end, raw clips → features → training → operating point → dashboard, with
every command and every config value that produced the committed numbers. Written
to be followed without any assistant: nothing here depends on knowledge that is
not in this file or the repo.

Read [`NEXT_STEPS.md`](NEXT_STEPS.md) for what to do next,
[`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md) for the top priority improvement, and
[`CHALLENGES.md`](CHALLENGES.md) for why things are built the way they are.

---

## 0. The committed result, in one table

Everything below reproduces this. If your numbers differ, something in the
config differs — check §7.

| quantity | value | where it lives |
|---|---|---|
| training clips | 1,365 (685 pos / 680 neg) | `data/features/train_all/` |
| held-out eval clips | 120 (60 pos / 60 neg), **frozen** | `data/features/eval/` |
| best epoch | 5 | `eval/train_log.txt` |
| useful-warning rate @ best epoch | 0.833 (50/60) | `eval/train_log.txt` |
| **committed operating point** | **threshold 0.70, confirm 5** | `eval/operating_point_sweep.md` |
| useful-warning @ operating point | 0.717 (43/60) | `eval/operating_point_sweep.md` |
| false-alarm @ operating point | **0.167** ✅ (target ≤ 0.20) | `eval/operating_point_sweep.md` |
| mean lead @ operating point | 1.58 s | `eval/operating_point_sweep.md` |
| mean AP (500/1000/1500 ms) | 0.693 (0.795 / 0.708 / 0.575) | `eval/train_log.txt` |
| checkpoint | 34,465 params | `notebooks/models/risk_gru_v1.pt` |

Documented alternative operating point: **0.60 / confirm 8** — useful 0.750,
lead 1.84 s, FA 0.217. Better detection *and* lead, but FA is over target. We
committed to 0.70/5 because a cleanly-met false-alarm target is the defensible
headline; 0.60/8 is a legitimate choice if you are prepared to defend FA 0.217.

---

## 1. Exact config that produced those numbers

Change any of these and the numbers change. They are recorded in the checkpoint
itself under `extra` (`torch.load(...)["extra"]`).

| parameter | value | set where |
|---|---|---|
| `kappa` | **3.0** | `--kappa`, default `DEFAULT_KAPPA` in `a3ps/risk/anticipation_loss.py` |
| `pre_alert_weight` | **0.5** | `--pre-alert-weight`, default `DEFAULT_PRE_ALERT_WEIGHT` same file |
| `pos_weight` | 1.0 | `--pos-weight` |
| `threshold` (decision) | **0.70** | `--threshold` (eval-time only) |
| `confirm` (frames) | **5** | `--confirm` (eval-time only) |
| `epochs` | 30 (max) | `--epochs` |
| `patience` | 8 | `--patience` |
| `batch_size` | 8 | `--batch-size` (see §8 — does not batch compute) |
| `lr` | 1e-3 | `--lr` |
| `weight_decay` | 1e-4 | `--weight-decay` |
| `hidden` / `layers` / `dropout` | 64 / 1 / 0.1 | `--hidden --layers --dropout` |
| `seed` | 1234 | `--seed` |
| feature rate | **10 Hz** | `--rate-hz` at extraction |
| feature window | 13 s (+1 s tail) | `--window-s 13 --tail-s 1` |
| feature width | 112 | derived; asserted at load |

**`threshold` and `confirm` are evaluation-time only.** They convert the head's
per-frame probability into a discrete alert and do not affect training at all —
which is why the operating point can be re-chosen for free (§5).

---

## 2. Dataset and the frozen split (do this first, once)

The usable pool is **1,500 labelled clips**. The `test/` folder's 1,344 clips
have **no labels** (`target` null on every row) — it is the Kaggle competition
holdout. A 1,500-row index is complete, not truncated.

```powershell
# index the labelled train clips, replaying the frozen dev/eval membership
python scripts/prepare_nexar.py --root data/nexar `
    --videos "D:\Coding\FYP\Nexar-Dataset\train" `
    --annotation "D:\Coding\FYP\Nexar-Dataset\train.csv" `
    --freeze eval/split_freeze.json

# positives absent from the freeze are left unassigned on purpose; name their split
python scripts/assign_unused_positives.py --index data/nexar/index.csv --split train_risk

# the split must still match the freeze, or no number is comparable to any other
python scripts/freeze_split.py verify --index data/nexar/index.csv
```

`verify` must print `OK: ... matches the 135 pinned clips`. **If it says
MISMATCH, stop.** The eval split has drifted and nothing downstream is
comparable. See [`CHALLENGES.md`](CHALLENGES.md) §1 for why this guard exists.

Expected splits: `train_risk` 685 pos, `train_traj` 680 neg, `eval` 60+60,
`dev` 5+10.

---

## 3. Feature extraction (GPU; ~3.5 h for the full set)

The only GPU step. Produces one `.npz` per clip holding a `T × 112` feature
matrix at 10 Hz over a 13 s window.

**All three splits must use the same `--rate-hz`.** Derivatives are central
differences over a `2/rate` span, so 15 Hz features are measurably noisier than
10 Hz ones; mixing rates trains on one noise distribution and evaluates on
another, and the resulting drop looks like a model failure rather than the
extraction artefact it is.

```powershell
# held-out eval FIRST -- 17 min, and it is the split that must succeed
python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split eval `
    --out data/features/eval --window-s 13 --tail-s 1 --rate-hz 10

python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split train_traj `
    --out data/features/train_neg --window-s 13 --tail-s 1 --rate-hz 10

python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split train_risk `
    --out data/features/train_pos --window-s 13 --tail-s 1 --rate-hz 10
```

Measured throughput: **8.4 s/clip** (~65 ms/frame — not the 20–40 ms/frame the
older docs assumed). Resumable: existing `.npz` files are skipped, so re-running
the same command continues after an interruption.

Watch the printed `ego fail` percentage. **0% is what you want** — it means the
optical-flow solve succeeded on every frame. It is a *failure* rate, not a
measure of how much ego motion there is. Above ~50% on a clip means that clip's
ego features are mostly zeros and it will look stationary to the model.

Sanity-check the output before training:

```powershell
python -c "
import glob, json, numpy as np, collections
for s in ('train_neg','train_pos','eval'):
    lab=collections.Counter(); ego=collections.Counter()
    for p in glob.glob(f'data/features/{s}/*.npz'):
        with np.load(p, allow_pickle=False) as d: m=json.loads(str(d['meta']))
        lab[m['label']]+=1; ego[bool(m['has_ego'])]+=1
    print(s, dict(lab), 'has_ego', dict(ego))
"
```

Expect `has_ego {True: N}` for all three. If `has_ego` is False anywhere, the
model is blind to ego motion on those clips and any result must say so.

---

## 4. Training (CPU — no GPU path exists)

`train_risk_head.py` contains no `.to(device)` and no `torch.cuda`: it is
CPU-only by construction. Running it on the GPU laptop is the same speed as
anywhere else.

`--features` takes **one** directory, so merge the two training halves. Do not
copy positives into `train_neg` — a directory named "neg" holding both classes
is how a mistake gets made three weeks later.

```powershell
mkdir data\features\train_all
Copy-Item data\features\train_pos\*.npz data\features\train_all\
Copy-Item data\features\train_neg\*.npz data\features\train_all\
# expect 1365 files
(Get-ChildItem data\features\train_all\*.npz | Measure-Object).Count
```

```powershell
python scripts/train_risk_head.py `
    --features data/features/train_all --val-features data/features/eval `
    --epochs 30 --patience 8 --batch-size 8 `
    --out notebooks/models/risk_gru_v1.pt `
    --history-json eval/risk_gru_history.json *>&1 | Tee-Object eval/train_log.txt
```

**Runtime ~2.2 min/epoch; ~50 min for 30 epochs**, less if early stopping fires.

Watch it live:

```powershell
Get-Content eval\train_log.txt -Wait
```

The loop **saves the checkpoint every time validation improves**, and rewrites
`risk_gru_history.json` every epoch, so it is safe to interrupt at any moment —
you keep the best model so far. (It did not always do this; see
[`CHALLENGES.md`](CHALLENGES.md) §16.)

Signs it worked:
- `<- best` appears on improving epochs, each followed by `saved best (epoch N)`.
- Early stopping prints `early stop: no improvement for 8 epoch(s)`.
- Final block prints `best epoch N` with the useful-warning rate, FA, lead, and
  the three cutoff APs.

Expect the best epoch to land early (ours was 5). Validation useful-warning
oscillates a lot epoch to epoch — 0.833 → 0.550 → 0.717 is normal at this
dataset size; judge the run by its best, not its last.

---

## 5. Choose the operating point (CPU; seconds)

Free, because `threshold`/`confirm` are applied after the model runs.

```powershell
python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_v1.pt `
    --features data/features/eval `
    --thresholds 0.5,0.6,0.7,0.8,0.9,0.95 --confirms 3,5,8 `
    --out eval/operating_point_sweep.md
```

How to read the result:

- **False-alarm rate is the gatekeeper.** Target ≤ 0.20. A high useful-warning
  rate at high FA is not a result — a system that alerts on most negatives will
  "catch" most positives too.
- **`mean AP` is constant across every row.** It ranks clips by peak probability
  and ignores the threshold, so it is the model's threshold-free discrimination:
  the ceiling no operating point can beat. Raising it needs better features or
  training, never a different threshold.
- **Lead time is the price.** Raising threshold/confirm buys FA back by alerting
  later. Below ~1 s of lead there is no time to react, so a row that fixes FA by
  collapsing lead has solved nothing.
- Pick the highest useful-warning rate subject to FA ≤ 0.20 **and** lead ≥ ~1.5 s.

---

## 6. Dashboard (CPU; ~2 min)

Shows the learned head against the old threshold system on the same clip, so
the improvement is visible rather than asserted. **No GPU:** the learned curve
comes from the cached `.npz` features and the old curve from cached
`eval/anticipation/<clip>/events.json`. Perception is never re-run.

```powershell
python scripts/build_dashboard_demo.py --clips auto --n 6
python scripts/serve_dashboard.py --port 8000
```

Then open <http://localhost:8000> and pick a clip from the dropdown (the demo
clips are listed first: 621, 488, 1004, 690, 1085, 1261).

What you should see:
- **MODEL COMPARISON** panel at the bottom: the learned head's per-frame risk in
  green, the old system's in red, the committed threshold as a dashed green
  line, `time_of_alert` as a blue dashed vertical, `time_of_event` as a white
  vertical, and a triangle where each system fired.
- **Console panel**: each system's verdict (`useful` / `too early` / `false
  alarm` / `clean`), when it fired and the lead time, and the natural-language
  explanation on intervention.

On the six demo clips: all four positives are `useful` for the learned head
while the old system is `too early` on every one (firing ~1.5 s into the clip,
18–24 s premature, 5–30 events per clip). On clip 1261 the old system false-alarms
where the learned head stays clean.

Pick specific clips with `--clips 237,311,1284`. A clip needs all three of:
extracted features, cached old-system output, and a local video (74 clips
qualify). Demo `raw.mp4` files are gitignored — re-run the builder to recreate
them after a fresh clone.

---

## 7. If your numbers differ

Check in this order:

1. **Split drift** — `python scripts/freeze_split.py verify --index data/nexar/index.csv`. Anything other than `OK` invalidates comparison.
2. **Feature rate** — all splits at the same `--rate-hz`? Check `rate_hz` in each `manifest.json` under `data/features/*/`.
3. **`has_ego`** — if False anywhere, ego features are zeros for those clips.
4. **kappa / pre_alert_weight** — read them back from the checkpoint:
   `python -c "import torch; print(torch.load('notebooks/models/risk_gru_v1.pt', map_location='cpu', weights_only=False)['extra'])"`
5. **Threshold/confirm** — the sweep table is only valid for the checkpoint named in its header.
6. **Seed** — `--seed 1234` produced the committed numbers. Different seeds move the useful-warning rate by a few points at this dataset size.

---

## 8. Known issues, deliberately not fixed

Both are documented tasks, not bugs to be surprised by.

**`--batch-size` does not batch the compute.** [`train_risk_head.py`](../../scripts/train_risk_head.py)
runs `logits = [model(c["X"]) for c in batch]` — one forward pass per clip in a
Python loop. So the effective batch size is 1 and `--batch-size` only controls
how many clips the loss aggregates over. Consequences: ~1.85× parallelism on 8
cores instead of near-linear, and ~2.2 min/epoch where a properly batched
version should manage well under a minute.

To fix: pad each batch's clips to a common `T` into a real `(B, T, D)` tensor,
run one forward, and mask the padding inside `batch_anticipation_loss` so padded
timesteps contribute no loss. Expect roughly 5× faster training. **Verify by
checking the loss matches the unbatched version on a fixed seed before trusting
any number from it** — a masking bug here would silently change the objective
rather than crash. Worth doing before a large hyperparameter sweep, since the
speedup compounds across every run.

**ReID appearance embeddings are not implemented.** The tracker is Ultralytics
BoT-SORT with `with_reid: False`, so no appearance embedding is ever computed.
Enabling it needs `onnxruntime` and changes association, which would make
features incomparable with everything measured so far. Left as a future ablation
arm — see [`FUTURE_WORK.md`](FUTURE_WORK.md).

---

## 9. What runs where

| step | machine | time | why |
|---|---|---|---|
| index + freeze verify (§2) | either | minutes | metadata only |
| **feature extraction (§3)** | **GPU** | **~3.5 h** | YOLO + tracking over video |
| training (§4) | either (CPU-only code) | ~50 min | no GPU path exists |
| operating-point sweep (§5) | either | seconds | forward pass on cached features |
| dashboard build (§6) | either | ~2 min | reads cached artefacts only |
| kappa retrain ([`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md)) | either | ~1 h | reuses existing features |

Only §3 needs the GPU. Everything after it runs on cached artefacts, which is
what makes iteration cheap: you can re-choose the operating point, rebuild the
dashboard, and retrain the head repeatedly without ever touching video again.
