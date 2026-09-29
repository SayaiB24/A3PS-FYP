# Reproduce and extend A3PS by hand

> **⚠️ Stops one step short of the current result (flagged 2026-09-29).** This
> file reproduces the v1 partition, up to the *clean-protocol* number (0.633).
> The current headline (**0.650** at FA **0.167**, `eval_v2`, checkpoint
> `risk_gru_k1p0_v2_selfix_s1234.pt`, threshold 0.70 / confirm 8) comes from the
> **v2 partition**, which this file does not cover yet. For the v2 steps, use:
> splits → [`../status/repartition_results.md`](../status/repartition_results.md)
> (`scripts/repartition_splits.py`, `eval/split_freeze_v2.json`); training on
> `train_core_v2` / `train_val_v2` with the fixed selector →
> [`../status/selection_fix_and_tradeoff.md`](../status/selection_fix_and_tradeoff.md);
> the one-shot held-out read → [`../status/final_eval_read.md`](../status/final_eval_read.md)
> (`scripts/final_eval_read.py`). The dataset, extraction, and training
> implementation sections (§2–3, §8) are still accurate.

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

> **The numbers in this section are selection-biased and superseded.** Every
> checkpoint here was early-stopped on the frozen eval split and the operating
> point was chosen there too, so 0.750 was selected on the test set rather than
> measured against it. See [`../status/eval_leakage_audit.md`](../status/eval_leakage_audit.md)
> for the evidence and [`../status/clean_protocol_results.md`](../status/clean_protocol_results.md)
> for the honest held-out number, which is **useful-warning 0.633 at FA 0.200**.
> The table below is retained for the record. §4–5 reproduce the *clean*
> protocol, not this one.

| quantity | value | where it lives |
|---|---|---|
| training clips | 1,365 (685 pos / 680 neg) | `data/features/train_all/` |
| held-out eval clips | 120 (60 pos / 60 neg), **frozen** | `data/features/eval/` |
| **checkpoint** | `risk_gru_k1p0.pt` — kappa 1.0, best epoch 8, 34,465 params | `notebooks/models/` |
| **committed operating point** | **threshold 0.60, confirm 8** | `eval/kappa_comparison.md` |
| useful-warning @ operating point | **0.750** (45/60) — *selection-biased* | `eval/operating_point_sweep_k1p0.md` |
| false-alarm @ operating point | **0.167** — *selection-biased* | `eval/operating_point_sweep_k1p0.md` |
| mean lead @ operating point | 1.67 s (target 2–6 s — **not met**) | `eval/operating_point_sweep_k1p0.md` |
| mean AP (500/1000/1500 ms) | 0.680 (0.785 / 0.697 / 0.557) | `eval/train_log_k1p0.txt` |
| too-early alerts | 1/60 | `eval/operating_point_sweep_k1p0.md` |

**Clean-protocol equivalent** (`risk_gru_k1p0_clean.pt`, trained on
`train_core`, selected on `train_val`, thr 0.70 / confirm 8, single eval read):
useful-warning **0.633** (38/60), false-alarm **0.200**, mean lead 1.62 s, mean
AP 0.661.

**Against the old threshold system** on the same frozen split
(`eval/phase4_comparison.md`): useful-warning **0.050 → 0.750**, false alarms
**0.767 → 0.167**, mean AP **0.546 → 0.680**. The old system's raw detection rate
was higher (0.917) only because it alerted on nearly everything, including 46 of
60 negatives, firing before `time_of_alert` on 52 of 60 positives.

Alternative operating points on the same checkpoint (see
`eval/operating_point_sweep_k1p0.md`): **0.60 / confirm 3** gives useful 0.833 and
lead 1.73 s at FA 0.300 — better on both, but FA is well over target. **0.70 / 5**
gives FA 0.133 at useful 0.600. We committed to 0.60/8 because it is the highest
useful-warning rate that still meets FA ≤ 0.20.

---

## 1. Exact config that produced those numbers

Change any of these and the numbers change. They are recorded in the checkpoint
itself under `extra` (`torch.load(...)["extra"]`).

| parameter | value | set where |
|---|---|---|
| `kappa` | **1.0** | `--kappa` (the file default is 3.0 — pass it explicitly) |
| `pre_alert_weight` | **0.5** | `--pre-alert-weight`, default `DEFAULT_PRE_ALERT_WEIGHT` in `a3ps/risk/anticipation_loss.py` |
| `pos_weight` | 1.0 | `--pos-weight` |
| `fa_target` | 0.20 | `--fa-target` (drives checkpoint selection) |
| `threshold` (decision) | **0.60** | `--threshold` (eval-time only) |
| `confirm` (frames) | **8** | `--confirm` (eval-time only) |
| `epochs` | 30 (max; stopped at 16) | `--epochs` |
| `patience` | 8 | `--patience` |
| `batch_size` | 8 | `--batch-size` |
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

Now split `train_all` into the set you train on and the set you select on. This
is not optional: `--val-features` drives early stopping and best-epoch choice,
so pointing it at `data/features/eval` *selects* on the held-out split and the
final number stops being held out. That is what earlier runs of this recipe did
— see [`../status/eval_leakage_audit.md`](../status/eval_leakage_audit.md).
`train_risk_head.py` now refuses it.

```powershell
python scripts/carve_train_val.py --frac 0.15 --seed 1234
# expect train_val 205 clips (103 pos / 102 neg), train_core 1160 (582 / 578)
```

The carve is idempotent-by-refusal: once `train_val` is in
`eval/split_freeze.json` the script will not re-draw it, because re-drawing
would move clips between the validation and training sets.

```powershell
python scripts/train_risk_head.py `
    --features data/features/train_core --val-features data/features/train_val `
    --kappa 1.0 --pre-alert-weight 0.5 --pos-weight 1.0 --fa-target 0.20 `
    --epochs 30 --patience 8 --batch-size 8 --seed 1234 `
    --out notebooks/models/risk_gru_k1p0_clean.pt `
    --history-json eval/risk_gru_history_k1p0_clean.json *>&1 `
    | Tee-Object eval/train_log_k1p0_clean.txt
```

> The superseded command trained on `data/features/train_all` and validated on
> `data/features/eval`, producing `risk_gru_k1p0.pt`. Its numbers are retained
> in [`../status/results.md`](../status/results.md) for the record, but they are
> selection-biased; do not reproduce it.

**Runtime ~45 s/epoch; ~12 min for a full run**, less when early stopping fires (ours stopped at epoch 16 of 30). The forward pass is batched — see §8.

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

Expect the best epoch to land early (ours was 8). Validation useful-warning
oscillates a lot epoch to epoch — the committed run went 0.800 → 0.767 → 0.567 →
0.667 → 0.817 across epochs 4–8, which is normal at this
dataset size; judge the run by its best, not its last.

---

## 5. Choose the operating point (CPU; seconds)

Free, because `threshold`/`confirm` are applied after the model runs.

```powershell
python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_k1p0_clean.pt `
    --features data/features/train_val `
    --thresholds 0.5,0.6,0.7,0.8,0.9 --confirms 3,5,8 `
    --out eval/operating_point_sweep_k1p0_clean_val.md
```

`--features` is required and has no default. It used to default to
`data/features/eval`, which is how the committed operating point came to be
chosen on the held-out split without anyone passing a flag. Choosing a
threshold is selection, so it happens on `train_val`.

Once the checkpoint **and** the operating point are fixed, score the held-out
split exactly once, with the flag that says you know what you are doing:

```powershell
python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_k1p0_clean.pt `
    --features data/features/eval --final-eval-report `
    --thresholds <chosen-thr> --confirms <chosen-confirm> `
    --out eval/final_eval_read_k1p0_clean.md
```

If that number disappoints, the honest move is to report it. Going back to
re-pick the epoch, threshold, confirm or seed and scoring again rebuilds exactly
the bias this protocol removes.

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
   `python -c "import torch; print(torch.load('notebooks/models/risk_gru_k1p0.pt', map_location='cpu', weights_only=False)['extra'])"`
5. **Threshold/confirm** — the sweep table is only valid for the checkpoint named in its header.
6. **Seed** — `--seed 1234` produced the committed numbers. Different seeds move the useful-warning rate by a few points at this dataset size.

---

## 8. Notes on the training implementation

**The forward pass is batched (done — was a known issue).**
[`train_risk_head.py`](../../scripts/train_risk_head.py) used to run
`[model(c["X"]) for c in batch]`, one forward per clip, giving an effective batch
size of 1. `batched_logits()` now pads each batch into one `(B, T_max, D)` tensor,
runs a single forward and slices each clip's valid prefix back out — **measured
6.14× faster** (75 s → 12 s per epoch).

No masking is needed in the loss, and end-padding is safe here for two specific
reasons worth knowing before you change the model:

* `RiskGRU` is **unidirectional**, so the output at step `t` depends only on steps
  `≤ t`. Padding appended after a clip cannot influence any valid position.
* Its input norm is `nn.LayerNorm(input_dim)`, which normalises across **features**
  within each timestep independently. Padding adds timesteps, not features.

**Swap in a bidirectional RNN, attention over the full sequence, or a BatchNorm /
any norm over the time axis, and this breaks silently** — padded positions would
leak into real ones and the objective would shift without any error.
[`tests/test_train_risk_head.py`](../../tests/test_train_risk_head.py) guards it:
logits, loss and **gradients** must all match the per-clip path exactly.

**Checkpoint selection respects the false-alarm target.** Ranking is
`(meets --fa-target, useful-warning, -FA)`, so an FA-compliant epoch always beats
a non-compliant one and ties break toward lower false alarms. Previously it
maximised useful-warning alone, which kept an epoch with worse FA over an
equal-scoring better one.

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
