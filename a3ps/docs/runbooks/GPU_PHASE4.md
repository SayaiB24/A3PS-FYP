# GPU laptop instructions — Phase IV (learned temporal risk head)

Everything in this file runs on the **GPU laptop**. Nothing here has been run on
the CPU laptop; the CPU-side work (split freeze, metric re-key, feature
extractor, ego helper, anticipation loss, temporal head, smoke train) is already
done and committed.

Run the steps in order. Steps 0–2 are indexing and a cheap measurement; do
**not** start step 4 (the 1,365-clip extraction) until step 3 has told us which
frame rate to use.

**Confirmed paths for this machine** (from the actual layout you have):

| what | path |
|---|---|
| repo | `D:\Coding\FYP\A3PS_MAIN\a3ps` |
| data root | `D:\Coding\FYP\Nexar-Dataset` |
| train videos | `D:\Coding\FYP\Nexar-Dataset\train` (1500 clips, **labelled**) |
| test videos | `D:\Coding\FYP\Nexar-Dataset\test` (1344 clips, **UNLABELLED**) |
| train labels | `D:\Coding\FYP\Nexar-Dataset\train.csv` |
| test labels | `D:\Coding\FYP\Nexar-Dataset\test.csv` (ids only, no targets) |

### The test split is unusable — this is expected, not a bug

`test.csv` has all 1344 rows present and every id matches a video on disk
exactly (0 unmatched either way), but **`target` is null and
`event_time_s`/`alert_time_s` are empty on every single row**. It is the Kaggle
competition holdout: predictions get submitted, labels are never published.

`prepare_nexar.py` therefore (correctly) indexes only the 1500 labelled train
clips. **Do not try to "fix" this by re-indexing** — there is no ground truth to
recover. Verified with:

```powershell
python -c "
import sys; sys.path.insert(0,'scripts')
from prepare_nexar import load_table
rows = load_table(r'D:\Coding\FYP\Nexar-Dataset\test.csv')
tgt = {}
for r in rows: tgt[r['target']] = tgt.get(r['target'],0)+1
print('rows:', len(rows), 'target distribution:', tgt)
"
# -> rows: 1344 target distribution: {None: 1344}
```

The usable pool is therefore **1,500 clips**, splitting as:

| split | clips | note |
|---|---|---|
| `train_risk` | 685 positive | for the learned head |
| `train_traj` | 680 negative | for the learned head |
| `eval` | 60 pos + 60 neg | frozen, held out |
| `dev` | 5 pos + 10 neg | frozen |

685/680 is near-perfectly class-balanced, so no aggressive class weighting is
needed — and 685 positives is a 10× improvement on the 65 available before.

The labels have been verified against known-trusted local data: 355 overlapping
clip ids checked against `eval/nexar_index_gpu.csv`, **0 mismatches**. Step 1's
overlap check below is kept for the record and does not need re-running.

The 1344 unlabelled test clips have exactly one potential use: generating a
Kaggle submission for an external, unbiased leaderboard AP. Worth considering
for the writeup if the competition is still open, but not required by anything
below.

---

## 0. Sync code and check CUDA

```powershell
cd D:\Coding\FYP\A3PS_MAIN\a3ps
git pull
```

Environment setup (venv, CUDA torch, requirements) is unchanged — follow
`GPU_HANDOFF.md` §2 if this is a fresh machine. Confirm CUDA is actually live,
because every runtime estimate below assumes it:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and your GPU name. If `False`, stop and fix torch first.

Then confirm the frozen split travelled with the repo:

```powershell
python scripts/freeze_split.py show
```

Expect `dev 15 clips (5 pos / 10 neg)`, `eval 120 clips (60 pos / 60 neg)`,
`135 clip ids pinned`.

---

## 1. Layout (already confirmed — nothing to run here)

`D:\Coding\FYP\Nexar-Dataset` holds separate train/test video folders and
separate train/test label sheets (`train.csv` / `test.csv`), rather than one
pooled folder + one table. `prepare_nexar.py` already supports this shape
directly — it takes multiple `--videos` dirs and multiple `--annotation` files
and merges them, so nothing needs to be combined by hand.

Already checked and confirmed:

- 1500 videos in `train\` (labelled), 1344 in `test\` (unlabelled -- see above).
- `train.csv` carries real `target` values; `test.csv` does NOT (all null).
- Label agreement: 355 overlapping clip ids matched `eval/nexar_index_gpu.csv`
  with 0 mismatches (see previous message).

For reference, this is the overlap check that was run (no need to re-run it —
kept here in case the dataset is ever replaced or re-downloaded):

```powershell
python -c "
import sys, csv
sys.path.insert(0, 'scripts')
from prepare_nexar import load_table, canon
known = {}
with open('eval/nexar_index_gpu.csv', encoding='utf-8-sig') as fh:
    for r in csv.DictReader(fh):
        known[canon(r['clip_id'])] = r
checked, mismatches = 0, []
for path in (sys.argv[1], sys.argv[2]):
    for row in load_table(path):
        cid = canon(row['id']) if row['id'] is not None else None
        if cid is None or cid not in known:
            continue
        checked += 1
        k = known[cid]
        exp_label = int(float(k['label']))
        if row['target'] != exp_label:
            mismatches.append((cid, 'label', exp_label, row['target']))
            continue
        if exp_label == 1:
            for field, mirror_val in (('event_time_s', row['event_time_s']), ('alert_time_s', row['alert_time_s'])):
                exp = k[field]
                if exp and mirror_val != '' and abs(float(exp) - float(mirror_val)) > 0.05:
                    mismatches.append((cid, field, exp, mirror_val))
print(f'checked {checked} overlapping clip ids against known-trusted local data')
print(f'mismatches: {len(mismatches)}')
for m in mismatches[:20]:
    print(' ', m)
" "D:\Coding\FYP\Nexar-Dataset\train.csv" "D:\Coding\FYP\Nexar-Dataset\test.csv"
```

---

## 2. Re-index WITHOUT re-drawing the eval split — DONE

Already run; `eval/nexar_index_full.csv` is committed (1500 rows, freeze
verified OK). Kept below as the record of what was done. Passing `test\` and
`test.csv` was harmless but contributed nothing, since those rows carry no
labels — see the note at the top of this file.

This is the step that would silently destroy comparability if done wrong.
`prepare_nexar.py` assigns splits with a pool-size-dependent shuffle, so simply
re-running it over a changed clip pool would hand us a different 120-clip held-out set —
invalidating the 3.7 GB cache and every number measured so far.

The `--freeze` flag (default `eval/split_freeze.json`, committed) prevents that:
pinned clips keep their split verbatim and new clips can only join `train_traj`.

`--videos` and `--annotation` both accept multiple paths and merge them, so
train/test don't need to be combined by hand first — train is passed before test
so it wins on any id collision (none expected — train and test are Nexar's own
disjoint split).

Copy the existing 155 already-cached videos into the pool first, so every clip
the freeze names is actually present as a file:

```powershell
Copy-Item D:\Coding\FYP\A3PS_MAIN\a3ps\data\nexar\videos\*.mp4 "D:\Coding\FYP\Nexar-Dataset\train" -Force
```

Then re-index over BOTH folders and BOTH label sheets in one pass:

```powershell
python scripts/prepare_nexar.py --root data/nexar `
    --videos "D:\Coding\FYP\Nexar-Dataset\train" "D:\Coding\FYP\Nexar-Dataset\test" `
    --annotation "D:\Coding\FYP\Nexar-Dataset\train.csv" "D:\Coding\FYP\Nexar-Dataset\test.csv" `
    --freeze eval/split_freeze.json
```

Expect `splits (frozen from eval/split_freeze.json): ...` and
`135 clip ids replayed from the freeze`. Runtime: a few minutes (it probes every
video's metadata across the 1500 labelled train clips, so a bit longer than a
155-clip pass).

**It printed `! 685 positive(s) are not in the freeze and were left UNUSED`.**
That is deliberate — positives are the scarce resource and the script
will not silently assign them. Give them a training split explicitly:

```powershell
python scripts/assign_unused_positives.py --index data/nexar/index.csv --split train_risk
```

Verify the held-out set survived, then commit the index so the CPU laptop has it:

```powershell
python scripts/freeze_split.py verify --index data/nexar/index.csv
Copy-Item data\nexar\index.csv eval\nexar_index_full.csv
git add eval/nexar_index_full.csv
git commit -m "eval: full 1500-clip labelled index with frozen dev/eval split"
git push
```

`verify` must print `OK: ... matches the 135 pinned clips`. If it reports
MISMATCH, **stop** and send me the output — do not proceed to extraction.

---

## 3. Association quality vs frame rate — DONE, decision recorded

This step has been run at all three rates. Results are committed as
`eval/assoc_rate_check.md` (30 vs 10 Hz) and `eval/assoc_rate_check2.md`
(30 vs 15 Hz). **No need to re-run** — the numbers and the decision are below.

| rate | tracks/clip | frac ≥2 s | actors/frame | switches/s | s/clip |
|---|---|---|---|---|---|
| 30 Hz | 44.7 | 0.231 | 4.35 | 3.45 | 19–21 |
| 15 Hz | 31.5 | 0.292 | 4.16 | 2.43 | 10.6 |
| 10 Hz | 25.2 | 0.349 | 4.00 | 1.95 | 8.4 |

**Read `tracks/clip × frac ≥2 s`, not `frac ≥2 s` alone.** The ratio *rises* as
the rate drops (+27% at 15 Hz, +51% at 10 Hz) purely because subsampling filters
out one-frame flicker detections that would never have become usable tracks
anyway — the switch rate falling 3.45 → 1.95/s is the same effect. What actually
determines training-data volume is the absolute count of forecastable tracks:

| rate | forecastable tracks/clip | vs 30 Hz |
|---|---|---|
| 30 Hz | ~10.3 | — |
| 15 Hz | ~9.2 | −11% |
| 10 Hz | ~8.8 | −15% |

So decimating costs real yield; it just costs far less than the raw `frac ≥2 s`
column suggests, and it is not the "borderline" the script's own verdict line
prints (that message fires on the `actors/frame` dip but is worded as if
`frac ≥2 s` had dropped — a wording bug in `check_assoc_rate.py`, now fixed).

### Decision: 10 Hz for EVERY split

**The rate must be identical across train and eval.** `extract.py` differentiates
with `_central_diff(values, times)` over real timestamps, so the feature *units*
are per-second and rate-invariant — but the central-difference span is `2/rate`
(0.200 s at 10 Hz, 0.133 s at 15 Hz). Over the same detector jitter, the shorter
span yields systematically noisier velocity/acceleration features. Extracting
training data at one rate and eval at another would hand the model a noise
distribution at eval it never saw in training, and the resulting drop would look
like a model deficiency rather than the extraction artefact it is.

Given one rate everywhere, **10 Hz**:

- Yield is immaterial between the two: 1,365 training clips gives ~12.0 k
  forecastable tracks at 10 Hz vs ~12.6 k at 15 Hz. Both are far more than a
  34 k-parameter GRU needs.
- 10 Hz is ~50 min faster over the full pass (~3.2 h vs ~4.0 h).

So the 4% density advantage of 15 Hz buys nothing, and the consistency
requirement removes the reason to mix. Step 4's commands all use `--rate-hz 10`.

---

## 4. Extract features over the full set

Windowed at 13 s. Positives are windowed around `event_time_s` so the window
always contains `time_of_alert` plus ~8 s of run-up; negatives get a
deterministic hash-seeded window.

All three splits use `--rate-hz 10` — identical by requirement, see step 3. If
you change it, change it for **all three** or the eval numbers become
meaningless.

```powershell
cd D:\Coding\FYP\A3PS_MAIN\a3ps

# training negatives
python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split train_traj `
    --out data/features/train_neg --window-s 13 --tail-s 1 --rate-hz 10

# training positives
python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split train_risk `
    --out data/features/train_pos --window-s 13 --tail-s 1 --rate-hz 10

# held-out eval (60 pos + 60 neg) -- the numbers that go in the writeup
python scripts/extract_features.py --from-video `
    --index data/nexar/index.csv --split eval `
    --out data/features/eval --window-s 13 --tail-s 1 --rate-hz 10
```

**Expected runtime.** Step 3 gives the first real measured throughput on this
GPU: **8.4 s/clip at 10 Hz** over a 13 s window (tracking only — feature
extraction adds the optical-flow pass on top).

| scope | clips | s/clip | estimated wall time |
|---|---|---|---|
| training (`train_traj` 680 + `train_risk` 685) | 1,365 | 8.4 | **~3.2 h** |
| held-out eval | 120 | 8.4 | **~17 min** |

That is slower per frame than the 20–40 ms/frame the docs guessed (8.4 s over
~130 frames is ~65 ms/frame), but the pool is only 1,500 labelled clips, so the
whole thing lands at **~3.5 h total**. Add 5–15% for the optical-flow pass on top.
`--ego-downscale 2` (the default) keeps that cost small; raise it to 4 if decode
dominates.

At ~3.5 h you can run it in one sitting; the script is also resumable (below), so
splitting it across sessions costs nothing if you prefer.
**Do the eval split first** (17 min): it is the one that must succeed, and any
extraction bug shows up there before you spend six hours on training data.

The script is **resumable** — it skips any clip whose `.npz` already exists, so a
crash or a reboot costs only the clip in flight. Re-run the same command to
continue; add `--force` only to deliberately re-extract.

Watch the printed `ego fail` percentage. A few percent is normal (night clips,
wipers). If a clip reports >50%, its ego features are mostly zeros and it will
look "stationary" to the model — note the clip ids and send them to me.

### What lands where

Each `.npz` is a few hundred KB and holds `X` (T×112 float32), `t`, `n_actors`
and a JSON `meta` with the label, event/alert times, schema version and
`has_ego`. For scale: the 120 cached eval clips compressed from 3.7 GB of JSON to
**2.3 MB** of features.

Full set ≈ **25–35 MB** total. Zip and send all three directories back:

```powershell
Compress-Archive -Path data\features\train_neg,data\features\train_pos,data\features\eval `
    -DestinationPath features_full.zip
```

Drop `features_full.zip` anywhere I can reach it (or commit it — at ~30 MB that's
acceptable, unlike the videos). I'll unpack to `a3ps/data/features/` on the CPU
laptop and take the training and evaluation from there.

---

## 5. (Optional) Train on the GPU laptop instead

Training this head is small enough that the CPU laptop can do it — 34,465
parameters, and the smoke run did 12 epochs over 120 clips in 60 s. So the
default plan is: you send features, I train here.

If you'd rather train there while you have the data local:

```powershell
python scripts/train_risk_head.py `
    --features data/features/train_neg --val-features data/features/eval `
    --epochs 40 --batch-size 16 --out notebooks/models/risk_gru_v1.pt `
    --history-json eval/risk_gru_history.json
```

Note `--features` takes ONE directory, so merge the two training dirs first
(`Copy-Item data\features\train_pos\*.npz data\features\train_neg\`) or the run
will have no positives. The script prints a loud warning if the features carry no
ego motion, and refuses to train and validate on the same directory without
`--allow-leakage`.

Send back the `.pt` and the history JSON.

---

## What NOT to do

- **Do not run `prepare_nexar.py` without `--freeze`.** It will re-draw the
  held-out split and quietly invalidate every prior number. The flag defaults to
  the committed freeze, so the only way to get this wrong is to pass `--freeze ''`.
- **Do not delete `data/nexar/videos/`** on the CPU laptop's side — the 155 clips
  there include all 60 held-out eval positives.
- **Do not enable ReID** (`with_reid: True`). We decided to skip it for v1;
  turning it on changes association and would make these features incomparable
  with the cached pass.
- **Do not re-run the old `eval_anticipation.py --run`** over the full set. Its
  cached output is keyed to the 120-clip split and the stale 0.75 threshold; the
  new pipeline is `extract_features.py`.
