# GPU laptop instructions — Phase IV (learned temporal risk head)

Everything in this file runs on the **GPU laptop**. Nothing here has been run on
the CPU laptop; the CPU-side work (split freeze, metric re-key, feature
extractor, ego helper, anticipation loss, temporal head, smoke train) is already
done and committed.

Run the steps in order. Steps 0–2 are the download and a cheap measurement; do
**not** start step 4 (the 1,500-clip pass) until step 3 has told us which frame
rate to use.

---

## 0. Sync code and check CUDA

```powershell
cd C:\path\to\A3ps-actual-P
git pull
cd a3ps
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

## 1. Already downloaded — confirm the layout

You already have the full set as **separate train/test video folders and
separate train/test label sheets**, rather than one pooled folder + one table.
`prepare_nexar.py` already supports this shape directly — it takes multiple
`--videos` dirs and multiple `--annotation` files and merges them — so nothing
needs to be combined by hand. Note the four paths, called below `<TRAIN_VIDEOS>`,
`<TEST_VIDEOS>`, `<TRAIN_XLSX>`, `<TEST_XLSX>`.

Sanity-check the layout and label coverage before indexing anything (cheap,
catches a wrong path or an unexpectedly-blank column immediately):

```powershell
Get-ChildItem <TRAIN_VIDEOS> -Filter *.mp4 | Measure-Object | Select-Object Count
Get-ChildItem <TEST_VIDEOS>  -Filter *.mp4 | Measure-Object | Select-Object Count
python -c "import openpyxl; wb=openpyxl.load_workbook(r'<TRAIN_XLSX>', read_only=True); ws=wb.active; rows=list(ws.iter_rows(values_only=True)); print('train:', ws.title, len(rows)-1, 'rows, header', rows[0])"
python -c "import openpyxl; wb=openpyxl.load_workbook(r'<TEST_XLSX>', read_only=True); ws=wb.active; rows=list(ws.iter_rows(values_only=True)); print('test: ', ws.title, len(rows)-1, 'rows, header', rows[0])"
```

Expect roughly 750+750 positive/negative videos across the two folders (or
however train/test are actually split — Nexar's own split, not ours) and a
`target` column with both 0s and 1s in the test sheet too, since you confirmed
it is labeled and not a blank Kaggle submission holdout.

Also check the two label tables don't silently disagree on an id that appears in
both (`prepare_nexar.py` gives the FIRST table passed precedence on a collision,
so this only matters if train/test aren't cleanly disjoint):

```powershell
@'
import sys
sys.path.insert(0, "scripts")
from prepare_nexar import load_table, canon
tr = {canon(r["id"]): r for r in load_table(r"<TRAIN_XLSX>") if r["id"] is not None}
te = {canon(r["id"]): r for r in load_table(r"<TEST_XLSX>") if r["id"] is not None}
overlap = set(tr) & set(te)
print(f"train {len(tr)} ids, test {len(te)} ids, overlap {len(overlap)}")
if overlap:
    mism = [i for i in overlap if tr[i]["target"] != te[i]["target"]]
    print(f"  of which {len(mism)} disagree on target -- e.g. {mism[:5]}")
'@ | python -
```

If `overlap` is 0, the two are disjoint splits as expected and nothing else to
worry about. If it's nonzero and `mism` is nonzero, stop and send me the output —
that means the same clip id carries two different labels depending on which
table you read, which is a data problem, not something to route around here.

---

## 2. Re-index WITHOUT re-drawing the eval split

This is the step that would silently destroy comparability if done wrong.
`prepare_nexar.py` assigns splits with a pool-size-dependent shuffle, so simply
re-running it over 1,500 clips would hand us a different 120-clip held-out set —
invalidating the 3.7 GB cache and every number measured so far.

The `--freeze` flag (default `eval/split_freeze.json`, committed) prevents that:
pinned clips keep their split verbatim and new clips can only join `train_traj`.

`--videos` and `--annotation` both accept multiple paths and merge them, so the
train/test folders and sheets don't need to be combined by hand first — pass
train before test so train wins on the (expected-to-be-zero, per step 1's check)
id collision case.

Copy the existing 155 videos into one of the pools first, so the pinned clips are
present under a folder `prepare_nexar.py` will scan (either works; `<TRAIN_VIDEOS>`
is shown here since that pool is usually larger):

```powershell
Copy-Item C:\path\to\A3ps-actual-P\a3ps\data\nexar\videos\*.mp4 <TRAIN_VIDEOS> -Force
```

Then re-index over BOTH splits' videos and BOTH label sheets in one pass:

```powershell
python scripts/prepare_nexar.py --root data/nexar `
    --videos <TRAIN_VIDEOS> <TEST_VIDEOS> `
    --annotation <TRAIN_XLSX> <TEST_XLSX> `
    --freeze eval/split_freeze.json
```

Expect `splits (frozen from eval/split_freeze.json): ...` and
`135 clip ids replayed from the freeze`. Runtime: a few minutes (it probes every
video's metadata across both folders).

**It will also print a warning like `! 1,435 positive(s) are not in the freeze and
were left UNUSED`.** That is deliberate — positives are the scarce resource and
the script will not silently assign them. Give them a training split explicitly:

```powershell
python scripts/assign_unused_positives.py --index data/nexar/index.csv --split train_risk
```

Verify the held-out set survived, then commit the index so the CPU laptop has it:

```powershell
python scripts/freeze_split.py verify --index data/nexar/index.csv
Copy-Item data\nexar\index.csv eval\nexar_index_full.csv
git add eval/nexar_index_full.csv
git commit -m "eval: full 1500-clip index with frozen dev/eval split"
git push
```

`verify` must print `OK: ... matches the 135 pinned clips`. If it reports
MISMATCH, **stop** and send me the output — do not proceed to extraction.

---

## 3. Measure association quality at 30 Hz vs 10 Hz (do this before step 4)

Decimating to 10 Hz is what turns a 9–19 h pass into 1–2 h. The risk is track
fragmentation: at 10 Hz an actor moves 3× further between frames, and the
trajectory buffer needs 2 s of *continuous* history before it forecasts at all,
so a fragmented track yields no features rather than noisier ones.

```powershell
python scripts/check_assoc_rate.py --index data/nexar/index.csv `
    --split eval --limit 10 --rates 30,10 `
    --window-s 13 --tail-s 1 --out eval/assoc_rate_check.md
```

Runtime: ~5.2 k frames of inference total, so **3–5 min** on a GPU.

Read the `frac >=2s` column — the fraction of tracks that live long enough to be
forecastable. The script prints its own verdict:

| result | meaning | action |
|---|---|---|
| within ~3% of 30 Hz | 10 Hz is safe | proceed to step 4 with `--rate-hz 10` |
| 3–10% worse | borderline | re-run with `--rates 30,15` and use 15 Hz |
| >10% worse | 10 Hz costs more than it saves | use 30 Hz, accept 9–19 h |

**Send me `eval/assoc_rate_check.md` before starting step 4.** If you'd rather not
wait on me, the table's verdict line is the decision — follow it.

---

## 4. Extract features over the full set

Windowed at 13 s and decimated to the rate chosen in step 3. Positives are
windowed around `event_time_s` so the window always contains `time_of_alert` plus
~8 s of run-up; negatives get a deterministic hash-seeded window.

Run the three splits separately (adjust `--rate-hz` per step 3):

```powershell
# training negatives (~750 + the existing 220)
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

**Expected runtime.** GPU throughput for this pipeline has never been measured
(`results.md:220` says so explicitly), so these come from the docs' 20–40 ms/frame
estimate and should be treated as a range, not a promise. Step 3 gives you the
first real per-clip number — use it to refine these.

| scope | frames @10 Hz | @20 ms | @40 ms |
|---|---|---|---|
| eval (120 clips) | ~15.6 k | ~5 min | ~10 min |
| full 1,500 clips | ~195 k | ~1.1 h | ~2.2 h |
| full 1,500 @30 Hz | ~585 k | ~3.3 h | ~6.5 h |

Add 5–15% for video decode and the optical-flow pass. `--ego-downscale 2` (the
default) keeps the flow cost small; raise it to 4 if decode dominates.

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
