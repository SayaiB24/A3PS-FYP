# GPU laptop handoff — what to do before the agentic-engine phase

This is the checklist for moving `a3ps` from this CPU laptop to your GPU
laptop, finishing the data-heavy work that CPU made impractical, and getting
to a clean starting line before we design the agentic engine (Phase 4/5:
risk decision layer + LLM explainer).

Everything here is either (a) infra you need once, or (b) a `python
scripts/....py` command you run and wait for. None of it requires new code —
the scripts already exist and are tested. Copy-paste the commands.

---

## 0. TL;DR — order of operations

1. Get the code onto the GPU laptop (git clone/pull).
2. Set up the Python env, **install CUDA-enabled torch first**.
3. Copy the data over (bigger than what's on this laptop, since GPU can handle it).
4. Re-run `prepare_nexar.py` (fast, seconds).
5. Run `mine_trajectories.py` for real this time (was interrupted here at 1/20 clips).
6. Train the LSTM (`train_forecaster.ipynb`) — now feasible locally, no need for Colab.
7. Run `eval_forecast.py` → real Kalman-vs-LSTM ADE/FDE numbers.
8. (Recommended) Run `run_pipeline.py` on more real clips now that it's fast.
9. Stop here — risk engine + agentic layer is the *next* phase, not a GPU prerequisite.

---

## 1. Get the project onto the GPU laptop

The repo has a GitHub remote already (`origin`). Data is gitignored (`data/`,
`dashboard/clips/`, `*.pt`, `.venv/`), so **code moves via git; data moves by hand**
(step 3).

On THIS laptop (commit what's pending):
```powershell
cd C:\Users\Sayali.bambal\Downloads\A3PS-Project
git add -A
git commit -m "wip: mining fix, seq2seq forecaster, eval_forecast, dashboard"
git push
```

On the GPU laptop:
```powershell
git clone https://github.com/SayaliB-04/A3ps-actual-P.git
cd A3ps-actual-P\a3ps
```

---

## 2. Python environment (GPU laptop)

Create a fresh venv — **do not reuse a CPU-only venv**, torch needs to be the
CUDA build.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Install torch WITH CUDA first**, before `requirements.txt` (otherwise pip
may grab the CPU wheel). Check your CUDA version:
```powershell
nvidia-smi
```
Look at the "CUDA Version" in the top-right of the output, then pick the
matching command from https://pytorch.org/get-started/locally/. Typically:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(swap `cu121` for `cu124`/`cu118`/etc. to match your driver)

Then everything else:
```powershell
pip install -r requirements.txt
```

**Verify CUDA is actually visible to torch** (this is the #1 way GPU runs
silently fall back to CPU):
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True` and your GPU's name. If `False`, stop and fix the torch
install before continuing — nothing downstream will be fast otherwise.

`Segmenter`/`Tracker` auto-detect CUDA and switch to `.cuda().half()` — no
config change needed.

---

## 3. Verify the environment

```powershell
python -m pytest -q
```
Expect `35 passed`. If this fails, fix it before touching data.
---------------------------------------------------------------------------------------------
Quick real-inference smoke test (downloads `yolov8s-seg.pt` on first run):
```powershell
python -m a3ps.perception.segmenter data\dev_clips\dev01.mp4
```
Look at the printed `mean inference: X ms/frame` — on a GPU this should be
roughly **20-40 ms/frame**, versus ~800-1000 ms/frame on this CPU laptop
(20-40x speedup). If it's still four digits, CUDA isn't being used — recheck
step 2.
---------------------------------------------------------------------------------------------

## 4. Copy the data over

**Only negatives need to increase.** Positives are already exactly sufficient
(65 = 5→dev + 60→eval, zero spare). Do NOT spend transfer time/space on more
positives unless convenient — it's optional and low priority.

**Why negatives are the bottleneck:** the split logic is `dev`=10 negatives
(fixed), `eval`=60 negatives (fixed), and **everything else negative goes to
`train_traj`** — the pool trajectory mining draws from. With 90 negatives
today, only `90 - 10 - 60 = 20` are left for `train_traj`. That's far too few
to hit the 10,000+ mined-window target.

**The exact numbers:**

| | value |
|---|---|
| Target mined windows | 10,000+ |
| Observed yield (1 real clip mined so far, `1875.npz`) | 22 windows/clip |
| `train_traj` clips needed at that yield | 10,000 / 22 ≈ 455 → round to **500-600** for buffer |
| Total negatives needed | 500-600 (train_traj) + 10 (dev) + 60 (eval) = **570-670** |
| Negatives already have | 90 |
| **Additional negatives to bring** | **480-580** |

⚠️ The 22 windows/clip figure is from a **single** clip — treat it as a rough
estimate, not a guarantee (clip length and traffic density vary). Recommended
approach, so you don't haul ~500 clips over on a guess:

1. Bring an initial batch of **~100 additional negatives** (on top of the 90
   already here) and run mining (step 6 below).
2. Check the real `windows/clip` from that run's `stats.json`
   (`total_windows / clips_processed`).
3. Compute the exact clips still needed: `(10000 - windows_so_far) /
   observed_windows_per_clip`, and bring that many more if short.

This turns a guess into a measurement after ~15 minutes of GPU mining, instead
of committing to a transfer size upfront.

**Target layout** (same as before, just more negatives):
```
a3ps/data/nexar/
├── videos/            # ALL clips, flat, original id filenames (e.g. 00042.mp4)
└── labels.xlsx        # bring BOTH train.xlsx and test.xlsx if you have them
```
No `positive/`/`negative/` folders, no pre-sorted split folders — the script
labels from the Excel's `target` column and computes splits itself. If you're
bringing more clips from the `test` set's Excel too, drop both tables side by
side in `data/nexar/` (e.g. `train_labels.xlsx`, `test_labels.xlsx`) —
`prepare_nexar.py` auto-detects and merges every `.xlsx/.csv/.json` it finds
there, matching by `id`.

---

## 5. Re-run prepare_nexar.py

```powershell
python scripts\prepare_nexar.py --root data\nexar
```
This is fast (just probes video metadata, no inference) — seconds even for
thousands of clips. It (re)writes:
- `data\nexar\index.csv` — master table: `clip_id, path, label, split, event_time_s, alert_time_s, duration_s, fps, width, height`
- `data\dev_clips\dev01.mp4 .. devNN.mp4` (+ README.md)

Splits: `dev` = 10 neg + 5 pos (shortest), `eval` = 60 neg + 60 pos, everything
else negative → `train_traj`. More data in step 4 means a bigger `train_traj`
pool — that's the whole point of bringing more.

Sanity check the split sizes:
```powershell
python -c "import csv; from collections import Counter; print(Counter(r['split'] for r in csv.DictReader(open('data/nexar/index.csv'))))"
```

---

## 6. Run trajectory mining for real

This was interrupted on the CPU laptop at 1 of 20 `train_traj` clips (22
windows mined, in `data/trajectories/1875.npz`). On GPU this should run in
minutes instead of hours.

```powershell
python scripts\mine_trajectories.py --plot
```

What to check when it finishes:
- **Total windows mined** (printed at the end) — target **10,000+**. If you
  brought the minimum subset you may land short of that; bring more
  negatives if so (see step 4).
- **`data\trajectories\windows_preview.png`** — 50 random windows, blue
  history / red future, normalized so history ends at the origin heading +y.
  Paths should look **smooth** (no zig-zags — the ID-switch filter worked)
  and **forward-biased** (red segments mostly extend +y).
- **`data\trajectories\stats.json`** — check `class_mix` (mostly car, some
  person/truck) and that `static_kept` is roughly 20% of
  `static_kept + static_dropped`.

Currently mines in **image pixels** (`forecast_space: img` in
`configs/default.yaml`), not BEV metres — that's an intentional decision
until per-clip BEV calibration happens (see `TODO.md`, "Week 4" section). Do
not switch this to `bev` for now.

If you deleted the leftover single shard from the interrupted CPU run and
want a clean start:
```powershell
Remove-Item -Recurse -Force data\trajectories
```
(the mining script overwrites per-clip shards and `stats.json` regardless, so
this is optional cleanup, not required)

---

## 7. Train the LSTM forecaster

`notebooks/train_forecaster.ipynb` was written Colab-ready (pip installs,
Drive mount) but **works identically as a local Jupyter notebook now that
you have a local GPU** — no need to upload anything to Colab.

```powershell
pip install jupyter
jupyter notebook notebooks\train_forecaster.ipynb
```
When it asks for `SHARD_DIR`, point it at `data/trajectories` (the local
fallback path already defaults there if Drive-mount fails, which it will
locally — just confirm the printed path is correct).

Run all cells. It will:
- load + concatenate all shards, 90/10 train/val split (seed 42),
- train `Seq2SeqNet` (encoder/decoder LSTM, hidden 64, 2 layers) with AdamW
  lr 1e-3, up to 30 epochs, early-stopping on validation ADE,
- plot loss curves + 12 sample predictions vs ground truth,
- save the best weights to `models/seq2seq_v1.pt`.

This file is picked up automatically by `Seq2SeqForecaster` and by
`eval_forecast.py` (default `--weights models/seq2seq_v1.pt`).

---

## 8. Run the forecaster evaluation

```powershell
python scripts\eval_forecast.py
```
This computes ADE/FDE at 1s/2s/4s for **both** Kalman-CV and (now that
weights exist) Seq2Seq-LSTM, on the same held-out val split the notebook
used. Prints a table and writes `eval/forecast_table.md`. This is the real
version of the 2-window sanity check we ran on the CPU laptop.

---

## 9. (Recommended, not required) Process more real clips through the pipeline

Now that inference is fast, it's worth generating real `annotated.mp4` +
`events.json` for more than the one test clip (`02134`) we have so far:

```powershell
python scripts\run_pipeline.py --video data\dev_clips\dev01.mp4 --out dashboard\clips\dev01\
```
Repeat for a few `dev0N.mp4` clips (or write a small loop). Add their ids to
`dashboard\clips\manifest.json` to see them in the dashboard. Note: every
track will still render `risk_level: safe` with `events: []` — the risk
engine isn't wired into the pipeline yet, this step only gets you real
detections/tracking/prediction overlays on more footage.

---

## 10. What NOT to do yet (this is the next phase, not a prerequisite)

These are genuinely blocked on **code that doesn't exist yet**, not on GPU or
data — don't spend GPU-laptop time chasing them before we design the agentic
engine together:

- **Wiring the risk engine into the pipeline** (`a3ps/risk/gaussian.py` →
  `collision.py` → `decision.py` exist but the `risk_engine` hook in
  `run_pipeline.py` is still `None`). This — plus the LLM explainer
  (`a3ps/explain/llm_client.py`) — **is** the agentic-engine design work.
- `scripts/eval_anticipation.py` (mTTA / AP / false-alarm rate) — needs the
  risk engine wired in, and reads `event_time_s`/`alert_time_s` from your
  Excel (already flowing into `index.csv`, ready whenever this is built).
- Dashboard real-events wiring (`fake_demo` in `manifest.json`) — same
  blocker; the dashboard already reads real clips fine (step 9), it just has
  nothing risk-related to show yet.
- Week 4 per-clip BEV calibration — do this only when prepping the 5 final
  demo clips, not now.

---

## 11. Required outputs — do not leave the GPU laptop without these

This is the exact deliverable checklist. If any of these are missing or fail
their check, that step isn't done yet.

| # | File(s) | Must satisfy |
|---|---|---|
| 5 | `data/nexar/index.csv` | Loads with csv/pandas; `split` column has `dev` (15), `eval` (120), `train_traj` (500+, was 20) |
| 5 | `data/dev_clips/dev01.mp4` … `dev15.mp4` + `README.md` | 15 files present, each opens with OpenCV |
| 6 | `data/trajectories/*.npz` (one per train_traj clip) | Each loads via `np.load`; `history` shape `(N,10,2)`, `future` shape `(N,20,2)` |
| 6 | `data/trajectories/stats.json` | `total_windows >= 10000`; `class_mix` non-empty; `static_kept / (static_kept+static_dropped) ≈ 0.20` |
| 6 | `data/trajectories/windows_preview.png` | Visually smooth + forward-biased (eyeball check, see step 6) |
| 7 | `models/seq2seq_v1.pt` | Loads via `Seq2SeqForecaster(weights_path=...)` without error; note the printed **best val ADE** from training — write it down, you'll want it for comparison |
| 8 | `eval/forecast_table.md` | Has a populated `Seq2Seq-LSTM` row (not just Kalman-CV) — this is the real Kalman-vs-LSTM comparison the whole GPU trip was for |
| 9 (optional) | `dashboard/clips/<id>/{annotated.mp4, events.json, meta.json}` for a few more dev clips + updated `dashboard/clips/manifest.json` | Each `events.json` loads via `ClipResult.from_dict` |

If step 8's table doesn't exist or only shows Kalman-CV, step 7 (training)
didn't actually save `models/seq2seq_v1.pt` — check the notebook ran to its
last cell without error.

---

## 12. Bringing it back to this (CPU) laptop

**Push code changes first** (same as step 1, reversed):
```powershell
git add -A
git commit -m "GPU laptop: full mining run, trained seq2seq, real eval"
git push
```
On this laptop: `git pull`. That brings back any code you touched (there
shouldn't be much — this phase is data/training, not code).

**Then manually copy back ONLY the derived artifacts below.** Everything else
in `data/nexar/videos/` (the 500+ negative clips) is disposable — it was only
needed to *produce* these outputs, and none of the upcoming agentic-engine
wiring work needs to re-run mining or bulk video processing. Leave the raw
video pool on the GPU laptop (or an external drive); do not copy 30GB+ back.

**Copy back (small, required):**
```
data/nexar/index.csv                  # the bigger, real split table
data/dev_clips/                       # dev01..dev15.mp4 + README.md
data/trajectories/                    # *.npz + stats.json + windows_preview.png
models/seq2seq_v1.pt                  # trained weights
eval/forecast_table.md                # real Kalman-vs-LSTM numbers
```
**Copy back (optional, only if you did step 9 and want them here too):**
```
dashboard/clips/<new ids>/{events.json, meta.json}   # small, worth bringing
dashboard/clips/<new ids>/{annotated.mp4, raw.mp4}   # bigger, optional — nice
                                                       # for demoing but not
                                                       # needed for wiring work
dashboard/clips/manifest.json                         # if you updated it
```
**Do NOT copy back:** `data/nexar/videos/` (the full clip pool).

### Verify on THIS laptop before starting agentic-engine work

Run these in order. All should be fast (no video re-processing needed) —
if any step suddenly wants to reprocess video, something didn't copy right.

```powershell
# 1. code + tests still pass
python -m pytest -q                                    # expect 35 passed

# 2. index.csv came back correctly
python -c "import csv; from collections import Counter; print(Counter(r['split'] for r in csv.DictReader(open('data/nexar/index.csv'))))"
#   expect: Counter({'eval': 120, 'train_traj': 500+, 'dev': 15})

# 3. dev clips are real files, not zero-byte copies
Get-ChildItem data\dev_clips\*.mp4 | Select-Object Name, Length

# 4. trajectory shards are intact
python -c "import glob, numpy as np; s=glob.glob('data/trajectories/*.npz'); print(len(s), 'shards'); d=np.load(s[0]); print(d['history'].shape, d['future'].shape)"

# 5. stats.json still shows the real numbers (not the old 22-window run)
python -c "import json; print(json.load(open('data/trajectories/stats.json'))['total_windows'])"

# 6. trained weights load and produce a sane prediction shape
python -c "from a3ps.forecasting.seq2seq import Seq2SeqForecaster; f=Seq2SeqForecaster(weights_path='models/seq2seq_v1.pt'); m,s=f.predict([[0.0,0.0],[1.0,1.0]],0.2,4.0); print(len(m), len(s))"
#   expect: 20 20

# 7. re-run eval on THIS machine to confirm cross-laptop reproducibility
#    (this is fast — it's LSTM forward passes on small arrays, not video,
#     so CPU is fine here)
python scripts\eval_forecast.py
#   expect the same table as eval/forecast_table.md from the GPU laptop

# 8. (if you brought step-9 clips) dashboard still renders them
python scripts\serve_dashboard.py
#   open localhost:8000, select the new clip id, confirm it plays
```

If all 8 pass, the data/forecasting track is genuinely done and verified end
to end — safe to start wiring the risk engine and agentic layer with
confidence that nothing upstream is silently broken.

---

## Quick command cheat-sheet

```powershell
# env
nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest -q

# data
python scripts\prepare_nexar.py --root data\nexar

# mining
python scripts\mine_trajectories.py --plot

# train (via notebook, or headless if you prefer — ask me for a script version)
jupyter notebook notebooks\train_forecaster.ipynb

# eval
python scripts\eval_forecast.py

# more real clips for the dashboard
python scripts\run_pipeline.py --video data\dev_clips\dev01.mp4 --out dashboard\clips\dev01\
```

Ping me once steps 1-8 are done (or if anything errors) and we'll move into
designing the agentic risk/decision + explainer layer.
