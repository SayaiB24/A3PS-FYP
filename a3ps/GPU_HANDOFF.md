# GPU laptop handoff — what to do for the rest of the project

Status snapshot as of this rewrite: **all five pipeline phases are implemented
and unit-tested on CPU** (perception → tracking → forecasting → risk/decision
→ explanation), **63 tests passing**. What's left is entirely *scale* work —
things that need either a GPU (fast YOLO/tracking inference over many real
clips) or a lot of real data (trajectory mining, LSTM training, eval over the
full 120-clip eval split). None of it needs new code.  
This laptop (CPU, no GPU) already has real Nexar data on it — **155 labelled
clips** (65 positive + 90 negative), the 15 dev clips, and one trained-but-tiny
LSTM checkpoint. The GPU laptop needs that data copied over (it's gitignored,
so `git clone` won't bring it), plus a few environment secrets that are also
gitignored.

---

## 0. TL;DR — order of operations

1. Pull the latest code (risk/decision/explain phases + two dashboard wiring
   fixes landed since the last handoff).
2. Set up the Python env, **install CUDA-enabled torch first**, recreate the
   gitignored `.env` (`GROQ_API_KEY`).
3. Copy `data/`, `dashboard/clips/`, `notebooks/models/` over from this laptop
   (bigger data can go straight to the GPU laptop instead of round-tripping
   here — see step 4).
4. Bring **~480-580 more negative clips** (the real bottleneck — see step 4's
   math; positives are already sufficient).
5. Re-run `prepare_nexar.py` (fast, seconds) — bigger `train_traj` split.
6. Resume `mine_trajectories.py` — the GPU laptop has already mined **~93
   clips**; re-running just skips those and continues with any remaining
   `train_traj` clips. Check `total_windows` in `stats.json` against the
   10,000+ target.
7. Retrain the LSTM (`train_forecaster.ipynb`) on the real mined set — the
   current `notebooks/models/seq2seq_v1.pt` was trained on 22 windows from a
   single clip and should be treated as a placeholder, not a real result.
8. Run `eval_forecast.py` → real Kalman-vs-LSTM ADE/FDE numbers.
9. Run `run_pipeline.py` on real clips — **risk_engine is now wired**, so this
   produces real ALERT/VIRTUAL_BRAKE events, not just tracking overlays.
   Update `dashboard/clips/manifest.json` to point at them.
10. Run `eval_anticipation.py --run` over the real 120-clip `eval` split — the
    headline mTTA / AP / false-alarm-rate numbers for the report.
11. (Optional, free) Run the Groq MLLM enrichment on 2-3 dev clips and
    eyeball the narratives for hallucination.
12. Week 4, only when prepping the 5 final demo clips: per-clip BEV
    calibration (unrelated to GPU — do this last, on any machine).

---

## 1. Sync code (and secrets) onto the GPU laptop

Code moves via git; data and the `.env` secrets file are gitignored and move
by hand.

On THIS laptop (commit what's pending):
```powershell
cd C:\Users\Sayali.bambal\Downloads\A3PS-Project
git add -A
git commit -m "risk/decision/explain phases, dashboard wiring fixes, Groq enrichment"
git push
```

On the GPU laptop:
```powershell
git clone https://github.com/SayaliB-04/A3ps-actual-P.git
cd A3ps-actual-P\a3ps
```

**Recreate `.env`** (it's gitignored on purpose — never commit a real key):
```powershell
"GROQ_API_KEY=your_real_key_here" | Out-File -Encoding utf8 .env
```
Get a free key at https://console.groq.com/keys if you don't already have
one. This is only needed for step 11 (optional) — everything else works
without it.

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
This now also installs `groq` (offline MLLM enrichment) and `python-dotenv`
(loads `.env` automatically) — neither needs a GPU.

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
Expect **`63 passed`**. All of these are CPU-only, data-free (synthetic
fixtures) — if this fails, fix it before touching real data or the GPU.

Quick real-inference smoke test (downloads `yolov8s-seg.pt` on first run if
you didn't copy it over — see step "what to copy" below):
```powershell
python -m a3ps.perception.segmenter data\dev_clips\dev01.mp4
```
Look at the printed `mean inference: X ms/frame` — on a GPU this should be
roughly **20-40 ms/frame**, versus ~800-1000 ms/frame on a CPU laptop
(20-40x speedup). If it's still four digits, CUDA isn't being used — recheck
step 2.

---

## 4. Copy the data over

**What this laptop already has** (copy these over as a starting point instead
of re-downloading): `data/nexar/videos/` (133 real clips), `data/nexar/index.csv`,
`data/nexar/labels.xlsx`, `data/dev_clips/` (all 15 dev clips + README),
`data/trajectories/1875.npz` + `stats.json` (1 clip mined, 22 windows —
treat as a smoke-test artifact, not real training data),
`notebooks/models/seq2seq_v1.pt` (trained on that same 22-window set — a
placeholder, will be overwritten in step 7), `yolov8s-seg.pt`.

**Only negatives need to increase.** Positives are already exactly sufficient
(65 = 5→dev + 60→eval, zero spare). Do NOT spend transfer time/space on more
positives unless convenient — it's optional and low priority.

**Why negatives are the bottleneck:** the split logic is `dev`=10 negatives
(fixed), `eval`=60 negatives (fixed), and **everything else negative goes to
`train_traj`** — the pool trajectory mining draws from. Current `index.csv`
confirms this exactly: `Counter({'eval': 120, 'train_traj': 20, 'dev': 15})` —
with 90 negatives total, only `90 - 10 - 60 = 20` are left for `train_traj`.
That's far too few to hit the 10,000+ mined-window target. (This reflects
*this* CPU laptop; the GPU laptop already brought more negatives and has mined
~93 clips — see step 6. Verify its `total_windows` before assuming more
negatives are still needed.)

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
Expect `dev: 15, eval: 120` unchanged, and `train_traj` much bigger than the
current `20`.

---

## 6. Resume trajectory mining

The GPU laptop has already mined **~93 `train_traj` clips** (this CPU laptop
only ever mined 1 — `clips_processed: 1`, 22 windows in its local
`data/trajectories/stats.json` — that figure is stale and refers to *this*
machine, not the GPU one). So mining is largely done; this step is now
**resume + verify**, not a from-scratch run.

**Re-running resumes automatically** — mining skips any clip whose
`data/trajectories/<clip_id>.npz` shard already exists (counted as
`clips_skipped_existing` in stats.json) and only mines the rest. So just
re-run it and it picks up where it left off:
```powershell
python scripts\mine_trajectories.py --plot
```
`stats.json`'s `clips_processed` / `total_windows` are **self-healed** each
run by recounting the actual shards on disk, so they reflect the true
cumulative total (all ~93 + any new ones), not a reset.

Two things to know about resume:
- **Force a full re-mine** with `--force` (re-processes every clip even if a
  shard exists), or wipe the dir for a truly clean start:
  ```powershell
  Remove-Item -Recurse -Force data\trajectories
  python scripts\mine_trajectories.py --plot
  ```
- A clip that yielded **zero** valid windows writes **no** shard, so it isn't
  remembered as done and gets re-tracked on every run — harmless, just some
  repeated work on empty/low-detection clips.

**The one number that matters now:** check `total_windows` in
`data/trajectories/stats.json` against the **10,000+** target. If it's already
there, mining is done — go to step 7. If it's short, bring more negatives
(step 4) and re-run (it'll resume, only mining the new clips).

What to check when it finishes:
- **Total windows mined** (printed at the end, and in `stats.json`) — target
  **10,000+**. If you brought the minimum subset you may land short of that;
  bring more negatives if so (see step 4).
- **`data\trajectories\windows_preview.png`** — 50 random windows, blue
  history / red future, normalized so history ends at the origin heading +y.
  Paths should look **smooth** (no zig-zags — the ID-switch filter worked)
  and **forward-biased** (red segments mostly extend +y).
- **`data\trajectories\stats.json`** — check `class_mix` (mostly car, some
  person/truck) and that `static_kept` is roughly 20% of
  `static_kept + static_dropped`.

Currently mines in **image pixels** (`forecast_space: img` in
`configs/default.yaml`), not BEV metres — that's an intentional decision
until per-clip BEV calibration happens (step 12 / `TODO.md` "Week 4"
section). Do not switch this to `bev` for now.

---

## 7. Retrain the LSTM forecaster

`notebooks/models/seq2seq_v1.pt` already exists but was trained on the
22-window/1-clip placeholder set — **retrain it** once step 6 produces the
real mined set, or the "LSTM vs Kalman" comparison in step 8 is meaningless.

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
- save the best weights to `notebooks/models/seq2seq_v1.pt` (overwrites the
  placeholder).

Note the printed **best val ADE** — you'll want it for comparison against the
old placeholder run and against Kalman-CV in step 8.

---

## 8. Run the forecaster evaluation

```powershell
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_v1.pt
```
⚠️ **Pass `--weights` explicitly** — the script's default (`models/seq2seq_v1.pt`)
does not match where the notebook actually saves the file
(`notebooks/models/seq2seq_v1.pt`). Without the flag it silently skips the
Seq2Seq row and only reports Kalman-CV.

This computes ADE/FDE at 1s/2s/4s for **both** Kalman-CV and Seq2Seq-LSTM, on
the same held-out val split the notebook used. Prints a table and writes
`eval/forecast_table.md`. The existing `eval/forecast_table.md` in the repo
right now was run against the tiny placeholder set — expect the real numbers
to shift once the LSTM is retrained on 10,000+ windows.

### 8.1 Comparing a previous vs a new model (fallback / A-B)

`eval_forecast.py` takes `--weights` (which checkpoint) and `--out` (which
table file), so you can keep more than one model and score each into its own
table without overwriting the other. Keep the two checkpoints under distinct
names, e.g. `seq2seq_prev.pt` (previous) and `seq2seq_new.pt` (retrained):

```powershell
# NEW model  -> main table
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_new.pt --out eval\forecast_table.md

# PREVIOUS model -> its own table
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_prev.pt --out eval\forecast_table_oldmodel.md
```

⚠️ **The numbers depend on the mined data in `--shards` (default
`data/trajectories`), not just the model.** Running the *previous* model on the
*current* (larger) mined set will NOT reproduce its original 51.64 figure —
that number was on the smaller earlier set and is archived verbatim in
`eval/forecast_table_PREVIOUS.md`. Both `--weights` commands above score on
whatever is in `data/trajectories` *now*, so they are a fair model-vs-model
comparison on today's data. To cite the old result in the report, just use the
archived `forecast_table_PREVIOUS.md`.

Only `eval_forecast.py` uses the LSTM at all — `run_pipeline.py` (step 9) and
`eval_anticipation.py` (step 10) always forecast with Kalman-CV and have no
model flag, so the choice of `.pt` never affects the dashboard or the
anticipation numbers.

---

## 9. Run the full pipeline on real clips (risk + decision are now wired)

This step changed meaningfully since the last handoff: **the risk/decision
engine is fully implemented and wired into `run_pipeline.py`**. Running the
pipeline now produces real `collision_prob`, `risk_level` (green/amber/red),
and ALERT/VIRTUAL_BRAKE/THRESHOLD_LOWERED events — not just tracking overlays.

```powershell
python scripts\run_pipeline.py --video data\dev_clips\dev01.mp4 --out dashboard\clips\dev01\
```
Repeat for the other dev clips (or loop over all 15):
```powershell
Get-ChildItem data\dev_clips\dev*.mp4 | ForEach-Object {
    $id = $_.BaseName
    python scripts\run_pipeline.py --video $_.FullName --out "dashboard\clips\$id\"
}
```

**Every clip currently in `dashboard/clips/` (`dev01`, `dev11`, `02134_pipe`)
was generated BEFORE risk/decision was wired in** — their `meta.json` shows
`"risk_engine": false` and their `events.json` has **zero events**. Re-run them
with the command above to get real events.

Once you have real clips, update the manifest (currently `["fake_demo",
"02134_pipe"]` — both stale/dummy):
```powershell
'["dev01", "dev11", "dev04"]' | Out-File -Encoding utf8 dashboard\clips\manifest.json
```
Pick a mix of positive and negative dev clips so the demo shows both a real
ALERT/VIRTUAL_BRAKE sequence and a clean run.

**Verify in the dashboard:**
```powershell
python scripts\serve_dashboard.py
```
Open `localhost:8000`, select a re-run clip, and confirm: masks/boxes color
green→amber→red as risk rises, a red border flashes for 0.5s after a
VIRTUAL_BRAKE, the event log shows real ALERT/VIRTUAL_BRAKE rows with
`explanation_template` text, the risk timeline's threshold line actually steps
down when a `crosswalk_ahead`/`intersection` context flag is active (this
requires a per-clip `dashboard/clips/<id>/context.json` — optional, only
needed to demo THRESHOLD_LOWERED), and the ▼ A3PS vs ▽ reactive-ADAS gap
markers appear on any VIRTUAL_BRAKE.

*(Two small wiring gaps between the Python side and the dashboard were fixed
alongside this handoff rewrite — `context.active_threshold` and
`ego.corridor_poly_bev` are now written every frame. If you're running code
from before this commit, `git pull` first.)*

### 9.1 How VIRTUAL_BRAKE is decided — the full logic and math

A VIRTUAL_BRAKE is **not** triggered by a real collision happening in the
video. It fires when the system *predicts* a future collision confidently
enough, for long enough. Forecasting uses **Kalman-CV** (not the LSTM). The
chain, per track, per frame:

**Step A — forecast the actor's future.** For any track with enough history,
Kalman-CV predicts its next `horizon_s` seconds (4.0 s) at `predict_hz`
(5 Hz) → **20 future steps, dt = 0.2 s apart**, each a Gaussian: a mean
position + a per-step std (uncertainty grows further out).
(`scripts/run_pipeline.py:make_forecaster_hook` → `a3ps/forecasting/kalman_cv.py`)

**Step B — per-step collision probability.**
(`a3ps/risk/collision.py:step_collision_prob`) For each of the 20 predicted
Gaussians, sample 5 **sigma points**, and compute the weighted fraction of
those points that land **inside the ego corridor, dilated by the actor's
radius**. Plain English: *"given where this actor will probably be and how
sure we are, how much of that probability mass sits in the car's path?"* →
a number in `[0, 1]` for that step.
- The **ego corridor** is the fixed trapezoid over the lower-centre of the
  frame (`ego_corridor` in `configs/default.yaml`).
- **Actor radius** dilates the corridor so a car counts as colliding when its
  *body* (not just its centre point) enters the path (`actor_radius_px` /
  `actor_radius_m`).

**Step C — collapse the trajectory to one number + a TTC.**
(`trajectory_collision_prob`) Take the 20 per-step probabilities, smooth them
with a 3-step moving average, then:
- `max_prob` = the highest smoothed probability over the 4 s horizon.
- `ttc_s` (time-to-collision) = the time of the **first** step whose prob
  exceeds 0.5; else the time of the peak-probability step. This is the
  "collision predicted in ~X s" estimate shown on events.

**Step D — smooth across frames.** (`RiskSmoother`, EMA `risk_ema_alpha =
0.4`) Blend this frame's `max_prob` with the running per-track average so a
single-frame glitch can't fire an intervention. The result is the track's
**`collision_prob`**.

**Step E — the decision state machine.** (`a3ps/risk/decision.py`) Each actor
climbs `SAFE → ALERT → BRAKE`, and each forward step must be *confirmed for
`frames_to_confirm` = 3 consecutive frames*:

| Transition | Condition (must hold 3 frames in a row) | Emits |
|---|---|---|
| SAFE → ALERT | `collision_prob ≥ active_threshold − alert_margin` (0.75 − 0.15 = **0.60**) | `ALERT` |
| ALERT → BRAKE | `collision_prob ≥ active_threshold` (**0.75**) | `VIRTUAL_BRAKE` |

- **`active_threshold`** starts at `base_threshold` (0.75) and is *lowered*
  under adverse context so we brake sooner: `crosswalk_ahead` −0.10,
  `intersection` −0.10, `dense_traffic` (≥8 tracks) −0.05, never below
  `threshold_floor` 0.45.
- **BRAKE is terminal** for that incident — one ALERT + one VIRTUAL_BRAKE per
  near-miss, not one per frame. The actor only re-arms after `collision_prob`
  falls below `caution_threshold` (0.4) **and** `event_cooldown_s` (2 s) has
  passed.

**So, in one sentence:** *VIRTUAL_BRAKE fires when a track's smoothed,
predicted collision probability stays at/above the danger threshold (0.75 by
default, lower in risky context) for 3 straight frames, after it has already
crossed the ALERT level (0.60).*

**Tuning knobs (all in `configs/default.yaml`, no retraining):** lower
`base_threshold` or raise `horizon_s` → brakes **earlier**; lower
`frames_to_confirm` → fires **sooner** (but jitterier); widen the
`ego_corridor` or `actor_radius_*` → more sensitive. If a clip brakes *after*
the collision (e.g. the dev14 case), it means `collision_prob` only crossed
0.75 late — usually because the actor entered the corridor / became
predictable only just before impact; lowering `base_threshold` and/or
lengthening `horizon_s` is the first thing to try.

---

## 10. Run the anticipation eval over the real eval split

```powershell
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --run
```
`--run` processes every one of the 120 real `eval` clips through the full
pipeline **with rendering disabled** (`render=False` — no annotated.mp4/raw.mp4,
just `events.json`), so it is far faster than step 9. Still the GPU-heavy part
of this step; strip `--run` to re-score clips already processed into
`--clips-dir`. Point `--clips-dir` at a smaller pre-processed directory to
iterate on a subset.

**It scores two methods side by side and writes two files:**
- **A3PS (proactive)** — the pipeline's forecast + risk ALERT/VIRTUAL_BRAKE
  events.
- **Reactive-proximity baseline** — a naive ADAS that "brakes" the first frame
  any actor comes physically close to the ego corridor (BEV < 2.0 m, else image
  < 8% of frame height — identical to the dashboard's reactive-ADAS marker). No
  forecasting, so it can only react late; A3PS's mTTA advantage over it is the
  headline "we warn N seconds earlier" number.

Outputs:
- `eval/anticipation.md` — the comparison table (detection rate, false-alarm
  rate, mTTA for **both** methods) plus the anticipation-gain line.
- `eval/anticipation_per_clip.csv` — one row per clip: label, event time, and
  each method's alert time / TTA / flagged / outcome.

Both methods share the same metric definitions (detection = alerted *before* the
annotated event time; false alarm = any alert on a negative), so the comparison
is apples-to-apples. Expect A3PS to show a **higher mTTA** (fires earlier) than
the reactive baseline — that gap is the whole point of the forecast layer.

---

## 11. (Optional, free) Offline MLLM enrichment via Groq

Not required — the deterministic `explanation_template` already satisfies the
XAI requirement — but if you want the richer scene-aware narrative:

```powershell
python -m a3ps.explain.llm_client dashboard\clips\dev01 --dry-run   # sanity check, no API call
python -m a3ps.explain.llm_client dashboard\clips\dev01             # real enrichment
```
Needs `GROQ_API_KEY` in `.env` (step 1) — Groq's free tier covers this easily.
Do this for 2-3 dev clips and read the `explanation_llm` strings: **pass
criterion is that narratives mention only real scene elements** (no invented
objects/numbers). If one hallucinates, tighten `SYSTEM_PROMPT` in
`a3ps/explain/llm_client.py` and re-run with `--overwrite`; keep one
before/after example for the report's hallucination-mitigation note.

---

## 12. Week 4 — per-clip BEV calibration (do this last, any machine)

Unrelated to GPU/data volume — do this only when prepping the 5 final demo
clips, not as part of the bulk GPU work above.

The default ground-plane trapezoid is generic and miscalibrated for real
clips. Pipeline default stays `forecast_space: img` until this is done.

For each of the 5 final demo clips:
- Pick 4 road points in the frame + estimate their real-world distances.
- Save `dashboard/clips/<clip>/ground.yaml` (image_points_frac + ground_points_m).
- Set `ground_plane_override` (or per-clip config) and `forecast_space: bev`.
- Re-run `scripts/verify_bev.py` — velocities should read ~5-20 m/s and
  predictions should stay in-frame (>=80%).

---

## 13. Required outputs — do not leave the GPU laptop without these

| # | File(s) | Must satisfy |
|---|---|---|
| 5 | `data/nexar/index.csv` | `split` column has `dev` (15), `eval` (120), `train_traj` (500+, was 20) |
| 6 | `data/trajectories/*.npz` (one per train_traj clip) | Each loads via `np.load`; `history` shape `(N,10,2)`, `future` shape `(N,20,2)` |
| 6 | `data/trajectories/stats.json` | `total_windows >= 10000`; `class_mix` non-empty; `static_kept / (static_kept+static_dropped) ≈ 0.20` |
| 6 | `data/trajectories/windows_preview.png` | Visually smooth + forward-biased |
| 7 | `notebooks/models/seq2seq_v1.pt` | Loads via `Seq2SeqForecaster(weights_path=...)`; note the printed **best val ADE** |
| 8 | `eval/forecast_table.md` | Has a populated `Seq2Seq-LSTM` row from the *retrained* model (not the 22-window placeholder) |
| 9 | `dashboard/clips/<id>/{annotated.mp4, events.json, meta.json}` for several dev clips + updated `dashboard/clips/manifest.json` | `meta.json.stages.risk_engine == true`; `events.json` has non-zero `events` for at least one positive clip |
| 10 | `eval/anticipation.md` + `eval/anticipation_per_clip.csv` | A3PS-vs-reactive-baseline detection rate / false-alarm rate / mTTA over the 120-clip eval split; A3PS mTTA > reactive mTTA (fires earlier); per-clip CSV has both methods |
| 11 (optional) | enriched `events.json` with populated `explanation_llm` on 2-3 clips | Narratives mention only real scene elements — no hallucinated objects/numbers |

---

## 14. Bringing it back to this (CPU) laptop

**Push code changes first:**
```powershell
git add -A
git commit -m "GPU laptop: full mining run, retrained seq2seq, real eval + anticipation numbers"
git push
```
On this laptop: `git pull`.

**Then manually copy back ONLY the derived artifacts below.** Everything else
in `data/nexar/videos/` (the 500+ negative clips) is disposable — it was only
needed to *produce* these outputs. Leave the raw video pool on the GPU laptop
(or an external drive); do not copy 30GB+ back.

**Copy back (small, required):**
```
data/nexar/index.csv                  # the bigger, real split table
data/dev_clips/                       # if it changed
data/trajectories/                    # *.npz + stats.json + windows_preview.png
notebooks/models/seq2seq_v1.pt        # retrained weights
eval/forecast_table.md                # real Kalman-vs-LSTM numbers
eval/anticipation.md                  # A3PS-vs-reactive mTTA/detection/false-alarm
eval/anticipation_per_clip.csv        # per-clip breakdown, both methods
```
**Copy back (optional, small-medium — for demoing on this laptop too):**
```
dashboard/clips/<ids>/{events.json, meta.json}       # small, worth bringing
dashboard/clips/<ids>/{annotated.mp4, raw.mp4}       # bigger, optional
dashboard/clips/manifest.json                        # updated clip list
```
**Do NOT copy back:** `data/nexar/videos/` (the full clip pool).

### Verify on THIS laptop before calling it done

```powershell
# 1. code + tests still pass
python -m pytest -q                                    # expect 63 passed

# 2. index.csv came back correctly
python -c "import csv; from collections import Counter; print(Counter(r['split'] for r in csv.DictReader(open('data/nexar/index.csv'))))"
#   expect: Counter({'eval': 120, 'train_traj': 500+, 'dev': 15})

# 3. trajectory shards are intact
python -c "import glob, numpy as np; s=glob.glob('data/trajectories/*.npz'); print(len(s), 'shards'); d=np.load(s[0]); print(d['history'].shape, d['future'].shape)"

# 4. stats.json shows the real numbers (not the old 22-window run)
python -c "import json; print(json.load(open('data/trajectories/stats.json'))['total_windows'])"

# 5. retrained weights load and produce a sane prediction shape
python -c "from a3ps.forecasting.seq2seq import Seq2SeqForecaster; f=Seq2SeqForecaster(weights_path='notebooks/models/seq2seq_v1.pt'); m,s=f.predict([[0.0,0.0],[1.0,1.0]],0.2,4.0); print(len(m), len(s))"
#   expect: 20 20

# 6. re-run eval on THIS machine to confirm cross-laptop reproducibility
#    (fast — LSTM forward passes on small arrays, not video, so CPU is fine)
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_v1.pt
#   expect the same table as eval/forecast_table.md from the GPU laptop

# 7. dashboard renders the re-run clips with real events
python scripts\serve_dashboard.py
#   open localhost:8000, pick a manifest clip, confirm ALERT/VIRTUAL_BRAKE
#   events appear in the log and risk colors change from green
```

If all 7 pass, the whole pipeline is genuinely done and verified end to end.

---

## Quick command cheat-sheet

```powershell
# env
nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
"GROQ_API_KEY=your_real_key_here" | Out-File -Encoding utf8 .env
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest -q                                             # expect 63 passed

# data
python scripts\prepare_nexar.py --root data\nexar

# mining
python scripts\mine_trajectories.py --plot

# train (via notebook)
jupyter notebook notebooks\train_forecaster.ipynb

# forecast eval (note the explicit --weights)
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_v1.pt

# real clips through the full pipeline (perception+tracking+forecast+risk+decision)
python scripts\run_pipeline.py --video data\dev_clips\dev01.mp4 --out dashboard\clips\dev01\

# anticipation eval over the real 120-clip eval split
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --run

# optional: Groq narrative enrichment
python -m a3ps.explain.llm_client dashboard\clips\dev01 --dry-run
python -m a3ps.explain.llm_client dashboard\clips\dev01

# dashboard
python scripts\serve_dashboard.py
```
