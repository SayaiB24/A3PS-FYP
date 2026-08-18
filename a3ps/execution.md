f# A3PS — Execution Guide (from scratch)

This is the "I just got this repo, how do I run everything" guide. Follow it
top to bottom on a fresh machine and you end up with the full pipeline
running, all evaluation numbers generated, and the dashboard demo working.
No prior knowledge of the project is assumed.

Companion docs:
- **`logic_pipeline.md`** — *how* each file works and why (read after this).
- **`metrics.md`** — deep reference for every metric/eval script.
- **`GPU_HANDOFF.md`** — the two-laptop (CPU dev / GPU run) workflow specifics.
- **`QnA.md`** — answers to every tricky question about the implementation.

---

## 0. What this project is, in one paragraph

**A3PS (Anticipatory Proactive Perception & Safety)** takes dashcam video and
produces a *proactive* collision warning: it detects and segments road actors
(YOLOv8-Seg), tracks them across frames (BoT-SORT), forecasts each actor's
next 4 seconds of motion (Kalman constant-velocity filter, with an optional
learned LSTM), converts those forecasts into a per-actor collision
probability against the ego vehicle's driving corridor, and runs a decision
state machine that emits `ALERT` and `VIRTUAL_BRAKE` events *before* the
collision — with a human-readable explanation attached to every event. A web
dashboard replays annotated clips with all of this overlaid.

```
video ─▶ perception+tracking (one fused YOLOv8-Seg model.track() call)
      ─▶ forecasting (Kalman-CV / Seq2Seq-LSTM, 4 s @ 5 Hz)
      ─▶ risk (sigma-point collision probability vs ego corridor)
      ─▶ decision (SAFE→ALERT→BRAKE state machine, context-aware thresholds)
      ─▶ explanation (deterministic template + optional Groq LLM narrative)
      ─▶ annotated.mp4 + events.json + meta.json  ─▶ dashboard
```

---

## 1. Prerequisites

| What | Why | Check |
|---|---|---|
| Python 3.10+ | everything | `python --version` |
| Git | clone the repo | `git --version` |
| NVIDIA GPU + driver (strongly recommended) | YOLO inference is ~20-40 ms/frame on GPU vs ~800-1000 ms/frame on CPU | `nvidia-smi` |
| ~30 GB disk (if using the full Nexar dataset) | raw video clips | — |
| Free Groq API key (optional) | only for the optional LLM explanation enrichment | https://console.groq.com/keys |

A CPU-only machine works for **development and unit tests** (all 90 tests are
CPU-only and data-free), but real clip processing / mining / eval runs need
the GPU machine.

## 2. Clone and set up the environment

```powershell
git clone https://github.com/SayaliB-04/A3ps-actual-P.git
cd A3ps-actual-P\a3ps
python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
```

**On a GPU machine, install CUDA-enabled torch FIRST** (before
requirements.txt, or pip may grab the CPU wheel). Check your CUDA version in
`nvidia-smi`'s top-right corner, then pick the matching index URL:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then everything else:

```powershell
pip install -r requirements.txt
```

**Verify CUDA is actually visible** (the #1 way GPU runs silently fall back
to CPU):

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` + your GPU name. The `Segmenter`/`Tracker` auto-detect CUDA
and switch to `.cuda().half()` — no config change needed.

**Optional — Groq key** (only needed for step 12):

```powershell
"GROQ_API_KEY=your_real_key_here" | Out-File -Encoding utf8 .env
```

The `.env` file is gitignored on purpose; never commit a real key.

## 3. Verify the install

```powershell
python -m pytest -q
```

Expect **90 passed**. These are all CPU-only and data-free (synthetic
fixtures) — if this fails, fix the environment before touching data.

Quick real-inference smoke test (auto-downloads `yolov8s-seg.pt` ~23 MB on
first run):

```powershell
python -m a3ps.perception.segmenter data\dev_clips\dev01.mp4
```

Look at `mean inference: X ms/frame`: **20-40 ms** = GPU working; four digits
= CPU (recheck step 2).

## 4. Get the data in place

The dataset is the **Nexar dashcam collision-prediction dataset**: short
clips, each labelled positive (leads to a collision/near-miss) or negative
(normal driving), with a labels Excel (`id, time_of_event, time_of_alert,
target`). Data is gitignored — it moves by hand, not through git.

**Layout (flat — do NOT pre-sort into folders):**

```
data/nexar/
├── videos/            # ALL clips, flat, original id filenames (e.g. 00042.mp4)
└── labels.xlsx        # the Excel; drop several side-by-side if you have more
                       # (train.xlsx + test.xlsx etc. — all are auto-merged)
```

**How many clips you need:**
- positives: **65** is exactly sufficient (5 → dev split, 60 → eval split)
- negatives: 70 minimum (10 dev + 60 eval); **everything above that** goes to
  `train_traj`, the pool trajectory mining draws from. To hit the 10,000+
  mined-window target for LSTM training you want **several hundred** negatives.

## 5. Build the index and splits

```powershell
python scripts\prepare_nexar.py --root data\nexar
```

Fast (seconds — only probes video metadata). Produces:
- `data/nexar/index.csv` — the master table every other script reads:
  `clip_id, path, label, split, event_time_s, alert_time_s, duration_s, fps, width, height`
- `data/dev_clips/dev01.mp4 … dev15.mp4` + `README.md` — the 15 shortest
  clips (10 neg + 5 pos) copied out and renamed for daily development. The
  README holds the `devNN ↔ original clip_id` mapping — you will need it.

Splits: `dev` = 15 (fixed), `eval` = 120 (fixed, 60 pos + 60 neg),
everything else negative → `train_traj`. Sanity check:

```powershell
python -c "import csv; from collections import Counter; print(Counter(r['split'] for r in csv.DictReader(open('data/nexar/index.csv'))))"
```

## 6. Run the pipeline on a clip (the core loop)

```powershell
python scripts\run_pipeline.py --video data\dev_clips\dev14.mp4 --out dashboard\clips\dev14\
```

Or all 15 dev clips:

```powershell
Get-ChildItem data\dev_clips\dev*.mp4 | ForEach-Object {
    python scripts\run_pipeline.py --video $_.FullName --out "dashboard\clips\$($_.BaseName)\"
}
```

Each run writes into the `--out` directory:
- `annotated.mp4` — the video with risk-colored masks/boxes, trails,
  predicted paths, ego corridor, and a red flash on VIRTUAL_BRAKE
- `raw.mp4` — verbatim copy of the source (dashboard + figure scripts need it)
- `events.json` — the full structured result (every frame's tracks +
  predictions + every emitted event with its explanation)
- `meta.json` — clip metadata **including `per_stage_ms`** (per-stage latency,
  written automatically — this feeds the paper's latency table)

Useful flags: `--max-seconds 5` (quick tests), `--config` (alternate YAML).

## 7. View it in the dashboard

```powershell
'["dev14", "dev11", "dev04"]' | Out-File -Encoding utf8 dashboard\clips\manifest.json
python scripts\serve_dashboard.py
```

Open `http://localhost:8000`, pick a clip. You should see: masks colored
green→amber→red as risk rises, the event log filling with ALERT /
VIRTUAL_BRAKE rows (each with its explanation sentence), the risk timeline
with the threshold line, a red border flash after a VIRTUAL_BRAKE, and the
▼ A3PS vs ▽ reactive-ADAS anticipation-gap markers. Pick a mix of positive
and negative clips so the demo shows both an intervention and a clean run.

## 8. Mine trajectories (training data for the LSTM)

```powershell
python scripts\mine_trajectories.py --plot
```

Runs the tracker over every `train_traj` clip and exports normalized
(history=10 pts / future=20 pts @ 5 Hz) trajectory windows to
`data/trajectories/*.npz` + `stats.json` + `windows_preview.png`.

- **Resumable:** clips whose `.npz` shard already exists are skipped — just
  re-run after an interruption. `--force` re-mines everything.
- **Target:** `total_windows >= 10000` in `stats.json`. If short, add more
  negative clips (step 4) and re-run `prepare_nexar.py` + mining.
- **Check the preview:** paths should look smooth (no zig-zags) and
  forward-biased (red futures mostly extend +y).

## 9. Train the LSTM forecaster (optional/stretch)

```powershell
pip install jupyter
jupyter notebook notebooks\train_forecaster.ipynb
```

Point `SHARD_DIR` at `data/trajectories`, run all cells. Trains a Seq2Seq
LSTM (hidden 64, 2 layers, Gaussian-NLL loss, early stopping on val ADE) and
saves `notebooks/models/seq2seq_v1.pt`.

⚠️ **Before retraining, back up the previous checkpoint** — the notebook
overwrites in place:

```powershell
Copy-Item notebooks\models\seq2seq_v1.pt notebooks\models\seq2seq_prev.pt
```

## 10. Run the evaluations

Full details for every metric are in **`metrics.md`** — this is the short
version, in the recommended order.

**(a) Forecast accuracy (ADE/FDE, Kalman vs LSTM):**

```powershell
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_v1.pt
```

⚠️ Always pass `--weights` explicitly (the script's default path doesn't
match where the notebook saves). Writes `eval/forecast_table.md`.

**(b) Anticipation eval (the headline numbers).** Sanity-check on dev first
(instant — reads the clips step 6 already produced), then the real 120-clip
run:

```powershell
python scripts\eval_anticipation.py --index data\nexar\index.csv --split dev --clips-dir dashboard\clips
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --run
```

`--run` is the GPU-heavy part (all 120 eval clips through the pipeline, no
rendering) and is **resumable**. Writes `eval/anticipation.md` (A3PS vs a
reactive-proximity baseline: detection rate / false-alarm rate / mTTA, plus
the **matched-subset mTTA** — the only number that fairly supports an
"A3PS warns N s earlier" claim) and `eval/anticipation_per_clip.csv`.

⚠️ Note on the dev sanity check: the eval scripts look clips up by their
**original clip_id** from index.csv, but the dashboard folders use `devNN`
names. Create directory junctions once (mapping from
`data\dev_clips\README.md`):

```powershell
cd dashboard\clips
cmd /c mklink /J <original_id> devNN     # one line per dev clip
```

**(c) Threshold sweep / PR curve** (cheap — reuses (b)'s cached results):

```powershell
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --sweep
```

Writes `eval/pr_curve_data.csv` + `eval/pr_curve.png` (base_threshold swept
0.40→0.90 in 0.05 steps, no GPU re-run).

**(d) Ablation study (Table II)** — budget ~4x the wall-clock of (b); good
overnight job:

```powershell
python scripts\run_ablations.py --index data\nexar\index.csv --split eval
```

Runs the eval once per config variant (Kalman/LSTM forecaster, dynamic vs
static threshold, EMA on/off), temporarily overriding `configs/default.yaml`
per variant and always restoring it. Writes `eval/ablation_table.md`.

**(e) Qualitative figure** (a 2×2 before/ALERT/BRAKE/after panel from a real
incident, print quality):

```powershell
python scripts\make_qual_figure.py --clip dashboard\clips\dev14
```

## 11. Collect everything for the paper/report

```powershell
python scripts\collect_paper_stats.py
```

Scans the repo (config, hardware, splits, per-stage latency from every
`meta.json`, all eval outputs) and writes `eval/paper_stats_summary.md`.
Sections whose inputs are missing say `NOT YET AVAILABLE` with the command to
run — it never fabricates a number.

## 12. Optional: LLM explanation enrichment (Groq, free tier)

```powershell
python -m a3ps.explain.llm_client dashboard\clips\dev14 --dry-run   # no API call
python -m a3ps.explain.llm_client dashboard\clips\dev14             # real
```

Adds a scene-aware 2-3 sentence `explanation_llm` to each event (keyframe +
structured facts → vision LLM). Read the outputs and confirm they mention
only real scene elements. For the paper's hallucination before/after example
there is a deliberate, illustration-only permissive variant:

```powershell
python -m a3ps.explain.llm_client_permissive_test dashboard\clips\dev14
```

(prints constrained-vs-unconstrained narratives side by side, then
auto-restores the real one on disk).

## 13. Week-4 polish: per-clip BEV calibration (only for final demo clips)

The default ground-plane trapezoid is generic; forecasting therefore runs in
image pixels (`forecast_space: img`) by default. For the ~5 final demo clips
only: pick 4 road points + real-world distances, save
`dashboard/clips/<id>/ground.yaml`, set `ground_plane_override` +
`forecast_space: bev`, and check with `python scripts\verify_bev.py`
(velocities should read ~5-20 m/s).

---

## Troubleshooting quick table

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` → `False` | CPU torch wheel installed | reinstall torch with the CUDA index URL (step 2) |
| ~1000 ms/frame inference | running on CPU | same as above |
| "No processed clips found" from eval | clip-id vs devNN folder-name mismatch | create the junctions (step 10b note) |
| Seq2Seq row missing from forecast table | `--weights` not passed | pass `--weights notebooks\models\seq2seq_v1.pt` |
| eval numbers differ from an older run | mined pool changed → different val split | only compare within one run (see `metrics.md` §1) |
| `pytest` failures on a fresh clone | broken env, not code | recreate the venv, reinstall requirements |
| mining seems to redo some clips | zero-window clips write no shard | harmless; they re-track each run |
| dashboard shows no events | clips processed before risk engine was wired, or negative clip | re-run step 6; pick a positive clip |
