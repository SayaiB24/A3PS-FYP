# a3ps — Anticipatory Traffic-Safety Pipeline

`a3ps` turns dashcam video into an annotated clip plus a structured
`events.json`: it detects and segments road actors, tracks them, forecasts
their short-term trajectories, estimates collision probability against the
ego corridor, and emits explained intervention events.

```
video ──▶ perception (YOLOv8-Seg) ──▶ tracking (BoT-SORT) ──▶ forecasting
      ──▶ risk (Gaussian + sigma-point collision) ──▶ decision ──▶ explain
      ──▶ annotated.mp4 + events.json
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py --video clip.mp4 --out dashboard/clips/clip/
```

To start dashboard development before the pipeline is ready:

```bash
python scripts/make_fake_events.py --out dashboard/clips/demo/
python scripts/serve_dashboard.py
```

## Results dashboard (v2)

A second, read-only dashboard for comparing training runs and inspecting their
results. The original dashboard (`dashboard/`, `scripts/serve_dashboard.py`) is
left exactly as it was, and both can run side by side.

```bash
python scripts/serve_dashboard_v2.py --port 8010   # then open http://127.0.0.1:8010/
python scripts/serve_dashboard.py --port 8000      # original dashboard, unchanged
```

Startup takes a few seconds because it reads every checkpoint's metadata through
`RiskGRU.load`. That needs torch; `--no-checkpoints` skips it, so the best epoch and
the validation scores saved in each checkpoint are not shown. Only the Python
standard library is added. There is no npm, no build step and no CDN, so it works
offline.

**Views**

- **Live replay** (the default tab): the original dashboard's player with a
  simulator beside it. Pick a clip from the same `dashboard/clips/manifest.json`
  dropdown. The left pane plays the dashcam video with labels, masks, trails,
  predicted paths, the ego corridor and the alert/brake banners. The right pane
  replays the same moment in a simulator, from a 3D chase camera or from above.
  The actor that raised the alert gets a pulsing ring, a TTC tag and a marker
  where its forecast crosses the ego path. Both panes run on the video's clock,
  so an alert fires in both at the same instant. `⏮ alert` / `alert ⏭` (or the
  `p`/`n` keys) jump to 1.5 s before the previous/next alert, and clicking an
  event in the log replays it. Simulator positions are the pipeline's
  `centroid_bev`. Where an overlay didn't store it (the Phase IV demo clips),
  it's projected with the pipeline's own `GroundPlane` (the default
  calibration, or a per-clip `ground.yaml` if one exists). The ego stays fixed
  at the origin because the pipeline doesn't estimate ego speed.
- **Compare runs:** tick up to 8 runs to overlay them. Each run has a show/hide
  checkbox and a dropdown that picks which evaluation record to plot. Charts:
  headline rates, lead time, training curves, the operating-point trade-off,
  precision–recall and confusion matrices. A warning appears when the visible
  runs were scored on different splits.
- **History:** a metric plotted run by run, in the order runs were committed,
  with one line per evaluation split. Below it is a table of every run.
- **Drill-down:** per-clip true/false positives and negatives, verdicts, fire
  times and peak scores, plus per-object risk over time for any clip that has
  track data.
- One filter row applies to every view: run date, family, split, clip ID, a
  risk threshold and object class. Every chart has a "table" button that shows
  its data as a table.

**Data it reads (and never writes)**

| Source | Used for |
|---|---|
| `notebooks/models/risk_gru_<tag>.pt` (`extra`, via `RiskGRU.load`) | run config, best epoch, selection-time validation metrics |
| `eval/risk_gru_history_<tag>.json` | per-epoch training curves |
| `eval/operating_point_sweep*.md`, `eval/final_eval_read_*.md` | threshold × confirm grids |
| `eval/final_eval_read_*.json` | held-out read with per-clip rows |
| `eval/ablation_rerun_*/*.json`, `eval/ensemble_calibration/comparison.json` | ablation and ensemble totals |
| `eval/anticipation_per_clip.csv` (through `eval_anticipation.compute_metrics`) | threshold-system and reactive baselines, per clip |
| `eval/anticipation/<id>/events.json` (through `ClipResult`) | full tracks + BEV for the 120 v1-eval clips |
| `dashboard/clips/manifest.json`, `dashboard/clips/<id>/{events,risk}.json`, `raw.mp4` | the original dashboard's clip list, overlays, alerts and learned-head curves |
| `configs/default.yaml` + `a3ps.common.geometry.GroundPlane` | ground-plane projection for simulator positions the overlay did not store |
| `eval/nexar_index_full.csv` + `build_dashboard_demo.resolve_video` | ground-truth times, video lookup |
| `git log` | run dates (first commit of each run's artifacts) |

**Adding a run so it shows up.** Runs are found by filename, so there is nothing to register:

1. Train with the usual naming: `scripts/train_risk_head.py --out
   notebooks/models/risk_gru_<tag>.pt --history-json eval/risk_gru_history_<tag>.json`.
   The run appears under `<tag>`.
2. Optional: sweep it with `scripts/sweep_operating_point.py`, writing to
   `eval/operating_point_sweep_<anything>.md`. The sweep is attached to the run by
   the checkpoint path named in the file's header, not by the filename.
3. Optional, and only for the committed checkpoint: a `scripts/final_eval_read.py`
   JSON in `eval/`. This is the only thing that enables per-clip drill-down, PR
   curves and confusion matrices for a learned run.
4. Restart the server. Discovery runs once, at startup.

**How runs are identified.** There is no run registry, so the dashboard relies
on these conventions:

- A run is the `<tag>` shared by its checkpoint and history file. `risk_gru_v1.pt`
  pairs with the tag-less `risk_gru_history.json`.
- The tag decodes as `k<κ>` (kappa), `paw<w>` (pre-alert weight), `_s<seed>`,
  `_clean` (v1 clean partition), `_v2` (v2 partition), `selfix` (the fixed
  checkpoint selector) and `abl_<group>` (ablation).
- A run's date is when its artifacts were first committed, not when it was
  trained.
- The validation split is inferred from the partition: `_v2` means
  train_val_v2, `_clean` means train_val, anything else means eval (v1).

**Metric names.** The pipeline's metrics keep their own names:
`useful_warning_rate_v2`, `useful_warning_rate`, `false_alarm_rate`,
`mean_lead_s`, `mean_AP`. Precision, recall, F1, PR curves and confusion
matrices are labelled *derived*. They are computed here from per-clip rows, and
"fired anywhere in the clip" counts as a positive prediction. They are not
time-aware metrics.

**Limits.**
- The dashboard never scores a checkpoint, so runs without per-clip rows show
  totals only.
- `mean_bev` is never filled in by the pipeline. The simulator's forecast
  paths are therefore the pipeline's image-space Kalman-CV forecast
  (`mean_img`) projected onto the ground, and the page says so.
- Of the 120 eval_v2 clips, only 12 have stored track data and 1 has a stored
  learned-head curve. The rest play in Live replay as video plus ground-truth markers.

**Possible later upgrade.** If a 3D scene is ever needed, MetaDrive or Duckietown
Gym are much lighter than CARLA. The current dashboard does not need either.

## Layout

- `a3ps/common/` — schema, geometry, video IO
- `a3ps/perception/` — YOLOv8-Seg segmenter wrapper
- `a3ps/tracking/` — BoT-SORT tracker + trajectory buffer
- `a3ps/forecasting/` — Kalman CV (core) and LSTM seq2seq (stretch)
- `a3ps/risk/` — Gaussian trajectories, collision probability, decision rules
- `a3ps/explain/` — deterministic templates + optional LLM enrichment
- `a3ps/pipeline.py` — orchestrator
- `a3ps/features/` — per-frame feature extraction + ego motion (Phase IV)
- `dashboard/` — static player with canvas overlay + agent console
- `dashboard_v2/` — read-only results dashboard (live replay + simulator, run comparison, history, drill-down)
- `scripts/` — CLI entry points and evaluation
- `tests/` — unit tests for schema, geometry, and risk math
- `docs/` — documentation, grouped by use (see below)
- `eval/` — generated evaluation output (written by scripts, not hand-edited)

## Documentation

**[`docs/INDEX.md`](docs/INDEX.md) is the master index** — what every doc is for,
an "I want to… → read this" lookup, and which docs are current vs. pre-Phase IV.
Start there if you are unsure where to look.

To continue the project, go straight to [`docs/handoff/`](docs/handoff/README.md).

In short:

- [`docs/runbooks/`](docs/runbooks/) — things you execute.
  [`execution.md`](docs/runbooks/execution.md) is the place to start;
  [`GPU_HANDOFF.md`](docs/runbooks/GPU_HANDOFF.md) covers the two-laptop
  workflow and [`GPU_PHASE4.md`](docs/runbooks/GPU_PHASE4.md) the current phase.
- [`docs/design/`](docs/design/) — how it works:
  [`logic_pipeline.md`](docs/design/logic_pipeline.md),
  [`metrics.md`](docs/design/metrics.md),
  [`explanation.md`](docs/design/explanation.md),
  [`QnA.md`](docs/design/QnA.md).
- [`docs/status/`](docs/status/) — [`results.md`](docs/status/results.md) and
  [`TODO.md`](docs/status/TODO.md).

## Config

See `configs/default.yaml` for model name, classes, thresholds, horizons, fps.

## Tests

```bash
pytest
```
