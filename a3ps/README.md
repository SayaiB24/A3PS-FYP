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

## Layout

- `a3ps/common/` — schema, geometry, video IO
- `a3ps/perception/` — YOLOv8-Seg segmenter wrapper
- `a3ps/tracking/` — BoT-SORT tracker + trajectory buffer
- `a3ps/forecasting/` — Kalman CV (core) and LSTM seq2seq (stretch)
- `a3ps/risk/` — Gaussian trajectories, collision probability, decision rules
- `a3ps/explain/` — deterministic templates + optional LLM enrichment
- `a3ps/pipeline.py` — orchestrator
- `dashboard/` — static player with canvas overlay + agent console
- `scripts/` — CLI entry points and evaluation
- `tests/` — unit tests for schema, geometry, and risk math

## Config

See `configs/default.yaml` for model name, classes, thresholds, horizons, fps.

## Tests

```bash
pytest
```
