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
- `a3ps/features/` — per-frame feature extraction + ego motion (Phase IV)
- `dashboard/` — static player with canvas overlay + agent console
- `scripts/` — CLI entry points and evaluation
- `tests/` — unit tests for schema, geometry, and risk math
- `docs/` — documentation, grouped by use (see below)
- `eval/` — generated evaluation output (written by scripts, not hand-edited)

## Documentation

See [`docs/`](docs/README.md) for the full index. In short:

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
