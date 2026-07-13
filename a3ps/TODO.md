# a3ps — TODO

## Status (updated 2026-07-10)

Done and unit-tested (CPU, no data needed):
- **Phase 4 — risk + decision.** `a3ps/risk/collision.py` (sigma-point collision
  prob, corridor dilation, `RiskSmoother` EMA, 4 analytic tests) and
  `a3ps/risk/decision.py` (`DecisionEngine` with context-aware thresholds +
  per-actor SAFE→ALERT→BRAKE state machine, 2 s cooldown, THRESHOLD_LOWERED on
  span onset). Wired into `pipeline.py` as a per-frame risk engine; renderer now
  colors green→amber→red by risk level and flashes a red border 0.5 s after a
  VIRTUAL_BRAKE. Per-clip context via `dashboard/clips/<id>/context.json`.
- **`scripts/eval_anticipation.py`** — detection rate / false-alarm rate / mTTA
  for **A3PS vs a naive reactive-proximity baseline** (dashboard corridor-distance
  logic), plus A3PS AP. `--run` processes every eval clip through the pipeline
  with rendering off (`render=False`); writes `eval/anticipation.md` +
  `eval/anticipation_per_clip.csv`. Metric logic verified locally on synthetic
  fixtures.
- **`scripts/eval_forecast.py`** — ADE/FDE Kalman vs Seq2Seq. Implemented;
  Kalman path runs now, Seq2Seq auto-skips until `models/seq2seq_v1.pt` exists.
- **Phase 5 — explanation (XAI).** `a3ps/explain/templates.py` `explain(event,
  track, context)` builds the deterministic one-liner (motion phrase from BEV
  velocity, threshold-lowered reason); the Pipeline fills `explanation_template`
  on every emitted event. `a3ps/explain/llm_client.py` `enrich_events(clip_dir)`
  is the OFFLINE MLLM pass — pulls each event's keyframe from `raw.mp4`,
  downsizes to 768 px, sends facts + image to **Groq's free-tier** vision API
  (`meta-llama/llama-4-scout-17b-16e-instruct`, OpenAI-compatible), writes
  `explanation_llm`; has `--dry-run`, `--overwrite`, retries, and skips
  already-enriched events. CLI: `python -m a3ps.explain.llm_client <clip_dir>`.
  Switched off Anthropic to avoid any paid API usage.
- Test suite: **60 passing** (`pytest a3ps/tests`).

Pending / next up:
- [ ] **Verify explanations on 3 dev clips** (GPU laptop): get a free key at
      https://console.groq.com/keys, `export GROQ_API_KEY=...`, run the
      pipeline, then `python -m a3ps.explain.llm_client dashboard/clips/<id>`
      and read the `explanation_llm` strings — confirm they mention only real
      scene elements. If one hallucinates, tighten `SYSTEM_PROMPT` in
      `llm_client.py` and keep one before/after example for the report's
      hallucination-mitigation note. Step can be cut entirely if it's simpler —
      the deterministic templates already carry the XAI requirement.
- [ ] Everything below (BEV calibration, mining, LSTM, dashboard, real data) is
      blocked on the GPU laptop + the real Nexar dataset landing.

## Data: feed in the actual Nexar dataset

Nexar clips are labelled by an Excel (`id, time_of_event, time_of_alert, target`),
not by folder. Upload a FLAT pool of clips + the Excel; `prepare_nexar.py` reads
`target` for the label and computes the dev/eval/train_traj splits.

**Where to put it (input — do NOT pre-sort by class or split)**

```
data/nexar/
├── videos/            # ALL clips, flat, original id filenames (e.g. 00042.mp4)
└── labels.xlsx        # the Excel (drop both if you have train + test Excels)
```

**Counts to upload** (from the Excel `target` column):
- positives (`target=1`): 65   (5 -> dev, 60 -> eval)
- negatives (`target=0`): 90   (10 -> dev, 60 -> eval, 20 -> train_traj)
- minimum viable: 20 pos + 30 neg (fills dev, partial eval)

**Then run** (paths can point anywhere; no need to move 30 GB):

```
python scripts/prepare_nexar.py --root data/nexar
```

Produces `data/nexar/index.csv` (master: clip_id, path, label, split,
event_time_s, alert_time_s, fps, w, h) and copies the 15 dev clips into
`data/dev_clips/` (+ README). eval / train_traj stay as rows in index.csv.

- [ ] FIRST: finish `prepare_nexar.py` rewrite — xlsx reader (openpyxl, installed),
      match videos by `id`, label from `target`. (folder-mode is the old path.)
- [ ] Remove the smoke-test copy `data/nexar/negative/02134.mp4` if present.

## Week 4: per-clip BEV calibration for the 5 final demo clips
When Option 2 becomes worth it: later, in Week 4, when you're preparing your 5 demo clips. At that point you have exactly 5 clips to worry about, and spending 30 minutes each to give them proper metric units for the final demo is a good investment. Do it then, not now.

The default ground-plane trapezoid is generic and miscalibrated for real clips
(verified on 02134.mp4: BEV velocities too low, some predicted depths negative).
Pipeline default is `forecast_space: img` (relative units) until then.

For each of the 5 final demo clips:
- [ ] Pick 4 road points in the frame + estimate their real-world distances.
- [ ] Save `dashboard/clips/<clip>/ground.yaml` (image_points_frac + ground_points_m).
- [ ] Set `ground_plane_override` (or per-clip config) and `forecast_space: bev`.
- [ ] Re-run `scripts/verify_bev.py` — velocities should read ~5-20 m/s and
      predictions should stay in-frame (>=80%).

## Trajectory mining (`scripts/mine_trajectories.py`) — NOT done in this step

The script + its windowing/normalization logic are implemented and unit-tested
(`tests/test_mining.py`, 8 tests). What remains — all blocked on the real
dataset, since `train_traj` is currently empty (only clip 02134 exists, in `dev`):

- [ ] Run on the real `train_traj` split and hit the target of **10k+ windows**
      (`python scripts/mine_trajectories.py`; overnight on the laptop is fine).
- [ ] Verify the `--plot` scatter on real traffic: paths should be **smooth**
      (jump filter worked) and **forward-biased** (red futures extend +y).
- [ ] Confirm `stats.json` class mix / filter tallies look sane on real data
      (~20% static retained; short-track & ID-switch drops non-trivial).
- [ ] Currently mines in **image pixels** (`forecast_space: img`, static_thresh
      15 px). Re-mine in **BEV metres** after Week 4 per-clip calibration if the
      LSTM is to train in metric space.
- [ ] Verified so far only via a `--split dev` smoke run on 02134 (not the real
      train set). Re-run once real data lands.

## LSTM forecaster (STRETCH, Week 3) — not started

- [ ] Train `notebooks/train_forecaster.ipynb` on the mined windows.
- [ ] Implement `a3ps/forecasting/seq2seq.py` (currently raises NotImplementedError).

## Dashboard: shift from dummy to real data (later)

The dashboard ships a dummy clip `fake_demo` (`scripts/make_demo_clip.py`) so the
console / timeline / banner can be built before the risk engine exists. It reads
ONLY the ClipResult schema, so the switch to real data needs no dashboard code
changes:

- [ ] After Phase 4 (risk engine) fills real `collision_prob` + emits events,
      run the pipeline on the demo clips.
- [ ] Edit `dashboard/clips/manifest.json` to list the real clip id(s); drop
      `fake_demo`.
- [ ] Verify threats / threshold / event-log / banner / risk-timeline (incl. the
      ▼ A3PS vs ▽ reactive-ADAS gap) all populate from the real events.

## Downstream (needs the real dataset in place)

Implemented and runnable now (on any clip, CPU): YOLOv8-Seg segmenter, BoT-SORT
tracker + TrajectoryBuffer, Kalman-CV forecaster, risk/decision engine, and
`pipeline.py` (-> `annotated.mp4` + `events.json`). Both eval scripts are
implemented (see Status). Remaining runs are gated on the GPU laptop + data:

- [ ] `scripts/eval_forecast.py` — run once mined shards exist; add the Seq2Seq
      column once `models/seq2seq_v1.pt` is trained.
- [ ] `scripts/eval_anticipation.py` — run with `--run` on the real `eval` split
      (60 pos + 60 neg) to get the headline mTTA / AP / false-alarm numbers.
