# A3PS — Architecture

This document describes the A3PS pipeline **as it currently runs**, phase by
phase, from raw dashcam video to an explained intervention event. It is a
architecture reference, not a history — for how the system arrived here, see
[`REPORT_evaluation_audit.md`](REPORT_evaluation_audit.md) and
[`handoff/CHALLENGES.md`](handoff/CHALLENGES.md).

---

## 1. End-to-end flow

```
video ──▶ Perception + Tracking (YOLOv8-Seg + BoT-SORT, fused)
       ──▶ Forecasting (Kalman-CV)
       ──▶ Per-frame feature extraction (a3ps/features/)
       ──▶ RiskGRU — learned causal temporal head (a3ps/risk/temporal.py)
       ──▶ Operating point (threshold, confirm-N)
       ──▶ Explanation (deterministic templates + optional offline LLM)
       ──▶ annotated.mp4 + events.json
```

---

## Phase 0 — Data & Splits

- **`a3ps/common/splits.py`** freezes `dev`, `eval`, and `train_val`
  membership in `eval/split_freeze.json`, kept outside the gitignored
  `data/` tree so it travels with the repo. Split assignment consults this
  file; clips named in it keep their split forever, and only `train_traj` is
  allowed to grow as the dataset grows.
- **`train_val` is frozen for a different reason than `dev`/`eval`**: it is
  what early stopping, best-epoch selection, kappa, and the operating point
  are chosen on. Held-out `eval` is never touched during selection.
- Current partition (`scripts/repartition_splits.py`, seed 1234): the 745
  timed positives are binned into deciles of alert-to-event window length,
  and each split is apportioned proportionally from every bin, so window
  length is not confounded with split membership. Splits in use:
  `train_core_v2`, `train_val_v2`, and `eval_v2` (120 clips, 60 positive /
  60 negative, frozen).

## Phase I+II — Perception + Tracking (fused)

`a3ps/tracking/tracker.py`. One YOLOv8-Seg call in tracking mode
(`model.track(persist=True, tracker='botsort_a3ps.yaml')`) returns
detections, segmentation masks, and BoT-SORT track IDs in a single inference
pass — a separate segmentation pass would double inference cost for
identical information.

- Class filtering happens *after* tracking, not inside `model.track()`,
  because passing `classes=` into the track call breaks BoT-SORT's
  association.
- `centroid_img` is the bbox foot point (bottom-center) — the actor's
  ground-contact point, meaningful for a ground-plane projection.
- `TrajectoryBuffer` decimates the 30 fps track stream into 5 Hz buckets and
  keeps the last 2 s (10 points) per track ID. A track becomes `ready()` for
  forecasting at ≥80% of that window; IDs unseen for 1 s are pruned.
- `configs/botsort_a3ps.yaml` raises `track_buffer` 30→90 frames so brief
  occlusions don't kill a track ID.

## Phase III — Forecasting

`a3ps/forecasting/kalman_cv.py` — constant-velocity Kalman filter, state
`[x, y, vx, vy]`: fit over the 10-point history, then rolled forward 20 steps
(4 s @ 5 Hz) without updates, so covariance — and the reported std — grows
with horizon. Reported std is the predicted measurement std
`sqrt(diag(HPHᵀ+R))`, tuned so growth over the horizon is a stable ~2.1×.
Histories under 4 points fall back to straight-line extrapolation with a
fixed large std.

`a3ps/forecasting/seq2seq.py` is a learned alternative (encoder-decoder
LSTM, hidden 64, 2 layers, native per-step uncertainty via a log-variance
head) behind the identical `predict(history, dt, horizon_s)` interface,
selectable via `configs/default.yaml → forecaster:`. It wins on 4 s trajectory
error but loses at 1–2 s (ADE@1s 18.85 vs Kalman's 12.94 px). Pushed end to end
through the risk head it tied Kalman on useful-warning and warned ~0.56 s
later, so Kalman stays the default
([`status/headline_number_attempts.md`](status/headline_number_attempts.md) §2).

## Phase IV — Feature Extraction + Learned Temporal Risk Head

The core of the risk system: a pooled per-frame feature vector feeding a
small causal model, trained end-to-end against Nexar's own annotated alert
times.

### `a3ps/features/extract.py` — per-frame feature vectors

- `actor_feature_matrix()` produces one row per (frame, actor): position and
  extent, velocity/acceleration of the foot point, looming cues (area
  growth, time-to-area-zero), collision-probability geometry from
  `a3ps/risk/collision.py` (`ttc_pred`, `collision_prob`, `prob_trend`),
  continuous corridor-distance geometry, and a one-hot class —
  `ACTOR_FEATURE_DIM` values per actor.
- `clip_features()` pools those rows into **one fixed-width vector per
  frame** — the label is per-frame, not per-actor. Pooling carries the
  top-3 actors by risk proxy individually, plus a max-pool over all actors,
  plus whole-frame scalars, rather than a plain mean (which would bury the
  one actor that matters in a crowded frame).
- **Ego motion** occupies the leading `EGO_FEATURE_DIM` slots
  (`a3ps/features/ego.py`) when supplied; a `has_ego` metadata flag records
  whether they were populated, so a run trained without ego motion is never
  silently reported as ego-aware.
- Extraction is pure NumPy over an already-computed `ClipResult` — CPU only,
  no model, no GPU, no video decode. Current extraction: 1,485 clips at
  10 Hz over 13 s windows, 112-dim, `has_ego=True` on every clip.

### `a3ps/risk/temporal.py` — `RiskGRU`

A **causal, unidirectional GRU**: per-frame features in, one risk logit per
frame out. Causality is the load-bearing constraint — frame `t`'s output may
only depend on frames `≤ t`, since the system's entire claim is
*anticipation*; a bidirectional or non-causal architecture would let the
model see the collision and "predict" it, producing excellent metrics on a
worthless system. `assert_causal()` is a runtime check that this holds.

- Hidden 64, ~34,465 parameters — deliberately small, matched in scale to
  the Phase III LSTM forecaster, sized for a ~1,500-sequence training pool.
- `LayerNorm` on the input, since the feature vector mixes log-areas
  (order 1–10), normalised positions (order 0.1), and saturating
  probabilities (order 1) — without it, large-magnitude columns dominate the
  first layer by scale alone.
- Trained with `binary_cross_entropy_with_logits`; `.probs()` applies
  sigmoid at eval time.

### `a3ps/risk/anticipation_loss.py` — the training objective

A per-frame BCE keyed to Nexar's own `time_of_alert`. For a positive clip
with alert time `ta` and event time `te`, normalised progress
`u(t) = clamp((t-ta)/(te-ta), 0, 1)` weights the positive-frame loss by
`w(t) = exp(kappa · (u(t) - 1))`, so a late miss is punished harder than an
early one, anchored to the dataset's own "this became actionable" moment.
Frames after `te` are masked out entirely; `pre_alert_weight` penalises
firing before `ta` in a positive clip, since that is, by the dataset's own
annotation, still ordinary driving.

### Operating point: threshold + confirm-N

Applied **after** the model runs, at eval time only — sweepable in one
forward pass per clip via `scripts/sweep_operating_point.py`. Model capacity
and loss shape (kappa, `pre_alert_weight`) require a retrain; threshold and
confirm-N do not, which keeps the expensive axis of tuning separate from the
cheap one.

## Phase V — Explanation

`a3ps/explain/templates.py` builds a grammar-based, deterministic one-liner
for every emitted event, using only facts the risk engine actually computed
— structurally incapable of stating something the pipeline didn't measure.
`a3ps/explain/llm_client.py` optionally enriches this after the pipeline
runs (never in the live loop): a keyframe plus structured facts go to a
vision LLM, and the resulting 2–3 sentence narrative is stored separately as
`explanation_llm`.

---

## 2. Evaluation protocol

- **Model selection happens only on `train_val_v2`** — kappa, capacity,
  feature choices, and the operating point are all decided there.
- **`eval_v2` is read at most once**, for the single locked configuration,
  at the very end. Reading a held-out split repeatedly across candidates
  reintroduces exactly the selection bias the protocol exists to prevent.
- **Checkpoint saving includes an unconditional degenerate-pick check**, so
  training cannot silently keep a chance-level epoch.
- **`useful_warning_rate_v2`** is the current episode-based metric: a clip
  only counts as a useful warning if it fired within the actionable window,
  not merely "fired somewhere in the clip."

**Current locked state** (`docs/status/final_eval_read.md`):

| quantity | value |
|---|---|
| checkpoint | `risk_gru_k1p0_v2_selfix_s1234.pt` |
| operating point | threshold 0.70, confirm 8 |
| split | `eval_v2` (120 clips, 60/60, frozen) |
| useful-warning rate | **0.650** (39/60) |
| false-alarm rate | **0.167** (10/60) |
| mean lead time | 1.51 s |
| mean AP | 0.646 |
| gross-premature alerts | 0 |

---

## 3. `dashboard_v2` — results/replay dashboard

`dashboard_v2/` (HTML/CSS/vanilla JS, no build step, no npm, no CDN) +
`scripts/dashboard_v2_data.py` (read-only data layer) +
`scripts/serve_dashboard_v2.py` (stdlib HTTP server).

- **Never writes, never scores a checkpoint.** It only reads artifacts other
  scripts already produced, through the repo's own readers
  (`ClipResult`, `RiskGRU.load`, `eval_anticipation.compute_metrics`,
  `splits.load_freeze`, `build_dashboard_demo.resolve_video`).
- **No run registry.** A "run" is the naming tag shared by
  `notebooks/models/risk_gru_<tag>.pt` and
  `eval/risk_gru_history_<tag>.json`; a run's evaluation artifacts (sweep,
  final read) are attached by the checkpoint path each file's own header
  names, not by filename matching.
- **Four views:** Live replay (video + simulator side by side, synced clock,
  alert/brake banners fire in both at once), Compare runs (up to 8, with a
  cross-split warning), History (a metric plotted run-by-run in commit
  order), Drill-down (per-clip verdicts, fire times, per-object risk over
  time).
- **The "simulator" (`dashboard_v2/js/sim.js`) is a custom, from-scratch
  scene renderer, not an external simulator.** No CARLA, MetaDrive,
  Duckietown, game engine, or physics engine, and no WebGL/Three.js — it is
  plain HTML5 Canvas 2D with hand-written camera math (position/forward/
  right/up vectors, perspective projection, near-plane polygon clipping)
  that draws each tracked actor as a 2.5D cuboid sized by class (car, truck,
  bus, motorcycle, bicycle, person — hardcoded width/length/height in
  metres). Two camera modes: a fixed **chase camera** behind the ego vehicle
  and a **top-down** orthographic view. It does not simulate anything
  independently — it only re-renders the pipeline's own `centroid_bev`
  ground-plane positions, frame by frame, on the video's own clock. A CARLA-
  scale 3D engine (or a lighter alternative like MetaDrive/Duckietown Gym)
  is noted in the project as a possible future upgrade, not something in
  use today.
- **Known limits, by design:** `mean_bev` is never populated by the
  pipeline, so simulator forecast paths are the image-space Kalman-CV
  forecast projected onto the ground; the ego stays fixed at the origin
  because the pipeline doesn't estimate ego speed; only 12 of the 120
  `eval_v2` clips have stored track data.

---

## 4. Module map

| module | role |
|---|---|
| `a3ps/tracking/tracker.py` | fused YOLOv8-Seg + BoT-SORT |
| `a3ps/forecasting/kalman_cv.py` | production forecaster |
| `a3ps/forecasting/seq2seq.py` | learned forecaster (selectable) |
| `a3ps/risk/collision.py`, `gaussian.py` | sigma-point collision-probability geometry, feeds the feature extractor |
| `a3ps/features/extract.py` | per-frame feature vectors |
| `a3ps/features/ego.py` | ego-motion features |
| `a3ps/risk/temporal.py` | `RiskGRU`, the learned risk head |
| `a3ps/risk/anticipation_loss.py` | alert-time-keyed training loss |
| `a3ps/common/splits.py` | frozen split membership |
| `a3ps/explain/templates.py`, `llm_client.py` | explanation |
| `dashboard_v2/` | results/replay dashboard |
| `scripts/train_risk_head.py`, `sweep_operating_point.py`, `final_eval_read.py` | train / sweep / one-shot held-out read |
| `scripts/repartition_splits.py` | decile-stratified split construction |
| `scripts/dashboard_v2_data.py`, `serve_dashboard_v2.py` | dashboard_v2 backend |

---

## 5. Where to go next

- Full methodology narrative: [`REPORT_evaluation_audit.md`](REPORT_evaluation_audit.md).
- What to do next / open gaps: [`handoff/NEXT_STEPS.md`](handoff/NEXT_STEPS.md),
  [`handoff/FUTURE_WORK.md`](handoff/FUTURE_WORK.md).
- Reproducing any number by hand: [`handoff/REPRODUCE_BY_HAND.md`](handoff/REPRODUCE_BY_HAND.md).
- Master index of every doc and its currency: [`INDEX.md`](INDEX.md).
