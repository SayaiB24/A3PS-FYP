# A3PS — Pipeline Logic End-to-End

**Purpose of this document:** a plain-English walkthrough of how the A3PS project works, phase by phase, **as it is actually implemented today** (status: 2026-09-21). Every teammate should read this once before touching the implementation. If you can explain each phase in your own words after reading this, you can defend the project in the viva.

> **What changed since the original version of this document.** The project pivoted (the "Phase IV pivot"): the hand-tuned threshold risk engine is no longer the decision-maker. A **learned causal GRU risk head** now makes the alert decision. Phases 1–3 are unchanged in purpose but now also act as **feature extractors** for that head. The old threshold engine is kept as (a) a feature source and (b) the "before" baseline in the dashboard. Sections that changed are marked **[CHANGED]**.

---

## Two systems, one codebase

| | Old system ("threshold engine") | Current system ("learned head") |
|---|---|---|
| Decides | hand-set `base_threshold` on sigma-point collision probability + ALERT/BRAKE state machine | causal GRU over per-frame features, thresholded at an operating point |
| Lives in | `a3ps/risk/collision.py`, `decision.py` | `a3ps/risk/temporal.py`, `anticipation_loss.py` |
| Runs | inside `scripts/run_pipeline.py` (online, per frame) | **offline** on cached feature files (`extract_features.py` → `train_risk_head.py` → `sweep_operating_point.py`) |
| Status | superseded, retained as baseline | **the headline system** |

Why the pivot: Nexar labels *both* collisions and near-misses as positive, so the task is "risky event vs ordinary driving", not "hit vs near-miss". No single hand-set threshold on one geometric feature separates those (old system: 46 of 60 negatives alerted, false-alarm rate 0.767).

---

## How the commands relate to each other

**Original two-command flow (still valid, this is the *old-system* pipeline + dashboard):**

```bash
# 1. The thinker — processes one video and writes files into the clip folder.
python scripts/run_pipeline.py --video data/dev_clips/dev01.mp4 --out dashboard/clips/dev01

# 2. The storyteller — tiny local web server for the dashboard.
python scripts/serve_dashboard.py     # then open http://localhost:8000
```

`run_pipeline.py` writes `events.json`, `meta.json`, `raw.mp4`, and (unless `render=False`) `annotated.mp4`. Optional input: `context.json` in the clip folder.

**[CHANGED] Learned-head flow (the current headline system):**

```
extract_features.py   (GPU, once)  -> data/features/*   112-dim per-frame vectors, 13 s windows @ 10 Hz
train_risk_head.py    (CPU, ~4 min)-> notebooks/models/risk_gru_k1p0.pt
sweep_operating_point.py (seconds) -> eval/operating_point_sweep_*.md   (threshold x confirm-frames grid)
build_dashboard_demo.py (no GPU)   -> dashboard/clips/<id>/{events.json, raw.mp4, risk.json}
```

The learned head is **not** called from `run_pipeline.py`. The dashboard's old-vs-new comparison is built from cached artefacts.

**Mental model:** the pipeline is the thinker; the dashboard is the storyteller. They never talk directly — they meet at files on disk. The dashboard runs no inference.

---

## Phase 1 — Perception

**Job.** For every frame, find which road actors are present and mark them at pixel level.

**Output.** Per detection: class (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`), confidence, bounding box, mask polygon.

**Logic.**

1. Load pretrained YOLOv8-Seg (`yolov8s-seg.pt`) once. No training; published COCO weights. CUDA auto-detected, `.half()` on GPU. `img_size` 1280.
2. Feed each frame in; keep only the six road classes with confidence ≥ 0.35.
3. Simplify each mask polygon to ≤ 40 points (OpenCV approximation) to keep `events.json` small.

**[CHANGED] Implementation note.** Perception and tracking are **one fused call** (`model.track()`), implemented in `a3ps/tracking/tracker.py`; there is no separate perception step at runtime (`a3ps/perception/segmenter.py` exists but the fused path is what runs). Class filtering happens **after** association — filtering before it caused a real ID-loss bug. Because they are fused, latency is reported as one combined figure (with an honesty note in `meta.json`).

**Why masks, not just boxes.** The mask says which exact pixels are the actor, giving a more honest "is it entering the corridor" answer.

---

## Phase 2 — Tracking

**Job.** Give each actor a persistent ID and a 2-second history.

**Logic.**

1. Ultralytics BoT-SORT via `model.track(...)`, config `configs/botsort_a3ps.yaml`.
2. Kalman-filter motion prediction matches new detections to existing IDs. **[CHANGED] `with_reid: False` — association is motion-only; the appearance cue this document previously described is NOT enabled.** A ReID arm is deferred future work (needs `onnxruntime`, changes association).
3. `track_buffer` raised 30 → 90 frames so IDs survive short occlusions.
4. Each track's **foot point** (bottom-centre of the box) is appended to a `TrajectoryBuffer` that decimates 30 fps → **5 Hz**, holding 2 s (10 points).
5. A track is "ready" after ≥ 8 points (`0.8 × 2 s × 5 Hz`).

**Where the difficulty is.** ID switches when actors overlap. Tolerated, and smoothed over downstream.

**[CHANGED] Frame-rate note.** The learned-head features are extracted at **10 Hz** (not 30 Hz) over 13 s windows. Rate is fixed at 10 Hz everywhere because mixing rates changes feature noise (central-difference span is `2/rate`). The 5 Hz history buffer above is the forecaster's rate, unchanged.

---

## Phase 3 — Forecasting

**Job.** Predict each ready track's position over the next 4 s (20 steps at 5 Hz) with a per-step standard deviation.

**Two interchangeable forecasters** (config key `forecaster: kalman_cv | seq2seq`, default `kalman_cv`):

- **Kalman-CV** (`kalman_cv.py`): state `[x, y, vx, vy]`; feed the 10 history points, then roll forward 20 steps without observations; covariance growth gives the uncertainty. Noise constants (`process_var=0.1`) were recalibrated against measured prediction error; falls back to a straight line under 4 history points.
- **[CHANGED] Seq2Seq-LSTM** (`seq2seq.py`): **no longer hypothetical/optional-Week-3 — it is implemented and trained** on trajectories mined from the Nexar data (`mine_trajectories.py`). Encoder/decoder with a per-step log-variance head (Gaussian NLL). On the 289-window val split it beats Kalman at 4 s (ADE 56.33 vs 61.27, FDE 99.74 vs 122.55). It is still not the pipeline default.

**Coordinate system.** **[CHANGED — status]** Forecasting runs in **image space** (`forecast_space: img`) using the "simple" path; corridor dilation uses pixel stand-in radii (`actor_radius_px`). The BEV/homography path is fully implemented and supported but **not calibrated per clip**, so it is not used in any headline result (deferred: `ground.yaml`, `verify_bev.py`).

**[CHANGED] Role after the pivot.** The Phase 3 collision probability and TTC are no longer the decision; they are **input features** to the learned head (`collision_prob`, `prob_trend`, `ttc_pred`).

---

## Phase 4 — Risk assessment and decision  **[CHANGED — rewritten]**

### 4a. Feature extraction (`a3ps/features/`, `scripts/extract_features.py`)

Each frame becomes a **112-dim vector**:

- **Ego motion** (4): from detection-masked sparse optical flow (`ego_tx, ego_ty, ego_log_scale, ego_rot`).
- **Global scalars** (4): `n_actors`, `frac_in_corridor`, `mean_prob`, `max_prob`.
- **Top-3 actors** by risk proxy, each with 19+ actor features (position/extent, velocity/acceleration, bbox growth, looming TTC `tau_area`/`tau_h`, `ttc_pred`, `collision_prob`, `prob_trend`, corridor distance / rate / in-corridor, class one-hot).
- **Pooled max** over all actors of the same actor features.

The head therefore sees **the scene, not one actor**: it pools over actors and cannot attribute risk to one (the dashboard's actor colour is scene-level for this reason; a per-actor head is future work).

Positives are windowed around `event_time_s` so the window always contains Nexar's `alert_time_s`; negatives get a deterministic pseudo-random window.

### 4b. Learned head (`a3ps/risk/temporal.py`, `scripts/train_risk_head.py`)

- Causal GRU, hidden 64, 1 layer, **34,465 parameters**, input LayerNorm, runtime causality assertion (a frame's output never depends on future frames).
- Outputs one risk logit per frame → probability.
- **Anticipation loss** (`anticipation_loss.py`): time-step weighting keyed to Nexar's `alert_time_s`, sharpness `kappa`, and a `pre_alert_weight` penalty for firing before anything is visible. `expected_lead_time()` reports what lead the loss is actually asking for.
- Checkpoint-on-improvement + early stopping; FA-aware selection ranks `(meets --fa-target, useful-warning, −FA)`. Batched forward pass (6.14× faster).
- Committed model: `notebooks/models/risk_gru_k1p0.pt`, kappa 1.0, best epoch 8.

### 4c. Decision = operating point (`scripts/sweep_operating_point.py`)

An alert fires when the head's probability stays ≥ **threshold** for **confirm** consecutive frames. Both are **eval-time only** — the whole trade-off curve costs one forward pass per clip, no retraining. Committed: **threshold 0.60, confirm 8**.

There is no SAFE→ALERT→BRAKE ladder and no dynamic/context thresholding in the learned system.

### 4d. Evaluation protocol

- Frozen split (`eval/split_freeze.json`, `a3ps/common/splits.py`): 15 dev clips, 60 positive + 60 negative held-out eval clips; drift guards refuse to run on a changed split. **Never tune on eval.**
- Training set: 685 positive / 680 negative labelled Nexar clips (1,485 feature-extracted). The 1,344 `test/` clips are unlabelled Kaggle holdout and excluded.
- Metrics: useful-warning rate (keyed to `alert_time_s`, with `n_too_early` as a separate failure mode), false-alarm rate, mean lead, mean AP, and official-style Nexar cutoff APs at 500/1000/1500 ms.

**Current results (held-out 120 clips):**

| metric | old threshold system | learned head | target |
|---|---|---|---|
| useful-warning | 0.050 (3/60) | **0.750 (45/60)** | ≥ 0.75 |
| false-alarm | 0.767 | **0.167** | ≤ 0.20 |
| mean AP | 0.546 | **0.680** | — |
| mean lead | — (not comparable) | **1.67 s** | 2–6 s ✗ **open gap** |

Never quote a useful-warning rate without the FA it was measured at (0.833 is available on the same checkpoint at FA 0.300). The two systems are scored on different observation windows (13 s @ 10 Hz vs full clips @ 30 Hz), so quote the useful-warning and AP gains, not the lead-time difference.

**What has been tried against the lead-time gap:** kappa ∈ {0.5, 1.0, 2.0, 3.0} (lead moved only 1.59–1.67 s, AP flat within 0.004); capacity `--hidden 128 --layers 2` (mean AP 0.702, but lead 1.59 s at FA ≤ 0.20 — no better). Remaining: 20 s feature window (GPU re-extraction) and better features. "This feature set supports ~1.7 s of warning" is the defensible finding until then.

**Feature ablations are NOT usable.** The four logged runs (ego / corridor / collision_prob / ttc) zeroed columns on training clips only, leaving validation intact — a train/eval shift. Fixed 2026-09-19, guarded by a test, **not yet re-run**. Do not quote the old numbers.

### 4e. The legacy threshold engine (baseline; still what `run_pipeline.py` runs)

Kept here because the pipeline still executes it and the dashboard shows it as "before":

1. Ego corridor trapezoid from `configs/default.yaml` (fractions of frame size: bottom 1.00 → top 0.60; x 0.39–0.61 bottom, 0.48–0.52 top).
2. Per track, per future step: 5 UKF sigma points from the step's Gaussian; corridor dilated by actor radius; weighted fraction inside = step probability. 3-step moving average, take max → `collision_prob`. `ttc_s` = first step whose smoothed probability exceeds 0.5, else the time of the max step, else none.
3. Cross-frame EMA (`RiskSmoother`, α = 0.4).
4. State machine per actor: `SAFE→ALERT` when prob ≥ (active_threshold − 0.15) for 3 consecutive frames; `ALERT→BRAKE` when ≥ active_threshold for 3 more; `BRAKE` is terminal for the incident, 2 s cooldown before re-arming.
5. Dynamic threshold: **`base_threshold` is 0.65** (recalibrated from 0.75 after the Kalman noise fix), minus context reductions (`crosswalk_ahead` −0.10, `intersection` −0.10, `dense_traffic` −0.05), floored at 0.45. Context flags come from optional hand-written `context.json`; detecting them from imagery is future work.
6. Ablation switches: `forecaster`, `dynamic_threshold`, `ema_smoothing`.

---

## Phase 5 — Explanation

**Job.** Turn the decision into a sentence a driver or evaluator can verify against the screen.

### Tier 1 — deterministic template (implemented, `a3ps/explain/templates.py`)

Pure string formatting from the event's own fields (action, actor class/ID, probability, TTC, threshold, context flags) plus a motion phrase derived from the track's velocity relative to the corridor:

> "Virtual braking engaged: pedestrian ID-4 trajectory crossing ego path in 1.2 s (P=0.81, threshold 0.65 — lowered due to crosswalk ahead)."

It cannot hallucinate: every number is the number that caused the trigger. **[CHANGED — scope note]** These events come from the *legacy threshold engine*. The learned head currently emits a probability curve and alert time, and the dashboard shows an explanation on intervention; the template's `threshold`/`TTC` fields describe the old engine's variables, not GRU internals. Explaining the GRU's decision is not separately implemented.

### Tier 2 — LLM enrichment (implemented, **not yet run for real**)

**[CHANGED] Uses Groq (OpenAI-compatible API, vision-capable Llama model), not the Anthropic API.** Offline script `a3ps/explain/llm_client.py`; needs `GROQ_API_KEY`; supports dry-run, overwrite, retries, and a facts-only constrained system prompt. A permissive-prompt variant (`llm_client_permissive_test.py`, self-restoring) exists to produce the paper's hallucination before/after example. Enrichment on 2–3 clips is still on the to-do list. Never runs during the pipeline loop; never blocks a demo. If unavailable, Tier 1 alone meets the requirement.

Viva answer: the safety-critical explanation is deterministic; the LLM narrative is labelled enrichment, constrained to the same facts and image.

---

## The whole pipeline in one paragraph

Read a frame; YOLOv8-Seg + BoT-SORT (one fused call, motion-only association) yield masked, ID-stable actors with a 2 s / 5 Hz history (**Phases 1–2**). A Kalman-CV (or trained Seq2Seq-LSTM) forecaster rolls each ready track 4 s ahead in image space with growing uncertainty (**Phase 3**). Per-frame, those tracks, the sigma-point collision probability, corridor geometry, looming TTC and optical-flow ego motion are packed into a 112-dim feature vector at 10 Hz; a 34k-parameter causal GRU trained with an anticipation loss turns the sequence into a risk probability, and an alert fires when it holds ≥ 0.60 for 8 frames (**Phase 4**, learned). The legacy threshold engine still runs in `run_pipeline.py` as the baseline. Events get a deterministic template explanation, with optional offline Groq enrichment (**Phase 5**). Everything is written to files; the dashboard replays old-vs-new side by side with nothing computed live.

---

## Data flow between phases

| Phase | Reads | Writes |
|---|---|---|
| 1 Perception (fused with 2) | raw frame | `cls`, `bbox`, `mask_poly` per detection |
| 2 Tracking | frame + detections | `id`, `centroid_img`, `history_img` per track |
| 3 Forecasting | `history_img` per ready track | `prediction.mean_img`, `prediction.std_*` (image space by default) |
| 4a Features | ClipResult + frame + optical flow | 112-dim × T array in `data/features/` |
| 4b/c Learned head | feature sequence | per-frame risk probability; alert time at the operating point |
| 4e Legacy engine | `prediction` + ego corridor | `collision_prob`, `ttc_s`, `risk_level`; appends `Event`s |
| 5 Explanation | each `Event` | `explanation_template`, optional `explanation_llm` |

The pipeline builds a single `ClipResult` (`a3ps/common/schema.py`, lossless JSON round-trip) and serialises `events.json` at the end. Dashboard clip folders for the comparison demo additionally hold `risk.json` (both systems' curves).

---

## Self-check questions

1. **"Where does the alert decision actually get made?"** — **[CHANGED]** In the learned GRU head: its per-frame probability must stay ≥ the threshold (0.60) for the confirm count (8) of consecutive frames. (Old answer — Phase 4's state machine — describes only the legacy baseline.)
2. **"Why does the dashboard never crash the demo?"** — It runs no inference; everything it shows was decided and written to disk in advance.
3. **"Why is the headline claim 0.750 at FA 0.167 and not just 0.750?"** — Because useful-warning without its false-alarm rate is meaningless; the old system alerted on 46/60 negatives.
4. **"What do you not claim?"** — Lead time (1.67 s vs 2–6 s target), the feature ablations, and any pre-pivot number from `status/results.md` or `design/explanation.md`.
