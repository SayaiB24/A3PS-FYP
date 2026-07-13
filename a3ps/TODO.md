# a3ps — TODO (regenerated 2026-07-14)

Two halves: **everything implemented so far** (the record), then **future
scope** (what's left, what's next, what could come after). For how to run
any of it: `execution.md`. For how it works: `logic_pipeline.md`.

---

## Part A — Implementation completed to date

### Core pipeline (all five phases, implemented + unit-tested)
- [x] **Phase 1+2 — Perception + Tracking** (`a3ps/tracking/tracker.py`):
      single fused YOLOv8-Seg `model.track()` call (detection + segmentation
      + BoT-SORT IDs in one pass); class filtering AFTER association (fixes a
      real ID-loss bug); foot-point centroids; `TrajectoryBuffer` decimating
      30 fps → 5 Hz with 2 s history per track; CUDA auto-detect + `.half()`;
      `track_buffer` 30→90 for occlusion robustness.
- [x] **Phase 3 — Forecasting** (`a3ps/forecasting/`): Kalman-CV (filterpy,
      `[x,y,vx,vy]`, measurement-covariance std reporting, tuned
      `process_var=0.1`, straight-line fallback under 4 history points) and
      Seq2Seq-LSTM (encoder/decoder, per-step log-variance head, drop-in
      `predict()` interface). **Trained on the real mined set**; on the
      289-window val split the LSTM beats Kalman at 4 s (ADE 56.33 vs 61.27,
      FDE 99.74 vs 122.55); both checkpoints kept (`seq2seq_v1.pt`,
      `seq2seq_prev.pt`) with both result tables archived.
- [x] **Phase 4 — Risk + Decision** (`a3ps/risk/`): sigma-point collision
      probability vs actor-radius-dilated ego corridor; per-trajectory
      max_prob + ttc_s; cross-frame EMA (`RiskSmoother`); `DecisionEngine`
      with context-lowered thresholds (floor 0.45), 3-frame confirmation,
      one-event-per-transition SAFE→ALERT→BRAKE, 2 s cooldown,
      THRESHOLD_LOWERED on span onset.
- [x] **Phase 5 — Explanation (XAI)** (`a3ps/explain/`): deterministic
      grammar templates on every event (motion phrase from BEV velocity,
      threshold-lowered reason); offline Groq vision-LLM enrichment
      (`llm_client.py`) with dry-run/overwrite/retries + facts-only
      constrained prompt; `system_prompt` override + permissive-prompt
      illustration variant (`llm_client_permissive_test.py`, self-restoring)
      for the paper's hallucination before/after example.
- [x] **Orchestrator** (`a3ps/pipeline.py`): injectable hooks per phase;
      `render=False` fast path for batch eval; **per-stage latency
      (`per_stage_ms`) written into meta.json** (ordering bug fixed — it
      used to be computed after the file write and never persisted), with an
      honesty note for the fused perception+tracking measurement.
- [x] **Shared schema** (`a3ps/common/schema.py`): lossless JSON round-trip
      contract consumed by every phase, every eval script, and the JS
      dashboard.

### Data & training infrastructure
- [x] `prepare_nexar.py` — Excel(s) auto-merge, flat-pool ingestion, split
      computation (dev 15 / eval 120 / rest→train_traj), dev-clip
      copy+rename with README mapping.
- [x] `mine_trajectories.py` — windowing, ID-switch + static filters (20%
      static kept), normalization with stored inverse transform, per-clip
      shards, **resume-by-shard-existence + self-healing stats**. Run for
      real on the GPU laptop (~93 clips mined; more negatives added).
- [x] `train_forecaster.ipynb` — Colab-ready but runs locally; early
      stopping on val ADE; retrained on the enlarged mined set.

### Evaluation suite (all implemented, tested on synthetic fixtures)
- [x] `eval_forecast.py` — ADE/FDE @1/2/4 s, both forecasters, same seeded
      val split; previous-model comparison workflow (`--weights`/`--out`).
- [x] `eval_anticipation.py` — detection rate / false-alarm rate / mTTA for
      **A3PS vs a reactive-proximity baseline** (exact port of the
      dashboard's marker logic); **matched-subset mTTA** (the fair
      anticipation-gain number); resumable `--run` with rendering off;
      **`--sweep`** threshold sweep (0.40→0.90) from a per-frame prob cache
      → PR curve CSV + PNG with zero extra GPU passes.
- [x] `run_ablations.py` — Table II in the paper's exact row order; config
      temporarily overridden and always restored; identical-config rows
      reuse the baseline run (4 real passes instead of 6).
- [x] `make_qual_figure.py` — 2×2 before/ALERT/BRAKE/after figure at
      300 DPI using the *actual* dashboard drawing code; verified on dev14.
- [x] `collect_paper_stats.py` — one-stop paper-stats scan (config,
      hardware/CUDA, splits, latency aggregation, all eval outputs); never
      fabricates missing numbers.
- [x] New ablation config switches wired through the pipeline:
      `forecaster: kalman_cv|seq2seq`, `dynamic_threshold`, `ema_smoothing`.

### Dashboard & demo
- [x] Static dashboard (canvas overlays, event log with explanations, risk
      timeline with live `active_threshold` line, brake banner/vignette,
      ▼ A3PS vs ▽ reactive-ADAS gap markers). All 15 dev clips processed
      with the full risk engine and committed.

### Documentation & quality
- [x] `execution.md` (from-scratch run guide), `logic_pipeline.md` (+ PDF),
      `metrics.md` (per-metric commands/reasoning/execution order),
      `QnA.md` (tricky-part Q&A), `GPU_HANDOFF.md` (kept current, incl.
      §9.1 VIRTUAL_BRAKE math and §10 matched-subset guidance), `results.md`
      + `explanation.md` updated for the retrained-LSTM outcome.
- [x] Test suite: **90 passing** (schema, geometry, collision, decision incl.
      new toggles, tracking buffer, forecasting, mining, eval metrics incl.
      sweep + matched-subset, qual-figure logic, pipeline latency, LLM
      client incl. prompt override).

---

## Part B — Future scope & future implementation

### B1. Immediate next runs (blocked only on GPU time, no new code)
- [ ] Finish `eval_anticipation.py --run` over the full 120-clip eval split;
      read the **matched-subset mTTA** and, if it's negative (A3PS later
      than the naive baseline, as seen on the tiny dev sample), tune
      `base_threshold` / `horizon_s` / `frames_to_confirm` and re-run until
      the operating point is defensible.
- [ ] `--sweep` on the cached eval results → PR curve for the paper.
- [ ] `run_ablations.py` overnight → Table II.
- [ ] 3+ clips re-run through `run_pipeline.py` **on the GPU laptop** so
      Table IV latency reflects GPU (CPU numbers are ~20-40x slower and must
      be labeled if used).
- [ ] Groq enrichment on 2-3 clips + the permissive-prompt before/after pair
      for Section VII.B.
- [ ] Decide (with supervisor) on the user study (Section VII.C): run a
      15-25-person Likert study, or state the limitation explicitly.
- [ ] Final `collect_paper_stats.py` pass; paste into the paper.

### B2. Week-4 planned work
- [ ] **Per-clip BEV calibration** for the ~5 final demo clips
      (`ground.yaml` per clip, `forecast_space: bev`, `verify_bev.py`
      check: 5-20 m/s velocities, ≥80% predictions in frame).
- [ ] Re-mine trajectories in BEV metres after calibration if the LSTM is to
      train in metric space (current mining is image-pixel space).
- [ ] Pick + polish the 5 demo clips (manifest, optional `context.json` for
      a THRESHOLD_LOWERED demo, one hero clip for the qualitative figure).

### B3. Model & algorithm improvements (post-deadline candidates)
- [ ] **Grow the mined set to the original 10k+ window target** and retrain
      the LSTM — its 4 s-horizon win should widen with more nonlinear-motion
      data; revisit making it the pipeline default if it also wins at 2 s.
- [ ] **Interaction-aware forecasting** (social pooling / attention over
      neighboring tracks) — current forecasters treat every actor
      independently.
- [ ] **Full-covariance uncertainty** — risk math currently assumes diagonal
      per-step Gaussians; correlated x-y uncertainty would sharpen corridor
      probabilities for turning actors.
- [ ] **Learned/adaptive decision thresholds** — replace the hand-tuned
      0.75/0.60/3-frame machine with thresholds optimized directly against
      mTTA-vs-false-alarm cost on the eval split (the PR-sweep
      infrastructure already computes the frontier).
- [ ] **Ego-motion compensation** — the ego corridor is static in the image;
      compensating camera motion (optical flow / homography between frames)
      would reduce false risk on turns.
- [ ] **Detect context flags from imagery** — crosswalk/intersection are
      currently per-clip annotations by design; a lightweight classifier
      would close that honesty gap.
- [ ] **Class-conditioned forecasting** — mined windows carry `class_ids`;
      conditioning the LSTM on actor class (pedestrian vs car dynamics) is
      unexploited.

### B4. Systems & engineering future work
- [ ] **True real-time mode** — frame-skipping / async stages / TensorRT or
      ONNX export of the YOLO model; target ≥15 FPS end-to-end on GPU.
- [ ] **Streaming input** (webcam/RTSP) instead of file-based clips.
- [ ] Fix the mining quirk where zero-window clips are re-tracked every run
      (write an empty marker shard).
- [ ] Batched forecasting: the per-track Python loop in the forecaster hook
      could batch all ready tracks per frame (matters for the LSTM path).
- [ ] Dashboard: side-by-side variant comparison (e.g. Kalman vs LSTM run of
      the same clip) and a PR-curve/ablation results page.
- [ ] CI (GitHub Actions) running the 90-test suite per push.

### B5. Research extensions (beyond the current paper)
- [ ] Evaluate on a second dataset (e.g. DoTA / DAD) to test generalization
      of thresholds tuned on Nexar.
- [ ] End-to-end learned risk (video → collision probability) as a
      comparison arm against this interpretable modular stack.
- [ ] Closed-loop simulation (CARLA): does the virtual brake actually avoid
      the collision when it fires at the observed mTTA?
- [ ] The user study, if not done now (three explanation conditions,
      clarity/trust Likert, paired significance test).
