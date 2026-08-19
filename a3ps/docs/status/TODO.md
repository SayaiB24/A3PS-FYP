# a3ps — TODO

Two halves: **everything implemented so far** (the record), then **future
scope** (what's left, what's next, what could come after). For how to run
any of it: `../runbooks/execution.md`. For how it works: `../design/logic_pipeline.md`.

> **⚠️ Part A below was written 2026-07-14, before the Phase IV pivot.** It is
> still an accurate record of Phases I–III, but where it says "Phase 4 — Risk +
> Decision" it means the **old hand-tuned threshold system**. That has been
> superseded by a learned temporal head. Part B has been rewritten for the
> current state.

---

## Status at a glance (2026-08-19)

**Where the project is:** the learned risk head is trained, evaluated on the
frozen held-out split, has a committed operating point, and the dashboard shows
it against the old system. Full reproduction steps:
[`../handoff/REPRODUCE_BY_HAND.md`](../handoff/REPRODUCE_BY_HAND.md).

| quantity | value | vs target |
|---|---|---|
| checkpoint | `notebooks/models/risk_gru_k1p0.pt` (34,465 params, kappa 1.0, best epoch 8) | — |
| operating point | threshold 0.60, confirm 8 | — |
| false-alarm rate | **0.167** | ✅ ≤ 0.20 |
| useful-warning rate | **0.750** (45/60) | ✅ meets 0.75 |
| mean lead time | 1.67 s | ✗ short of 2–6 s — **the open gap** |
| mean AP | 0.680 (0.785 / 0.697 / 0.557 @ 500/1000/1500 ms) | ceiling; threshold cannot raise it |
| too-early alerts | 1/60 | ✅ |
| tests | 170 passing | — |

**vs the old threshold system**, same frozen split (`eval/phase4_comparison.md`):

| metric | threshold system | learned head |
|---|---|---|
| useful-warning rate | 0.050 (3/60) | **0.750 (45/60)** |
| false-alarm rate | 0.767 | **0.167** |
| mean AP | 0.546 | **0.680** |
| fired before `time_of_alert` | 52/60 | 1/60 |

The old system's raw detection rate was higher (0.917 vs 0.750) only because it
alerted on nearly everything — including 46 of 60 negatives.

### ✅ Done since the pivot

- [x] **Split freeze** (`eval/split_freeze.json`, `a3ps/common/splits.py`) — dev/eval membership pinned; `prepare_nexar.py` and `eval_anticipation.py` refuse to run on a drifted split.
- [x] **Full dataset indexed** — 1,500 labelled clips (685 pos / 680 neg train, 60+60 frozen eval, 5+10 dev). Positives went 65 → 685. The 1,344 `test/` clips are unlabelled (Kaggle holdout) and correctly excluded.
- [x] **Feature extraction** — 1,485 clips at 10 Hz over 13 s windows, 112-dim, 0 failures, `has_ego=True` on every clip (first time the model has had real ego motion). Rate fixed at 10 Hz everywhere: mixing rates changes feature noise, since the central-difference span is `2/rate`.
- [x] **Learned risk head trained** (`a3ps/risk/temporal.py`, `scripts/train_risk_head.py`) with an `alert_time_s`-keyed anticipation loss.
- [x] **Checkpoint-on-improvement + early stopping** — runs are safe to interrupt (an 85-min run lost its best model before this).
- [x] **Operating-point sweep** (`scripts/sweep_operating_point.py`) — threshold/confirm are eval-time only, so the whole trade-off curve costs one forward pass per clip rather than a retrain per point.
- [x] **AP tie-order bug fixed** — the old 0.978 was an artefact of CSV row order (0.371 with the same scores reversed); corrected mean AP 0.546.
- [x] **`useful_warning_rate` re-keyed** to Nexar's own `alert_time_s`, replacing an invented 0.5–6.0 s window; `n_too_early` exposed as its own failure mode.
- [x] **Official-style Nexar cutoff APs** at 500 / 1000 / 1500 ms.
- [x] **Dashboard** — learned head vs old threshold system on the same clip, ground-truth markers, verdicts, explanation on intervention, slim overlays (0.1–2.4 MB vs 20–50 MB), grouped dropdown, correct threshold readout, self-explaining failure states.
- [x] **Handoff docs** — [`../handoff/`](../handoff/): reproduce-by-hand, next steps, kappa runbook, 19 challenges, future work; plus `metrics.md` §9–10 and a quick start in `execution.md`.

### ✅ Also done (2026-08-19, later)

- [x] **Kappa sweep** — trained kappa ∈ {0.5, 1.0, 2.0, 3.0}, swept the operating point for each, picked by the documented rule. **kappa 1.0 wins** and is now the committed model. Results: `eval/kappa_comparison.md`.
- [x] **Batched forward pass** — `batched_logits()`, **6.14× faster** (75 s → 12 s per epoch), verified against the per-clip path on logits, loss *and* gradients.
- [x] **FA-aware checkpoint selection** — ranking is `(meets --fa-target, useful-warning, -FA)`.
- [x] **Paper tables regenerated** — `eval/anticipation.md`, `eval/paper_stats_summary.md`, plus the new `eval/phase4_comparison.md` head-to-head.

**Kappa outcome, stated honestly:** lowering kappa did **not** fix lead time.
Across all four values lead moved only 1.59–1.67 s against a 2–6 s target, and
mean AP stayed flat within 0.004. A parameter that changes only *when* the model
fires leaving AP unmoved means the ceiling is the model's **discrimination** — a
feature/capacity limit, not a loss-tuning one. kappa 1.0 still won on the stated
rule and gave a modest real gain over the previous kappa 3.0 model at identical
FA: useful 0.717 → 0.750, lead 1.58 → 1.67 s.

### ⬜ Remaining — see [`../handoff/NEXT_STEPS.md`](../handoff/NEXT_STEPS.md)

1. **Lead time still misses target** (1.67 s vs 2–6 s). Kappa is exhausted as a lever. Next candidates in order: **more capacity** (`--hidden 128 --layers 2`, ~15 min now that training is 6× faster), **a longer feature window** (20 s, needs a GPU re-extraction), or **better features**. See [`../handoff/FUTURE_WORK.md`](../handoff/FUTURE_WORK.md) Tier 1.
2. **Write the numbers up** — quote 0.750 @ FA 0.167, and state that lead time is not met. Never quote a useful-warning rate without the false-alarm rate it was measured at.
3. **Strict old-vs-new comparison** — the two systems are scored under different observation conditions (13 s @ 10 Hz vs full clips @ 30 Hz). Re-running the threshold system on matched windows would make it exact; until then quote the useful-warning and AP gains (robust to this) rather than the lead-time difference (not).
4. **Optional** — see [`../handoff/FUTURE_WORK.md`](../handoff/FUTURE_WORK.md): Kaggle submission for an external AP, ReID arm, per-actor head, BEV calibration, user study.

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
- [x] **Phase 4 — Risk + Decision** (`a3ps/risk/`) — **⚠️ SUPERSEDED by the
      learned head; retained as a feature source and as the before/after
      baseline in the dashboard.** Sigma-point collision
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

### Phase IV — learned temporal risk head (added 2026-08-19)

Phases I–III above are retained as **feature extractors**; the hand-tuned
threshold risk formula is replaced by a learned discriminator. The motivating
correction: Nexar labels **both collisions and near-misses as positive**, so the
task is *risky-event vs ordinary-driving*, not *hit vs near-miss* — and no single
hand-set threshold on one geometric feature separates those, which is why the
field learns it. See [`../handoff/CHALLENGES.md`](../handoff/CHALLENGES.md) §6.

- [x] **Split freeze** (`a3ps/common/splits.py`, `scripts/freeze_split.py`) —
      dev/eval membership pinned outside gitignored `data/`; new clips may only
      join `train_traj`; new positives are left unassigned rather than
      auto-placed. Both `prepare_nexar.py` and `eval_anticipation.py` refuse to
      run on a drifted split.
- [x] **Feature extraction** (`a3ps/features/`, `scripts/extract_features.py`) —
      112-dim per-frame vectors: kinematics, bbox growth, looming TTC, corridor
      distance, the Phase III collision probability, class one-hot, and
      ego motion from detection-masked sparse optical flow (`ego.py`).
- [x] **Anticipation loss** (`a3ps/risk/anticipation_loss.py`) — timestep
      weighting keyed to `alert_time_s`, sharpness `kappa`, `pre_alert_weight`
      penalty for firing before anything is visible, and
      `expected_lead_time()` to report what lead the loss is actually asking for.
- [x] **Temporal head** (`a3ps/risk/temporal.py`) — causal GRU (hidden 64,
      1 layer, 34,465 params) with a runtime causality assertion.
- [x] **Training + selection** (`scripts/train_risk_head.py`) —
      checkpoint-on-improvement, early stopping, leakage guard.
- [x] **Operating-point sweep** (`scripts/sweep_operating_point.py`).
- [x] **Dashboard comparison** (`scripts/build_dashboard_demo.py`) — both
      systems' curves on the same clip from cached artefacts, no GPU.
- [x] **Association-rate check** (`scripts/check_assoc_rate.py`) — reports
      absolute forecastable tracks per clip, not the ratio that misleadingly
      improves as the frame rate drops.

### Documentation & quality
- [x] `../runbooks/execution.md` (from-scratch run guide), `../design/logic_pipeline.md` (+ PDF),
      `../design/metrics.md` (per-metric commands/reasoning/execution order),
      `../design/QnA.md` (tricky-part Q&A), `../runbooks/GPU_HANDOFF.md` (kept current, incl.
      §9.1 VIRTUAL_BRAKE math and §10 matched-subset guidance), `results.md`
      + `../design/explanation.md` updated for the retrained-LSTM outcome.
- [x] Test suite: **90 passing** (schema, geometry, collision, decision incl.
      new toggles, tracking buffer, forecasting, mining, eval metrics incl.
      sweep + matched-subset, qual-figure logic, pipeline latency, LLM
      client incl. prompt override).

---

## Part B — Future scope & future implementation

> **B1 and B2 as originally written are superseded.** They planned to tune the
> old threshold system's `base_threshold` / `horizon_s` / `frames_to_confirm`
> until its operating point became defensible. The Phase IV pivot replaced that
> system with a learned head, so those runs no longer describe the work. The
> current plan is in [`../handoff/NEXT_STEPS.md`](../handoff/NEXT_STEPS.md);
> B3–B5 below still stand as longer-range ideas and overlap with
> [`../handoff/FUTURE_WORK.md`](../handoff/FUTURE_WORK.md), which has costs and
> risks attached.

### B1. Immediate next work (current)

Summarised in "Status at a glance" above; full detail in the handoff docs.

- [ ] **Kappa retrain** to fix lead time — [`../handoff/KAPPA_RETRAIN.md`](../handoff/KAPPA_RETRAIN.md).
- [ ] **Regenerate the paper tables** against the learned head (`eval_anticipation.py`, then `collect_paper_stats.py`).
- [ ] **Groq enrichment** on 2–3 clips + the permissive-prompt before/after pair for Section VII.B. *(Still valid — the explanation layer is unchanged by the pivot.)*
- [ ] **Decide on the user study** (Section VII.C): run a 15–25-person Likert study, or state the limitation explicitly. *(Still valid; see `../design/metrics.md` §8.)*

### B2. Superseded / deferred

- [x] ~~Tune the old threshold system to a defensible operating point~~ — superseded: the learned head now provides the operating point (0.60/8, FA 0.167).
- [ ] **Per-clip BEV calibration** for the final demo clips (`ground.yaml`, `forecast_space: bev`, `verify_bev.py`). Still worthwhile for interpretability — it makes distances metric instead of pixel stand-ins — but it affects no headline metric. Deferred, listed in `../handoff/FUTURE_WORK.md`.
- [ ] Re-mine trajectories in BEV metres if the LSTM is ever to train in metric space. Only relevant to the Phase III forecaster, which the learned head now consumes as a feature rather than depending on directly.
- [x] ~~Pick + polish demo clips~~ — done: 6 Phase IV comparison clips (621, 488, 1004, 690, 1085, 1261) built by `scripts/build_dashboard_demo.py`, chosen for the sharpest old-vs-new contrast.

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
