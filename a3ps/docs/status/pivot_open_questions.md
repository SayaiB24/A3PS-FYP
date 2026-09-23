# Phase IV pivot — four open design questions, investigated (2026-09-22)

A status writeup of this project stated four things as facts without giving a
reason, or as "not yet done" without a stated blocker. This document gives the
actual reasons, found in the code and git history, not restatements of the
symptom. Each section cites the file:line evidence it is based on.

Investigated: why Seq2Seq-LSTM never reaches headline (GRU) results despite
beating Kalman on ADE/FDE; why Kalman is still the pipeline default; why
BEV/homography is not used for headline results; and whether the feature
ablations are actually rerunnable now (they were — new numbers are in §4).

---

## 1. Why Seq2Seq-LSTM never reaches headline results, despite beating Kalman on ADE/FDE

**Finding.**

*It is wired correctly end-to-end at the code level.* `scripts/run_pipeline.py`
picks the forecaster from `config["forecaster"]`
([run_pipeline.py:45-51](../../scripts/run_pipeline.py#L45-L51)) and both
forecasters implement the identical
`predict(history, dt, horizon_s) -> (means, stds)` interface
([forecasting/base.py](../../a3ps/forecasting/base.py)). The GRU feature
pipeline (`scripts/extract_features.py`) does not have its own forecaster
logic — it imports and reuses the exact same `make_forecaster_hook` /
`make_risk_hook` from `run_pipeline.py`
([extract_features.py:225-229](../../scripts/extract_features.py#L225-L229)),
so `forecaster: seq2seq` would flow through feature extraction unchanged if
set. Downstream, `collision_prob`/`ttc_pred`/`tau_area`/`tau_h` are computed by
`trajectory_collision_prob()` / `step_collision_prob()`
([risk/collision.py:122-206](../../a3ps/risk/collision.py#L122-L206)), which
consume only a diagonal `(mean, std)` pair per step via
`sigma_points_2d()` ([risk/gaussian.py:44-60](../../a3ps/risk/gaussian.py#L44-L60))
— there is no code path anywhere in this chain that reads a Kalman state
vector or a covariance cross-term. `Seq2SeqForecaster.predict()` explicitly
converts its per-step log-variance head into the same diagonal `[sx, sy]`
shape before returning
([forecasting/seq2seq.py:144-158](../../a3ps/forecasting/seq2seq.py#L144-L158)),
so the "does the legacy engine assume a Kalman covariance" concern the audit
raised does not hold — it was designed and built as a drop-in, and the code
confirms it behaves as one.

*But it has never actually been run through that chain.* Every feature file in
`data/features/{train_all,eval,...}` was produced by
`scripts/extract_features.py::run_from_video`, which calls
`load_config(args.config)` with no forecaster override
([extract_features.py:220-229](../../scripts/extract_features.py#L220-L229)) —
i.e. every training and eval clip used whatever `configs/default.yaml` says,
which is `forecaster: kalman_cv`. `data/features/eval/manifest.json` doesn't
even record which forecaster ran (no `forecaster` key at all), because there
has only ever been one value to record. No `data/features/*_seq2seq` directory,
no alternate `notebooks/models/risk_gru_*seq2seq*.pt` checkpoint, and no
`eval/*seq2seq*` GRU result exist anywhere in the repo. The only place
`forecaster: seq2seq` is actually exercised is
`scripts/run_ablations.py`'s `VARIANTS` list
([run_ablations.py:52-58](../../scripts/run_ablations.py#L52-L58)), and that
targets the **legacy** `eval_anticipation.py` threshold pipeline (mTTA /
detection / false-alarm, pre-pivot), not the GRU risk head — and even that run
left no surviving artifact: `eval/ablation_table.md` and `eval/ablation/`,
which `docs/design/metrics.md` §4 documents as "✅ actually re-run", do not
exist anywhere in this checkout. So Seq2Seq's evaluation, in this repository,
stops at trajectory error (`eval/forecast_table.md`) — it was never
integration-tested through feature extraction, let alone through the GRU.

*The ADE/FDE win was measured on full histories only, not the padded case
production actually hits.* `scripts/eval_forecast.py::load_val` scores
directly on `data/trajectories/*.npz` shard arrays
([eval_forecast.py:29-46](../../scripts/eval_forecast.py#L29-L46)). Those
shards are produced by `mine_trajectories.py::contiguous_windows`, which only
accepts a fixed-length, fully-contiguous window of `WIN = HIST + FUT = 30`
points (`HIST=10`, `FUT=20`)
([mine_trajectories.py:35-37,64](../../scripts/mine_trajectories.py#L35-L37))
— there is no code path in mining that keeps a short or gappy history. In
production, though, a track becomes eligible to forecast as soon as
`TrackBuffer.ready()` sees `0.8 * history_s * predict_hz = 8` points, not the
full 10 (`history_s=2.0`, `predict_hz=5` in `configs/default.yaml`; gate at
[tracker.py:54,90-92](../../a3ps/tracking/tracker.py#L54)). At that moment
`Seq2SeqForecaster.predict()` left-pads by repeating the *oldest* history
point once or twice to reach `HIST=10`
([seq2seq.py:134-137](../../a3ps/forecasting/seq2seq.py#L134-L137)) — a
different, degraded input than anything in the 289-window val set, which is
built exclusively from real, unpadded 10-point histories. So the reported
56.33-vs-61.27 px ADE@4s advantage is real for the condition it was measured
under, but that condition (a full, contiguous history) is not the condition
new tracks actually forecast from in the first 0.4-1.2s after becoming
eligible — the comparison doesn't necessarily transfer, and nothing in the
repo tests whether it does.

*No compute/latency reason is documented or evident.* Neither forecaster has
a benchmarking script, and `eval/latency_table4.md` (Table IV, CPU-only) times
perception/tracking, not forecasting; nothing in the repo measures or
comments on Seq2Seq's inference cost relative to Kalman's.

**Verdict:** unresolved-nobody-checked. The plumbing is genuinely
forecaster-agnostic (not a stale assumption problem), but nobody has run the
GRU pipeline with it, and the one number that exists in its favor (ADE/FDE)
was measured under easier conditions than production readiness actually
produces.

**Recommendation:** worth pursuing, but only after confirming the readiness-gate
gap matters. Cheapest next step is not a full retrain: extend
`scripts/eval_forecast.py` (or a small variant) to also score ADE/FDE on
Seq2Seq using **padded** 8-9-point histories drawn from the same shards
(truncate the stored 10-point history before predicting), and compare against
Kalman on the identical truncated inputs. If Seq2Seq's edge survives
truncation, it's worth the ~20-30 min cost of extracting one
`data/features/eval_seq2seq/` set (`--config` override with `forecaster:
seq2seq`) and sweeping a GRU trained on it against the committed 0.680 mean AP
— that is the only way to know whether it moves the actual headline numbers
rather than a proxy metric.

---

## 2. Why Kalman is the default when it underperforms Seq2Seq

**Finding.** `forecaster: kalman_cv` was introduced in commit `57e82da`
("Final dump", 2026-07-14) together with `seq2seq_weights`,
`dynamic_threshold` and `ema_smoothing`, all under one comment: "Ablation
switches (`scripts/run_ablations.py` flips these one variant at a time;
defaults below reproduce the standard pipeline)"
([configs/default.yaml:31-35](../../configs/default.yaml#L31-L35), see also
`git show 57e82da -- configs/default.yaml`). That framing is explicit: these
four keys were added *so the ablation study would have something to flip*,
with Kalman as "the standard pipeline" from day one — not chosen by
comparing ADE/FDE. `docs/design/QnA.md` confirms the same framing directly:
"**Which forecaster does the live pipeline/dashboard actually use?** A:
Kalman-CV, by config default... The LSTM is selectable... added for the
ablation study — but retraining the LSTM never changed the dashboard, because
the dashboard path never loaded it."
([QnA.md:85-89](../design/QnA.md#L85-L89)).

The ADE/FDE numbers that show Seq2Seq winning at 4s (56.33 vs 61.27 px) came
later, from commit `2a04195` ("Calibrate forecaster uncertainty...",
2026-08-18) — over a month after the default was set — and the default was
never revisited afterward; `git log -- configs/default.yaml` shows no commit
touching the `forecaster` key since `57e82da`. There is no hard technical
dependency forcing Kalman either: as shown in §1, the risk/decision code
downstream is forecaster-agnostic by construction
([risk/collision.py](../../a3ps/risk/collision.py),
[risk/gaussian.py](../../a3ps/risk/gaussian.py)).

**Verdict:** both at once, and worth being precise about which part is which.
*Deliberate*: Kalman as the always-on baseline (deterministic, no training/GPU
dependency, no risk of stale/mismatched weights) is a reasonable default for
a system meant to "just work" without a training step — this part of the
choice has a real rationale even though it isn't spelled out as a design
decision anywhere. *Stale*: the specific claim "Kalman is better" or "the
choice was compared" is not true and was never true; the default predates the
comparison, and the comparison's result (in Seq2Seq's favor, at the horizon
that matters for this project) was never fed back into the decision.

**Recommendation:** leave the config default as-is until §1's transfer
question is answered — switching now would trade a determinism guarantee for
an advantage that hasn't been shown to survive real (padded-history)
conditions. But the documentation should stop implying the default reflects a
considered performance judgment; it doesn't, and QnA.md already says so
plainly if anyone reads that far.

---

## 3. Why BEV/homography is not used for headline results

**Finding.** Per-clip calibration in this codebase means exactly one thing: a
manual 4-point homography (`image_points_frac` -> `ground_points_m`) in
`configs/default.yaml`'s `ground_plane` block, or a per-clip override file
loaded by `GroundPlane.from_config()` / `load_ground_override()`
([common/geometry.py:159-179,211](../../a3ps/common/geometry.py#L159)). There
is **no automatic vanishing-point or camera-intrinsics estimation anywhere in
the repo** — `grep`ing the codebase for any such logic finds none. The single
default trapezoid, added once in the first commit (`d009d1b`) and never
changed since (`git log -p -- configs/default.yaml` shows only one
`ground_plane` block, ever), is a generic guess, not measured from any real
camera. The override mechanism exists and is documented
(`ground_plane_override: null  # e.g. dashboard/clips/<clip>/ground.yaml`,
[default.yaml:77,89](../../configs/default.yaml#L89)) but **zero
`ground.yaml` files exist anywhere in the repository** (`find . -iname
ground.yaml` returns nothing) — the override path is wired but has never
been used, for even one clip.

`scripts/verify_bev.py` is not a broken or silent-failure tool — it runs, and
I ran it in this session against every real `events.json` present in
`dashboard/clips/`:

| clip | verdict |
|---|---|
| `02134_pipe` | OK |
| `1004` | SUSPECT |
| `1042` | SUSPECT |
| `1054` | SUSPECT |
| `1085` | SUSPECT |
| `dev01` | SUSPECT |
| `dev05` | SUSPECT |
| `dev10` | OK |
| `demo` | OK (has no `meta.json` — likely `make_demo_clip.py` synthetic output, not a real pipeline run) |

6 of 8 real-pipeline clips fail plausibility under the one fixed default
trapezoid. This directly confirms the calibration problem is dataset-wide, not
a one-clip fluke (worth noting: `docs/design/explanation.md`'s specific claim
that verification "failed this check on real footage" [§12.4,
explanation.md:1207-1209](../design/explanation.md#L1207) does not reproduce
on the one artifact its own example command names, `02134_pipe/events.json`
— that one comes back OK today. The underlying conclusion the doc draws
holds, just not from the specific clip it cites; I verified this by running
the tool myself rather than trusting the doc's citation). Note also that
`centroid_bev`/`velocity_bev` — what `verify_bev.py` checks — are computed
unconditionally by `Pipeline._update_bev()` every frame regardless of
`forecast_space`
([pipeline.py:120-132](../../a3ps/pipeline.py#L120)), so this is a live,
continuously-running check, not a dead code path nobody exercises.

**Is it wired into `extract_features.py`?** Partially, and correctly for what
exists: `forecast_space` is a config key read the same way by both
`run_pipeline.py` and `extract_features.py` (they share `make_forecaster_hook`
/ `make_risk_hook`), so setting `forecast_space: bev` would take effect in
feature extraction with zero code changes. The gap isn't wiring — it's that
there is no calibration *data* to switch to, since no per-clip `ground.yaml`
exists for any of the 1,365+120 extraction clips, and building one manually
per clip at that scale is not a "smaller gap" (a missing wiring step); it's a
real per-camera, per-video calibration problem across a dataset "varied
cameras" (Nexar dashcams, unknown mount angle/height per submission) with no
automatic estimator implemented to amortize it.

**Verdict:** genuine blocker, correctly scoped as such. `docs/status/TODO.md`
and `docs/handoff/FUTURE_WORK.md` already say this precisely: "Per-clip BEV
calibration for the final demo clips... affects no headline metric. Deferred."
That is the accurate framing, not a placeholder excuse — there is no
lower-effort fix available, and it doesn't gate anything the pivot's headline
numbers depend on (mean AP / useful-warning / FA are all computed on pixel
features, and `img_to_bev` outputs are logged, not consumed by the GRU).

**Recommendation:** correctly left alone for headline results. Worth pursuing
only at the scale FUTURE_WORK.md already scopes it — a handful of final demo
clips, for interpretability (metric distances instead of pixel stand-ins), not
as a blanket per-clip effort across the dataset. If pursued, the first real
step is deciding whether to hand-annotate ~4 points per demo clip (cheap, a
few minutes each) or invest in an automatic estimator (nontrivial, and
currently unbuilt) before writing any `ground.yaml` files.

---

## 4. Feature ablations — fix verification and rerun

### 4.1 Was the fix actually applied correctly, on both sides?

Yes, verified by reading the diff, not just noting a doc's claim. The
uncommitted change to `scripts/train_risk_head.py` now calls
`apply_ablation()` on **both** `clips` (train) and `val_clips`
([train_risk_head.py:469-470](../../scripts/train_risk_head.py#L469-L470)),
with a comment explaining exactly the failure mode being fixed. Before the
fix, only `apply_ablation(clips, abl_cols)` ran, and — critically — with
`--val-features` given, `val_clips` is loaded from an *entirely separate* file
list than `clips` ([train_risk_head.py:434-443](../../scripts/train_risk_head.py#L434)),
so the old code genuinely left validation untouched; this isn't a
double-zeroing-is-harmless case, it's a real train/eval mismatch as the
banner in `docs/status/ablations.md` describes.

The regression test,
`tests/test_train_risk_head.py::test_ablation_zeroes_val_features_too`
([tests/test_train_risk_head.py:164-206](../../tests/test_train_risk_head.py#L164)),
does not just check the code runs — it builds separate synthetic train/`va`
`.npz` clips, monkeypatches `train()` to capture the exact `train_clips` and
`val_clips` lists `main()` passes it, and asserts the ablated columns are zero
**and** every other column is untouched, on **both** sides. I ran it directly
(`pytest -k ablat`) and it passes against the current code
(`1 passed`). Reading the pre-fix code confirms it would fail: `val_clips`
would still carry ones in the ego columns the test checks. This is a real,
targeted regression guard, not a rubber stamp.

### 4.2 Is the current committed model still compatible?

Yes. `frame_feature_dim()` returns 112 and every checkpoint — the committed
`risk_gru_k1p0.pt` and all four `risk_gru_abl_*.pt` — report
`feature_dim: 112` in their saved `extra` dict, matching
`data/features/eval`'s actual array width (verified directly: `X.shape[1] ==
112` for every `.npz` in that directory). `eval/split_freeze.json`'s 120 `eval`
clip IDs match exactly the 120 `.npz` files in `data/features/eval` (set
equality, verified), with no label mismatches and zero overlap with
`data/features/train_all`. Nothing about the fix or a rerun touches the
frozen split or the feature schema.

### 4.3 Were the ablations already rerun?

**Yes — this had already been done, uncommitted, in the working tree, before
this investigation started.** The four `eval/train_log_abl_*.txt` /
`eval/risk_gru_history_abl_*.json` / `notebooks/models/risk_gru_abl_*.pt`
files on disk right now are **not** the invalid pre-fix runs `git show HEAD:`
still holds — they are already the post-fix reruns (file mtimes: fix at
20:24, ego/corridor/collprob/ttc logs written 20:50-21:58, all 2026-09-19,
strictly after). The `docs/status/ablations.md` banner claiming "have **not**
been re-run" was written mid-sequence (20:25, before the corridor/collprob/ttc
runs finished at 20:53/21:55/21:58) and was simply never updated afterward —
it is now stale relative to the working tree, not an accurate description of
the current state. I did not need to retrain anything; I verified the
training-time numbers matched (see below) and then completed the one thing
the doc itself still called for: sweeping each retrained checkpoint to the
committed operating point.

**Training-time numbers** (threshold 0.5, confirm 3 — the training script's
scoring default, from `eval/train_log_abl_*.txt`, post-fix):

| group | columns zeroed | best epoch | useful-warning | false-alarm | mean lead | mean AP |
|---|---|---|---|---|---|---|
| `ego` | 4 | 8 | 0.817 (49/60) | 0.583 | 2.13s | 0.689 |
| `corridor` | 13 | 4 | 0.767 (46/60) | 0.400 | 1.72s | 0.654 |
| `collision_prob` | 6 | 4 | 0.800 (48/60) | 0.583 | 2.15s | 0.659 |
| `ttc` | 12 | 10 | 0.850 (51/60) | 0.600 | 2.24s | 0.650 |

None of these are comparable to the committed model's headline numbers at
this threshold — as `docs/status/ablations.md` itself says, they need
sweeping to the committed operating point first. I did that.

### 4.4 Rerun at the committed operating point (threshold 0.60, confirm 8)

I wrote `scripts/rerun_ablation_eval.py` (new file; does not modify any
existing script) to score each already-retrained checkpoint
(`notebooks/models/risk_gru_abl_<group>.pt`) on `data/features/eval` — the
frozen 120-clip eval split, untouched — with the **same** feature columns
zeroed at eval time as were zeroed for that arm during training (using
`resolve_ablation()`/`apply_ablation()` from `train_risk_head.py` directly, so
the eval-time ablation is defined identically to the training-time one, not
reimplemented). Output written to `eval/ablation_rerun_2026-09-22/` — a new
directory, so the pre-fix invalid `eval/train_log_abl_*.txt` /
`eval/risk_gru_history_abl_*.json` artifacts (still recoverable from `git show
HEAD:`) are left alone and the two states stay distinguishable. No existing
`eval/` file was modified; `eval/split_freeze.json`,
`notebooks/models/risk_gru_k1p0.pt`, and the four ablation checkpoints
themselves were only read, not written.

**Committed full-feature model, for reference** (`eval/operating_point_sweep_k1p0.md`):
useful-warning 0.750 (45/60), false-alarm 0.167, mean lead 1.67s, mean AP
0.680.

**New ablation results — threshold 0.60, confirm 8, on the frozen eval split:**

| group | columns zeroed | useful-warning | false-alarm | mean lead (s) | mean AP | Δ mean AP vs committed |
|---|---|---|---|---|---|---|
| `ego` | 4 | 0.683 (41/60) | 0.150 | 1.68 | 0.689 | +0.009 |
| `corridor` | 13 | 0.583 (35/60) | 0.100 | 1.35 | 0.654 | −0.026 |
| `collision_prob` | 6 | 0.717 (43/60) | 0.267 | 1.65 | 0.659 | −0.021 |
| `ttc` | 12 | 0.833 (50/60) | 0.383 | 1.95 | 0.650 | −0.030 |

Sanity check: mean AP is threshold-free by construction (it ranks by peak
probability, per `sweep_operating_point.py`'s own header comment), so it
should be identical to the training-time validation number for that arm
regardless of confirm/threshold — it is, for all four groups (0.689 / 0.654 /
0.659 / 0.650 match §4.3 exactly), which cross-validates that this rerun is
scoring the same ablated inputs the training run used, not a different or
inconsistent set.

**Reading it.** At the committed operating point, `corridor`,
`collision_prob` and `ttc` all cost mean AP relative to the full-feature
model (−0.021 to −0.030), consistent with each carrying real signal — `ttc`
costs the most, matching the pre-fix run's qualitative direction (it was the
most damaged arm there too, though that number was invalid). `ego` is the one
surprising result: mean AP is *slightly higher* without it (+0.009) but
useful-warning drops sharply (0.750 → 0.683) at a much lower false-alarm rate
(0.167 → 0.150) — removing ego features doesn't hurt ranking quality but does
change where the operating point lands, which is a different effect than a
pure ablation cost. With one seed and one checkpoint per arm this is not
strong enough to draw a "ego doesn't matter" conclusion; re-running with 2-3
more seeds per group would be needed before quoting a ranked feature
importance.

**Caveat:** all four ablation checkpoints (`notebooks/models/risk_gru_abl_*.pt`)
and their training logs are currently **uncommitted**. If this project wants
to keep the post-fix ablation numbers, those files plus the new
`eval/ablation_rerun_2026-09-22/` directory and `docs/status/ablations.md`'s
banner all need to be committed together, and the banner text updated — right
now it still says the ablations were never rerun, which this investigation
found to be false.
