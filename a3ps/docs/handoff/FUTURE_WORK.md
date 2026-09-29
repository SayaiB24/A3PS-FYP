# Future work — better-to-have improvements

Not required by anything committed. Ordered by expected value per unit effort, so
working top-down is a reasonable default. Each entry says what it would buy, what
it costs, and the specific risk that makes it non-trivial.

**Where this starts from (2026-09-29).** The locked result is useful-warning
**0.650** at FA **0.167**, mean lead **1.51 s**, mean AP **0.646** on `eval_v2`,
read once ([`../REPORT_evaluation_audit.md`](../REPORT_evaluation_audit.md)).
Five levers have been tried on validation and none moved it beyond noise:
kappa, ensembling, temperature calibration, capacity, and the Seq2Seq-LSTM
forecaster ([`../status/headline_number_attempts.md`](../status/headline_number_attempts.md)).
The remaining open target is **lead time**, and the evidence says the limit is
the **training objective**, not the model: the loss gives zero signal outside
`[alert_t, event_t]` and that window averages ~1.6 s
(`REPORT_evaluation_audit.md` §5.3). Tier 1 is ordered with that in mind.

For the one-week judges plan, see [`JUDGES_PREP_PLAN.md`](JUDGES_PREP_PLAN.md);
for the post-judges order of work, [`NEXT_STEPS.md`](NEXT_STEPS.md).

**Before comparing any run below with the baseline:**
`sweep_operating_point.py` prints the **legacy** useful-warning only. Compare with
`scripts/rescore_candidates_both_defs.py`, which scores both definitions and the
selection rule on one footing. A legacy-vs-v2 mismatch already produced one wrong
conclusion (`JUDGES_PREP_PLAN.md` §1.11).

---

## Done or ruled out

### ~~Batched training forward pass~~ — ✅ DONE

`batched_logits()` pads each batch into one `(B, T_max, D)` tensor and slices the
valid prefixes back out. **Measured 6.14×** (75 s → 12 s per epoch). No loss
masking was needed: the GRU is causal and its input norm is per-timestep, so
end-padding cannot reach valid positions. Logits, loss and gradients are all
asserted equal to the per-clip path in `tests/test_train_risk_head.py`.

### ~~Model selection respecting the false-alarm constraint~~ — ✅ DONE

Selection is now `(meets --fa-target, useful-warning, -FA)`: a compliant epoch
always beats a non-compliant one and ties break toward lower false alarms. An
unconditional near-chance check flags degenerate picks on top of that. See
[`CHALLENGES.md`](CHALLENGES.md) §15 and `REPORT_evaluation_audit.md` §2.4.

### ~~More model capacity~~ — ❌ TRIED, NO GAIN (2026-09-28)

`--hidden 128 --layers 2` and `--hidden 256 --layers 2` on
`train_core_v2`/`train_val_v2`, seed 1234. At each run's best passing cell,
useful_v2 0.553 and 0.573 vs the baseline's 0.612, with shorter lead (1.48 s,
1.28 s vs 1.83 s); mean AP 0.601 and 0.641 vs 0.630. All within single-seed
noise. No evidence a bigger head helps on ~1,160 training clips. Keep
`--hidden 64 --layers 1`. (An earlier write-up called this a "collapse"; that
compared legacy numbers against a v2 baseline.)

### ~~Seq2Seq-LSTM forecaster in production~~ — ❌ TRIED END TO END, NO GAIN (2026-09-28)

Re-extracted all 1,485 clips with `forecaster: seq2seq` and trained the same
GRU. Useful_v2 **0.621 vs 0.612** (a tie), FA 0.176 vs 0.196, but lead **1.27 s
vs 1.83 s**; mean AP 0.622 vs 0.630. Kalman is ~31% more accurate at 1 s, where
alerts live; the LSTM only wins at 4 s. Kalman stays the default. Revisit only
via the "horizon-blended forecaster" entry in Tier 3.

### ~~Ensembling and temperature calibration~~ — ❌ TRIED, NO GAIN (Step 10.6)

Five-seed ensemble +0.019 useful_v2 (CI straddles zero) at 0.44 s less lead;
temperature calibration −0.019 and did not narrow the seed spread
([`../status/ensemble_calibration_results.md`](../status/ensemble_calibration_results.md)).
Still useful as a variance reducer; see Tier 2.

---

## Tier 1 — cheap, and likely to change what can be claimed

### Feature-group ablations on `train_val_v2`

**~1 h · CPU.**

The only ablations on file were selected under the flawed v1 protocol
([`../status/ablations.md`](../status/ablations.md)), and the paper has a
`[PENDING]` for them. Four retrains (`--ablate ego|collision_prob|ttc|corridor`),
same recipe as the baseline; commands in `JUDGES_PREP_PLAN.md` §3.2.

**Risk:** one seed per arm. The v1 post-fix rerun found every group inside a
±0.03 mean-AP noise band, so the likely finding is "no single group is
necessary". Read by mean AP, not useful-warning.

### Simple baselines on the same features

**~2 h · CPU.**

Score `max_prob` (frame-level max collision probability) thresholded with
confirm-N, and a per-frame logistic regression on the 112-d features, through the
same `evaluate()` and grid. Answers "does the GRU add anything over its best input
or a linear model?", which the paper currently can't ("No comparable baseline
yet").

**Risk:** none worth naming; if the GRU barely beats logistic regression, that is
a result, and it's better to find out before a judge asks.

### An objective that rewards warning before `alert_t`

**~1 day · CPU.**

**The actual lever for lead time**, and first on the paper's own future-work list.
`alert_weights()` (`a3ps/risk/anticipation_loss.py`) multiplies every frame
before `alert_t` by zero (positives) or penalises it via `pre_alert_weight`, so the
model is never asked to fire early. Options: a decaying positive ramp over
`[alert_t − Δ, alert_t]`, or an exponential time-to-event weighting in the style of
the DSA/AdaLEA anticipation losses, keyed to `event_t` rather than `alert_t`.

**Risk:** this is exactly the axis on which the old threshold engine failed
(median 14 s premature). Gate every candidate on the v2 prematurity axis
(`frac_premature`, `n_gross_premature`), not just useful-warning and FA, and
decide the acceptable prematurity before looking at results.

### Kaggle submission for an external AP

**~2 h · GPU for extraction.**

The 1,344 unlabelled `test/` clips have no ground truth locally, but the
competition can score them. That yields an **externally validated** AP with no
possibility of us having leaked or mis-scored anything, and it's the only way to
get a new held-out number without spending the last `eval_v2` read. It also
answers "how do you compare with published work?"

Needs: extract features for the test clips (`--split` will need assigning, as they
are currently unindexed), run the head, emit predictions in the competition's
submission format. Check the competition is still accepting submissions first.
Extraction measured ~23.5 s/clip on the RTX 3050, so budget ~9 h of GPU.

---

## Tier 2 — moderate cost, addresses a known limitation

### Multi-seed reporting on the held-out split

**~1 h CPU + the last `eval_v2` read.**

The headline is one seed, and seeds of the same recipe spread 0.39–0.86 on
validation useful-warning at the locked operating point. Reporting mean ± sd over
five seeds of the **same locked recipe** on `eval_v2` would replace the single draw
with a distribution; the paper promises it.

**Risk:** it spends the one remaining read. It's not selection (nothing is being
chosen among the seeds), so it doesn't reintroduce the leak, but decide this in
writing beforehand and don't combine it with any other candidate.

### Seed-variance reduction

**~half day · CPU.**

Seed choice moves useful-warning more than any lever tried. Candidates:
stochastic weight averaging over the last K epochs, or the five-seed ensemble
recast as a robustness measure (it was smoother across cells, at 5× inference
cost). Judge by the spread across seeds at a fixed cell, not by the best seed.

**Risk:** the calibration attempt showed seeds differ in *which clips* they fire
on, not just scale, so averaging may reduce spread without raising the mean.

### Per-actor temporal head

**~2 days.**

The head pools features across actors into one frame-level vector, so it cannot
say *which* actor is dangerous. The dashboard's explanation names an actor from
the cached tracks, which is presentation, not a model output, and the code says so.

A per-actor head (run the GRU per track, aggregate to a frame-level risk) would
make attribution a genuine model output and make explanations honest. It also
enables per-actor risk colouring in the dashboard overlay.

**Risk:** a substantial change to the feature pipeline; `extract.py` already emits
per-actor rows before pooling, so the data exists, but training and evaluation
both need reworking, and variable actor counts per frame need handling.

### Feature attribution for the GRU

**~half day · CPU.**

The paper defers "a faithful attribution of the GRU's decision to its input
feature groups". Integrated gradients (or occlusion by feature group) at the alert
frame would give per-alert attribution without retraining, and would let the
explanation template name the cue that actually drove the alert.

**Risk:** attributions on a pooled vector attribute to feature *groups* and pooled
slots (top-3 actors, max-pool), not to named actors. Pair with the per-actor head
for actor-level answers.

### Real-time path

**~1–2 days · GPU.**

Measured on the RTX 3050 Laptop: fused perception + tracking 61.9 ms/frame, of
which ~27 ms is BoT-SORT association + sparse-optical-flow CMC **on CPU**
(`results/latency_gpu_summary.md`); end to end ≈ 12 fps. Routes to camera rate:
TensorRT export of the detector, `yolov8n-seg` or a lower `imgsz`, and running
CMC on a downscaled frame. Needed for the paper's "live on-vehicle deployment".

**Risk:** every one of these changes detections or association, so features must
be re-extracted and the head retrained before any accuracy claim carries over.

### ReID appearance arm

**~1 day.**

Deliberately deferred from v1 ([`CHALLENGES.md`](CHALLENGES.md) §7). The tracker
is BoT-SORT with `with_reid: False`, so no appearance embedding exists.
Ultralytics ships `yolo26{n,s,m,l,x}-reid.onnx` and wires it via
`build_encoder(args.with_reid, ...)`.

**Run it as a separate ablation arm, not a replacement**, so the appearance
contribution is measurable rather than confounded with the association change
that enabling ReID causes.

**Risks:** needs `onnxruntime` (not currently a dependency); changes association,
so features become incomparable with the cached pass and the eval split must be
re-extracted; and conceptually, a ReID embedding is trained for *identity
invariance*, the same vector for the same car whether or not it is about to
crash, so it is a weak prior for risk. Expect a small effect.

### Strict old-vs-new comparison on `eval_v2`

**~1 h GPU + minutes CPU, and a decision.**

The threshold engine has only been scored on the superseded split, on full clips at
30 Hz, so the paper states "No comparable baseline yet". Re-run it on the same
13 s windows as the learned head. On `train_val_v2` this is free of any read
budget; on `eval_v2` it is scoring a frozen, untuned baseline (nothing selected),
but decide in writing whether that counts against the one-read rule first.

---

## Tier 3 — completeness and rigour

### Longer feature window

**GPU re-extraction ~10 h at the measured 23.5 s/clip.**

Features cover 13 s at 10 Hz. `--window-s 20 --tail-s 1` was the original idea
for lead time. **Deprioritised:** 13 s already covers ~11 s of run-up before a
~1.6 s annotated window, and the loss gives no signal before `alert_t`, so a
longer window changes what the model can see, not what it is asked to predict.
Only worth trying *after* a pre-alert objective (Tier 1) shows the model can use
earlier context.

**Risk:** a full re-extraction, and nothing measured on 13 s windows is comparable
afterwards unless every split is re-extracted.

### Strict static-scene ego flow

**~half day.**

`ego.py` masks detections already, which is the important part. But BoT-SORT's own
GMC uses `sparseOptFlow`, which ignores its `detections` argument entirely, so the
tracker's internal motion compensation is whole-frame flow made robust only by
RANSAC. Switching `gmc_method` to `orb`/`sift` routes through `apply_features`,
which does mask detections.

**Risk:** ORB/SIFT is meaningfully slower than sparse LK (CMC is already ~27 ms of
the 62 ms GPU frame) and changes association, so it needs re-extraction. Given our
own estimator already reports 0% failure, the expected gain is small.

### Horizon-blended or calibrated learned forecaster

**~1 day.**

Kalman wins at 1–2 s, the LSTM at 4 s (`eval/forecast_table.md`). A blend (Kalman
up to ~2 s, LSTM beyond) or an LSTM whose log-variance head is calibrated the way
Kalman's std was (`RMSE(k)/std(k) ≈ 1`) is the only version of the learned
forecaster worth re-testing end to end.

**Risk:** the end-to-end test already showed no coverage gain, so the upside is
bounded; this is for completeness, not the headline.

### Frame-rate ablation (10 / 15 / 30 Hz)

**GPU, ~10 h for all three.**

10 Hz was chosen from a 10-clip association probe, not from downstream
performance. An ablation training the head on each rate would show whether the
−15% forecastable-track yield actually costs accuracy. Mostly of methodological
interest; the rate-consistency constraint ([`CHALLENGES.md`](CHALLENGES.md) §11)
means each arm needs its own matched eval extraction.

### Transformer variant of the temporal head

**~1 day.**

A small causal transformer instead of the GRU, matching the Phase III forecaster's
architecture family. At ~130 timesteps and ~1,160 training clips, and with larger
GRUs showing no gain, the head is unlikely to be the bottleneck; mean AP 0.646
more plausibly reflects feature quality and the objective. Treat it as a
comparison point, not an expected win.

### Cross-dataset evaluation

**Days · GPU.**

Every result is on Nexar. Running the frozen pipeline on DAD, CCD or DoTA would
test whether the geometric features transfer across cameras and annotation
conventions. Addresses the paper's "one dataset" limitation.

**Risk:** each dataset annotates "alert" differently (or not at all), so the
useful-warning metric may not transfer; AP at time-to-accident cutoffs is the
portable measure.

### BEV per-clip calibration and ego speed

**~1 day, any machine.**

The pipeline forecasts in image space (`forecast_space: img`) because the ground
plane is uncalibrated per clip, so `std_bev` holds pixel standard deviations, and
`verify_bev.py` marks most clips `SUSPECT`. Calibrating a homography per demo
clip would make distances metric and collision radii physical. Estimating ego
speed would also let the dashboard simulator move the ego vehicle instead of
pinning it at the origin.

Only worth doing for the handful of final demo clips; it improves
interpretability rather than any headline metric.

### Dashboard: full `eval_v2` coverage and a real simulator

**~half day for coverage; days for a simulator.**

Only 12 of the 120 `eval_v2` clips have stored track data, so Drill-down can't
open most held-out clips. Storing tracks for all 120 is a pipeline pass, not new
code. Separately, `new-architecture.md` notes a CARLA / MetaDrive-scale 3D scene
as a possible upgrade to the Canvas-2D simulator; it would be presentation only.

### User study

**Weeks; a scoping decision, not a script.**

[`../design/metrics.md`](../design/metrics.md) §8 covers this. The explanation
quality claim currently has no human evaluation behind it. Either run a real study
or state the absence as an explicit limitation; the latter is entirely defensible
for a thesis of this scope, and the paper already does.

### LLM explanation enrichment at scale

**Free tier, ~2 h.**

`a3ps/explain/llm_client.py` exists with a Groq client and a permissive-prompt
hallucination test, but has only been eyeballed on a few dev clips. Running it
across the demo clips and checking for hallucination against the deterministic
templates would let you say something evidence-based about the explanation layer.

**Risk:** the whole point is checking for hallucination. Treat the LLM output as
untrusted and always keep the deterministic template alongside it, which the
schema already supports (`explanation_template` and `explanation_llm` are separate
fields).

### Continuous integration

**~1 h.**

194 tests run in ~2 minutes on CPU. A GitHub Actions workflow running `pytest` on
push would keep the bit-for-bit legacy-metric regression and the leakage guards
enforced without anyone remembering to run them.
