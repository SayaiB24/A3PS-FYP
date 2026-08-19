# Future work — better-to-have improvements

Not required by anything committed. Ordered by expected value per unit effort, so
working top-down is a reasonable default. Each entry says what it would buy, what
it costs, and the specific risk that makes it non-trivial.

For what to do **next**, see [`NEXT_STEPS.md`](NEXT_STEPS.md); the kappa retrain
in [`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md) is the one genuinely prioritised item
and is not repeated here.

---

## Tier 1 — cheap, and likely to move a reported number

### Batched training forward pass

**~1–2 h · CPU · ~5× faster training.**

`--batch-size` does not batch the compute: `[model(c["X"]) for c in batch]` runs
one forward per clip, giving effective batch size 1 and ~1.85× parallelism on 8
cores. Pad each batch to a common `T` into a real `(B, T, D)` tensor, run one
forward, and mask padding inside `batch_anticipation_loss`.

**Risk:** a masking bug changes the objective silently instead of crashing.
Verify the batched loss equals the unbatched loss on a fixed seed before trusting
any result. Do this before any large sweep — the speedup compounds across runs.

### More model capacity

**~1 h · CPU.**

The head is 34,465 parameters (`--hidden 64 --layers 1`). Mean AP is 0.693 and is
the ceiling no operating point can exceed, so if the model is underfitting, this
is the cheapest way to raise it. Try `--hidden 128 --layers 2`, then `--hidden 256`.

**Risk:** 1,365 training clips is small; larger capacity may simply overfit
faster. Watch the gap between train loss and validation useful-warning — the
existing run already peaked at epoch 5, which suggests overfitting is close.
Keep `--patience` and judge by best epoch, not last.

### Model selection that respects the false-alarm constraint

**~30 min · CPU.**

Selection is currently `useful_warning_rate` alone, strictly greater. That
discarded epoch 8 (useful 0.833, FA 0.550) in favour of epoch 5 (useful 0.833, FA
0.583) — equal on the criterion, worse on the metric that decides publishability.
See [`CHALLENGES.md`](CHALLENGES.md) §15.

Change the criterion to useful-warning subject to `FA ≤ target`, or tie-break on
FA. Cheap, and it makes every subsequent run select a better checkpoint.

### Kaggle submission for an external AP

**~2 h · GPU for extraction.**

The 1,344 unlabelled `test/` clips have no ground truth locally, but the
competition can score them. That yields an **externally validated** AP with no
possibility of us having leaked or mis-scored anything — valuable for a thesis,
since every other number here is self-reported.

Needs: extract features for the test clips (`--split` will need assigning, as
they are currently unindexed), run the head, emit predictions in the competition's
submission format. Check the competition is still accepting submissions first.

---

## Tier 2 — moderate cost, addresses a known limitation

### Longer feature window

**GPU re-extraction ~4 h.**

Features cover 13 s at 10 Hz (~130 timesteps). Mean lead is 1.58 s against a 2–6 s
target; more run-up context may help the model commit earlier. Try
`--window-s 20 --tail-s 1`.

**Risk:** a full re-extraction, and the result is not comparable to anything
measured on 13 s windows unless you re-run the eval split too. Do the kappa sweep
first — it targets the same weakness for a fraction of the cost.

### Per-actor temporal head

**~2 days.**

The head pools features across actors into one frame-level vector, so it cannot
say *which* actor is dangerous. The dashboard's explanation currently names an
actor pulled from the old system's cached tracks — presentation, not a model
output, and the code says so.

A per-actor head (run the GRU per track, aggregate to a frame-level risk) would
make attribution a genuine model output and make explanations honest. It also
enables per-actor risk colouring in the dashboard overlay.

**Risk:** a substantial change to the feature pipeline; `extract.py` already emits
per-actor rows before pooling, so the data exists — but training and evaluation
both need reworking, and variable actor counts per frame need handling.

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
invariance* — it aims to be the same vector for the same car whether or not it is
about to crash — so it is a weak prior for risk. Expect a small effect.

### Strict static-scene ego flow

**~half day.**

`ego.py` masks detections already, which is the important part. But BoT-SORT's own
GMC uses `sparseOptFlow`, which ignores its `detections` argument entirely — so the
tracker's internal motion compensation is whole-frame flow made robust only by
RANSAC. Switching `gmc_method` to `orb`/`sift` routes through `apply_features`,
which does mask detections.

**Risk:** ORB/SIFT is meaningfully slower than sparse LK and changes association,
so it needs re-extraction. Given our own estimator already reports 0% failure,
the expected gain is small.

---

## Tier 3 — completeness and rigour

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
architecture family. Would let the head attend over the full 13 s window rather
than carrying state. At 130 timesteps and 1,365 clips the GRU is unlikely to be
the bottleneck — mean AP 0.693 more plausibly reflects feature quality — so treat
this as a comparison point rather than an expected win.

### BEV per-clip calibration

**~1 day, any machine.**

The pipeline forecasts in image space (`forecast_space: img`) because the ground
plane is uncalibrated per clip, so `std_bev` currently holds pixel standard
deviations. Calibrating a homography per demo clip would make distances metric and
collision radii physical rather than pixel stand-ins.

Only worth doing for the handful of final demo clips, and it improves
interpretability rather than any headline metric.

### User study

**Weeks; a scoping decision, not a script.**

[`../design/metrics.md`](../design/metrics.md) §8 covers this. The explanation
quality claim currently has no human evaluation behind it. Either run a real study
or state the absence as an explicit limitation — the latter is entirely
defensible for a thesis of this scope, but it should be stated rather than left
implicit.

### LLM explanation enrichment at scale

**Free tier, ~2 h.**

`a3ps/explain/llm_client.py` exists with a Groq client and a permissive-prompt
hallucination test, but has only been eyeballed on a few dev clips. Running it
across the demo clips and checking for hallucination against the deterministic
templates would let you say something evidence-based about the explanation layer.

**Risk:** the whole point is checking for hallucination — treat the LLM output as
untrusted and always keep the deterministic template alongside it, which the
schema already supports (`explanation_template` and `explanation_llm` are separate
fields).
