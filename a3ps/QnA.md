# A3PS — Q&A: every tricky part of this implementation

Question-and-answer format, grouped by area. If a reviewer, teammate, or
future-you asks "wait, why does it do X?" — the answer should be here.

---

## Perception & tracking

**Q: Why is there no separate perception stage? The README diagram shows one.**
A: Conceptually there are two stages; in code they are ONE fused call.
YOLOv8-Seg run in *tracking mode* (`model.track(persist=True)`) returns
detections, segmentation masks, and BoT-SORT track IDs from a single
inference pass. Running a separate `Segmenter` would double inference cost
for identical outputs. This is also why per-stage latency reports
`perception: 0.0` — the combined cost is measured under `tracking`, with a
`per_stage_ms_note` in every meta.json explaining it.

**Q: Why not pass `classes=` into `model.track()` to filter to road actors?**
A: It breaks BoT-SORT. Filtering detections *inside* the tracker call
disrupts association: on dev01, only 10/540 frames got track IDs with the
filter vs ~274-327 without it. So the tracker sees ALL classes (stable
association), and the target-class filter is applied to the *output* list
afterwards.

**Q: Why is `centroid_img` the bbox bottom-center instead of its center?**
A: It's the *foot point* — the actor's ground-contact point. A ground-plane
homography can only meaningfully project points that are actually on the
road plane; a bbox center is meters above the ground and would project to
the wrong BEV location.

**Q: Why does the trajectory buffer decimate 30 fps to 5 Hz?**
A: Forecasting operates at `predict_hz=5` (0.2 s steps, 4 s = 20 steps).
Feeding 30 fps history would mostly add redundant near-duplicate points;
binning to 0.2 s buckets (first sample per bucket) gives a clean 10-point /
2-second history matched to the forecast step size — and it's the exact same
resampling mining uses, so train and inference distributions match.

**Q: Why was `track_buffer` raised 30 → 90 frames?**
A: A diagnosed night-clip had long detection dropouts; 90 frames (3 s at
30 fps) lets BoT-SORT re-associate a briefly lost actor with its old ID
instead of minting a new one. Safe for daytime clips too.

## Forecasting

**Q: Why does Kalman-CV report `sqrt(diag(HPHᵀ+R))` instead of the raw state covariance std?**
A: Raw state covariance starts near zero right after fitting, so the
std-growth ratio over the horizon was unstable (3.6x–7.9x depending on the
clip). Adding measurement noise R keeps step-0 std off the floor, giving a
stable ~2.1x growth over 4 s — an uncertainty signal downstream risk math
can actually rely on. `process_var` was likewise tuned (4.0 → 0.1) by
ablation.

**Q: What happens when a track has almost no history?**
A: Under `MIN_HISTORY=4` points, no filter is run at all — `_straight_line()`
extrapolates from the last two points and reports a fixed large std
(`FALLBACK_STD=40 px`) for every step: an honest "I barely know" signal
rather than a fake-confident filter output.

**Q: `Prediction.std_bev` holds PIXEL values — isn't that a bug?**
A: Deliberate naming-stability decision. While `forecast_space: img`, all
geometry (means, stds, corridor, radii) is in pixels; the field name is kept
so that when per-clip BEV calibration lands (Week 4), nothing downstream has
to rename anything — the units change, the schema doesn't.

**Q: Is the LSTM better than Kalman or not? I've seen both claims.**
A: Both are true — on *different validation sets*, which is the trap. On the
early 172-window split they tied (51.24 vs 51.64 px ADE@4s, Kalman ahead).
After ~100 more negatives were mined and the model retrained, on the
289-window split the **LSTM wins at 4 s** (56.33 vs 61.27 ADE, 99.74 vs
122.55 FDE) while Kalman stays marginally ahead at 1-2 s. ADE is an absolute
error on whatever windows are in the val split — enlarging the pool changed
the split for BOTH models (Kalman's own number rose 51.24 → 61.27). Rule:
**never compare ADE across runs with different mined pools; only compare the
two models within one run.** Old table archived as
`eval/forecast_table_PREVIOUS.md`.

**Q: Why does the LSTM only win at the long horizon?**
A: Over 1-2 s, real motion is nearly constant-velocity — which Kalman-CV
computes *optimally in closed form*, so a learned model can at best tie it.
The LSTM's edge appears where trajectories curve/decelerate (4 s), which a
constant-velocity model cannot represent. That is also why the first
(under-trained, car-dominated) LSTM just re-learned constant velocity.

**Q: Which forecaster does the live pipeline/dashboard actually use?**
A: **Kalman-CV**, by config default (`forecaster: kalman_cv`). The LSTM is
selectable (`forecaster: seq2seq` + `seq2seq_weights`) — added for the
ablation study — but retraining the LSTM never changed the dashboard,
because the dashboard path never loaded it. If braking behavior changes,
look at config/thresholds or re-picked dev splits, not the LSTM.

## Risk & decision

**Q: How exactly does a VIRTUAL_BRAKE get decided?**
A: Chain per track, per frame: Kalman forecast (20 Gaussian steps) → per
step, weighted fraction of 5 sigma points inside the actor-radius-dilated
ego corridor → 3-step moving average, take the max (= `max_prob`) and the
first step above 0.5 (= `ttc_s`) → EMA across frames (α=0.4) →
`DecisionEngine`: prob ≥ (threshold − 0.15) for 3 consecutive frames fires
ALERT; then prob ≥ threshold for 3 consecutive frames fires VIRTUAL_BRAKE.
One event per transition; BRAKE is terminal until prob < 0.4 AND 2 s pass.

**Q: Why dilate the corridor by an actor radius instead of testing the centroid?**
A: A car collides when its *body* enters the path, not its center point.
Dilation = Minkowski sum of the corridor polygon with a disk of the actor's
class radius (person 8 px … truck 28 px in image space; metric values exist
for BEV). Implemented as signed-distance ≤ radius in pure Python — same
criterion as `cv2.pointPolygonTest(...) >= -radius` but dependency-free and
deterministic for tests.

**Q: Why sigma points instead of Monte-Carlo sampling or an analytic integral?**
A: 5 deterministic UKF points capture a diagonal 2D Gaussian's mass well
enough for a polygon-membership test, cost is fixed and tiny (5 tests/step),
and the result is reproducible (no RNG in the safety-critical path). An
analytic Gaussian-polygon integral is expensive and unnecessary at this
fidelity.

**Q: Why the EMA + 3-frame confirmation? Doesn't that delay braking?**
A: Yes — deliberately. A single-frame forecast glitch (mask flicker, ID
jitter) must not fire a brake; the EMA damps spikes and the 3-frame streak
requires persistence. The cost is ~0.6 s of reaction lag at 5 Hz, which is
exactly the trade the ablation switches (`ema_smoothing`,
`frames_to_confirm`) and the threshold sweep exist to quantify. If eval
shows A3PS reacting too late, these are the knobs.

**Q: Are crosswalk/intersection context flags detected from video?**
A: **No — and the code says so honestly.** They come from an optional
hand-written `dashboard/clips/<id>/context.json` span file; only
`dense_traffic` is derived from perception (≥8 tracks in frame). Detecting
them from imagery is listed future work. The threshold floor (0.45) caps how
far stacked flags can lower the bar.

**Q: Why did dev14 brake AFTER the collision at one point?**
A: Not an LSTM issue (the pipeline uses Kalman) — the smoothed probability
only crossed 0.75 late: the actor entered the corridor / became predictable
just before impact, and confirmation+EMA added lag on top. Fixes are config:
lower `base_threshold`, raise `horizon_s`, lower `frames_to_confirm`. Also
note: after adding negatives, `prepare_nexar.py` re-picks dev clips, so
"dev14" may not be the same video it was before.

## Data, mining & training

**Q: Does re-running mining redo everything?**
A: No — resume is keyed on shard existence: a clip is skipped iff
`data/trajectories/<clip_id>.npz` exists. `stats.json` counters are
*self-healed* every checkpoint by recounting shards actually on disk, so an
interrupted run can't leave lying stats. `--force` re-mines. One quirk: a
clip that produced 0 windows writes no shard, so it re-tracks every run —
wasteful but harmless.

**Q: Why keep 20% of near-static windows instead of dropping all of them?**
A: The forecaster must learn "stationary stays stationary." Dropping all
static windows would bias the model to always predict motion.

**Q: How does the ID-switch filter work in mining?**
A: A window is dropped if any consecutive step exceeds 3x the window's
median step — a tracking ID jump teleports the trajectory, and one such
window would teach the model impossible dynamics. (Degenerate near-static
windows use an absolute 5-unit jump test instead, since their median ~0.)

**Q: Why is each window normalized (origin + rotate-to-+y)?**
A: So the model learns motion *patterns*, not absolute screen positions —
every training sample says "history arrives at the origin heading up; where
does it go next?" The inverse transform (origin_x, origin_y, theta) is
stored per window so predictions can be mapped back to absolute coordinates.

**Q: Why did adding 100 negative clips change which clips are dev01–dev15?**
A: `prepare_nexar.py` defines dev as the 15 *shortest* clips — a
data-dependent choice. New negatives can displace old ones. The
`data/dev_clips/README.md` mapping table is regenerated each run; trust it,
not memory.

**Q: The eval script said "No processed clips found" but dashboard/clips is full — why?**
A: Name mismatch: `index.csv` refers to original clip IDs (`838`, `1372`…),
your dashboard folders are `devNN`. Fix: directory junctions
(`cmd /c mklink /J <id> devNN`) using the README mapping — the same data
becomes reachable under both names, dashboard untouched.

## Evaluation

**Q: What's the reactive-proximity baseline and where does it come from?**
A: A naive no-forecasting ADAS: it "brakes" the first frame ANY actor is
physically close to the ego corridor (BEV < 2.0 m, else image < 8% of frame
height). It is an exact Python port of the dashboard's own reactive-ADAS
marker (`computeReactiveMarkers()` in `app.js`) — same thresholds, same
distance test — so the paper's baseline and the demo's ▽ marker can never
disagree. It isolates the value of the forecasting layer.

**Q: Why can't I just compare the two mTTA numbers in the main table?**
A: Because each method's mTTA averages over its *own* true-positive set —
different sizes, different clips. A method with worse recall and a 60%
false-alarm rate can post a *better-looking* mTTA simply by which easy clips
it happened to catch (this actually occurred: the dev run showed reactive
mTTA 7.13 s vs A3PS 3.50 s at wildly different recall/FAR). The
**matched-subset mTTA** — both methods restricted to the positives BOTH
anticipate — is the only apples-to-apples timing comparison, and the only
number allowed to back a "warns N s earlier" claim. And don't assume its
sign: on the 4-clip dev overlap A3PS was ~2.9 s *later* — a real,
tunable finding, not a bug.

**Q: How can the threshold sweep test 11 thresholds without 11 GPU runs?**
A: Because the threshold is applied to *already-computed* probabilities.
Each clip's per-frame max collision_prob is cached once
(`eval/cache/<clip>.csv`, extracted from the ClipResult already loaded for
scoring); each candidate threshold is then just an in-memory comparison
against those numbers. The sweep deliberately ignores the
frames-to-confirm/cooldown debounce — it measures the pure probability
decision boundary with everything else held fixed.

**Q: The ablation table has 6 rows but you said only 4 runs — how?**
A: "Dynamic threshold" and "EMA on" are configurationally identical to the
Kalman-CV baseline row (both switches default true), so they *reuse* its
computed numbers — marked `(= Kalman-CV)` in the output for transparency.
Re-processing 120 clips twice more for identical configs would be pure
waste.

**Q: Can `run_ablations.py` corrupt my config if it crashes?**
A: No — three layers of protection: the original text is restored after
*every* variant (in a `finally`), a `.ablation_backup` file exists on disk
during the sweep, and a final `finally` restores again at the end.

**Q: Why does the per-stage latency table show perception = 0.0 ms?**
A: See the first Q&A — fused perception+tracking call. Report it in the
paper as "perception and tracking are fused into a single detector-tracker
pass; combined cost reported under Tracking." Never present it as
"detection is free." Also: latency is hardware-bound — CPU numbers are
~20-40x slower than GPU; `collect_paper_stats.py` records
`torch.cuda.is_available()` so you can state which machine produced Table IV.

**Q: There used to be NO timing in meta.json at all — why?**
A: A genuine ordering bug: the timing dict was attached to `result.meta`
*after* `json.dump` had already written meta.json, so only the in-memory
object ever had it. Fixed by computing `per_stage_ms` before the write; a
unit test pins the aggregation math.

## Explanation / XAI

**Q: Why have deterministic templates when there's an LLM?**
A: The template is the *faithfulness guarantee*: it is grammar-built from
the exact numbers the risk engine computed (P, threshold, TTC, motion
inferred from BEV velocity) — it cannot hallucinate. The LLM narrative is
optional enrichment, run offline after the pipeline, never in the live
loop; if the API is unavailable the XAI requirement is still met.

**Q: How is the LLM prevented from hallucinating?**
A: The system prompt is facts-only constrained ("Using ONLY the structured
facts provided and what is visible in the image… Do not invent objects or
numbers"), and the input is a structured-facts JSON + the actual keyframe.

**Q: Isn't the "permissive prompt" hallucination example fabricated?**
A: It's *reproduced under a controlled relaxation*, and disclosed as such:
`llm_client_permissive_test.py` runs the same real event/keyframe under the
real prompt and under a deliberately unconstrained one, prints both, then
auto-restores the constrained narrative to disk so the relaxed output can't
leak into anything real. The paper states the "before" was produced under a
relaxed prompt for illustration — methodologically honest, and it
demonstrates exactly the failure mode the tightened prompt prevents.

## Architecture & engineering

**Q: Why hooks (`forecaster=`, `risk_engine=`, `explainer=`) instead of hard-wiring the phases?**
A: Each phase was built and tested in isolation weeks apart. Injectable
hooks meant the pipeline ran (tracking-only) before forecasting existed,
forecasting ran before risk existed, and the eval scripts can rebuild the
exact same hooks from config. It's also what makes the ablation switches a
config change instead of a code fork.

**Q: How does `make_qual_figure.py` guarantee its overlays match the dashboard?**
A: It doesn't re-implement drawing — it instantiates a bare Pipeline via
`Pipeline.__new__` (skipping `__init__`, so no YOLO/tracker load) and calls
the *actual* `_draw_frame`/`_draw_prediction` methods. Safe because those
two methods touch no other instance state; by construction the figure can't
drift from the real renderer.

**Q: Why does `eval_anticipation.py --run` use `render=False`?**
A: Batch scoring needs only events.json; skipping the VideoWriter and
raw.mp4 copy removes all encoding cost from a 120-clip run. Consequence:
those clips have no raw.mp4, so `make_qual_figure.py` and the LLM enrichment
(both need frames) require a clip processed by `run_pipeline.py` instead.

**Q: Why is everything resumable (mining, eval --run)?**
A: Long GPU jobs on a borrowed/shared laptop get interrupted. Both scripts
key resume on per-clip output-file existence, so re-running the same command
continues instead of restarting — and both self-report how many clips were
skipped.

**Q: One schema file for everything — what does that actually buy?**
A: `schema.py` is the contract between five pipeline phases, five eval
scripts, and a JavaScript dashboard. Because everything round-trips through
the same JSON shape (with optional BEV fields as `null`, never dropped), the
dashboard was built against synthetic data before the risk engine existed,
and switching to real data needed zero dashboard changes.

**Q: Why Groq instead of OpenAI/Anthropic for the LLM?**
A: Free tier covers the workload (a handful of events across demo clips),
it exposes an OpenAI-compatible API with a vision-capable Llama model, and
the project constraint was zero paid API usage.

**Q: The `.env` file / model weights / data aren't in git — where are they?**
A: Deliberately gitignored: secrets (`.env` with `GROQ_API_KEY`), bulk video
data (`data/nexar/videos/`), and large artifacts move by hand between the
CPU and GPU laptops. `GPU_HANDOFF.md` documents exactly what to copy each
direction. Code moves only via git.

**Q: How do I trust all this still works after a change?**
A: `python -m pytest -q` → **90 tests**, all CPU-only and data-free
(synthetic fixtures): schema round-trip, geometry/BEV, collision math,
decision state machine (incl. the new toggles), buffer decimation,
forecasters, mining filters + self-heal, eval metrics (incl. AP,
matched-subset, sweep cache/precision-recall), qual-figure frame picking,
per-stage latency aggregation, and LLM-client behavior (fake client, incl.
the prompt override). If those pass, the math and contracts are intact.
