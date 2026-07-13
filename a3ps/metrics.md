# Metrics reference — which script, which parameters, why, and what you need

One section per metric. Each section gives the exact command(s), what every
parameter means, **why** that metric exists (what question it answers), and
what has to be true on disk before you run it. This is the single place to
look up "how do I (re)generate metric X" — the individual scripts' own
docstrings have more implementation detail if you need it.

---

## Quick reference

| # | Metric | Script | Needs | Output |
|---|---|---|---|---|
| 1 | Forecast accuracy (ADE/FDE, Kalman-CV vs Seq2Seq-LSTM) | `scripts/eval_forecast.py` | mined shards (`data/trajectories`), trained weights | `eval/forecast_table.md` |
| 2 | Anticipation (detection rate / false-alarm rate / mTTA, A3PS vs reactive baseline) | `scripts/eval_anticipation.py` | eval-split clips (GPU once) | `eval/anticipation.md`, `eval/anticipation_per_clip.csv` |
| 3 | Threshold sweep / PR curve | `scripts/eval_anticipation.py --sweep` | clips already processed by #2 | `eval/pr_curve_data.csv`, `eval/pr_curve.png` |
| 4 | Ablation study (Table II) | `scripts/run_ablations.py` | same as #2 (re-run per variant) | `eval/ablation_table.md` |
| 5 | Qualitative figure (before/ALERT/BRAKE/after) | `scripts/make_qual_figure.py` | one clip processed **with rendering** (`run_pipeline.py`) | `eval/figures/<clip>_qual.png` |
| 6 | Per-stage latency (Table IV) | `scripts/run_pipeline.py` (instrumentation is automatic) | any clip run through the pipeline | `meta.json`'s `per_stage_ms` field |
| 7 | Hallucination before/after example | `python -m a3ps.explain.llm_client_permissive_test` | `GROQ_API_KEY`, one enriched clip | printed side-by-side narratives |
| 8 | User study (C3) | *(no script — human participants)* | a scoping decision, see §8 | either real data or an explicit limitation statement |

---

## 1. Forecasting accuracy — ADE/FDE (Kalman-CV vs Seq2Seq-LSTM)

**File:** `scripts/eval_forecast.py`

```powershell
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_v1.pt
```

**Parameters:**
- `--shards` (default `data/trajectories`) — the mined trajectory windows (from `scripts/mine_trajectories.py`) to draw the held-out validation split from.
- `--weights` — path to the trained Seq2Seq checkpoint. **Must be passed explicitly** — the script's own default (`models/seq2seq_v1.pt`) doesn't match where the training notebook actually saves the file (`notebooks/models/seq2seq_v1.pt`); without this flag the Seq2Seq row is silently skipped and only Kalman-CV prints.
- `--out` (default `eval/forecast_table.md`) — where the table is written.

**Comparing two model checkpoints** (e.g. previous vs. newly retrained): run it twice with different `--weights`/`--out`:
```powershell
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_new.pt --out eval\forecast_table.md
python scripts\eval_forecast.py --weights notebooks\models\seq2seq_prev.pt --out eval\forecast_table_oldmodel.md
```

**What it computes:** ADE (Average Displacement Error) and FDE (Final Displacement Error) at 1 s / 2 s / 4 s horizons, on the same 90/10 (seed 42) held-out validation split of mined windows, for both forecasters.

**Why this metric exists:** it's the *only* way to judge raw positional-prediction quality in isolation, independent of the risk/decision layer's thresholds. A forecaster can only be as good as its trajectory predictions — this tells you whether the learned model (Seq2Seq-LSTM) is actually better at predicting *where an actor will be* than the closed-form physics baseline (Kalman-CV), before any of that feeds into collision-probability or braking decisions.

**⚠️ Only compare ADE/FDE numbers from runs against the *same* `--shards`.** Growing the mined pool changes which windows fall in the validation split (harder/easier), so absolute numbers shift for *both* models — always compare Kalman vs. Seq2Seq within one run, never across two runs with different data sizes.

**Needs on disk:** `data/trajectories/*.npz` (run `mine_trajectories.py` first) and a trained `.pt` weights file (run `notebooks/train_forecaster.ipynb` first). Without weights, only the Kalman-CV row prints.

---

## 2. Anticipation performance — detection rate / false-alarm rate / mTTA

**File:** `scripts/eval_anticipation.py`

```powershell
# sanity check on dev (instant, no GPU -- clips already processed by run_pipeline.py)
python scripts\eval_anticipation.py --index data\nexar\index.csv --split dev --clips-dir dashboard\clips

# the real run (GPU-heavy, resumable)
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --run
```

**Parameters:**
- `--index` — the master clip table (`data/nexar/index.csv`), which supplies each clip's `split`, `label` (0=negative/1=positive), and `event_time_s`.
- `--split` — `dev` (15 clips, sanity-check only — too few positives to draw conclusions from), `eval` (120 clips, the real report numbers), or `demo`.
- `--run` — process any clip whose `events.json` is missing, through the full pipeline **with rendering disabled** (`render=False`, no `annotated.mp4`/`raw.mp4` — just the JSON needed to score). This is the GPU-heavy step. **Resumable**: clips with an existing `events.json` are skipped, so an interrupted run can just be re-launched.
- `--clips-dir` (default `eval/anticipation`) — where processed `<clip_id>/events.json` live.
- `--config` (default `configs/default.yaml`) — pipeline config used when `--run` processes a clip (thresholds, forecaster choice, etc.).
- `--alert-types` (default `ALERT,VIRTUAL_BRAKE`) — which event types count as "an alert" for scoring.
- `--out` (default `eval/anticipation.md`), `--per-clip-csv` (default `eval/anticipation_per_clip.csv`).

**What it computes, for *two* methods side by side:**
- **A3PS (proactive)** — the pipeline's actual forecast + risk-decision `ALERT`/`VIRTUAL_BRAKE` events.
- **Reactive-proximity baseline** — a naive ADAS with *no forecasting*: it "brakes" the first frame any actor is physically close to the ego corridor (identical thresholds to the dashboard's own reactive-ADAS marker: BEV < 2.0 m, else image < 8% of frame height).

For both: **detection rate** (fraction of positives alerted *before* the annotated event), **false-alarm rate** (fraction of negatives that raised any alert), **mTTA** (mean seconds between the alert and the event, over correctly-anticipated positives).

**Why this metric exists:** it's the project's core anticipation claim — does the system warn about a collision *before* it happens, and how early, compared to a system with no predictive component? Detection rate and false-alarm rate alone don't capture *earliness*; mTTA does.

**⚠️ Matched-subset mTTA is the number to cite for "A3PS warns N s earlier."** The raw per-method mTTA row is confounded: each method's mTTA averages over its *own*, differently-sized true-positive set, so a method with lower recall/higher false-alarm rate can show a misleadingly early mTTA just by which (fewer/easier) clips it happens to count. `eval/anticipation.md`'s `## Matched-subset mTTA` section restricts both methods to only the positives **both** correctly anticipate, so the timing comparison is apples-to-apples. **Do not assume the sign is positive** — check the actual number every time; a negative matched-subset gain (A3PS *later* than the naive baseline) is a real, actionable finding, not a bug (see §9.1/§10.2 of `GPU_HANDOFF.md` for the exact tuning knobs to fix it: lower `base_threshold`, raise `horizon_s`, lower `frames_to_confirm`).

**Needs on disk:** `data/nexar/index.csv` with the target split populated (`prepare_nexar.py`); for `--run`, the raw eval-split videos in `data/nexar/videos/`.

---

## 3. Threshold sweep / precision-recall curve (Step 7.1)

**File:** `scripts/eval_anticipation.py --sweep` (same script, additional flag)

```powershell
python scripts\eval_anticipation.py --index data\nexar\index.csv --split eval --sweep
```
(Add `--run` too the first time, if the eval split isn't processed yet — `--sweep` piggybacks on the same clip-processing loop as metric #2.)

**Parameters (in addition to #2's):**
- `--sweep` — turns on caching + the threshold sweep.
- `--cache-dir` (default `eval/cache`) — per-clip, per-frame max-collision-probability cache, one small CSV per clip (`t, max_prob`). Built **once** per clip (skipped on subsequent `--sweep` runs unless the clip's `events.json` changed) directly from the `ClipResult` object already loaded for metric #2's scoring — no extra parsing, no GPU work.
- `--force-cache` — rebuild a clip's cache even if it already exists (use after re-running the pipeline with different config, e.g. after an ablation).
- `--pr-csv` (default `eval/pr_curve_data.csv`), `--pr-png` (default `eval/pr_curve.png`).

**What it computes:** precision and recall at every `base_threshold` in **0.40 → 0.90, step 0.05** (11 points), computed **entirely from the cache, in memory** — no re-running perception/tracking/forecasting. For each candidate threshold *T*: a positive clip counts as a hit if any cached frame at or before its `event_time_s` has `max_prob >= T`; a negative clip counts as a false alarm if *any* cached frame anywhere has `max_prob >= T`. This intentionally ignores the decision engine's `frames_to_confirm`/cooldown debounce — the sweep isolates the pure probability decision boundary, holding everything else in the pipeline fixed.

**Why this metric exists:** `base_threshold = 0.75` is one specific operating point. A PR sweep shows the full precision/recall trade-off curve around it — is 0.75 near the "knee" of the curve, or could a different threshold buy meaningfully more recall for an acceptable precision cost (or vice versa)? This is the standard way to justify *why* a specific threshold was chosen, rather than asserting it.

**Why the caching step matters:** re-scoring at 11 different thresholds the "obvious" way would mean re-running the full GPU pipeline 11 times (11x the cost of metric #2). Caching each frame's max probability once means the sweep is instead 11 cheap in-memory comparisons over already-extracted numbers — the entire sweep costs nothing beyond the one pipeline pass already paid for by metric #2.

**Needs on disk:** clips already processed by metric #2 (`events.json` in `--clips-dir`). No extra GPU time beyond that.

---

## 4. Ablation study — Table II (forecaster / dynamic threshold / EMA smoothing)

**File:** `scripts/run_ablations.py`

```powershell
python scripts\run_ablations.py --index data\nexar\index.csv --split eval
```

**Parameters:**
- `--index`, `--split` — same meaning as metric #2.
- `--config` (default `configs/default.yaml`) — **temporarily overridden** once per variant, then restored immediately after (belt-and-suspenders: restored after every variant, plus a `.ablation_backup` file and a `finally` block, so a crash mid-sweep can never leave the real config mutated).
- `--clips-root` (default `eval/ablation`) — each variant's processed clips + per-variant report/CSV go under `eval/ablation/<variant_slug>/`.
- `--out` (default `eval/ablation_table.md`).

**The exact Table II row order** (hardcoded in `VARIANTS` at the top of the script — edit there if the paper draft's table changes):

| Row | Config override | Actually re-run? |
|---|---|---|
| Kalman-CV | *(none — the baseline)* | ✅ yes |
| Seq2Seq-LSTM | `forecaster: seq2seq` | ✅ yes |
| Static threshold | `dynamic_threshold: false` | ✅ yes |
| Dynamic threshold | *(identical to Kalman-CV row)* | ❌ reused |
| EMA off | `ema_smoothing: false` | ✅ yes |
| EMA on | *(identical to Kalman-CV row)* | ❌ reused |

"Dynamic threshold" and "EMA on" are configurationally identical to the baseline `Kalman-CV` row (both switches default to `true`), so those two rows **reuse** the baseline's already-computed numbers instead of burning GPU time re-processing all 120 clips a second/third time — only **4 of the 6 rows** trigger an actual pipeline run. Reused rows are marked `EMA on (= Kalman-CV)` in the output table for transparency.

**The three ablation switches** (in `configs/default.yaml`, read by `a3ps/risk/decision.py` and `scripts/run_pipeline.py`):
- `forecaster: kalman_cv | seq2seq` — which model predicts future trajectories.
- `dynamic_threshold: true | false` — whether the danger threshold is lowered under adverse context (crosswalk/intersection/dense traffic); `false` = a flat `base_threshold` always.
- `ema_smoothing: true | false` — whether each track's collision probability is smoothed across frames (EMA, α=0.4) before being compared to the threshold; `false` = each frame's raw probability is used directly.

**What it computes:** each variant's own A3PS detection rate / false-alarm rate / mTTA (not compared against the reactive baseline — that's metric #2; this is purely "how much does removing/changing component X hurt or help A3PS's own numbers").

**Why this metric exists:** demonstrates which architectural choices actually matter. Without it, "we use a dynamic threshold and EMA smoothing" is an unverified design choice; the ablation table quantifies what each one buys (or costs) in detection/false-alarm/timing terms — the standard way to justify each component's presence in the final pipeline.

**Needs on disk:** same as metric #2. **Expect roughly 4x the GPU time of one `eval_anticipation.py --run`**, since it processes all 120 eval clips 4 separate times (once per non-reused variant).

---

## 5. Qualitative figure — before/ALERT/VIRTUAL_BRAKE/after

**File:** `scripts/make_qual_figure.py`

```powershell
python scripts\make_qual_figure.py --clip dashboard\clips\dev14
python scripts\make_qual_figure.py --clip dashboard\clips\dev14 --out eval\figures\dev14_qual.pdf
```

**Parameters:**
- `--clip` — a clip directory that has **both** `raw.mp4` and `events.json` — i.e. processed by `scripts/run_pipeline.py` (rendering **on**), not `eval_anticipation.py` (rendering **off**, no `raw.mp4`).
- `--actor-id` — pick a specific actor's incident if the clip has more than one `VIRTUAL_BRAKE` (default: the earliest one in the clip).
- `--out` (default `eval/figures/<clip_id>_qual.png`) — `.png` or `.pdf`, chosen by extension.
- `--dpi` (default `300`) — print-quality resolution.

**What it does:** finds one `ALERT` → `VIRTUAL_BRAKE` incident (matched by `actor_id`, since `VIRTUAL_BRAKE` only ever fires after that same actor's own `ALERT` per the decision-engine state machine), extracts exactly four real frames from `raw.mp4` via OpenCV — one frame just before the `SAFE→ALERT` transition, the `ALERT` frame itself, the `VIRTUAL_BRAKE` frame, and one frame just after it — draws each with the **exact same overlays the dashboard uses** (risk-tinted mask, predicted path, ego corridor, the red brake-flash border), and arranges them in a 2x2 grid, labeled with `risk_level` and timestamp, with the `VIRTUAL_BRAKE` panel captioned by that event's real `explanation_template` string.

**Why the overlays are guaranteed dashboard-identical (not a re-implementation):** the drawing code isn't copy-pasted — the script instantiates a bare `a3ps.pipeline.Pipeline` object (via `Pipeline.__new__`, skipping the heavy `__init__` that would otherwise load YOLO/the tracker) purely to call its real `_draw_frame`/`_draw_prediction` methods directly. Those two methods touch no other instance state, so this is safe and means the qualitative figure can never visually drift out of sync with what the live dashboard actually renders.

**Why this exists:** the report needs a concrete, honest "here is what the system actually saw and did" visual for one real incident — not a schematic — including the literal deterministic explanation text a user would have seen, for the XAI/explainability requirement.

**Needs on disk:** one positive clip already run through `scripts/run_pipeline.py` (with rendering), containing at least one `VIRTUAL_BRAKE` event. (Verified end-to-end against the real `dashboard/clips/dev14` clip.)

---

## 6. Per-stage latency — Table IV

**File:** `a3ps/pipeline.py` (instrumentation — no new script; runs automatically inside `scripts/run_pipeline.py`)

```powershell
python scripts\run_pipeline.py --video data\dev_clips\dev01.mp4 --out dashboard\clips\dev01\
python scripts\run_pipeline.py --video data\dev_clips\dev02.mp4 --out dashboard\clips\dev02\
python scripts\run_pipeline.py --video data\dev_clips\dev03.mp4 --out dashboard\clips\dev03\
```
(any 3+ clips — `scripts/collect_paper_stats.py` averages across every `dashboard/clips/*/meta.json` it finds, so more clips = a more representative mean/std)

**Nothing to configure** — every `run_pipeline.py` invocation now writes timing automatically. There used to be a real bug here worth knowing about (not just "the feature didn't exist yet"): the timing dict was computed and attached to `result.meta` **after** `meta.json` was already written to disk, so the on-disk file never actually contained timing data — only the in-memory object handed back to the caller did. That's fixed: timing is now computed and attached **before** the write.

**What's in `meta.json` now:**
```json
"per_stage_ms": {
  "perception": 0.0,
  "tracking": 1315.45,
  "forecasting": 0.0,
  "risk_decision": 0.05
},
"per_stage_ms_note": "perception (YOLO detection + segmentation) and tracking (BoT-SORT association) run as a SINGLE model.track() call in this architecture ..."
```

**⚠️ Why `perception` is always `0.0` — read before reporting Table IV.** This project's `Tracker` (see `a3ps/tracking/tracker.py`'s own docstring) deliberately runs YOLO in **tracking mode**, which returns detection + segmentation + track-association in **one single `model.track()` call** — "there is no need to also run a separate Segmenter — that would double the inference cost." So there is no independent "perception" call to time; the entire combined cost is measured under `tracking`. Reporting `perception: 0.0` is the honest choice (keeps the 4-key schema Table IV expects, keeps the end-to-end sum correct — no double-counting a cost under two labels). **State this explicitly in the paper** next to Table IV — e.g. "perception and tracking are fused into a single detector/tracker pass in this architecture; their combined cost is reported under Tracking" — rather than silently listing perception as ~0 ms, which would misleadingly imply detection is free.

**⚠️ CPU vs. GPU — state which one you measured on, explicitly.** These are wall-clock numbers, so they only mean anything alongside the hardware they were measured on. `scripts/collect_paper_stats.py`'s **Section 2 (Hardware & Environment)** already auto-detects `torch.cuda.is_available()` and the GPU name — **check that section every time before pasting Table IV into the paper.** If you develop/test on the CPU laptop, expect roughly **20-40x slower** per-frame numbers than the GPU laptop (per `GPU_HANDOFF.md` §3: ~800-1000 ms/frame CPU vs. ~20-40 ms/frame GPU) — CPU-measured latency must be labeled as such, never presented as if it reflects the deployed system's real-time performance. The headline Table IV numbers for the paper should come from the **GPU laptop**, with CUDA confirmed active (`python -c "import torch; print(torch.cuda.is_available())"` → `True`) before you run the 3+ clips above.

**Why this metric exists:** a claimed real-time safety system needs to show it can actually run fast enough to matter (i.e., faster than the reaction time it's trying to beat). Per-stage breakdown also shows *where* the cost is — useful for knowing what to optimize if latency is ever a problem.

**Needs on disk:** nothing beyond what `run_pipeline.py` already needs (a video + config). Verified end-to-end on this checkout: ran 3 real dev clips, confirmed `per_stage_ms` is now genuinely written to `meta.json` on disk, and confirmed `scripts/collect_paper_stats.py`'s Section 4 correctly aggregates it (mean/std/n across clips, plus end-to-end mean and implied FPS).

---

## 7. Hallucination before/after example (Section VII.B)

**File:** `a3ps/explain/llm_client_permissive_test.py` — **TEMPORARY, illustration-only.** Not used by the real pipeline; only exists to generate one deliberate comparison pair for the paper, then can be deleted.

```powershell
python -m a3ps.explain.llm_client_permissive_test dashboard\clips\ped_crossing_01
```

**What it needs:** `GROQ_API_KEY` set (`.env` or environment — same as the real `llm_client.py`), and a clip directory with `raw.mp4` + `events.json` containing at least one event (any clip already run through `run_pipeline.py`).

**What it does, in order:**
1. Runs the **real**, constrained `llm_client.SYSTEM_PROMPT` on every event in the clip (skips events already enriched, unless `--overwrite`).
2. Runs the **same** events again with a deliberately unconstrained, permissive prompt (no facts-only rule) — the one variable that changes.
3. Prints both narratives side by side, per event — this pair is what goes in the paper's before/after example.
4. **Automatically restores** the real constrained narrative back into `events.json` afterward, so this illustration run never leaves a hallucination-prone string sitting in the file the dashboard/report would otherwise read.

**Why this exists / the reasoning behind it:** Section VII.B needs one concrete "the model hallucinated, here's how tightening the prompt fixed it" example. If the current, already-tightened `SYSTEM_PROMPT` doesn't happen to hallucinate on the clips tried so far (plausible — it's specifically designed not to), waiting for a *naturally occurring* hallucination isn't practical on a fixed timeline. Deliberately relaxing the prompt on a throwaway variant reproduces the *same failure mode* the tightened prompt was written to prevent, on the *same* real event/keyframe — a controlled, honest before/after, not a fabricated one. **This is fine to disclose in the paper** — state plainly that the "before" example was reproduced under a deliberately relaxed prompt for illustration, not a naturally-occurring failure.

**What "reasoning" to check when you read the output:** the permissive narrative should mention something **not present** in the real structured facts/keyframe (an invented object, a fabricated number, an assumed cause) — that's the hallucination to quote. If the permissive prompt *also* comes back accurate, the underlying vision model may just be reliably grounded regardless of prompt — try a busier/more ambiguous clip, or note in the paper that a hallucination could not be reproduced even under a relaxed prompt (still a legitimate, honestly-reported finding).

---

## 8. User study (Section VII.C, claim C3) — a decision, not a script

**No script exists or should be written for this** — it requires actual human participants rating actual clips; nothing here can be automated or generated on your behalf.

**Decide with your supervisor, early (this needs lead time to recruit people):**
- **If you have time (Week 3–4):** recruit 15–25 people, show each the same 4–5 clips under three conditions (no explanation / template-only / template+LLM), collect Likert ratings (1–5) on clarity and trust, run a paired t-test or Wilcoxon signed-rank across conditions.
- **If you don't:** state plainly in the paper that the user study is left to future work, and that claim C3 currently rests on the structural-faithfulness argument (deterministic `explanation_template` always populated, LLM narrative is optional enrichment) plus the hallucination-mitigation example from §7 above.

**Do not leave Section VII.C blank either way** — an explicit limitation statement reads as a considered scope decision; a blank section reads as unfinished.

---

## Recommended order of execution (on the GPU laptop)

Numbers refer to the sections above.

1. **§6 (latency)** first — it's a pure code change already in place; run 3+ dev clips through `run_pipeline.py` to populate `meta.json`. Quick, and blocks nothing else.
2. **§2 (anticipation eval)** — the real 120-clip `--run`. Gives you Table I and unblocks §3/§4 (both read its cached `events.json`).
3. **§4 (ablations)** — start this once §2 finishes; it reprocesses the eval split 4 more times (once per non-reused variant), so budget roughly **4x** the wall-clock of step 2. Good candidate to kick off and leave running overnight.
4. **§3 (threshold sweep)** — cheap once §2's clips are cached; run it the same day as §4, no need to wait for the ablation sweep to finish first.
5. **§5 (qualitative figure)** — quick, once you've picked which real incident (e.g. a `dev1x` clip) to feature.
6. **§7 (hallucination example)** — quick, needs `GROQ_API_KEY` and your own eyes on the output.
7. **§8 (user study)** — decide *now*, in parallel with the above, since recruiting people needs lead time; don't leave this until everything else is done.

**Finally:** re-run `scripts/collect_paper_stats.py` once all of the above have produced real output. It only reads files it finds on disk (never fabricates a number) — everything currently marked `NOT YET AVAILABLE` in `eval/paper_stats_summary.md` will now have real content, ready to paste into the paper draft in place of each `[RESULT]` tag.
```powershell
python scripts\collect_paper_stats.py
```
