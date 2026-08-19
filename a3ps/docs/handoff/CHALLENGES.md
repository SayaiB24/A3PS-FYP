# Challenges faced and how they were solved

Written for the thesis. Each entry gives the symptom, the diagnosis, the fix, and
what generalises — because several of these are methodological traps rather than
code bugs, and those are the ones worth writing up.

The ones with the most thesis value are **§1 (silent split re-draw)**,
**§4 (AP tie-order artefact)**, **§5 (metric keyed to an invented window)**,
**§6 (task mis-framing)**, **§10 (ratio metric that improves as data degrades)**
and **§13 (loss that silently optimised the wrong objective)**. Those are all
cases where the code was working exactly as written and the *measurement* was
wrong — the failure mode that survives code review.

---

## Part A — Measurement and methodology traps

### 1. The held-out split re-drew itself whenever the dataset grew

**Symptom.** None. That is the problem.

**Diagnosis.** `prepare_nexar.py` assigns splits with
`random.Random(SEED).shuffle()` over the *remaining* clip pool. The shuffle order
therefore depends on how many clips are in the pool. Re-running it after
downloading more clips silently produces a **different** 120-clip eval split.
Every previously measured number would then refer to a different test set, and
3.7 GB of cached pipeline output would be invalidated — with no error raised and
no diff in the code.

Worse, this had already partly happened: the repo contained two indices
(`index.csv` and `index2.csv`) whose eval splits overlapped on only **69 of 120
clips**, and the default `--index` pointed at the stale one.

**Fix.** A freeze mechanism ([`a3ps/common/splits.py`](../../a3ps/common/splits.py),
[`scripts/freeze_split.py`](../../scripts/freeze_split.py)): dev/eval membership is
recorded once to `eval/split_freeze.json`, committed outside the gitignored
`data/` tree. Clips named there keep their split permanently; new clips can only
join `train_traj`. New *positives* are deliberately left unassigned rather than
auto-placed, because positives are scarce and belong to an explicit decision.
`prepare_nexar.py` and `eval_anticipation.py` both consult the freeze and
**refuse to run** on a drifted split.

**Generalises.** A held-out split is only held out if something enforces it. Any
pipeline where split assignment is recomputed from the data is one dataset
addition away from invalidating its own results.

### 2. Zero training positives were available locally

**Symptom.** The plan called for training a classifier; there was nothing to
train it on.

**Diagnosis.** All 65 local positive clips were already consumed by `dev` (5) and
`eval` (60). The training split contained negatives only, so a classifier trained
on it would learn to output "safe" always — and would look perfectly calibrated
against a negatives-only validation set.

**Fix.** Downloaded the full labelled set, taking positives from 65 → 685 while
the freeze protected the existing eval membership. Also a guard in
`train_risk_head.py` that refuses to train and validate on the same directory
without an explicit `--allow-leakage` flag, which prints a loud "this is a
plumbing check, not a result" warning.

### 3. Two indices, two different eval splits

**Symptom.** Scoring silently reported 69 of 120 clips as processed.

**Diagnosis.** `index.csv` (155 rows, CPU laptop) and `index2.csv` (355 rows, GPU
laptop) encoded different splits. `eval/anticipation`'s cached output matched
`index2` exactly; the script defaulted to `index.csv`. All 60 positives were
present in both, but only 9 of 60 negatives — so the **false-alarm rate was
computed over 9 negatives while appearing to use 60**.

**Fix.** Default `--index` changed to the tracked `eval/nexar_index_gpu.csv`, and
the freeze check (§1) now catches the mismatch as an error instead of a silently
smaller denominator.

**Generalises.** A metric whose denominator can quietly shrink is more dangerous
than one that crashes.

### 4. Average precision was an artefact of row order

**Symptom.** Reported AP of **0.978** — implausibly good.

**Diagnosis.** `average_precision()` sorted by score with ties broken by *input
order*, and the index lists all positives before all negatives. With many tied
scores, every tied positive was therefore ranked above every tied negative. Feeding
the same scores in reverse produced **0.371**. The metric was measuring the CSV's
row order, not the model.

**Fix.** Deterministic tie handling. The corrected mean AP over the same cached
120-clip eval split was **0.546** — less than half the headline it replaced.

**Generalises.** Any ranking metric on a dataset sorted by label will flatter the
model through ties alone. Test ranking metrics by shuffling the input and
confirming the number does not move.

### 5. The "useful warning" window was invented, not measured

**Symptom.** `useful_warning_rate` was computed against constants
`USEFUL_WINDOW_LO_S = 0.5`, `USEFUL_WINDOW_HI_S = 6.0` — chosen by us.

**Diagnosis.** Nexar already annotates the answer: `time_of_alert`, the
ground-truth earliest actionable moment, present per positive clip and already
carried in the index as `alert_time_s` — loaded and never used. Measured over the
65 local positives, the real lead (`time_of_event − time_of_alert`) is
**2.97–4.47 s (mean 3.49, sd 0.40)**, so the invented 6.0 s ceiling was ~2.5 s too
generous and the 0.5 s floor corresponded to nothing.

**Fix.** Re-keyed to a per-clip window `[alert_time_s − slack, event_time_s]` with
`slack` defaulting to 0. Firing before `time_of_alert` is no longer credited: by
the dataset's own annotation there was nothing actionable to see, so an earlier
alarm is luck or a standing false alarm. A separate `n_too_early` counter exposes
that failure mode instead of hiding it.

**Generalises.** Prefer the dataset's own annotation of "what counts" over a
threshold you picked. If you must pick one, report it as a parameter.

### 6. The task itself was mis-framed

**Symptom.** An earlier three-feature ablation produced a "negative result" —
no geometric feature separated collisions from near-misses.

**Diagnosis.** Nexar labels **both collisions and near-misses as positive**. So
"hit vs near-miss" was never the task; a near-miss that looks geometrically like
a collision is *supposed* to fire. The real task is **risky-event vs
ordinary-driving**.

**Fix.** Re-framed, and the ablation's correct conclusion recovered: no single
hand-set threshold on one geometric feature separates risk from normal driving,
because risk is a nonlinear combination of cues — which is precisely why the
literature learns the discrimination. That motivated the Phase IV pivot from a
hand-tuned threshold formula to a small learned temporal head, with Phases I–III
retained as feature extractors.

**Generalises.** Read the label definition before designing the metric. A
"negative result" is often a correct answer to the wrong question.

---

## Part B — Wrong assumptions about our own stack

### 7. The appearance embeddings we planned to reuse did not exist

**Symptom.** The plan called for reusing DeepSORT appearance embeddings as
features.

**Diagnosis.** The tracker is Ultralytics **BoT-SORT**, not DeepSORT, and
`configs/botsort_a3ps.yaml` sets `with_reid: False`. No appearance embedding is
ever computed; association is IoU + score fusion only. There was nothing to reuse.

**Fix.** Deferred ReID to a future ablation arm rather than adopting it blindly.
Two reasons: it needs `onnxruntime` (not a dependency) and enabling it changes
association, which would make new features incomparable with the cached pass. Also
a conceptual reason worth stating — a ReID embedding is trained for *identity
invariance*, i.e. to be the same vector for the same car whether or not it is
about to hit you, which makes it a weak prior for risk.

### 8. Ego-motion was already being computed and thrown away

**Symptom.** Ego motion was listed as a missing feature class requiring new work.

**Diagnosis.** BoT-SORT's global motion compensation already runs sparse optical
flow every frame and produces a 2×3 affine warp (`byte_tracker.py:353`), then
discards it. The signal we wanted was being computed and dropped.

**Fix.** Built [`a3ps/features/ego.py`](../../a3ps/features/ego.py) with its own
estimator rather than tapping GMC's, for a specific reason: GMC's `sparseOptFlow`
path ignores its `detections` argument, so it is whole-frame flow made robust by
RANSAC, **not** static-scene flow. Ours masks detections, so moving actors do not
contaminate the ego estimate. Result: `has_ego=True` with a **0% failure rate**
across all 1,485 extracted clips.

**Generalises.** Before building a signal, check whether a dependency already
computes it internally — and check what it actually computes rather than what its
name implies.

### 9. The test split had no labels

**Symptom.** After indexing, only 1,500 of 2,844 clips appeared. Initially
diagnosed (by me) as ~47% of the data going missing.

**Diagnosis.** Wrong diagnosis. All 1,344 `test/` ids matched their videos
exactly — nothing failed to match. But `target` was **null on every row**, with
empty event/alert times: it is the Kaggle competition holdout, where labels are
never published. `prepare_nexar.py` had correctly excluded unusable rows.

**Fix.** None needed; a 1,500-row index is complete. Documented prominently so it
is not re-investigated. The unlabelled clips retain one use: generating a Kaggle
submission for an external unbiased AP.

**Generalises.** "Data is missing" and "data is unusable" produce the same row
count and need opposite responses. Check *why* rows were dropped before fixing it.

---

## Part C — Empirical traps during tuning

### 10. The frame-rate metric improved as the data got worse

**Symptom.** Decimating 30 Hz → 10 Hz *increased* the fraction of forecastable
tracks by 51%, suggesting lower frame rates were strictly better.

**Diagnosis.** `frac ≥ 2s` is a **ratio**. Subsampling removes short-lived
flicker detections that would never have become usable tracks, so it shrinks the
denominator faster than the numerator. Read alone, the ratio argues for
decimating forever. The quantity that actually determines training data volume is
the **absolute** count:

| rate | tracks/clip | frac ≥2 s | forecastable/clip | vs 30 Hz |
|---|---|---|---|---|
| 30 Hz | 44.7 | 0.231 | 10.3 | — |
| 15 Hz | 31.5 | 0.292 | 9.2 | −11% |
| 10 Hz | 25.2 | 0.349 | 8.8 | −15% |

**Fix.** `check_assoc_rate.py` now reports `forecastable/clip` as the headline
column and explains why the ratio is misleading. Chose 10 Hz deliberately: −15%
yield for 2.5× throughput, immaterial when 1,365 clips still yield ~12k tracks.

**Generalises.** A ratio can improve because its denominator degraded. Always
check the absolute numerator too.

### 11. Feature noise depends on frame rate, so rates cannot be mixed

**Symptom.** A plan to extract training data at 10 Hz and eval at 15 Hz — cheaper
where volume matters, denser where accuracy matters.

**Diagnosis.** Derivatives are central differences over a `2/rate` span: 0.200 s
at 10 Hz versus 0.133 s at 15 Hz. Over the same detector jitter, the shorter span
yields **systematically noisier** velocity and acceleration features. Training on
one noise distribution and evaluating on another would produce a performance drop
attributable to extraction, not to the model — and it would have looked like a
model deficiency.

**Fix.** One rate everywhere (10 Hz), enforced in the docs with an explicit
warning. The units are rate-invariant (per-second, from real timestamps); it is
the *noise characteristics* that are not.

**Generalises.** Rate-invariant units do not imply rate-invariant distributions.

### 12. Runtime estimates were wrong twice, in different ways

**Symptom.** Predicted "~6–8 min" for training; it took 85 min. Then predicted
"~15 min remaining"; also wrong.

**Diagnosis.** Two independent errors. First, the benchmark used plain
`BCEWithLogits` rather than the real anticipation loss, which computes per-timestep
weights per clip. Second and larger: `--batch-size 8` does **not** batch the
compute — `[model(c["X"]) for c in batch]` is a Python loop of single-sequence
forwards, so effective batch size is 1 and a GRU at batch 1 is bound by 130
sequential timestep kernel launches. Measured parallelism was 1.85× on 8 cores,
not near-linear.

Also: documented GPU throughput was assumed at 20–40 ms/frame; measured was
**65 ms/frame**, making the extraction pass ~3.5 h rather than ~2 h.

**Fix.** Benchmarked the real loss at real shapes. Batching inefficiency
documented as an explicit task with a verification requirement, rather than
patched mid-tuning. Measured throughput recorded in the runbooks so estimates
start from data.

**Generalises.** Benchmark the real code path, not a simplified stand-in. And a
parameter named `batch_size` does not necessarily batch anything.

### 13. The loss was optimising for a ~1 s warning

**Symptom.** Mean lead time stuck at 1.5–2.1 s against a 2–6 s target.

**Diagnosis.** The anticipation loss weights timesteps by proximity to the event,
sharpened by `kappa`. At the default `kappa=3.0`, `expected_lead_time()` reports
the loss is asking for a warning **0.98 s** before impact. The model was doing
exactly what it was told — we had trained a detector and measured it as an
anticipator. The function's own docstring warned of this: *"a large kappa quietly
turns an anticipation objective into a detection one."*

**Fix.** Diagnosed and documented as the top next step
([`KAPPA_RETRAIN.md`](KAPPA_RETRAIN.md)), with the kappa-to-lead mapping tabulated
so the choice is explicit. Deliberately **not** changed mid-analysis, to keep the
committed baseline reproducible.

**Generalises.** Print what your objective is actually asking for, in the units
you care about. A hyperparameter that silently redefines the task is worse than
one that fails loudly.

### 14. The premature-alarm problem was a threshold artefact

**Symptom.** 21 of 60 positives fired before `time_of_alert` — a third of them
warning before anything was there to see. The apparent fix was to retrain with a
higher `pre_alert_weight` (~21 min per value).

**Diagnosis.** `threshold` and `confirm` are **evaluation-time** parameters; they
do not affect the weights. So the entire trade-off curve could be swept on the
existing checkpoint for the cost of one forward pass per clip. Doing so showed
too-early alerts falling **5 → 1** purely by raising the threshold from 0.5 to
0.70, and false alarms falling **0.583 → 0.167**. No retraining was needed at all.

**Fix.** [`scripts/sweep_operating_point.py`](../../scripts/sweep_operating_point.py),
and the operating point committed at 0.70/5. `pre_alert_weight` was aimed at an
already-solved problem; the real remaining gap is lead time (§13).

**Generalises.** Separate parameters that require retraining from those that do
not, and exhaust the free ones first. It cost minutes instead of hours and
redirected the diagnosis.

### 15. Model selection ignored the gatekeeper metric

**Symptom.** Best epoch chosen as 5 (useful 0.833, FA 0.583). But epoch 8 scored
useful 0.833 with FA **0.550** — equal on the selection metric, strictly better on
false alarms — and was discarded.

**Diagnosis.** Selection is `score > best["score"]`, strictly greater on
useful-warning rate alone. Ties never update, and false-alarm rate is not part of
the criterion at all — even though it is the metric that decides publishability.

**Fix.** Not yet changed; documented so the operating-point sweep (§14) compensates
by re-choosing the decision boundary afterwards. A better criterion would be
useful-warning subject to an FA constraint, or a tie-break on FA.

**Generalises.** The model-selection criterion must include the constraint that
makes a result acceptable, or selection will optimise a number you cannot report.

---

## Part D — Engineering and process

### 16. An 85-minute training run lost its best model

**Symptom.** A 40-epoch run was killed at epoch 30 holding a best of 0.850
useful-warning from epoch 24. None of it survived.

**Diagnosis.** Two compounding faults. The training loop kept best weights **in
memory** and wrote the checkpoint only after the final epoch, so any interruption
lost everything. And the run's output had been piped through `tail`, which
withholds all output until the process exits — so there was no per-epoch progress
signal either, making it impossible to judge whether stopping was safe. (The best
epoch had to be recovered by attaching `py-spy` to the live process to read the
loop's local variables.)

A third contributing error was mine: I reported the run as "peaked at epoch 5 and
cannot improve" based on a mid-run snapshot, and the decision to kill it was made
on that incorrect premise. It had in fact improved to epoch 24.

**Fix.** Checkpoint-on-improvement via an `on_improve` callback; history rewritten
every epoch; `flush=True` on every print with a `<- best` marker; `--patience`
early stopping so runs end when they stop improving rather than burning the full
budget. Runs are now safe to interrupt at any moment. Logs are written with `>`
redirection, never through a buffering pipe.

**Generalises.** Persist the best result the moment it exists, not at the end. And
never make an irreversible decision from a snapshot of a process you cannot see
the history of.

### 17. Instructions that could not have run

**Symptom.** The GPU handoff doc contained a bash heredoc (`python - <<'PY'`) in a
PowerShell-only runbook.

**Fix.** Extracted to a real script
([`scripts/assign_unused_positives.py`](../../scripts/assign_unused_positives.py))
and converted the remaining inline snippets to PowerShell here-strings. Also
corrected repeated stale figures (a "2,844-clip pool" that was actually 1,500;
runtime tables scaled to the wrong pool size).

**Generalises.** Runbook commands are code. If they have not been executed in the
target shell, assume they do not work.

### 18. A tracked file inside the Python package

**Symptom.** `a3ps/a3ps/eval/forecast_table_PREVIOUS.md` — a documentation file
inside the package directory.

**Diagnosis.** A truncated copy of `eval/forecast_table.md` (same header and
Kalman row, Seq2Seq row missing), misnamed `_PREVIOUS` while holding current
numbers. Evidently written by a script run from the wrong working directory.

**Fix.** Deleted; the genuine previous-run numbers remain in
`eval/forecast_table_PREVIOUS.md`. Docs were then grouped into
`docs/{runbooks,design,status,handoff}` so there is an obvious correct home for
any new document.

### 19. A benign warning that looked like a failure

**Symptom.** `WARNING not enough matching points` recurring throughout feature
extraction.

**Diagnosis.** Ultralytics' internal BoT-SORT motion compensation
(`gmc.py:326`), not our code. When optical flow tracks ≤4 keypoints between
frames it cannot fit an affine transform, logs this, and returns identity — no
motion compensation for that one frame, association continuing on IoU. Graceful
degradation. Our own ego estimator is a separate pass and reported a **0% failure
rate** throughout.

**Fix.** No code change; documented, along with the distinction that `ego fail 0%`
is a *failure rate* (0% is ideal) rather than a measure of how much ego motion was
found — a reading that had initially caused concern.

**Generalises.** Know which component a warning comes from before acting on it,
and make health metrics unambiguous about their direction.
