# Step 11: the single eval_v2 read (2026-09-23)

One scoring pass. Checkpoint `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt`,
threshold 0.70, confirm 8, on `data/features/eval_v2` (120 clips, 60 positive,
60 negative), `--final-eval-report`. No sweep, no alternate cell, no retrain, no
retry. Raw output: `eval/final_eval_read_v2_s1234.json`. Nothing here was
adjusted after seeing it.

Process note: `sweep_operating_point.py` prints only legacy-definition columns,
so a single run of it could not have yielded both definitions. I used a new
script, `scripts/final_eval_read.py`, which calls the same `evaluate()` and the
same held-out guard, does one pass, and refuses to overwrite an existing output.
It was dry-run on train_val_v2 first (reproducing 0.612 / FA 0.196 / AP 0.630
exactly) so the real read could not fail and tempt a second one.

## 1. eval_v2 numbers, both definitions

| metric | legacy (first-crossing) | v2 (episode-based) |
|---|---|---|
| useful-warning | **0.317** (19/60) | **0.650** (39/60) |
| false-alarm rate | 0.167 (10/60) | 0.167 (same) |
| mean lead vs event (onset) | 1.72 s | 1.51 s |
| mean lead vs alert | not defined | +0.07 s |
| frac_premature | (prematurity axis is v2-only) | 0.367 |
| median seconds early, premature clips | | 1.12 s |
| n_gross_premature | | **0** |
| mean AP (cutoffs 0.5 / 1.0 / 1.5 s) | 0.646 (0.735 / 0.642 / 0.561) | same, threshold-free |
| too-early (legacy) | 22 of 60 | |

"Lead vs alert" is measured from the dataset's `time_of_alert`, so +0.07 s
means the average useful warning lands almost exactly at the annotated alert
moment, not well before it. Legacy lead is over clips whose first crossing was
useful or too early; v2 lead is over v2-useful clips.

## 2. What each successive fix changed the number to

| row | useful-warning | FA | lead | definition / partition | status |
|---|---|---|---|---|---|
| Original v1-partition headline (thr 0.60/c8) | 0.750 (45/60) | 0.167 | 1.67 s | legacy; 65-clip non-representative split, selected on the test set | **superseded** |
| Intermediate clean protocol (thr 0.70/c8) | 0.633 (38/60) | 0.200 | 1.62 s | legacy; selection clean, partition still broken | **superseded** |
| train_val_v2 estimate (this config) | 0.612 | 0.196 | 1.83 s | v2; fixed partition and metric | validation estimate |
| **eval_v2 actual (this read)** | **0.650** (39/60) | **0.167** | **1.51 s** | v2; fixed partition and metric, held out | **final** |

Careful reading of that table, because it is easy to misread:

- The first two rows are **legacy-definition** numbers; the last two are **v2**.
  They are different metrics, so the rows are a story about what each fix
  changed, not a like-for-like trend. The like-for-like legacy figure on the
  fair held-out set is **0.317**, which is far below 0.633 and 0.750. Under v2
  it is 0.650. Neither the 0.75 nor the 0.633 should be quoted against the
  eval_v2 result without naming the definition.
- Validation to held-out: 0.612 -> 0.650 useful, FA 0.196 -> 0.167, lead
  1.83 -> 1.51 s, AP 0.630 -> 0.646. Useful and FA moved in the favourable
  direction, lead in the unfavourable one. All are well inside the noise
  already established (seed sd ~0.10-0.12 on useful; FA on 60 negatives has a
  standard error near 0.05). The right reading is "the validation estimate held
  up," not "the model is better than validation suggested."

## 3. The defensible headline

On a fair held-out set (eval_v2, disjoint from every set used for training,
early stopping, checkpoint selection or operating-point choice), the locked
configuration issues a useful warning on **65% of timed positive clips (39/60)**
under the corrected episode-based definition, at a **false-alarm rate of 16.7%**
(10/60 negatives), with a **mean lead of 1.5 s before the collision**, zero
grossly premature alerts, and threshold-free discrimination of mean AP 0.646.
Under the original stricter first-crossing definition the same behaviour scores
31.7% useful, because 22 of the 60 positives fire before `time_of_alert` and are
not credited; that gap is a property of the metric, and both are reported.
The typical useful warning arrives about at the annotated alert time
(+0.07 s), not meaningfully ahead of it. These are single-seed, single-split
numbers: the seed-to-seed sd is ~0.10-0.12 on useful-warning, so the honest
uncertainty on 0.650 is about that wide.

## 4. Targets not met, and what was achieved

The original targets (2-6 s lead; useful-warning >= 0.75, later >= 0.70) were
set **before** the partition bug, the scoring bug and the selection-bias bug
were found. They were never validated as achievable under fair conditions, and
**the corrected numbers do not meet them**: mean lead is 1.51 s (below 2 s), and
v2 useful-warning is 0.650 (0.05 short of 0.70; 0.10 short of 0.75). Under the
legacy definition it is 0.317.

What was achieved:

- **The false-alarm target is met on held-out data:** 0.167 against a <= 0.20
  target, and this was not tuned on eval_v2.
- **The result is trustworthy.** The held-out number is a genuine held-out
  number for the first time in this project, and it is consistent with the
  validation estimate rather than collapsing.
- **Failure modes are bounded:** zero gross-premature alerts, and no
  degenerate checkpoint (mean AP 0.646, well above chance).
- The model discriminates real risk (AP 0.65 vs 0.50 chance) and warns at
  roughly the annotated actionable moment, at an FA the metrics spec accepts.

What was not achieved: warnings that are earlier than about 1.5 s on average,
and the 0.70 useful-warning floor.

## 5. Chain of corrections

1. **Partition (Step 6).** `labels.xlsx`'s sort-and-truncate had put every clip
   with a window over 2.97 s in the eval split, biasing every prior number; the
   splits were redrawn (`repartition_results.md`, `label_window_audit.md`).
2. **Selection leakage (earlier).** Checkpoints, thresholds and kappa had been
   chosen on the test set; selection moved to train_val and eval was guarded
   (`eval_leakage_audit.md`, `clean_protocol_results.md`: 0.750 -> 0.633).
3. **Metric (Step 8).** The first-crossing useful-warning definition could score
   a clip that warned continuously through the whole window the same as one that
   never warned; the episode-based v2 definition and prematurity axis were added,
   with 0/45 superset violations against legacy
   (`metric_fix_results.md`, `early_firing_diagnosis.md`).
4. **Selection fallback (Steps 9 and 10.5).** A near-chance epoch could be kept
   as the checkpoint, on both selection paths; fixed and every existing
   checkpoint re-audited (`selection_fix_and_tradeoff.md`,
   `instability_and_checkpoint_audit.md`). Training confirmed bit-for-bit
   deterministic.
5. **Operating point (Step 10).** The documented rule had zero survivors across
   45 cells; seed 1234 at 0.70/c8 (0.612) was carried forward as a documented
   near-miss (`operating_point_selection.md`).
6. **Ensemble and calibration (Step 10.6).** A 5-seed ensemble (0.631) and
   temperature scaling (0.592) did not beat the baseline beyond noise; rejected
   (`ensemble_calibration_results.md`).
7. **Final read (this step).** One pass on eval_v2: v2 useful 0.650, FA 0.167,
   lead 1.51 s, AP 0.646.

## 6. Provenance

Nothing was deleted or overwritten. New files: `scripts/final_eval_read.py`,
`eval/final_eval_read_v2_s1234.json`, this document. eval_v2 was read exactly
once, by that script, in that one invocation; no other read of it occurred at
any point in the investigation.
