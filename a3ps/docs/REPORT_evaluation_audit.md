# A3PS evaluation audit: what was wrong, what was fixed, what the number is

This report assembles the eleven-step evaluation investigation into one
narrative. It is a synthesis: every number below is taken from the status or
design document named next to it, and none of those documents has been changed.
No experiment was run and `eval_v2` was not read to write it. Where I computed a
figure that no source document states (a confidence interval, a per-clip
sensitivity), it is marked **[derived]** and the arithmetic is shown.

Document paths are relative to `docs/`. Step numbers are the ones the status
documents use.

---

## 1. Summary

**The defensible result:** on a fair 120-clip held-out split (`eval_v2`, 60
positive / 60 negative), the locked configuration issues a useful warning on
**0.650 of timed positives (39/60)** under the project's corrected
episode-based metric, at a **false-alarm rate of 0.167 (10/60)**, with a **mean
lead of 1.51 s** before the collision, **zero** grossly premature alerts, and a
threshold-free mean AP of **0.646** against a chance level of 0.50. The
checkpoint is `risk_gru_k1p0_v2_selfix_s1234.pt` at threshold 0.70 / confirm 8,
scored once
([`status/final_eval_read.md`](status/final_eval_read.md) §1, §3).

**The original targets were not met.** Useful-warning of 0.70–0.75 was missed
(0.650 under the corrected metric, 0.317 under the original stricter one) and
the 2–6 s lead target was missed (1.51 s). The false-alarm target (≤ 0.20) was
met, as a point estimate, with wide uncertainty (§6). Both misses are reported
here at face value; §5 explains the lead-time shortfall and is careful about
how much of it the data can be blamed for.

**Why the number is different from the ones quoted earlier.** The project
previously reported useful-warning 0.750 at FA 0.167. That figure was wrong in
three separable ways, and a fourth defect could silently corrupt any given
training run:

1. It was **selected on the test set**, not measured against it (§2.1).
2. The test set was **not representative**: a hand-built spreadsheet had been
   sorted and truncated so that the held-out split held every long-warning-window
   clip in the dataset (§2.2).
3. The metric **could score a clip that warned continuously through the whole
   actionable window the same as one that never warned** (§2.3).
4. Checkpoint selection could **silently keep a chance-level epoch** (§2.4).

Each was found, evidenced, and fixed, and the final number was read once, after
everything it depends on was frozen. It is lower than the old headline on the
original definition (0.317 vs 0.750) and lower-to-comparable on the corrected
one (0.650), and it is the first number in the project that is a genuine
held-out measurement.

**Read the headline with its error bar.** 60 positives means one clip is worth
1.7 points of useful-warning; the binomial 95% interval on 39/60 is roughly
0.52–0.76 **[derived, §6]**, and seed-to-seed variation at this operating point
on validation was wider still (§6). This is one seed, one split, one read.

---

## 2. The problems found

Ordered by discovery. (`final_eval_read.md` §5 lists the corrections in a
different order; this report follows the order in which each was found.)

The brief for this report described three problems. The record contains
**four**: selection on the test set, the split construction, the scoring rule,
and the checkpoint-selection gap. All four are included; the first is the one
most easily overlooked because it was found before the split problem and is
independent of it.

### 2.1 The headline was selected on the test set (Steps 3–4)

**Symptom.** The documented retraining command for the headline checkpoint
passed `--val-features data/features/eval`, and `eval` is the frozen 120-clip
held-out split. A number reported on a set that also chose the model is not a
held-out number.

**Investigation.** [`status/eval_leakage_audit.md`](status/eval_leakage_audit.md)
established what happened from evidence rather than from what the docs claimed:

- The command in `docs/handoff/REPRODUCE_BY_HAND.md` (lines 177–183 at the time
  of the audit) used `--features data/features/train_all --val-features
  data/features/eval`.
- The checkpoint's own saved `extra` dict, read directly, recorded `n_clips:
  120` for validation, the exact size of the frozen eval split
  (`eval/split_freeze.json`: 60 + 60) and unlike any other directory
  (`train_all` is 1,365 clips) (§1).
- `train()` in `scripts/train_risk_head.py` scores `val_clips` every epoch and
  uses that score both to decide which weights to save and when to stop early
  (§1).

It went further than early stopping. The **kappa choice**, the **operating
point** (threshold 0.60 / confirm 8), the **capacity check** (hidden 64 vs
128×2) and the **feature ablations** were each decided by eval-scored numbers
(§2, §3, §6). The audit's classification table concluded: "every decision that
shaped the final model and its operating point ... passed through eval-scored
numbers."

**Cause.** Two guards existed and neither caught it. `verify_freeze()`
(`a3ps/common/splits.py`) checks split *membership* drift, not what a training
run points `--val-features` at. The `--allow-leakage` guard in
`train_risk_head.py` only refused runs where `--features` and `--val-features`
were the *identical directory*; `train_all` vs `eval` are different directories,
so the guard was silent and the checkpoint duly recorded `leakage_warning:
False` (§5). Separately, `sweep_operating_point.py` defaulted `--features` to
`data/features/eval`, which is how the operating point came to be chosen on the
test set without anyone passing a flag
([`status/clean_protocol_results.md`](status/clean_protocol_results.md), "What
changed" item 3).

**Fix.** A stratified validation split, `train_val` (205 clips, 103 pos / 102
neg), was carved from `train_all` and frozen in `eval/split_freeze.json`.
`train_risk_head.py` and `sweep_operating_point.py` now consult the freeze and
refuse any clip frozen as eval unless `--final-eval-report` is passed
(`train_risk_head.py:726-736`). Selection moved to `train_val`.

**What changed.** On the same v1 eval split, the clean protocol scored
useful-warning **0.633 (38/60)** at FA **0.200**, against the biased **0.750
(45/60)** at 0.167. Kappa, re-selected on `train_val`, chose 0.5 by a 0.007 s
margin and then scored *worse* on eval (0.583) than the config it replaced,
"which is what a margin that size should be expected to do"
(`clean_protocol_results.md`). The clean document already flagged that not all
of the drop was bias: the training set shrank 1,365 → 1,160, and "no attempt is
made here to separate the two."

**Verification.** The guard was exercised directly: `train_core_v2` and
`train_val_v2` allowed, `eval_v2`, `train_all` and the v1 `eval` directory
refused (`status/repartition_results.md` §3, "The guard").

### 2.2 The held-out split was the easiest slice by construction (Steps 4–6)

**Symptom.** While asking why validation useful-warning (~0.37) sat so far below
eval's (0.633), Step 4 tabulated the alert-to-event window per split
(`clean_protocol_results.md`):

| split | timed positives | window mean | min | max |
|---|---|---|---|---|
| `train_risk` | 582 | 1.41 s | 0.03 | 2.97 |
| `train_val` | 103 | 1.50 s | 0.03 | 2.97 |
| `eval` | 60 | 3.51 s | **2.97** | 4.47 |

The ranges do not overlap: every training positive under 2.97 s, every eval
positive over it. A useful warning must land inside that window, so eval's
positives were structurally easier to warn about.

**Investigation, including a retracted theory.** The first guess was that two
annotation sources with different conventions had been mixed. That guess was
**wrong and was retracted**
([`status/label_window_audit.md`](status/label_window_audit.md), intro). Tracing
ten eval and ten training positives through raw `train.csv` → `labels.xlsx` →
`index.csv` → `.npz` metadata showed the times were identical at every layer;
across all 1,485 extracted clips the mismatch counts were 0 for
`event_time_s`, 0 for `alert_time_s`, 0 for label (§1). No code in the
repository altered a time. What differed was which rows existed: **every eval
positive was present in `labels.xlsx` and no training positive was.**

**Cause.** `data/nexar/labels.xlsx` is a 355-row table (65 positives + 290
negatives) that `prepare_nexar.py` consumed for the original download
(`prepare_nexar.py:14` documents it as an input). Its positive rows are sorted
by window length descending — verified programmatically — and are **exactly the
65 longest windows in the 750-positive dataset** (set equality verified). The
cut is 4 ms wide: 65th-largest window 2.971 s, 66th 2.967 s
(`label_window_audit.md` §2). No script in the repository writes this file; it
was built by hand. `assign_splits()` (`prepare_nexar.py:247`) then shuffled
randomly, but "a random draw from a pre-filtered pool is still a filtered
sample": all 65 went to `dev` (5) and `eval` (60). The 685 positives added later
were all ≤ 2.967 s. The raw table itself is continuous (mean 1.60 s, median
1.43 s, 0.033–4.466 s, no gap at 2.97 s), so the disjointness was manufactured
by row selection, not present in the data.

A second consequence made this more than a fairness issue. `alert_weights()`
(`a3ps/risk/anticipation_loss.py:104-110`) multiplies every frame outside
`[alert_t, event_t]` by zero, so the loss provides **no positive signal for
firing earlier than a clip's own annotated window**. With no training clip above
2.967 s, the objective could never request more than that much lead. The
documented "requested lead" was also computed on the wrong population:
`compare_kappa.py:43` hard-codes `MEAN_ALERT_LEAD_S = 3.49` ("measured over the
65 local positives"). On the real training pool the loss asks for about **0.67 s**
at kappa 1.0, not the documented 1.46 s (`label_window_audit.md` §5;
`repartition_results.md` §6).

**Fix.** `scripts/repartition_splits.py` (seed 1234) binned the 745 pooled
positives into deciles of window length (74–75 per bin) and apportioned each
split from every bin by largest remainder. Per-bin rounding was tried first and
rejected because patching its overshoot took the excess from the top decile,
"precisely the clips this repartition exists to spread" (2 eval positives in
decile 10 instead of 6). New splits: `train_core_v2` 1,160 (582/578),
`train_val_v2` 205 (103/102), `eval_v2` 120 (60/60), pairwise disjoint, frozen
in `eval/split_freeze_v2.json`. No features were re-extracted, because
`event_time_s` was unchanged and every existing feature window was already
correctly placed (`repartition_results.md` §3; `label_window_audit.md` §3).

**What changed.**

| | v1 (superseded) | v2 |
|---|---|---|
| train window max | 2.967 s | 4.466 s |
| eval window mean / min | 3.51 s / 2.971 s | 1.587 s / 0.200 s |
| ranges overlap? | no, disjoint at 2.97 s | yes, all three span the pool |

(`repartition_results.md` §3, "Old vs new".) Because 97 of v1's 120 eval clips
moved into `train_core_v2`, **every earlier eval number, the old threshold
engine's included, became incomparable to anything produced afterwards**,
and cannot be rehabilitated by rescoring (§7). Two further things did *not*
change and are easy to overstate: the repartition barely moved the training
window mean (1.41 → 1.60 s), and the dataset still has only 63 positives over 3 s
of 750. "Redrawing the split fixed *who gets the long-window clips*; it did not
create more of them" (`repartition_results.md` §6).

**Not fixed by this step:** the hard-coded 3.49 in `compare_kappa.py:43` and
`train_risk_head.py:800` was still in the tree when this report was written.
*Update: fixed afterwards in commit `3a4b349`. Both callers now compute the mean
window from the loaded clips (`mean_alert_lead_s()` in `anticipation_loss.py`).
Diagnostic output only; no scored number changed.*

### 2.3 The metric could not tell "warned throughout" from "never warned" (Steps 7–8)

**Symptom.** At operating points with any useful lead, 40–60 of 103 validation
positives were classified `too_early` and earned no credit
(`repartition_results.md` §5). A model firing "too early" on more clips than it
got right looked like a timing failure.

**Investigation.**
[`status/early_firing_diagnosis.md`](status/early_firing_diagnosis.md) checked
whether the early fires were noise. They were not: zero early fires landed in
the first 15% of the clip at either operating point tested (0/21 and 0/55); they
clustered at 60–88% of the way through; and the risk curves for useful,
too-early and missed clips all had the same shape (near zero, then a smooth
multi-second rise peaking near `alert_t`/`event_t`) (§2–3). Clip 1013 was traced
frame by frame at threshold 0.60: probability above threshold **continuously
from 17.33 s through `alert_t` (18.83 s) and `event_t` (19.30 s) to the end of
the clip**, scored `too_early` regardless (§6).

**Cause.** `first_alert_time()` (`a3ps/risk/anticipation_loss.py:224-250`) scans
once and returns on the first sustained run:

```python
for i, hot in enumerate(over):
    if hot:
        run += 1
        if run >= need:
            return float(t[i - need + 1])   # returns immediately
    else:
        run = 0
return None
```

`evaluate()` (`scripts/train_risk_head.py:147`, call at `:197`) classifies the
whole clip from that single timestamp. What the curve does afterwards,
including staying above threshold through the entire window, is never inspected.
Under this rule "warned continuously through the window" and "never warned"
both score zero
([`design/useful_warning_definition.md`](design/useful_warning_definition.md)
§1–2). The recoverable headroom was large: at 0.60/confirm 5, 52 of 55
too-early clips had a qualifying run covering part of the window; at
0.80/confirm 8, 15 of 21 (`early_firing_diagnosis.md` §6).

**Fix, and how it was justified.** The design document was **committed before
any scoring code changed**, on the stated principle that the justification for
changing a self-defined metric must exist independently of the number the change
produces. The argument rests only on operational meaning: the question is
whether a warning was *active* during the window in which the driver could still
act. The v2 definition scores a clip useful if any confirmed alert episode's
active interval intersects `[alert_t, event_t]`. The legacy definition is
never removed, both are reported side by side, and prematurity (fraction of
positives whose earliest episode begins before `alert_t`, median seconds early,
and a gross-premature count for episodes over 5 s early or within the first 20%
of the clip) is a separate first-class axis so that firing from frame 0 cannot
buy a high score unseen (`useful_warning_definition.md` §4–5). The Nexar
competition AP-at-cutoffs metric is untouched.

Two candour points belong here. First, the design document itself records that
the headroom (15/21, 52/55) "was already known, from diagnosis, before this
document or any code change": the metric was redefined *after* the diagnosis
showed what it would gain, and the document says so rather than pretending
otherwise (§2). Second, the first implementation credited an episode at the
frame its confirm-debounce *completed*, which broke the guarantee that v2 ≥
legacy: on 6 of 45 grid cells v2 credited fewer clips (worst case 32/103 →
24/103). The fix credits an episode at its own first frame, matching
`first_alert_time`'s convention, and was verified across the full grid:
**0 of 45 cells have v2 below legacy** for all three checkpoints
([`status/metric_fix_results.md`](status/metric_fix_results.md), "A correction
found while producing this document"). Legacy numbers reproduce bit for bit,
locked by `tests/test_train_risk_head.py::test_legacy_numbers_reproduce_committed_checkpoint_bit_for_bit`.

**What changed.** The same behaviour scores very differently under the two
definitions. Final read: **0.317 (19/60) legacy vs 0.650 (39/60) v2**, with
22 of 60 positives firing before `time_of_alert` under the legacy rule
(`final_eval_read.md` §1). A loose metric is also gameable: on validation, seed
1234 at threshold 0.50 / confirm 3 reaches v2 useful-warning 0.942 at FA 0.667
(`metric_fix_results.md`, seed 1234 grid). That is why FA and the prematurity
axis gate every operating-point choice and why v2 useful-warning is never
quoted alone.

### 2.4 Checkpoint selection could silently keep a chance-level epoch (Steps 9, 10.5)

**Symptom.** Seed 1236's checkpoint had validation mean AP **0.501** on a
103/102 set, where chance is about 0.50 (`early_firing_diagnosis.md` §4). Its
output was not constant (probability spanned 0.062–0.896 over 24,543 frames) but
carried no label information: "collapse to an uninformative output, not a
constant one."

**Investigation (Step 9).** `train()` ranks epochs by `(meets fa_target,
useful-warning, −FA)` (`scripts/train_risk_head.py:583`). Checked against the
actual histories, **none of the three seeds ever met `fa_target`** at the
training loop's scoring point, so the first tuple element tied at 0 and
selection degraded to useful-warning alone. For seed 1236 that picked epoch 1
(AP 0.501) over epoch 6 (AP 0.648)
([`status/selection_fix_and_tradeoff.md`](status/selection_fix_and_tradeoff.md)
§1). The obvious fix, "select by mean AP whenever nothing is FA-compliant," was
checked against all three histories *before* implementing and would have moved
seeds 1234 and 1235 to different epochs, trading real useful-warning for
marginal AP on runs that were never broken (1234: epoch 4, useful 0.330 →
epoch 3, useful 0.117). It was therefore scoped narrowly: override only when the
primary pick never met `fa_target` **and** its AP is within
`NEAR_CHANCE_AP_MARGIN` (0.05) of chance, with chance computed as the
validation set's own positive prevalence (`train_risk_head.py:444`,
`:630-641`).

**Second gap (Step 10.5).** Seed 1238 exposed that the fallback did not cover
everything. Its selected epoch 1 had AP 0.490 but *did* meet `fa_target` (FA
0.127) because an undertrained model fires weakly, so the primary rule ran
normally and the fallback, which only activates when no epoch is compliant,
never triggered. Meeting the FA gate had been treated as sufficient evidence of
a legitimate result. It is not
([`status/instability_and_checkpoint_audit.md`](status/instability_and_checkpoint_audit.md)
§1).

**Cause, precisely.** The selection metric is an operating-point pair at a fixed
threshold (0.5, confirm 3), not a ranking metric, and a model that is barely off
initialisation is "compliant" for the wrong reason.

**Fix.** An unconditional check after selection settles: if the chosen epoch's
AP is within 0.05 of chance, the checkpoint's `extra` gets
`degenerate_checkpoint: True` regardless of `fa_target` or the fallback
(`train_risk_head.py:654-663`). It flags and does not reselect. A test
reproduces the seed-1238 pattern (`test_train_flags_degenerate_even_when_fa_target_met`).

**Verification and what it turned out to mean.** All eight checkpoints then on
file were re-audited: **only seed 1238 was degenerate** (chance on
`train_val_v2` 0.5024, threshold 0.5524; the least comfortable non-degenerate
one is `paw2p0_s1234` at 0.578) (`instability_and_checkpoint_audit.md` §1).
Training was confirmed bit-for-bit deterministic by rerunning seed 1234 and
comparing every epoch's history (`==` on all 12 epochs), which rules
non-reproducibility out as the cause of seed variation. And the mean-AP
trajectory of all five base seeds has the same shape (chance at epoch 1, rapid
rise over epochs 2–3, plateau of 0.58–0.68), including seed 1238, whose
epoch-9 AP was 0.645. So the "collapse" was a **selection artifact on healthy
training, not a training failure**: 0 of 5 seeds collapsed; 1 of 5 produced a
degenerate checkpoint selection (§2). The batch-size/learning-rate hypothesis
was checked against those curves and "not strongly supported"; it was not acted
on.

Two things were left open and stay open. The mean-AP-vs-useful-warning
discrepancy across seeds is not fully explained (§5 below and §3.2). And the
seed-1238 rerun used for the ensemble required a new opt-in flag,
`--skip-near-chance-epochs`, described in its own document as "my judgment call"
(`ensemble_calibration_results.md` §4).

---

## 3. What was tried and did not help

Each item below was tested on `train_val_v2` and rejected or left unadopted. Eval
was not consulted for any of them.

### 3.1 Ensembling and temperature calibration (Step 10.6)

Five-seed probability averaging and temperature scaling, against the seed-1234
baseline at its own best cell. "Best cell" is the highest v2 useful-warning
among grid cells passing FA ≤ 0.20, lead ≥ 1.0 s and zero gross-premature
([`status/ensemble_calibration_results.md`](status/ensemble_calibration_results.md)
§1):

| candidate | cell | useful_v2 | FA | lead | mean AP | paired diff vs baseline (95% CI) |
|---|---|---|---|---|---|---|
| **Baseline: seed 1234** | 0.70 / c8 | **0.612** | 0.196 | 1.83 s | 0.630 | – |
| Ensemble, uncalibrated | 0.70 / c5 | 0.631 | 0.196 | 1.39 s | 0.611 | +0.019 [−0.039, +0.078] |
| Ensemble, temp-calibrated | 0.70 / c8 | 0.592 | 0.176 | 1.53 s | 0.611 | −0.019 [−0.068, +0.029] |
| Mean of calibrated members | 0.70 / c5 | 0.631 | 0.196 | 1.38 s | 0.613 | +0.019 [−0.039, +0.078] |
| Ensemble, temp fit 5-fold CV | 0.70 / c8 | 0.602 | 0.186 | 1.53 s | 0.614 | −0.010 [−0.049, +0.029] |

All intervals straddle zero. The ensemble's "win" is two positive clips out of
103, bought with 0.44 s less lead and 0.019 less AP. It was not adopted. The one
honest thing to say in its favour is robustness (smoother across cells, does not
hinge on one seed) at 5× inference cost, and the document treats that as a
robustness argument, not a performance one.

The calibration *hypothesis* was tested directly and **not supported**. The
idea (from the Step 10.5 audit) was that seeds with similar ranking quality but
very different useful-warning at a shared threshold differ in probability scale.
Temperature scaling did not narrow the spread across seeds at threshold 0.70 /
confirm 8: the standard deviation **rose from 0.159 to 0.200**
(`ensemble_calibration_results.md` §2). The fitted temperatures lay within
0.75–1.36, and seed 1236 still fired at FA 0.43 after calibration, so the seeds
seem to differ in *when and on which clips* they fire, not just in global scale.
Platt scaling was not tried; the document is explicit that "I would not expect it
to change this, but that is an expectation, not a result."

### 3.2 The `pre_alert_weight` sweep (Step 9, seed 1234 only)

Retrained seed 1234 at `pre_alert_weight` ∈ {0.5 (baseline), 1.0, 2.0, 4.0}
([`status/selection_fix_and_tradeoff.md`](status/selection_fix_and_tradeoff.md)
§4):

| pre_alert_weight | mean AP | useful_v2 @ 0.80/c8 | useful_v2 @ 0.60/c5 | frac_premature @ 0.60/c5 | lead-from-onset @ 0.60/c5 |
|---|---|---|---|---|---|
| **0.5** | 0.630 | 0.447 | 0.786 | 0.534 | 2.40 s |
| 1.0 | 0.593 | 0.621 | 0.825 | 0.466 | 1.83 s |
| 2.0 | 0.578 | 0.291 | 0.573 | 0.223 | 1.10 s |
| 4.0 | 0.586 | 0.505 | 0.728 | 0.437 | 1.85 s |

It is **not a clean tradeoff curve**. 0.5 → 2.0 shows the expected pattern
(premature firing falls, coverage and lead fall with it). But 1.0 *raises*
useful-warning over baseline, and 4.0 partially reverses 2.0. With one seed per
setting and a seed-to-seed sd of ~0.10–0.12 on useful_v2, "part of what looks
like a `pre_alert_weight` effect could be ordinary run-to-run training
variance." Gross-premature was 0 or 1 of 103 at every setting, so the lever was
never fixing a gross-prematurity problem. The baseline 0.5 was retained; the
sweep was stated, not decided, in the source document.

### 3.3 Kappa re-selection (Step 4)

Kappa 1.0 had itself been chosen on eval, so all four values were retrained and
re-selected on `train_val`: kappa 0.5 won by **0.007 s of mean lead**, "far
inside the seed spread," and its single eval read scored *worse* on useful-warning
(0.583 vs 0.633). The project's decision rule also could not be applied as
written on that split (no row of any kappa reached useful-warning 0.70; the
maximum in the grid was 0.418) and the gate was dropped and recorded rather than
adjusted silently (`clean_protocol_results.md`).

### 3.4 Not attempted or not redone

- The four feature-ablation checkpoints and the hidden-128 capacity run were
  selected under the biased protocol and, per the documents reviewed, have not
  been redone under the clean one; their absolute numbers carry the old caveat
  and only their relative ranking is "less affected" (`clean_protocol_results.md`,
  "What has *not* been redone").
- Batch size / learning rate were examined only against training curves and
  not varied (`instability_and_checkpoint_audit.md` §2).

---

## 4. The final result

### 4.1 Both definitions, `eval_v2`, single read

`risk_gru_k1p0_v2_selfix_s1234.pt`, threshold 0.70 / confirm 8, 120 clips
(60 pos / 60 neg), `--final-eval-report`, one pass
([`status/final_eval_read.md`](status/final_eval_read.md) §1):

| metric | legacy (first-crossing) | v2 (episode-based) |
|---|---|---|
| useful-warning | **0.317** (19/60) | **0.650** (39/60) |
| false-alarm rate | 0.167 (10/60) | 0.167 (unchanged) |
| mean lead vs event | 1.72 s | 1.51 s |
| mean lead vs alert | not defined | +0.07 s |
| frac_premature | (v2-only axis) | 0.367 |
| median seconds early, premature clips | | 1.12 s |
| n_gross_premature | | **0** |
| mean AP (cutoffs 0.5 / 1.0 / 1.5 s) | 0.646 (0.735 / 0.642 / 0.561) | same, threshold-free |
| too-early (legacy) | 22 of 60 | |

The legacy lead is over clips whose first crossing was useful or too early; the
v2 lead is over v2-useful clips, so the two lead figures are over different
clip sets. "+0.07 s vs alert" means the typical useful warning arrives almost
exactly at the annotated alert moment, not meaningfully ahead of it.

### 4.2 What each fix changed the number to

The four rows below come from `final_eval_read.md` §2. **They are not a
like-for-like trend**: rows 1–2 use the legacy definition, rows 3–4 use v2, and
rows 1–2 are on the old, unrepresentative split.

| # | row | useful-warning | FA | lead | definition / partition | status |
|---|---|---|---|---|---|---|
| 1 | v1-biased headline (thr 0.60/c8) | 0.750 (45/60) | 0.167 | 1.67 s | legacy; 65-clip non-representative split; **selected on the test set** | superseded |
| 2 | clean protocol, still broken (thr 0.70/c8) | 0.633 (38/60) | 0.200 | 1.62 s | legacy; selection clean, **partition still broken** | superseded |
| 3 | `train_val_v2` estimate, this config | 0.612 | 0.196 | 1.83 s | v2; fixed partition and metric | validation estimate |
| 4 | **`eval_v2` actual (this read)** | **0.650** (39/60) | **0.167** | **1.51 s** | v2; fixed partition and metric, held out | **final** |

The like-for-like legacy figure on the fair held-out set is **0.317**, far
below 0.633 and 0.750. Under v2 it is 0.650. Neither 0.750 nor 0.633 may be
quoted against the `eval_v2` result without naming the definition and the
partition.

Validation to held-out: useful 0.612 → 0.650, FA 0.196 → 0.167, lead
1.83 → 1.51 s, AP 0.630 → 0.646. "The right reading is 'the validation estimate
held up,' not 'the model is better than validation suggested'"; the moves are
inside the established noise (`final_eval_read.md` §2).

---

## 5. What the targets were and why they were not met

### 5.1 The targets and when they were set

- **Mean lead / mTTA of roughly 2–6 s.** The 2–6 s range first appears in the
  project's metrics document (`metrics.md`, added in commit `2269ae6`,
  2026-07-14) as the target for the *threshold-engine* anticipation evaluation,
  alongside detection ≥ 0.75 at false-alarm ≤ 0.20 (`design/metrics.md` §2,
  "Expected good outcome"). It was later carried over as the lead target for the
  learned risk head (`design/metrics.md` §9: "mean lead 2–6 s (committed result:
  1.67 s — short)").
- **Useful-warning ≥ 0.75** for the learned head, with a "committed result:
  0.750 — met" (`design/metrics.md` §9). The **0.70** hard gate in the kappa
  decision rule comes from `docs/handoff/KAPPA_RETRAIN.md` §5 (committed
  2026-08-19), with 0.75 "the stated target."

Every one of these was set under conditions since found broken: before the
selection-on-eval finding, on a split that held every long-window clip, and
under a metric that scored continuous warning as no warning. `operating_point_selection.md`
§2 makes the point directly about the 0.70 floor: it "was never independently
re-validated as achievable, or even as the right number, under any of those ...
corrected conditions."

### 5.2 Were they met?

| target | source | result | met? |
|---|---|---|---|
| FA ≤ 0.20 | `design/metrics.md` | 0.167 (10/60) | **yes, point estimate** (see §6: the interval reaches ~0.28) |
| useful-warning ≥ 0.75 | `design/metrics.md` §9 | 0.650 v2 / 0.317 legacy | **no** (0.10 short under v2) |
| useful-warning ≥ 0.70 | `KAPPA_RETRAIN.md` §5 | 0.650 v2 / 0.317 legacy | **no** (0.05 short under v2) |
| mean lead 2–6 s | `design/metrics.md` §9 | 1.51 s (v2, vs event) | **no** (below the 2 s floor) |

Selection also could not satisfy the rule as written. On `train_val_v2` the
documented rule (FA ≤ 0.20, useful ≥ 0.70, lead ≥ 1.0 s, zero gross-premature)
had **zero survivors across all 45 cells and all three seeds**; seed 1234 at
0.70/c8 (0.612, gap 0.088) was carried forward "by explicit instruction" as a
documented near-miss, not as the rule succeeding
(`operating_point_selection.md` §1–2).

### 5.3 Why the lead is short: what the data supports, and what it does not

**What the data does establish.** The lead-time shortfall is not primarily a
feature or architecture failure, for these structural reasons:

- The dataset's own annotated windows are short. Across all 750 positives the
  window (`time_of_event − time_of_alert`) has **mean 1.60 s, median 1.43 s**,
  range 0.033–4.466 s (`label_window_audit.md` §2); the v2 pool of 745 has mean
  1.589 s and median 1.433 s (`repartition_results.md` §3). Only 63 positives
  exceed 3 s. (The brief for this report put the median at about 1.6 s; the
  source documents give 1.6 s as the *mean* and 1.43 s as the median.)
- The training objective cannot request more lead than that. `alert_weights()`
  zeroes every frame outside `[alert_t, event_t]`
  (`anticipation_loss.py:104-110`), and on the real training pool the loss
  requests about 0.67 s at kappa 1.0 (mean window 1.596 s), not the documented
  1.46 s. Requested lead falls with kappa (0.5 → 0.732 s ... 3.0 → 0.448 s)
  (`repartition_results.md` §6).
- The measured leads sit at the annotated-window scale, not above it: 1.59,
  1.62, 1.67 s under the biased regime and 1.51 s on `eval_v2`, against a
  mean window of 1.587 s on that split. Under the old partition they sat near the
  *training* window mean (1.41 s), and nowhere near the old eval's own 3.51 s
  mean, which is what one would expect if the objective, and not the features,
  set the ceiling (`label_window_audit.md` §5).
- The kappa sweep moved lead only 1.59–1.67 s while mean AP stayed flat
  (`design/metrics.md` §9 note).

So the fair statement is: **a 2–6 s mean lead is not something this dataset's
labels ask any model to produce.** The annotated window averages 1.6 s.

**What it does not establish, and I would not claim in a viva.** "Data-driven
ceiling, not a model deficiency" is supported at the level of the *objective*,
not proven at the level of the *model*:

1. The v2 lead is measured from the onset of the earliest confirmed episode, and
   a warning may begin before `alert_t` and still count. Onset-to-event lead is
   therefore not capped by the annotated window. On validation, at the best of
   six seed/operating-point combinations, the median onset-to-event lead was
   2.24 s with 48.8% of fired clips inside 2–6 s; at the stricter reference point
   it was 1.44 s (`metric_fix_results.md`, "Onset-to-event lead"). That document
   concludes the 2–6 s target is "reachable for a meaningful minority to
   near-half of positives depending on operating point, not for a typical or
   median positive," and warns that both "met" and "unreachable" would
   overstate.
2. The per-clip question, whether lead scales with each clip's own window, was
   flagged as unanswerable from existing artifacts
   (`label_window_audit.md` §5) and no document I reviewed records it being
   answered.
3. Because the loss gives no signal earlier than `alert_t`, the model was never
   *trained* to warn earlier, so the data cannot show whether the features could
   support it. That is a limit on what was tested, not a finding that the
   features are the limit.

The defensible sentence is therefore: *the targets were set against a regime
later found broken; the dataset's annotated window averages about 1.6 s and the
training objective is bounded by it; the measured 1.51 s lead is consistent with
that bound; whether a model trained with a different objective could reliably
warn earlier is untested.*

---

## 6. Threats to validity

**Sample size and per-clip sensitivity.** `eval_v2` has 60 positives, so **one
clip is 1/60 = 0.0167 of useful-warning [derived]**. Reaching 0.70 would have
needed 42 clips, three more than the 39 achieved; reaching 0.75 would have needed
45, six more. The 95% Wilson interval on 39/60 is about **0.52–0.76 [derived]**,
which straddles both missed targets. For false alarms, 10/60 gives an interval of
about **0.09–0.28 [derived]**, so "FA ≤ 0.20 met" is a point estimate whose
plausible range extends above the target. The source document's own figure for
the standard error on FA over 60 negatives is "near 0.05" (`final_eval_read.md`
§2), consistent with this. These intervals are binomial only and ignore seed
variance, so they are a lower bound on the uncertainty.

**One read, no error bar by construction.** `eval_v2` was read exactly once, by
one script, for one checkpoint at one operating point, and the script refuses to
overwrite its output (`final_eval_read.md` §6). That is what keeps the number
honest, and it is also why there is no second draw, no spread and no confidence
band from repeats on the held-out set. It was dry-run on `train_val_v2` first
(reproducing 0.612 / FA 0.196 / AP 0.630 exactly) so a failed run could not
tempt a rerun.

**Seed variance is characterised, not eliminated, and is wider than the
headline text implies.** `final_eval_read.md` cites a seed sd of "~0.10–0.12" on
useful-warning. That figure comes from three seeds at fixed reference points
(`selection_fix_and_tradeoff.md` §3). At the *locked* cell (0.70 / c8) on
validation, the picture is wider:

| seeds | useful_v2 at 0.70/c8 | spread | source |
|---|---|---|---|
| 1234, 1235, 1236, 1237, 1238 (as first trained) | 0.612, 0.495, 0.864, 0.388, 0.107 | range 0.107–0.864, pop. sd 0.250 | `operating_point_selection.md` §3 |
| 1234–1237 only | 0.388–0.864 | pop. sd 0.177, mean 0.590 | same |
| 1234–1238 with valid 1238 replacing the degenerate one | 0.612, 0.495, 0.864, 0.388, 0.612 | sd 0.159 | `ensemble_calibration_results.md` §2 |

Two consequences. The seed-1234 result on `eval_v2` (0.650) is a single draw
from a distribution roughly this wide, and other seeds at this exact operating
point would have produced very different useful-warning, largely by firing more
or less freely (seed 1236 reaches 0.864 at FA 0.431). And the operating-point
choice was itself a single-seed choice. The 0.088 near-miss gap is smaller than
the seed spread, so seed 1234 clearing or missing 0.70 says little about the
architecture. Mean AP is much tighter (0.58–0.68 across converged seeds), which
supports "the models rank risk similarly, and differ in how they fire."

**The metric was changed after seeing what it would gain.** Covered in §2.3 and
stated by the design document itself. The mitigation is real but partial: the
justification was written and committed before implementation, the change is
purely a superset (0 of 45 violations), the legacy score (0.317) is always
reported beside it, FA is unchanged by construction, and prematurity is
reported separately. A sceptic can still argue the corrected definition was
adopted because the old one was disappointing. The honest answer is that the
diagnosis motivated it and the operational argument justifies it; the reader
should weigh the legacy 0.317 alongside the 0.650.

**Under v2, "useful" does not mean "early."** The mean lead vs alert is
+0.07 s, and 36.7% of positives have their earliest episode starting before
`alert_t` (median 1.12 s early). The system warns at roughly the annotated
moment; it does not warn ahead of it. The zero gross-premature count says this
is not the always-on failure of the old threshold engine (median 14.4 s
premature), but it says nothing about whether ~1.5 s is enough for a driver.

**Repeated looks at the validation set.** `train_val_v2` chose early stopping,
the checkpoint, the operating point, and was the scoring set for the ensemble
and calibration tests. The ensemble document notes that taking a maximum over 15
cells on the set everything was tuned on is "optimistic for every row, the
baseline included." The `eval_v2` read shows the estimate held up (0.612 vs
0.650), which is reassuring but a single agreement, not a guarantee.

**Residual indirect exposure of hyperparameters to old eval clips.** Kappa 1.0,
hidden 64, the feature set and `pre_alert_weight` 0.5 were chosen in the v1 era
on the v1 eval split, and were carried into the v2 configuration rather than
re-derived (`clean_protocol_results.md` notes kappa was re-selected on
`train_val`, chose 0.5 by a negligible margin, and the headline config kept 1.0).
The v1 `eval` directory contains 12 clips that v2 freezes as `eval_v2`
(`repartition_results.md` §3, guard table), so 12 of the 120 held-out clips were
in a set that influenced architecture-level choices. The effect is probably
small; it is not zero, and I have not seen it quantified.

**Things not redone.** Feature ablations and the capacity check were not redone
under the clean protocol (§3.4). The 15 `dev` clips have no extracted features
and were not part of any of this. The hard-coded `MEAN_ALERT_LEAD_S = 3.49`
in `compare_kappa.py:43` and `train_risk_head.py:800` has since been fixed
(`3a4b349`), but the committed "requested lead" column of
`eval/kappa_comparison.md` predates the fix and stays stated against the wrong
population until that table is regenerated. And the source documents describe one dataset, one
annotation set (Nexar's own `time_of_alert`), one architecture (a 64-unit GRU on
112-dimensional features).

**Small demo sets.** The dashboard demo (six `train_val_v2` clips, after the
first pool was found to have 6 of 8 clips in the model's own training set) is
illustrative only and is not evidence (`instability_and_checkpoint_audit.md` §3).

---

## 7. Viva-ready answer

*"What is A3PS's actual performance, and why should I trust that number?"*

On a 120-clip held-out split that shares no clips with anything used for training,
early stopping, checkpoint selection or operating-point choice, A3PS issues a
useful warning on 39 of 60 collision clips, 65%, at a false-alarm rate of 16.7%
(10 of 60 negatives), with a mean lead of about 1.5 seconds and a mean AP of
0.65 against a chance level of 0.50. That is under our corrected metric, which
credits a warning that is active during the driver's actionable window; under
the original stricter first-crossing rule the same behaviour scores 32%, and I
report both. I would not present it as meeting our original targets: the
useful-warning goal of 70–75% and the 2–6 second lead goal were both missed, and
only the false-alarm goal was met, as a point estimate whose confidence interval
reaches about 28%. You can trust the number because of how it was produced: I
found that my earlier 75% had been selected on the test set, that the test set
was a hand-truncated slice containing every long-warning-window clip, that the
metric scored continuous warnings as no warning, and that checkpoint selection
could silently keep a chance-level model; I fixed each, wrote the metric
justification before changing any code, tested ensembles, calibration and
`pre_alert_weight` and reported that none of them beat the baseline beyond noise,
and read the final split exactly once, after freezing the checkpoint, threshold
and definition. What I would ask you to hold it to: it is one seed on 60
positives, so one clip is 1.7 points and the interval is roughly 52–76%; other
seeds at the same operating point ranged much wider on validation; and the 1.5
second lead reflects a dataset whose annotated warning window averages about 1.6
seconds and a training objective bounded by it, so it shows what this labelling
supports, not what a differently-trained model could do.
