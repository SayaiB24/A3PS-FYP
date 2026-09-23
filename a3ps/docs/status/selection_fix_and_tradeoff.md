# Checkpoint-selection fix, an honest seed spread, and the pre_alert_weight tradeoff (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document. Everything
below is scored on `data/features/train_val_v2` only. **No operating point is
selected here.**

This document does two unrelated things, kept separate as instructed:

1. **A bug fix** to `train_risk_head.py`'s checkpoint selection, which had
   silently kept a chance-level checkpoint for seed 1236.
2. **A tradeoff exploration** of `pre_alert_weight` on seed 1234 only — not a
   fix, not a recommendation, a curve for you to judge.

## 1. The checkpoint-selection fallback fix

### What was broken

`train()`'s primary selection rule ranks epochs by `(meets fa_target,
useful-warning, −FA)`
(`scripts/train_risk_head.py`, `train()`, the per-epoch loop). Checked against
the actual training histories: **none of the three existing seeds ever had an
epoch meet `fa_target` at the training loop's own scoring point** (threshold
0.5, confirm 3) — not just seed 1236. When no epoch is compliant, the rule's
first tuple element ties at 0 for everyone, so selection degrades to ranking
purely by useful-warning with FA as a tiebreak. For seed 1236 that picked
epoch 1 (mean AP 0.501 — chance, on a ~50/50 validation set) over epoch 6
(mean AP 0.648 — real separation), because epoch 1's lower confidence
happened to give it a marginally better useful/FA pair despite carrying
almost no label information.

### What was proposed, and what changed after investigation

The obvious fix — "whenever nothing is FA-compliant, select by mean AP
instead" — was checked against all three seeds' actual histories before
implementing. Because *no* seed ever met `fa_target`, applying that rule
uniformly would have also swapped seeds 1234 and 1235's checkpoints to
different epochs, trading real useful-warning for marginal AP gains on runs
that were never broken:

| seed | current pick | AP-fallback would pick |
|---|---|---|
| 1234 | epoch 4 (useful 0.330, AP 0.630) | epoch 3 (useful 0.117, AP 0.681) |
| 1235 | epoch 13 (useful 0.340, AP 0.590) | epoch 18 (useful 0.233, AP 0.655) |
| 1236 | epoch 1 (useful 0.340, AP 0.501) | epoch 6 (useful 0.204, AP 0.648) |

Only the third row is the bug this section exists to fix. Surfaced this
before implementing rather than applying the literal rule; asked which scope
was wanted.

### The fix as implemented

The fallback only overrides the primary pick when **both** are true: (a) the
primary pick never met `fa_target` during the run, **and** (b) the primary
pick's own mean AP is within `NEAR_CHANCE_AP_MARGIN` (0.05) of chance — where
chance is computed as the validation set's own positive prevalence (the exact
expected AP of a random ranking), not a bare 0.5. This is narrow by
construction: seeds 1234/1235's useful-first picks (AP 0.59–0.63) are nowhere
near that margin and are returned completely unchanged.

Mechanism (`train()`'s per-epoch loop and post-loop check,
`scripts/train_risk_head.py`):

```python
# NEW: computed once, before the epoch loop.
n_pos_val = sum(1 for c in val_clips if c["label"] == 1)
n_neg_val = sum(1 for c in val_clips if c["label"] == 0)
chance_ap = (n_pos_val / (n_pos_val + n_neg_val)) if (n_pos_val + n_neg_val) else 0.5
near_chance_ap = chance_ap + NEAR_CHANCE_AP_MARGIN   # 0.05

# ...inside the loop, alongside the existing primary `best` tracking...
# NEW: an independent second running best, ranked by mean AP alone. Never
# touches early stopping or the primary pick.
ap = val["mean_AP"]
if ap == ap and (best_by_ap is None or ap > best_by_ap["val"]["mean_AP"]):
    best_by_ap = {"epoch": epoch, "val": val, "state": ...}

# ...after the loop ends...
fallback_used = False
if (best["rank"][0] == 0                          # never FA-compliant
        and best["val"]["mean_AP"] < near_chance_ap  # AND near chance
        and best_by_ap is not None
        and best_by_ap["epoch"] != best["epoch"]):
    fallback_used = True
    best = {...best_by_ap..., "rank": best["rank"]}   # swap in
    if on_improve is not None:
        on_improve(best)                              # persist the swap

return best, history, fallback_used
```

`train()` now returns a third value, `fallback_used`. The checkpoint's `extra`
dict gains two booleans: `fa_target_met` (was any epoch ever FA-compliant —
derived from `best["rank"][0] == 1`, correct even after a swap since the swap
preserves the original rank) and `selection_fallback_used` (did the near-chance
override actually fire). Call sites updated: `main()`'s `train()` invocation,
and the two places in `tests/test_train_risk_head.py` that monkeypatch
`train()` with a fake (both needed a third return value).

Two new tests reproduce the fallback deterministically via a scripted
`evaluate()`, so the training dynamics don't need to be real:
`test_train_selection_fallback_picks_best_ap_not_near_chance_epoch` (the
seed-1236 pattern: near-chance pick gets overridden) and
`test_train_no_fallback_when_useful_first_pick_is_not_near_chance` (a real
useful-first pick, also never FA-compliant, is left alone — guarding the
narrow scope that was chosen). Full suite: 186 passed.

Nothing existing was deleted. `risk_gru_k1p0_v2_s1236.pt` (the old, broken
selection) stays on disk; the new checkpoints below use a `_selfix` suffix.

## 2. The three seeds, retrained under the fixed rule

Same hyperparameters as before (kappa 1.0, pre-alert-weight 0.5, pos-weight
1.0, fa-target 0.20, epochs 30, patience 8, batch-size 8, hidden 64/layers 1),
`train_core_v2` → `train_val_v2`. New files:
`notebooks/models/risk_gru_k1p0_v2_selfix_s{1234,1235,1236}.pt`.

| seed | best epoch | fa_target_met | fallback used | val mean AP |
|---|---|---|---|---|
| 1234 | 4 | False | **False** | 0.630 |
| 1235 | 13 | False | **False** | 0.590 |
| 1236 | 6 | False | **True** | **0.648** |

Seeds 1234 and 1235 are **byte-identical** to the checkpoints already
committed in Step 8 — same epoch, same numbers — confirming the fix does not
touch runs it shouldn't. Seed 1236 moved from epoch 1 (chance) to epoch 6
(mean AP 0.648), triggering the near-chance override exactly as designed.

**Seed 1236 is now a real, non-degenerate run.** Mean AP 0.648 is not only
far from chance (0.502), it is the *highest* of the three seeds — higher than
both 1234 (0.630) and 1235 (0.590).

Both definitions at the two reference operating points:

| seed | op point | useful (legacy) | useful (v2) | FA | frac_premature | n_gross_premature |
|---|---|---|---|---|---|---|
| 1234 | thr 0.80/c8 | 0.252 (26) | 0.447 (46) | 0.088 | 0.204 | 0 |
| 1234 | thr 0.60/c5 | 0.252 (26) | 0.786 (81) | 0.461 | 0.534 | 1 |
| 1235 | thr 0.80/c8 | 0.311 (32) | 0.398 (41) | 0.078 | 0.107 | 0 |
| 1235 | thr 0.60/c5 | 0.350 (36) | 0.709 (73) | 0.265 | 0.369 | 0 |
| 1236 | thr 0.80/c8 | 0.223 (23) | 0.680 (70) | 0.284 | 0.456 | 0 |
| 1236 | thr 0.60/c5 | 0.243 (25) | 0.951 (98) | 0.647 | 0.709 | 2 |

Seed 1236's v2 useful-warning is now the highest of the three at both
reference points — consistent with it having the best raw discrimination
(mean AP). It also has the highest FA and `frac_premature`, which is the same
kind of tradeoff seen throughout this project: a model that fires more freely
scores well on a loose, coverage-based metric and pays for it in false alarms.
None of this is a reason to prefer seed 1236 for anything — no operating point
or "best seed" is selected in this document.

## 3. The honest seed spread

This replaces the 0.501–0.630 spread reported in earlier documents, which
included the broken run.

**Mean AP across the three valid seeds:** 0.590, 0.630, 0.648 — **mean 0.623,
range 0.590–0.648, population sd 0.024** (sample sd 0.030). Compare to the
previous, contaminated spread (0.501–0.630, sd 0.056): the corrected spread is
both higher and roughly half as wide. The old spread's width was mostly the
gap between "trained" and "not trained," not genuine seed-to-seed variance.

**`useful_warning_rate_v2` across the three seeds, at the reference points**
(this one *is* threshold-dependent, unlike mean AP):

| op point | mean | range | population sd |
|---|---|---|---|
| thr 0.80/confirm 8 | 0.508 | 0.398–0.680 | 0.123 |
| thr 0.60/confirm 5 | 0.815 | 0.709–0.951 | 0.101 |

This spread is much wider than the mean-AP spread, because `useful_warning_rate_v2`
compounds two things that vary across seeds: raw discrimination (mean AP) and
how conservatively/liberally each seed's checkpoint fires at a fixed
threshold. It should not be read as "the model" having a ±0.10–0.12 spread in
some single underlying quality — it is the spread of *this specific
threshold/confirm choice* applied to three different weight sets.

## 4. pre_alert_weight tradeoff (seed 1234 only)

`pre_alert_weight` penalises firing before `alert_time_s`. Retrained seed 1234
at `pre_alert_weight ∈ {0.5 (baseline, already have it), 1.0, 2.0, 4.0}`, same
other hyperparameters, same splits. New files:
`notebooks/models/risk_gru_k1p0_v2_selfix_paw{1p0,2p0,4p0}_s1234.pt`.

| pre_alert_weight | best epoch | fallback used | mean AP | op point | useful (v2) | frac_premature | n_gross_premature | mean lead-from-onset (s) |
|---|---|---|---|---|---|---|---|---|
| **0.5** (baseline) | 4 | No | 0.630 | thr 0.80/c8 | 0.447 | 0.204 | 0 | 1.60 |
| | | | | thr 0.60/c5 | 0.786 | 0.534 | 1 | 2.40 |
| **1.0** | 23 | No | 0.593 | thr 0.80/c8 | **0.621** | 0.233 | 0 | 1.32 |
| | | | | thr 0.60/c5 | **0.825** | 0.466 | 0 | 1.83 |
| **2.0** | 15 | No | 0.578 | thr 0.80/c8 | **0.291** | **0.058** | 0 | 0.85 |
| | | | | thr 0.60/c5 | **0.573** | **0.223** | 0 | 1.10 |
| **4.0** | 7 | **Yes** | 0.586 | thr 0.80/c8 | 0.505 | 0.204 | 0 | 1.53 |
| | | | | thr 0.60/c5 | 0.728 | 0.437 | 0 | 1.85 |

`n_gross_premature` is 0 or 1 out of 103 at every setting tested — it was
already near-zero at the baseline (§4 of `metric_fix_results.md` found the
same), so `pre_alert_weight` has essentially no room to move it further down;
whatever this lever does, it is not fixing a gross-prematurity problem,
because there mostly isn't one to fix.

### The tradeoff is real, but it is not a clean curve

**This is not monotonic**, and that has to be stated plainly rather than
smoothed over. Naively, more `pre_alert_weight` should push onset later
(lower `frac_premature`, lower lead) at some cost to coverage (`useful_v2`).
That pattern shows up clearly between 0.5 → 2.0 (`frac_premature` 0.534 → 0.223
at thr 0.60/c5, `useful_v2` 0.786 → 0.573, lead 2.40 → 1.10) — a real and
fairly large cost paid for a real and fairly large reduction in premature
firing. But **1.0 breaks the pattern**: `useful_v2` goes *up* relative to
baseline (0.786 → 0.825) while `frac_premature` barely moves (0.534 → 0.466)
and lead drops only modestly. And **4.0 partially reverses 2.0** rather than
continuing the trend (`useful_v2` 0.573 → 0.728, `frac_premature` 0.223 →
0.437) despite being twice the pre_alert_weight.

**This is a single seed per setting**, and Section 3 just established that
`useful_warning_rate_v2` alone has a population sd of ~0.10–0.12 *across
seeds at fixed hyperparameters* — comparable in size to some of the swings
between adjacent `pre_alert_weight` values here (e.g. 1.0 → 2.0 is a 0.25 drop
at thr 0.60/c5, bigger than the seed sd; but 0.5 → 1.0's 0.04 *rise* is well
within it). Part of what looks like a `pre_alert_weight` effect could be
ordinary run-to-run training variance — which epoch early stopping happens to
land on — rather than a systematic consequence of the hyperparameter. A single
seed per value cannot distinguish these. Separating them would need multiple
seeds per `pre_alert_weight` value, which is out of this task's stated scope.

### Stated, not decided

At `pre_alert_weight = 2.0`, `frac_premature` drops from 0.534 to 0.223
(baseline vs. 2.0, thr 0.60/confirm 5) — a real reduction in how often the
model flickers a warning before the driver-relevant window opens — at a cost
of 0.213 points of `useful_warning_rate_v2` (0.786 → 0.573) and about 1.3 s of
mean lead-from-onset (2.40 → 1.10 s). Whether that trade is worth paying
depends on how much operational weight premature flickering carries versus
warning coverage and lead time — a judgment this document does not make.
`pre_alert_weight = 1.0` is a different kind of data point: it does not clearly
trade anything for anything in this single-seed measurement, which is itself
informative about how noisy this sweep is at n=1 per setting.
