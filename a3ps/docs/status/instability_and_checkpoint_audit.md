# Checkpoint-safety gap, seed instability diagnosis, demo decontamination (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document. No
operating point is selected here.

## 1. The checkpoint-safety gap, closed properly

`docs/status/operating_point_selection.md` found seed 1238's selected
checkpoint had mean AP 0.490 (chance) despite passing the primary selection
rule's `fa_target` check — the Step 9 near-chance fallback never triggered
for it, because that fallback only activates when `best["rank"][0] == 0`
(**no** epoch ever met `fa_target`). Seed 1238 had one that did (epoch 1, FA
0.127), so the primary rule ran normally and picked a chance-level epoch
anyway. Meeting the FA gate was being treated as sufficient evidence of a
legitimate result; it is not.

**Fix:** `train()` now runs one more, unconditional check after selection
settles, regardless of which rule produced the final pick: if the selected
epoch's mean AP is within `NEAR_CHANCE_AP_MARGIN` (0.05) of chance, a new
`degenerate` flag is set `True` and the checkpoint's `extra` dict gets
`degenerate_checkpoint: True` — whether or not `fa_target` was ever met,
whether or not the earlier fallback fired. This does not retrain or reselect
anything; it makes a chance-level checkpoint impossible to consume silently.

Added `test_train_flags_degenerate_even_when_fa_target_met`, which reproduces
the exact seed-1238 gap deterministically (a scripted epoch that meets
`fa_target` and is chance-level beats a scripted better epoch on the primary
rule): confirms `fallback_used` stays `False` (correct — its own scope didn't
change) while `degenerate` is now `True`. Full suite: 187 passed.

### Re-audit of every checkpoint produced so far

| checkpoint | mean AP | fa_target_met | fallback_used | degenerate (corrected) |
|---|---|---|---|---|
| `risk_gru_k1p0_v2_selfix_s1234.pt` | 0.6296 | False | False | **False** |
| `risk_gru_k1p0_v2_selfix_s1235.pt` | 0.5898 | False | False | **False** |
| `risk_gru_k1p0_v2_selfix_s1236.pt` | 0.6478 | False | True | **False** |
| `risk_gru_k1p0_v2_selfix_s1237.pt` | 0.5803 | False | False | **False** |
| `risk_gru_k1p0_v2_selfix_s1238.pt` | 0.4897 | **True** | False | **True** |
| `risk_gru_k1p0_v2_selfix_paw1p0_s1234.pt` | 0.5926 | False | False | **False** |
| `risk_gru_k1p0_v2_selfix_paw2p0_s1234.pt` | 0.5780 | False | False | **False** |
| `risk_gru_k1p0_v2_selfix_paw4p0_s1234.pt` | 0.5861 | False | True | **False** |

Chance on `train_val_v2` is 0.5024 (103/205 positives); the near-chance
threshold is 0.5524. **Only seed 1238 is degenerate. No other checkpoint
produced so far — across all 5 base seeds and all 3 `pre_alert_weight`
values — was silently affected.** `paw2p0_s1234` is the closest of the
non-degenerate ones (0.578, a 0.026 margin above the threshold), worth
noting as the least comfortable "real" result on file, but it is not flagged.

## 2. Diagnosing the instability

### Determinism check: training is bit-for-bit reproducible

Reran seed 1234 with identical hyperparameters and command. Compared its
12-epoch history (`train_loss`, `val_useful`, `val_mean_AP`,
`val_false_alarm`, every epoch) against the already-committed
`eval/risk_gru_history_k1p0_v2_selfix_s1234.json`:

```
same length: True 12 12
bit-for-bit identical (==): True
```

**Training is fully deterministic.** `torch.manual_seed(args.seed)` and
`random.seed(args.seed)` are set once at the top of `main()`
(`train_risk_head.py`), model weight initialisation draws from the seeded
torch RNG, the per-epoch training shuffle uses `random.Random(args.seed +
epoch)` (deterministic given `args.seed`), there is no `DataLoader` and no
CUDA path (`train_risk_head.py` is CPU-only by construction, per its own
module docstring), and no other randomness source is touched during
training. **The determinism check does not implicate reproducibility as the
root cause** — it rules it out. "Seed 1234" reliably means the same run,
every time.

### Collapse-rate classification: 1 of 5 base seeds, and it is a selection
artifact, not a training failure

Eight training runs exist under this recipe so far: seeds 1234/1235/1236/
1237/1238 at `pre_alert_weight=0.5`, plus seed 1234 at
`pre_alert_weight ∈ {1.0, 2.0, 4.0}`. Per the corrected check in Section 1,
**one of the eight — seed 1238 — is degenerate. That is 1/5 = 20% among the
five distinct-seed runs at fixed hyperparameters, or 1/8 = 12.5% across every
run on file including the `pre_alert_weight` sweep.**

**But looking at the actual training curves changes what that number means.**
Full mean-AP trajectory, every epoch, all five base seeds:

| seed | epoch-by-epoch mean AP |
|---|---|
| 1234 | 0.490 0.592 **0.681** 0.630 0.652 0.630 0.638 0.644 0.610 0.654 0.628 0.640 |
| 1235 | 0.506 0.520 0.592 0.630 0.514 0.603 0.598 0.571 0.575 0.587 0.632 0.634 0.590 0.644 0.615 0.632 0.608 **0.655** 0.609 0.624 0.602 |
| 1236 | 0.501 0.598 0.646 0.613 0.646 **0.648** 0.631 0.643 0.634 |
| 1237 | 0.504 0.594 0.621 0.580 0.652 0.633 0.628 0.609 **0.662** 0.639 0.607 0.629 |
| 1238 | 0.490 0.519 0.605 0.626 0.631 0.645 0.615 0.618 **0.645** (peaks at epoch 9, 0.6307 seen twice) |

**All five seeds show the identical shape**: chance at epoch 1 (0.49–0.51, as
expected — the model has barely moved from initialisation), a rapid rise
over epochs 2–3, then a plateau in the 0.58–0.68 range with ordinary
epoch-to-epoch noise (±0.02–0.05) for the rest of the run. **None of the five
runs — including seed 1238 — show a flat, stuck-at-chance curve, and none
show a rise followed by degradation.** Seed 1238's *training* is
indistinguishable from the other four: by epoch 9 its mean AP is 0.645,
squarely inside the other seeds' range.

**Seed 1238's checkpoint is degenerate not because training collapsed, but
because the selection rule picked epoch 1** — the one epoch of the run still
near initialisation — purely because that epoch's low confidence
(everything undertrained fires weakly) happened to keep its false-alarm rate
under 0.20 at the training loop's threshold/confirm scoring point, and
meeting that gate outranks every later, properly-trained epoch's
non-compliant score on the primary rule's tuple comparison. **This is a
selection-rule fragility, not a training-stability problem**, and it is
exactly what Section 1's fix now catches directly — before this fix, it was
silent regardless of how the curve itself looked.

**Revised framing of the collapse rate: 0 of 5 base seeds show genuine
training collapse (declining or stuck-at-chance curves). 1 of 5 produced a
degenerate *checkpoint selection* despite healthy training underneath.**
These are different failure modes and the difference matters for what to fix
next — Section 1 already fixes the one that actually occurred.

### Where the real instability lives: calibration, not convergence

If training itself is this consistent (mean AP 0.58–0.68 on every converged
seed, a span of 0.10), why did `docs/status/operating_point_selection.md`
find `useful_warning_rate_v2` ranging 0.388–0.864 — over four times wider —
across the same four converged seeds at one fixed operating point (thr
0.70/confirm 8)?

The mean-AP spread measures *ranking quality*: how well each model orders
risky clips above ordinary ones, independent of any threshold. The
useful-warning spread measures something additionally sensitive to *where
each model's raw probabilities sit* relative to one shared, hardcoded
threshold (0.70). Two models with near-identical ranking quality can fire at
very different rates against the same fixed cutoff if their output
probabilities are calibrated differently — one seed's "risky" clips might
cluster around 0.75, another's around 0.55, even if both rank clips in
nearly the same order. That is consistent with what was measured: tight mean
AP, wide useful-warning-at-a-fixed-threshold.

**This reframes the instability question.** The four converged seeds are not
four models of wildly different quality; they are four models of similar
quality whose absolute output scale happens to differ, interacting with one
threshold chosen without per-seed calibration. This does not by itself prove
the `pre_alert_weight`/threshold interaction is the cause, but it does
explain why AP was stable while useful-warning was not, which the raw
0.107–0.864 headline number alone did not make clear.

### The batch_size=8 / learning-rate hypothesis

**Not strongly supported by the training curves as measured.** If gradient
noise from a small batch size (8, on ~1,160 training clips) or too high a
learning rate were destabilising *optimisation*, the expected signature would
be large epoch-to-epoch swings, divergence, or a curve that rises and then
degrades. None of that appears: every seed's post-epoch-3 plateau is smooth,
within a normal ±0.02–0.05 band, and none trend downward over their
remaining epochs (12, 21, 9, 12, and 9 epochs respectively — seed 1235's
21-epoch run is the longest look available and stays just as flat as the
shorter ones). The curves look like ordinary, well-behaved convergence to a
local optimum, repeated five times with five different optima of similar
quality. This is *consistent with* batch size and learning rate being fine
for this dataset size; it does not rule out that a different setting could
reduce the *calibration* spread discussed above, which is a different
question this task was not asked to test. **Not acted on, per instruction —
reported only.**

## 3. Demo decontamination

The previous demo pool (`docs/status/operating_point_selection.md` §4) drew
from `data/features/eval`, and 6 of its 8 clips turned out to be inside seed
1234's own `train_core_v2` by clip identity.

**Selection criterion used here:** clips present in `data/features/train_val_v2`
that also have cached old-system output (`eval/anticipation/<id>/events.json`)
and a local video file — i.e. `--features data/features/train_val_v2` passed
to the unchanged `pick_auto()` logic, rather than the default
`data/features/eval`. Every clip in `train_val_v2` is, by construction of
`scripts/repartition_splits.py`, disjoint from `train_core_v2` — the three v2
splits are pairwise disjoint — so satisfying this membership criterion is
sufficient to guarantee the model never saw the clip's gradient during
training. **These are validation clips, used only for operating-point
selection and model comparison in earlier steps — not the frozen `eval_v2`
test set, which remains untouched.**

Checked the qualifying pool before building: of `train_val_v2`'s 205 clips,
11 also have cached old-system output and local video (7 positive, 4
negative) — this is the actual overlap available to draw a demo from, not
the full 205. `--clips auto --n 6` drew 6 of those 11. None of the
previously-named clips (621, 488, 1004, 690, 1085, 1261) qualify under this
constraint except 621, which was already `train_val_v2` (confirmed clean
even before this task) and is included below.

### Decontaminated demo set

Rebuilt `dashboard/clips/` with `scripts/build_dashboard_demo.py
--checkpoint notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt --features
data/features/train_val_v2 --threshold 0.70 --confirm 8 --clips auto --n 6`,
then `scripts/update_manifest.py --require-video`.

| clip | label | legacy verdict | v2 verdict | window covered (v2) | fire time | lead |
|---|---|---|---|---|---|---|
| 621 | pos | useful | useful | True | 22.80 s | 2.90 s |
| 206 | pos | useful | useful | True | 18.73 s | 2.00 s |
| 630 | pos | miss | missed | False | — | — |
| 932 | pos | useful | useful | True | 19.83 s | 0.20 s |
| 1426 | neg | clean | clean | — | — | — |
| 1564 | neg | clean | clean | — | — | — |

Legacy and v2 agree on every clip in this set too, same as the contaminated
one did — a demo pool this small still isn't where the clip-1013-style
legacy/v2 disagreement would reliably show up (see
`early_firing_diagnosis.md` and `metric_fix_results.md` for that evidence).
Clip 630 is a genuine miss under both definitions (no episode reaches
threshold/confirm anywhere relative to its window) — the first demo clip in
this project showing the model's failure mode directly, rather than only
successes. Still illustrative, not evidentiary: 6 clips is far too few to
read as a result, and the underlying operating point remains an unvalidated
near-miss (`operating_point_selection.md`).

**Not deleted:** the previous, contaminated demo clips (`488`, `1004`, `690`,
`1085`, `1261`, `1054`, `1042`) remain in `dashboard/clips/` and will still
appear in the manifest/dropdown, per the hard rule against deletion. They are
**stale artifacts from before this decontamination and should not be treated
as the current demo set** — this document is the record of which clip IDs
(621, 206, 630, 932, 1426, 1564) are the clean ones.

### Sanity check

Rebuild completed with 6/6 clips built, zero skipped, zero errors.
`update_manifest.py --require-video` completed with no error. Served locally
and verified directly: all six clips' `risk.json` and `raw.mp4` return HTTP
200; clip 630's `risk.json` parses with `verdict_v2: "missed"`,
`window_covered_v2: false`, confirming the v2 fields are present and correct
on the decontaminated build too.

`data/features/eval_v2` has not been read at any point in this task.
