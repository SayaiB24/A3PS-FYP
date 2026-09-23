# Why positives fire "too early" — diagnosis, not tuning (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document. Everything
below is `risk_gru_k1p0_v2_s1234.pt` / `_s1235.pt` / `_s1236.pt` scored on
`data/features/train_val_v2` only, using `evaluate(..., dump_rows=True)`
([train_risk_head.py:141-240](../../scripts/train_risk_head.py)) added for this
task. No hyperparameter was tuned, no model retrained, no operating point
selected, no scoring rule changed.

**Verdict up front: (A).** The model is anticipating a real, gradually
developing signal. The "too early" verdicts are overwhelmingly a single
continuous rise that starts slightly before the annotated `alert_time_s` and
stays elevated through the correct window — not noise, not clip-start
saturation, and not a degenerate output. The scoring rule then discards that
correct window entirely because it only ever looks at the *first* threshold
crossing. Section 6 quantifies how much useful-warning that costs.

## 1. Fire-time distribution (all 103 validation positives, seed 1234)

At each operating point, every positive falls into exactly one bucket:

| threshold/confirm | useful | too_early | missed (never fires) | late (fires after event) |
|---|---|---|---|---|
| 0.80 / 8 | 26 | 21 | 55 | 1 |
| 0.60 / 5 | 26 | 55 | 19 | 3 |

Looser scoring does not raise useful-warning at all here (26 in both cases) —
it only converts "missed" into "too_early". That is itself informative: the
model *is* producing signal on most of those clips, just not landing it inside
`[alert_t, event_t]` under the current scoring.

**Offset from `alert_t`** (`first_fire_t − alert_t`, seconds; negative = fired
before `alert_t`), over every positive that fired at all:

| op point | min | Q1 | median | Q3 | max |
|---|---|---|---|---|---|
| 0.80 / 8 | −3.13 | −0.73 | +0.32 | +0.77 | +2.41 |
| 0.60 / 5 | −5.10 | −1.70 | −0.60 | +0.47 | +3.30 |

At the stricter point the median fire is already *after* `alert_t`. At the
looser point the median is 0.6 s early — inside the pool's own 1.6 s mean
window, i.e. a timing error of well under one window-length, not a
qualitatively different behaviour.

## 2. Where the early fires land

For the `too_early` positives specifically:

| op point | n | fraction of clip elapsed at first fire (Q1/median/Q3) | seconds before `alert_t` (mean/median) | fired in first 15% of clip |
|---|---|---|---|---|
| 0.80 / 8 | 21 | 0.72 / 0.75 / 0.81 | 1.03 / 0.78 | **0/21** |
| 0.60 / 5 | 55 | 0.61 / 0.70 / 0.79 | 1.73 / 1.33 | **0/55** |

**Zero early fires land in the first 15% of the clip, at either operating
point.** They cluster at 60–88% of the way through — i.e. in the final few
seconds, close to `alert_t`, not at clip start. This alone rules out "the model
fires on anything, regardless of content": a content-blind detector would show
no relationship between fire position and clip progress.

Correlation of fire-fraction with window length: **+0.62** (thr 0.80/8), +0.36
(thr 0.60/5) — longer-window clips fire relatively later as a *fraction* of the
clip, consistent with the model tracking something that develops on the
dataset's own timeline rather than a fixed clip-relative habit. Correlation
with peak probability: ≈0 at both points — how early a clip fires is unrelated
to how confident the model ever gets on it.

## 3. The risk curves

`eval/early_firing_curves_thr080.png` — nine clips scored by seed 1234 at
thr 0.80/confirm 8: 2 useful, 2 too_early, 2 missed, 2 clean negatives, 1 false
alarm.

Every positive curve (useful, too_early, and missed alike) shows the same
shape: **near zero for most of the clip, then a smooth multi-second rise that
peaks close to `alert_t`/`event_t`.** None start high. None are noisy spikes
uncorrelated with position — clip `14` and `143` (useful) ramp from ~0.02 at
clip start to ~0.85–0.9 by `event_t`; clip `1013` and `104` (too_early) show
the identical ramp shape, just crossing 0.80 a fraction of a second to ~1 s
before the green `alert_t` line instead of after it.

Clip `1013` is the clearest case and is traced frame-by-frame in §6: its
probability is **continuously above 0.60 from 17.33 s onward**, straight
through `alert_t` (18.83 s) and `event_t` (19.3 s) to the end of the clip. One
uninterrupted rise, scored `too_early` purely because the 5-frame confirm
window closed before `alert_t`.

The two `missed` clips are genuinely weaker: `10` produces only a brief spike
that clears 0.8 for less than the 8-frame confirm requirement (a real near-miss
on the confirm debounce, not absence of signal); `1011` never exceeds ~0.6 at
all before the clip ends just after `event_t`. Negatives `1041`/`1156` show the
same late-clip rise shape as positives, peaking in the 0.5–0.7 range without
crossing 0.8 — partial, not zero, signal on hard negatives. `1225` (a false
alarm) shows a clean single rise-and-fall with no relation to any annotated
time (negatives have none), peaking at 0.85 mid-clip.

**Nothing here looks saturated or content-blind.** Every curve, across every
verdict, is doing the same qualitative thing: quiet, then a genuine multi-second
build that tracks the clip's own timeline.

## 4. Separation and the seed spread

Peak probability, positives vs. negatives, `train_val_v2`, no threshold applied:

| seed | best epoch | val mean AP | pos peak (mean / median / sd) | neg peak (mean / median / sd) |
|---|---|---|---|---|
| 1234 | 4 | **0.630** | 0.811 / 0.868 / 0.147 | 0.631 / 0.641 / 0.171 |
| 1235 | 13 | **0.590** | 0.789 / 0.841 / 0.195 | 0.549 / 0.559 / 0.222 |
| 1236 | 1 | **0.501** | 0.617 / 0.636 / 0.125 | 0.545 / 0.504 / 0.098 |

Chance on this exactly-balanced (103/102) validation set is mean AP ≈ 0.50.
**Seeds 1234 and 1235 show real, substantial separation** (pos median 0.84–0.87
vs. neg median 0.56–0.64). **Seed 1236 is indistinguishable from chance**
(0.501) — its positive and negative peak distributions barely separate (median
0.64 vs. 0.50).

### Seed 1236, investigated

Full validation history (`eval/risk_gru_history_k1p0_v2_s1236.json`), scored at
the training loop's default point (threshold 0.5, confirm 3, `fa_target=0.20`):

| epoch | val useful | val FA | val mean AP |
|---|---|---|---|
| **1 (selected)** | **0.340** | **0.402** | 0.501 |
| 2 | 0.262 | 0.608 | 0.598 |
| 3 | 0.146 | 0.794 | 0.646 |
| 4 | 0.282 | 0.510 | 0.613 |
| 5 | 0.233 | 0.853 | 0.646 |
| 6 | 0.204 | 0.784 | 0.648 |
| 7 | 0.301 | 0.588 | 0.631 |
| 8 | 0.184 | 0.676 | 0.643 |
| 9 | 0.252 | 0.676 | 0.634 |

**No epoch of this run ever reaches the FA ≤ 0.20 gate** — the *minimum* FA
across all 9 epochs is 0.402. The selection rule
(`train_risk_head.py:391-404`) is `rank = (ok_fa, useful, −FA)`; since
`ok_fa` is `False` for every epoch here, the tuple's first element ties across
the board and selection falls through to whichever epoch has the best
`useful`/`FA` pair. That is epoch 1 — not because it fires rarely, but because
it happens to have **both** the highest `useful` (0.340) **and** the lowest FA
(0.402) among nine epochs that are all non-compliant.

This is a real but narrow failure mode: the training loop's per-epoch
selection metric is the *operating-point* useful/FA pair at a fixed threshold,
not mean AP, and the two disagree here. Epochs 2–9 have **higher** mean AP
(0.60–0.65, comparable to seeds 1234/1235) but **worse** useful/FA at
threshold 0.5 — likely because as training proceeds the model becomes more
confident in both directions, and FA grows faster than useful at this
particular fixed threshold for this particular seed's weight trajectory.
Epoch 1 wins by default, not by genuine quality.

**Is epoch 1's output degenerate/near-constant?** No, not literally: over
24,543 scored frames its probability spans 0.062–0.896 (mean 0.248, sd 0.155),
with real per-frame variance. But that variance carries almost no label
information — mean AP 0.501 means ranking clips by peak probability sorts them
no better than a coin flip. It is **collapse to an uninformative output, not a
constant one**: the model moved off initialization but did not learn a
label-relevant signal at this seed's first epoch, and the selection rule has
no way to notice, since it never looks at AP.

## 5. Scoring semantics

### Does a too-early fire on a positive also count as a false alarm?

**No.** `false_alarm_rate` is accumulated only over label-0 clips:

```python
# train_risk_head.py:172-181 (evaluate())
if c["label"] == 0:
    fa_neg += int(fa is not None)
    ...
    continue
```

A positive clip never reaches this branch regardless of when it fires — it
falls through to the `label == 1` path below, where it is scored as `useful`,
`too_early`, `missed`, or `late`, and `fa_neg`/`false_alarm_rate` (computed at
`train_risk_head.py:233` as `fa_neg / n_neg`, `n_neg` counting only label-0
rows at `train_risk_head.py:226`) never sees it. A too-early positive costs
`useful_warning_rate` only, nothing else.

### After firing too early, can a clip still earn a useful warning later?

**No.** `first_alert_time` returns on the first sustained run it finds and
never looks further:

```python
# a3ps/risk/anticipation_loss.py:224-250
def first_alert_time(probs, t, threshold, frames_to_confirm=1):
    over = (probs >= float(threshold)).tolist()
    need = max(1, int(frames_to_confirm))
    run = 0
    for i, hot in enumerate(over):
        if hot:
            run += 1
            if run >= need:
                return float(t[i - need + 1])   # <- returns immediately
        else:
            run = 0
    return None
```

The loop terminates at the first `run >= need`. Whatever the probability curve
does afterward — including staying above threshold continuously through
`[alert_t, event_t]` — is never inspected. `evaluate()` then classifies the
clip using this single `first_alert_t` (`train_risk_head.py:190-199`): if it is
before `alert_t`, the clip is `too_early` and stays `too_early`, full stop.
**One early spike permanently forfeits a clip the model would otherwise have
scored correctly**, exactly as the task's hypothesis states.

## 6. Recoverable headroom

For every `too_early` positive, checked whether **any maximal run of
consecutive over-threshold frames has ≥ `confirm` of its frames falling inside
`[alert_t, event_t]`** — i.e., whether the window itself contains a qualifying
sustained crossing, whether or not that run started earlier. (A narrower check
— only counting a *new* run that starts inside the window — undercounts,
because most too-early clips are one continuous run that simply began before
`alert_t` and never dropped; see clip `1013` below.)

| op point | too_early | recoverable | current useful | ceiling if pre-alert firing were suppressed |
|---|---|---|---|---|
| 0.80 / 8 | 21 | **15** | 26/103 = 0.252 | **41/103 = 0.398** |
| 0.60 / 5 | 55 | **52** | 26/103 = 0.252 | **78/103 = 0.757** |

At the looser operating point, **52 of 55 too-early clips are cases where the
window itself is already correctly covered** — the "early firing" is just the
leading edge of a rise that persists through the whole actionable period.
Clip `1013`, frame-by-frame (`alert_t=18.833`, `event_t=19.300`, thr 0.60):

```
17.33  p=0.782  over          <- run already sustained here
18.83  p=0.856  over   ALERT
19.33  p=0.910  over   EVENT
19.73  p=0.909  over          <- clip ends still over threshold
```

One continuous crossing from 17.33 s to the end of the clip. Scored
`too_early` because the confirm-window closed at 17.33 s, 1.5 s before
`alert_t` — despite covering `[alert_t, event_t]` completely.

**This ceiling is a diagnostic upper bound, not a proposal.** It assumes a
hypothetical rule that ignores all firing before `alert_t` when scoring
positives; it says nothing about how such a rule would affect negatives (which
have no `alert_t` to anchor to), and it is not something this task implements,
retrains toward, or recommends selecting. At the looser point the ceiling
(0.757) would clear the documented ≥0.70 gate; at the stricter point (0.398) it
would not. That gap between operating points is itself informative: how much
headroom exists depends heavily on where the threshold sits, which is a
downstream decision, not one made here.

## Summary

The model is not firing blindly or near-chance (seeds 1234/1235; seed 1236 is
the documented exception and is explained, not hidden). Its risk curves rise
smoothly over multiple seconds and peak close to the annotated event, on
positives and — partially — on hard negatives alike. Most of what today's
metric calls "too early" is a correctly-timed, sustained rise that starts
slightly ahead of the annotated `alert_time_s` and is then thrown away entirely
by a first-crossing-only scoring rule that cannot credit anything after the
first mistake. That is a scoring-and-possibly-training-emphasis question, not
evidence the features or architecture lack signal.
