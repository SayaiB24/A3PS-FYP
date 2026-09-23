# The useful-warning definition: what it measures and why it is being corrected

This document is written and committed **before any scoring code changes**.
The order matters: the justification for changing a self-defined metric has to
exist independently of, and prior to, any number the change produces. If the
justification were written after seeing that the new definition scores better,
it would not be a justification at all.

## 1. What the current rule does

`useful-warning` is computed from a single timestamp, `first_alert_t`, produced
by `first_alert_time()`:

```python
# a3ps/risk/anticipation_loss.py:224-250
def first_alert_time(probs, t, threshold, frames_to_confirm=1):
    """Timestamp of the first sustained threshold crossing, else None."""
    over = (probs >= float(threshold)).tolist()
    need = max(1, int(frames_to_confirm))
    run = 0
    for i, hot in enumerate(over):
        if hot:
            run += 1
            if run >= need:
                return float(t[i - need + 1])   # <- returns on the FIRST hit
        else:
            run = 0
    return None
```

The loop stops at the first confirmed run. `evaluate()` then classifies the
whole clip from that one timestamp
(`scripts/train_risk_head.py:190-199`, current at time of writing): `useful` if
it falls in `[alert_t, event_t]`, `too_early` if before `alert_t`, `late` if
after `event_t`, `missed` if the model never fires at all. Whatever the
probability curve does after the first crossing — including staying above
threshold continuously through the entire actionable window — is never
inspected.

`docs/status/early_firing_diagnosis.md` traced clip `1013` frame by frame under
this rule (`alert_t = 18.833`, `event_t = 19.300`, threshold 0.60):

```
17.33  p=0.782  over          <- run already sustained here
18.83  p=0.856  over   ALERT
19.33  p=0.910  over   EVENT
19.73  p=0.909  over          <- clip ends still over threshold
```

One continuous crossing, unbroken, from 17.33 s to the end of the clip. It
fully covers `[alert_t, event_t]`. It is scored `too_early`, and nothing about
that verdict changes even though the model was actively warning for the entire
actionable window plus 1.5 s beforehand. Under the current rule this clip is
indistinguishable from a clip on which the model never fired at all — both
score zero toward `useful_warning_rate`.

The diagnosis found this is not rare: at threshold 0.60 / confirm 5, 52 of 55
`too_early` positives on `train_val_v2` have a qualifying sustained run that
also covers part of `[alert_t, event_t]`; at threshold 0.80 / confirm 8, 15 of
21 do.

## 2. Why this is operationally wrong

A driver-assistance system does not fire once and fall silent. A warning stays
active on the display or the alert channel while the hazard persists. The
question a useful-warning metric should answer is: **"was a warning active
during the window in which the driver could still act?"** — not "did the very
first threshold crossing of the clip happen to land inside that window."

Under the current rule, a system that displays a warning throughout the entire
actionable window — arguably the best possible behaviour, warning early and
never dropping it — scores **identically** to a system that never warns at
all. Both are `0` toward `useful_warning_rate`. That equivalence is wrong on
its face, independent of which checkpoint it happens to help or hurt: a metric
that cannot distinguish "warned continuously through the correct window" from
"never warned" is not measuring what it claims to measure.

This is the only argument this change rests on. It is not "the new number is
higher" — §6 of `early_firing_diagnosis.md` is explicit that the headroom
figures above (15/21, 52/55) were already known, from diagnosis, before this
document or any code change. Recording that plainly, rather than treating the
prior knowledge as if it weren't there, is the point of this section existing.

## 3. `useful-warning` is this project's own metric

The official Nexar competition metric is the AP-at-cutoffs figure
(`AP @ -500/1000/1500 ms`, already reported separately in every eval table).
It is a ranking metric over peak probability against a fixed pre-event cutoff
and is **completely unaffected by this change** — nothing below touches it.
`useful_warning_rate` and `false_alarm_rate` are metrics this project defined
itself, to answer an operational question the AP figures don't: not just
"does the model rank risky clips higher" but "would a driver watching this
system's output actually have gotten a usable warning." Fixing a self-defined
operational metric, with the justification made explicit and public, is a
legitimate part of developing that metric. Changing it without saying so, or
changing it because it scores better, would not be.

## 4. The corrected definition

An **alert episode** is a maximal run of consecutive frames with `p ≥
threshold`, and it must be at least `confirm` frames long to count at all (the
existing debounce). An episode is **confirmed at its own first frame** — the
same crediting convention `first_alert_time` already uses and explains in its
own comment: "the model actually knew at i-need+1, and charging it the
debounce delay would understate its lead time"
(`a3ps/risk/anticipation_loss.py:244-246`). It is **active** from that first
frame through the run's last frame (i.e. until `p` drops below `threshold`).

**Correction, found during implementation:** an earlier draft of this
definition read "confirmed at the frame where the run reaches `confirm`
consecutive frames" — i.e. the frame at which the debounce completes, not the
run's own start. Implementing that literally produced a real defect: on 6 of
45 grid points across the three `train_val_v2` checkpoints, a long `confirm`
value could push the debounce-completion frame of an early-starting run past
`event_t`, so the corrected definition credited *fewer* clips as `useful` than
the legacy one (worst case 32/103 → 24/103 at one operating point) — the
opposite of what this definition exists to fix, and a violation of §5's
superset guarantee below. Crediting the run's own start, matching
`first_alert_time`, closes that gap: every clip `useful` under the legacy
definition is provably `useful` under the corrected one, because the legacy
fire time and the corrected confirmation time are now the same quantity.

- **useful**: at least one episode's *active interval* — `[confirmation frame,
  last frame of the run]` — intersects `[alert_t, event_t]`. Intersects means
  the two closed intervals overlap by at least one frame; an episode that ends
  exactly at `alert_t` or begins exactly at `event_t` counts (frame-level
  boundary, no separate tolerance).
- **missed**: no episode's active interval intersects `[alert_t, event_t]`,
  including the case where the model never fires at all.
- **lead**: measured from the confirmation frame of the **earliest** episode
  whose active interval intersects the window, to `event_t`. Two lead
  quantities are reported separately, because they answer different questions:
  - `lead_vs_event = event_t − confirmation_t` — how long before the crash the
    driver had a warning up.
  - `lead_vs_alert = confirmation_t − alert_t` — how the warning's onset
    compares to the dataset's own annotated actionable moment. Positive means
    the model confirmed after `alert_t` (inside the window already); negative
    means it confirmed before `alert_t` and stayed active into the window (the
    clip-1013 case).
- **prematurity** (reported over every positive clip, useful or not):
  - (a) the fraction of positives whose *earliest confirmed episode* begins
    before `alert_t`;
  - (b) the median number of seconds that early episode begins before
    `alert_t`, over the clips in (a);
  - (c) a **gross-prematurity** count: episodes beginning more than 5 s before
    `alert_t`, **or** within the first 20% of the clip's own duration —
    whichever condition trips. This is deliberately a superset of "just early
    inside the window": it is aimed squarely at the old threshold engine's
    failure mode (median 14.4 s premature, firing from frame 0), and it must
    stay visible under the new definition exactly as it would have under the
    old one. A model whose episodes satisfy `useful` only by virtue of firing
    from clip start is still exposed by this count, on its own axis, whether
    or not it also clears the `useful` bar.
- **false alarms**: unchanged — accumulated over label-0 clips only, by the
  existing rule (a negative clip is a false alarm if it produces any confirmed
  episode at all, regardless of timing, since negatives have no `alert_t`/
  `event_t` to be early or late relative to).

## 5. What stays, in perpetuity

1. **The legacy definition is never removed.** The original first-crossing
   rule remains a selectable mode of `evaluate()`/`first_alert_time()`,
   reproducing its existing numbers exactly. It is not deprecated code kept for
   nostalgia — it is the number every prior document in this project
   (`clean_protocol_results.md`, `repartition_results.md`,
   `eval_leakage_audit.md`, and the original committed headline) was reported
   under, and those documents are not retroactively rewritten.
2. **Both definitions are reported side by side, always**, in every future
   table that reports `useful_warning_rate`, not just in the document that
   introduces the change. A table with only the new number is not an
   acceptable output of this change.
3. **Prematurity is a first-class reported axis, not folded into `useful`.**
   The corrected definition can only ever make more clips `useful`, never
   fewer (any clip `useful` under the legacy rule is a fortiori `useful` under
   the new one, since the legacy fire is itself a confirmed episode whose
   active interval starts inside the window). Without a separate prematurity
   report, a model that games "useful" by firing continuously from clip 0 on
   every positive would look strictly better under the new definition with no
   visible cost. The gross-prematurity count in §4(c) is what keeps that
   failure mode visible regardless of which definition's `useful_warning_rate`
   is being read.

## 6. Scope of this task

This document authorizes implementing §4 as a second mode alongside the
existing one, re-scoring the three existing `train_val_v2` checkpoints under
both, and reporting the prematurity axis. It does **not** authorize selecting
an operating point, tuning any hyperparameter, retraining, or reading
`data/features/eval_v2`. Those remain separate, later decisions.
