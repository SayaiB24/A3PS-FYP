# Anticipation eval (split: eval)

- clips scored: **120** (60 positive, 60 negative)
- thresholds (A3PS): base **0.75**, alert **0.6** (base - alert_margin), floor **0.45**
- reactive baseline: BEV < 2.0 m (else img < 8% of frame height) to the ego corridor

| method | detection rate | false-alarm rate | mTTA (s) |
|---|---|---|---|
| **A3PS (proactive)** | 0.917 | 0.767 | 16.11 (n=55) |
| Reactive-proximity (baseline) | 0.967 | 0.783 | 17.92 (n=58) |

_The mTTA row above is averaged over each method's own true-positive set (different n, different clips) -- do NOT read the difference between these two mTTA values as an anticipation-gain claim. See the matched-subset comparison below for that._

## Matched-subset mTTA (fair, same-clips comparison)

Over the **54** positive clip(s) BOTH methods correctly anticipate (removes the recall/false-alarm-driven bias of comparing mTTA across each method's own, differently-sized true-positive set):

- A3PS mTTA: **16.17 s**
- Reactive-baseline mTTA: **18.24 s**
- **Anticipation gain: A3PS is 2.07 s later** than the reactive baseline on this matched subset.

_A3PS AP (peak-prob ranking): 0.978_
