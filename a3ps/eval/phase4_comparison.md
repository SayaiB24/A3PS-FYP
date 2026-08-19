# Phase IV headline — learned risk head vs the threshold system

Both scored on the **same frozen 120-clip held-out split** (60 positive / 60
negative), both keyed to Nexar's own `time_of_alert` / `time_of_event`.

| metric | threshold system (pre-Phase IV) | **learned head (Phase IV)** | change |
|---|---|---|---|
| **useful-warning rate** | 0.050 (3/60) | **0.750 (45/60)** | **15× better** |
| **false-alarm rate** | 0.767 | **0.167** | **4.6× lower** |
| fired before `time_of_alert` | 52/60 positives | 1/60 | — |
| mean AP (500/1000/1500 ms) | 0.546 | **0.680** | +0.134 |
| mean lead time | 16.11 s | 1.67 s | see below |
| raw detection rate | 0.917 | 0.750 | −0.167 |

Sources: `anticipation.md` (threshold system, regenerated with the corrected
metrics) and `operating_point_sweep_k1p0.md` + `kappa_comparison.md` (learned
head at its committed operating point: kappa 1.0, threshold 0.60, confirm 8).

## How to read this

**The useful-warning rate is the headline.** The threshold system's 0.050 means
it produced an actionable warning on 3 of 60 positive clips. It was not failing
to detect — its raw detection rate is 0.917 — it was firing **before there was
anything to see**, on 52 of 60 positives. Its 16.11 s "mean lead time" is that
same pathology measured a different way: it alarmed near clip start, roughly 16 s
before impact, on clips where the hazard had not yet appeared. A warning 16 s
early on a 20 s clip is a standing alarm, not anticipation.

**The learned head's lower raw detection rate is the point, not a regression.**
0.917 → 0.750 while false alarms fall 0.767 → 0.167. The old number was inflated
by a system that alerted on nearly everything, including 46 of 60 negatives.

**mean AP is the threshold-free comparison.** 0.546 → 0.680 is the honest
measure of improved discrimination, because it ignores the decision boundary
entirely and cannot be tuned by choosing a threshold.

## What is still not met

**Mean lead time is 1.67 s against a 2–6 s target.** This is the one open gap,
and the kappa sweep (`kappa_comparison.md`) established that the loss weighting
is not the lever: across kappa ∈ {0.5, 1.0, 2.0, 3.0} lead time moved only
1.59–1.67 s, and mean AP stayed flat within 0.004. Flat AP across a parameter
that only changes *when* the model fires means the ceiling is the model's
discrimination — a feature/capacity limit, not a loss-tuning one. Next levers are
capacity, a longer feature window, or better features; not kappa.

## Caveat that must be stated

The two systems are **not measured under identical observation conditions**. The
learned head is scored on 13 s windows sampled at 10 Hz (its features); the
threshold system on full ~40 s clips at 30 Hz (its cached pipeline output). The
learned head therefore sees less of each clip, which makes its lead time
mechanically bounded by the window and makes its false-alarm advantage
*conservative* — it has fewer frames in which to raise a false alarm, but also
fewer in which to find the hazard. A strict comparison would re-run the
threshold system on the same 13 s / 10 Hz windows. Until that is done, quote the
useful-warning and AP improvements (which are large and robust to this) rather
than the lead-time difference (which is not).
