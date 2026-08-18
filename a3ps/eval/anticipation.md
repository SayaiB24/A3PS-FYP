# Anticipation eval (split: eval)

- clips scored: **120** (60 positive, 60 negative)
- thresholds (A3PS): base **0.75**, alert **0.6** (base - alert_margin), floor **0.45**
- reactive baseline: BEV < 2.0 m (else img < 8% of frame height) to the ego corridor

| method | detection rate | false-alarm rate | mTTA (s) | useful warning rate | fired too early |
|---|---|---|---|---|---|
| **A3PS (proactive)** | 0.917 | 0.767 | 16.11 (n=55) | 0.050 (3/60) | 52/60 |
| Reactive-proximity (baseline) | 0.967 | 0.783 | 17.92 (n=58) | 0.033 (2/60) | 56/60 |

_Useful-warning window: **[alert_time_s - 0.00s, event_time_s] per clip (ground truth, not a chosen window)**. `time_of_alert` is Nexar's own annotation of the earliest actionable moment, so this window is the dataset's, not ours. The measured lead (`time_of_event - time_of_alert`) is 2.97-4.47 s over the 65 local positives (mean 3.49, sd 0.40)._

_The mTTA row above is averaged over each method's own true-positive set (different n, different clips) -- do NOT read the difference between these two mTTA values as an anticipation-gain claim. See the matched-subset comparison below for that._

_**Read the useful-warning-rate column, not mTTA, to judge whether warnings are actionable.** mTTA rewards warning early without bound, so a method that alarms on the first frame of every clip maximises it while telling the driver nothing -- which is exactly what the reactive baseline does here. The useful warning rate instead counts positives warned inside the dataset's own actionable window; misses, too-late alarms and alarms fired before `time_of_alert` all count against it. The 'fired too early' column isolates that last failure mode._

## Matched-subset mTTA (fair, same-clips comparison)

Over the **54** positive clip(s) BOTH methods correctly anticipate (removes the recall/false-alarm-driven bias of comparing mTTA across each method's own, differently-sized true-positive set):

- A3PS mTTA: **16.17 s**
- Reactive-baseline mTTA: **18.24 s**
- **Anticipation gain: A3PS is 2.07 s later** than the reactive baseline on this matched subset.

## Official-style Nexar AP at pre-event cutoffs

Each clip is ranked by the highest collision probability the model held using ONLY the frames it would have seen had the video been cut the stated interval before the annotated event. This is the dataset's own anticipation protocol, independent of our alert thresholds and decision state machine.

| cutoff before event | AP | clips scored |
|---|---|---|
| 500 ms | 0.547 | 120 (60 pos) |
| 1000 ms | 0.546 | 120 (60 pos) |
| 1500 ms | 0.546 | 120 (60 pos) |

**mean AP over the three cutoffs: 0.546**

_negatives scored over their full clip (official protocol; gives negatives more frames than positives, so this AP is a lower bound)._

_A3PS AP (whole-clip peak-prob ranking): 0.562 -- uses every frame including post-event ones, so it is NOT an anticipation number; compare the cutoff APs above instead._
