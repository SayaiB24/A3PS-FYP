# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0.pt` (best epoch 8)
- scored on: `data/features/eval` — 120 clips, 60 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.817 | 49/60 | 6 | **0.517** | 2.18 | 0.680 |
| 0.60 | 3 | 0.833 | 50/60 | 1 | **0.300** | 1.73 | 0.680 |
| 0.70 | 3 | 0.667 | 40/60 | 1 | **0.167** ✅ | 1.45 | 0.680 |
| 0.80 | 3 | 0.450 | 27/60 | 0 | **0.083** ✅ | 1.13 | 0.680 |
| 0.50 | 5 | 0.783 | 47/60 | 6 | **0.417** | 2.13 | 0.680 |
| 0.60 | 5 | 0.817 | 49/60 | 1 | **0.217** | 1.64 | 0.680 |
| 0.70 | 5 | 0.600 | 36/60 | 1 | **0.133** ✅ | 1.49 | 0.680 |
| 0.80 | 5 | 0.333 | 20/60 | 0 | **0.033** ✅ | 1.25 | 0.680 |
| 0.50 | 8 | 0.817 | 49/60 | 2 | **0.317** | 1.99 | 0.680 |
| 0.60 | 8 | 0.750 | 45/60 | 1 | **0.167** ✅ | 1.67 | 0.680 |
| 0.70 | 8 | 0.483 | 29/60 | 1 | **0.117** ✅ | 1.59 | 0.680 |
| 0.80 | 8 | 0.300 | 18/60 | 0 | **0.017** ✅ | 1.29 | 0.680 |

## How to read this

- **`mean AP` is constant across rows** (0.680). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.60, confirm 8 → useful 0.750, FA 0.167, lead 1.67 s.
