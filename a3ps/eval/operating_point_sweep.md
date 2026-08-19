# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_v1.pt` (best epoch 5)
- scored on: `data/features/eval` — 120 clips, 60 positive
- trained with: `pre_alert_weight=0.5`, `kappa=3.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.833 | 50/60 | 5 | **0.583** | 2.15 | 0.693 |
| 0.60 | 3 | 0.750 | 45/60 | 5 | **0.417** | 1.99 | 0.693 |
| 0.70 | 3 | 0.767 | 46/60 | 1 | **0.250** | 1.61 | 0.693 |
| 0.80 | 3 | 0.633 | 38/60 | 0 | **0.100** ✅ | 1.41 | 0.693 |
| 0.90 | 3 | 0.250 | 15/60 | 0 | **0.017** ✅ | 1.01 | 0.693 |
| 0.95 | 3 | 0.100 | 6/60 | 0 | **0.000** ✅ | 1.01 | 0.693 |
| 0.50 | 5 | 0.833 | 50/60 | 5 | **0.517** | 2.10 | 0.693 |
| 0.60 | 5 | 0.767 | 46/60 | 3 | **0.317** | 1.92 | 0.693 |
| 0.70 | 5 | 0.717 | 43/60 | 1 | **0.167** ✅ | 1.58 | 0.693 |
| 0.80 | 5 | 0.550 | 33/60 | 0 | **0.083** ✅ | 1.46 | 0.693 |
| 0.90 | 5 | 0.183 | 11/60 | 0 | **0.017** ✅ | 1.19 | 0.693 |
| 0.95 | 5 | 0.083 | 5/60 | 0 | **0.000** ✅ | 1.17 | 0.693 |
| 0.50 | 8 | 0.817 | 49/60 | 5 | **0.400** | 2.04 | 0.693 |
| 0.60 | 8 | 0.750 | 45/60 | 2 | **0.217** | 1.84 | 0.693 |
| 0.70 | 8 | 0.667 | 40/60 | 1 | **0.167** ✅ | 1.61 | 0.693 |
| 0.80 | 8 | 0.483 | 29/60 | 0 | **0.050** ✅ | 1.46 | 0.693 |
| 0.90 | 8 | 0.167 | 10/60 | 0 | **0.000** ✅ | 1.22 | 0.693 |
| 0.95 | 8 | 0.083 | 5/60 | 0 | **0.000** ✅ | 1.17 | 0.693 |

## How to read this

- **`mean AP` is constant across rows** (0.693). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.70, confirm 5 → useful 0.717, FA 0.167, lead 1.58 s.
