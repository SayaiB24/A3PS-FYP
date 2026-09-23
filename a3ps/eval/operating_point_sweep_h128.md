# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_cap_h128.pt` (best epoch 21)
- scored on: `data/features/eval` — 120 clips, 60 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.850 | 51/60 | 5 | **0.400** | 2.11 | 0.702 |
| 0.60 | 3 | 0.817 | 49/60 | 4 | **0.350** | 1.88 | 0.702 |
| 0.70 | 3 | 0.767 | 46/60 | 2 | **0.283** | 1.60 | 0.702 |
| 0.80 | 3 | 0.700 | 42/60 | 1 | **0.183** ✅ | 1.52 | 0.702 |
| 0.50 | 5 | 0.800 | 48/60 | 5 | **0.333** | 2.09 | 0.702 |
| 0.60 | 5 | 0.750 | 45/60 | 4 | **0.300** | 1.87 | 0.702 |
| 0.70 | 5 | 0.750 | 45/60 | 2 | **0.200** ✅ | 1.59 | 0.702 |
| 0.80 | 5 | 0.683 | 41/60 | 1 | **0.150** ✅ | 1.52 | 0.702 |
| 0.50 | 8 | 0.750 | 45/60 | 5 | **0.317** | 2.07 | 0.702 |
| 0.60 | 8 | 0.767 | 46/60 | 2 | **0.233** | 1.75 | 0.702 |
| 0.70 | 8 | 0.650 | 39/60 | 2 | **0.150** ✅ | 1.74 | 0.702 |
| 0.80 | 8 | 0.633 | 38/60 | 1 | **0.067** ✅ | 1.47 | 0.702 |

## How to read this

- **`mean AP` is constant across rows** (0.702). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.70, confirm 5 → useful 0.750, FA 0.200, lead 1.59 s.
