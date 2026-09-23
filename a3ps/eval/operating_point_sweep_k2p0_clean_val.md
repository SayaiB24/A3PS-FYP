# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k2p0_clean.pt` (best epoch 7)
- scored on: `data/features/train_val` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=2.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.350 | 36/103 | 48 | **0.539** | 2.06 | 0.590 |
| 0.60 | 3 | 0.359 | 37/103 | 41 | **0.490** | 1.79 | 0.590 |
| 0.70 | 3 | 0.388 | 40/103 | 32 | **0.402** | 1.57 | 0.590 |
| 0.80 | 3 | 0.379 | 39/103 | 22 | **0.245** | 1.38 | 0.590 |
| 0.90 | 3 | 0.340 | 35/103 | 9 | **0.088** ✅ | 0.91 | 0.590 |
| 0.50 | 5 | 0.359 | 37/103 | 47 | **0.529** | 1.99 | 0.590 |
| 0.60 | 5 | 0.379 | 39/103 | 38 | **0.431** | 1.76 | 0.590 |
| 0.70 | 5 | 0.369 | 38/103 | 32 | **0.353** | 1.54 | 0.590 |
| 0.80 | 5 | 0.388 | 40/103 | 20 | **0.186** ✅ | 1.26 | 0.590 |
| 0.90 | 5 | 0.311 | 32/103 | 7 | **0.069** ✅ | 0.90 | 0.590 |
| 0.50 | 8 | 0.379 | 39/103 | 43 | **0.461** | 1.90 | 0.590 |
| 0.60 | 8 | 0.350 | 36/103 | 38 | **0.363** | 1.76 | 0.590 |
| 0.70 | 8 | 0.350 | 36/103 | 30 | **0.275** | 1.48 | 0.590 |
| 0.80 | 8 | 0.340 | 35/103 | 18 | **0.118** ✅ | 1.26 | 0.590 |
| 0.90 | 8 | 0.252 | 26/103 | 5 | **0.049** ✅ | 0.89 | 0.590 |

## How to read this

- **`mean AP` is constant across rows** (0.590). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.80, confirm 5 → useful 0.388, FA 0.186, lead 1.26 s.
