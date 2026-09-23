# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_clean.pt` (best epoch 6)
- scored on: `data/features/train_val` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.340 | 35/103 | 52 | **0.588** | 2.11 | 0.596 |
| 0.60 | 3 | 0.359 | 37/103 | 39 | **0.480** | 1.91 | 0.596 |
| 0.70 | 3 | 0.398 | 41/103 | 28 | **0.324** | 1.51 | 0.596 |
| 0.80 | 3 | 0.388 | 40/103 | 14 | **0.157** ✅ | 1.12 | 0.596 |
| 0.90 | 3 | 0.223 | 23/103 | 4 | **0.020** ✅ | 0.83 | 0.596 |
| 0.50 | 5 | 0.320 | 33/103 | 49 | **0.559** | 2.07 | 0.596 |
| 0.60 | 5 | 0.369 | 38/103 | 35 | **0.422** | 1.76 | 0.596 |
| 0.70 | 5 | 0.398 | 41/103 | 26 | **0.284** | 1.43 | 0.596 |
| 0.80 | 5 | 0.369 | 38/103 | 12 | **0.108** ✅ | 1.05 | 0.596 |
| 0.90 | 5 | 0.184 | 19/103 | 4 | **0.010** ✅ | 0.89 | 0.596 |
| 0.50 | 8 | 0.311 | 32/103 | 46 | **0.490** | 2.03 | 0.596 |
| 0.60 | 8 | 0.369 | 38/103 | 32 | **0.333** | 1.73 | 0.596 |
| 0.70 | 8 | 0.369 | 38/103 | 19 | **0.196** ✅ | 1.34 | 0.596 |
| 0.80 | 8 | 0.301 | 31/103 | 11 | **0.069** ✅ | 1.14 | 0.596 |
| 0.90 | 8 | 0.146 | 15/103 | 3 | **0.000** ✅ | 0.91 | 0.596 |

## How to read this

- **`mean AP` is constant across rows** (0.596). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.80, confirm 3 → useful 0.388, FA 0.157, lead 1.12 s.
