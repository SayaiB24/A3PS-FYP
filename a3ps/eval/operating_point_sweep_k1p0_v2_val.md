# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_v2_s1234.pt` (best epoch 4)
- scored on: `data/features/train_val_v2` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.330 | 34/103 | 63 | **0.667** | 2.70 | 0.630 |
| 0.60 | 3 | 0.252 | 26/103 | 56 | **0.490** | 2.52 | 0.630 |
| 0.70 | 3 | 0.282 | 29/103 | 43 | **0.304** | 2.13 | 0.630 |
| 0.80 | 3 | 0.311 | 32/103 | 23 | **0.147** ✅ | 1.62 | 0.630 |
| 0.90 | 3 | 0.194 | 20/103 | 9 | **0.029** ✅ | 1.14 | 0.630 |
| 0.50 | 5 | 0.291 | 30/103 | 63 | **0.627** | 2.75 | 0.630 |
| 0.60 | 5 | 0.252 | 26/103 | 55 | **0.461** | 2.49 | 0.630 |
| 0.70 | 5 | 0.301 | 31/103 | 41 | **0.275** | 1.95 | 0.630 |
| 0.80 | 5 | 0.291 | 30/103 | 22 | **0.088** ✅ | 1.56 | 0.630 |
| 0.90 | 5 | 0.136 | 14/103 | 8 | **0.029** ✅ | 1.14 | 0.630 |
| 0.50 | 8 | 0.291 | 30/103 | 63 | **0.569** | 2.69 | 0.630 |
| 0.60 | 8 | 0.233 | 24/103 | 53 | **0.373** | 2.48 | 0.630 |
| 0.70 | 8 | 0.243 | 25/103 | 39 | **0.196** ✅ | 1.95 | 0.630 |
| 0.80 | 8 | 0.252 | 26/103 | 21 | **0.088** ✅ | 1.64 | 0.630 |
| 0.90 | 8 | 0.087 | 9/103 | 6 | **0.020** ✅ | 1.30 | 0.630 |

## How to read this

- **`mean AP` is constant across rows** (0.630). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.80, confirm 3 → useful 0.311, FA 0.147, lead 1.62 s.
