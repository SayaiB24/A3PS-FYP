# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_v2_seq2seq_s1234.pt` (best epoch 15)
- scored on: `data/features/train_val_v2_seq2seq` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.369 | 38/103 | 48 | **0.392** | 1.97 | 0.622 |
| 0.60 | 3 | 0.427 | 44/103 | 34 | **0.275** | 1.73 | 0.622 |
| 0.70 | 3 | 0.398 | 41/103 | 18 | **0.167** ✅ | 1.45 | 0.622 |
| 0.80 | 3 | 0.252 | 26/103 | 9 | **0.049** ✅ | 1.15 | 0.622 |
| 0.90 | 3 | 0.155 | 16/103 | 1 | **0.000** ✅ | 0.68 | 0.622 |
| 0.50 | 5 | 0.350 | 36/103 | 46 | **0.333** | 1.97 | 0.622 |
| 0.60 | 5 | 0.417 | 43/103 | 30 | **0.216** | 1.61 | 0.622 |
| 0.70 | 5 | 0.369 | 38/103 | 15 | **0.127** ✅ | 1.35 | 0.622 |
| 0.80 | 5 | 0.223 | 23/103 | 9 | **0.029** ✅ | 1.15 | 0.622 |
| 0.90 | 5 | 0.117 | 12/103 | 0 | **0.000** ✅ | 0.67 | 0.622 |
| 0.50 | 8 | 0.330 | 34/103 | 44 | **0.265** | 1.92 | 0.622 |
| 0.60 | 8 | 0.408 | 42/103 | 25 | **0.176** ✅ | 1.41 | 0.622 |
| 0.70 | 8 | 0.311 | 32/103 | 13 | **0.069** ✅ | 1.27 | 0.622 |
| 0.80 | 8 | 0.214 | 22/103 | 8 | **0.010** ✅ | 1.16 | 0.622 |
| 0.90 | 8 | 0.058 | 6/103 | 0 | **0.000** ✅ | 0.83 | 0.622 |

## How to read this

- **`mean AP` is constant across rows** (0.622). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.60, confirm 8 → useful 0.408, FA 0.176, lead 1.41 s.
