# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k0p5_clean.pt` (best epoch 6)
- scored on: `data/features/train_val` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=0.5`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.369 | 38/103 | 47 | **0.578** | 2.01 | 0.592 |
| 0.60 | 3 | 0.340 | 35/103 | 38 | **0.490** | 1.81 | 0.592 |
| 0.70 | 3 | 0.388 | 40/103 | 28 | **0.314** | 1.49 | 0.592 |
| 0.80 | 3 | 0.379 | 39/103 | 13 | **0.137** ✅ | 1.12 | 0.592 |
| 0.90 | 3 | 0.204 | 21/103 | 4 | **0.020** ✅ | 0.87 | 0.592 |
| 0.50 | 5 | 0.350 | 36/103 | 45 | **0.529** | 2.01 | 0.592 |
| 0.60 | 5 | 0.369 | 38/103 | 34 | **0.402** | 1.73 | 0.592 |
| 0.70 | 5 | 0.379 | 39/103 | 26 | **0.265** | 1.46 | 0.592 |
| 0.80 | 5 | 0.330 | 34/103 | 11 | **0.108** ✅ | 1.09 | 0.592 |
| 0.90 | 5 | 0.165 | 17/103 | 4 | **0.010** ✅ | 0.96 | 0.592 |
| 0.50 | 8 | 0.320 | 33/103 | 44 | **0.480** | 1.99 | 0.592 |
| 0.60 | 8 | 0.350 | 36/103 | 31 | **0.314** | 1.69 | 0.592 |
| 0.70 | 8 | 0.359 | 37/103 | 19 | **0.167** ✅ | 1.35 | 0.592 |
| 0.80 | 8 | 0.252 | 26/103 | 11 | **0.059** ✅ | 1.23 | 0.592 |
| 0.90 | 8 | 0.146 | 15/103 | 3 | **0.000** ✅ | 0.92 | 0.592 |

## How to read this

- **`mean AP` is constant across rows** (0.592). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.80, confirm 3 → useful 0.379, FA 0.137, lead 1.12 s.
