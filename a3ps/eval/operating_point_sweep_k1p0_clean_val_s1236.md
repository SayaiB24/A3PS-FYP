# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_clean_s1236.pt` (best epoch 12)
- scored on: `data/features/train_val` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.60 | 3 | 0.379 | 39/103 | 30 | **0.324** | 1.59 | 0.579 |
| 0.70 | 3 | 0.359 | 37/103 | 24 | **0.255** | 1.47 | 0.579 |
| 0.80 | 3 | 0.398 | 41/103 | 12 | **0.127** ✅ | 1.15 | 0.579 |
| 0.60 | 5 | 0.359 | 37/103 | 29 | **0.294** | 1.60 | 0.579 |
| 0.70 | 5 | 0.369 | 38/103 | 22 | **0.196** ✅ | 1.41 | 0.579 |
| 0.80 | 5 | 0.388 | 40/103 | 7 | **0.108** ✅ | 0.95 | 0.579 |
| 0.60 | 8 | 0.340 | 35/103 | 28 | **0.245** | 1.58 | 0.579 |
| 0.70 | 8 | 0.359 | 37/103 | 16 | **0.127** ✅ | 1.25 | 0.579 |
| 0.80 | 8 | 0.291 | 30/103 | 7 | **0.069** ✅ | 1.09 | 0.579 |

## How to read this

- **`mean AP` is constant across rows** (0.579). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.80, confirm 3 → useful 0.398, FA 0.127, lead 1.15 s.
