# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_h128_v2_s1234.pt` (best epoch 4)
- scored on: `data/features/train_val_v2` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.330 | 34/103 | 43 | **0.353** | 1.91 | 0.601 |
| 0.60 | 3 | 0.330 | 34/103 | 33 | **0.225** | 1.70 | 0.601 |
| 0.70 | 3 | 0.301 | 31/103 | 22 | **0.118** ✅ | 1.35 | 0.601 |
| 0.80 | 3 | 0.252 | 26/103 | 8 | **0.049** ✅ | 1.04 | 0.601 |
| 0.50 | 5 | 0.311 | 32/103 | 41 | **0.314** | 1.91 | 0.601 |
| 0.60 | 5 | 0.330 | 34/103 | 31 | **0.206** | 1.66 | 0.601 |
| 0.70 | 5 | 0.301 | 31/103 | 20 | **0.078** ✅ | 1.32 | 0.601 |
| 0.80 | 5 | 0.204 | 21/103 | 7 | **0.049** ✅ | 1.07 | 0.601 |
| 0.50 | 8 | 0.311 | 32/103 | 39 | **0.275** | 1.87 | 0.601 |
| 0.60 | 8 | 0.291 | 30/103 | 29 | **0.147** ✅ | 1.65 | 0.601 |
| 0.70 | 8 | 0.282 | 29/103 | 17 | **0.078** ✅ | 1.24 | 0.601 |
| 0.80 | 8 | 0.175 | 18/103 | 6 | **0.029** ✅ | 1.18 | 0.601 |

## How to read this

- **`mean AP` is constant across rows** (0.601). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.70, confirm 3 → useful 0.301, FA 0.118, lead 1.35 s.
