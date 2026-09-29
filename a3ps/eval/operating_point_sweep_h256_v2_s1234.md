# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_h256_v2_s1234.pt` (best epoch 8)
- scored on: `data/features/train_val_v2` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.272 | 28/103 | 42 | **0.294** | 1.85 | 0.641 |
| 0.60 | 3 | 0.291 | 30/103 | 31 | **0.186** ✅ | 1.58 | 0.641 |
| 0.70 | 3 | 0.282 | 29/103 | 18 | **0.137** ✅ | 1.39 | 0.641 |
| 0.80 | 3 | 0.223 | 23/103 | 10 | **0.039** ✅ | 1.23 | 0.641 |
| 0.50 | 5 | 0.262 | 27/103 | 40 | **0.255** | 1.80 | 0.641 |
| 0.60 | 5 | 0.272 | 28/103 | 28 | **0.167** ✅ | 1.48 | 0.641 |
| 0.70 | 5 | 0.272 | 28/103 | 16 | **0.098** ✅ | 1.35 | 0.641 |
| 0.80 | 5 | 0.214 | 22/103 | 8 | **0.029** ✅ | 1.12 | 0.641 |
| 0.50 | 8 | 0.243 | 25/103 | 38 | **0.206** | 1.73 | 0.641 |
| 0.60 | 8 | 0.282 | 29/103 | 25 | **0.127** ✅ | 1.45 | 0.641 |
| 0.70 | 8 | 0.252 | 26/103 | 15 | **0.069** ✅ | 1.36 | 0.641 |
| 0.80 | 8 | 0.184 | 19/103 | 7 | **0.010** ✅ | 1.10 | 0.641 |

## How to read this

- **`mean AP` is constant across rows** (0.641). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.60, confirm 3 → useful 0.291, FA 0.186, lead 1.58 s.
