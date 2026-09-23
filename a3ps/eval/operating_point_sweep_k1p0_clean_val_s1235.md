# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_clean_s1235.pt` (best epoch 1)
- scored on: `data/features/train_val` — 205 clips, 103 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.60 | 3 | 0.359 | 37/103 | 25 | **0.373** | 1.75 | 0.538 |
| 0.70 | 3 | 0.272 | 28/103 | 11 | **0.206** | 1.33 | 0.538 |
| 0.80 | 3 | 0.087 | 9/103 | 1 | **0.010** ✅ | 0.70 | 0.538 |
| 0.60 | 5 | 0.320 | 33/103 | 21 | **0.275** | 1.64 | 0.538 |
| 0.70 | 5 | 0.204 | 21/103 | 6 | **0.078** ✅ | 1.04 | 0.538 |
| 0.80 | 5 | 0.049 | 5/103 | 0 | **0.000** ✅ | 0.68 | 0.538 |
| 0.60 | 8 | 0.252 | 26/103 | 19 | **0.176** ✅ | 1.68 | 0.538 |
| 0.70 | 8 | 0.097 | 10/103 | 5 | **0.039** ✅ | 1.36 | 0.538 |
| 0.80 | 8 | 0.010 | 1/103 | 0 | **0.000** ✅ | 1.16 | 0.538 |

## How to read this

- **`mean AP` is constant across rows** (0.538). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.60, confirm 8 → useful 0.252, FA 0.176, lead 1.68 s.
