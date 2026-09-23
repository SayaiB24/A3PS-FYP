# Operating-point sweep (decision threshold x confirm frames)

- checkpoint: `notebooks/models/risk_gru_k1p0_clean.pt` (best epoch 6)
- scored on: `data/features/eval` — 120 clips, 60 positive
- trained with: `pre_alert_weight=0.5`, `kappa=1.0`

These two parameters are applied **after** the model runs, so every row below is the same weights at a different decision boundary — no retraining involved.

| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |
|---|---|---|---|---|---|---|---|
| 0.70 | 8 | 0.633 | 38/60 | 1 | **0.200** ✅ | 1.62 | 0.661 |

## How to read this

- **`mean AP` is constant across rows** (0.661). It ranks clips by peak probability and ignores the threshold entirely, so it is the model's threshold-free discrimination — the ceiling no operating point can beat. Improving it needs better features or training, not a different threshold.
- **False-alarm rate is the gatekeeper** — target ≤ 0.20. A high useful-warning rate at a high FA is not a result: a system that alerts on most negatives will also 'catch' most positives.
- **Lead time is the price.** Raising the threshold or confirm count buys FA back by alerting later. Below ~1 s of lead there is no time to react, so a row that fixes FA by collapsing lead has not solved anything.
- **`too early`** counts positives alerted before the dataset's own `time_of_alert`. Those are not credited as useful, by design.

**Best row meeting FA ≤ 0.20:** threshold 0.70, confirm 8 → useful 0.633, FA 0.200, lead 1.62 s.
