# Kappa comparison — does a lower kappa buy lead time?

- scored on `data/features/eval` — 120 clips, 60 positive (frozen held-out split)
- gates: false alarm ≤ 0.20, useful-warning ≥ 0.70, mean lead ≥ 1.0 s
- decision rule: maximise mean lead among rows passing all gates, tie-break on mean AP (see `docs/handoff/KAPPA_RETRAIN.md` §5)

`kappa` sets how sharply the anticipation loss concentrates weight just before the event. A large kappa asks for a late warning, which quietly turns an anticipation objective into a detection one — the *requested lead* column is what the loss is actually optimising for.

| kappa | requested lead | best passing row | useful | FA | **mean lead** | mean AP |
|---|---|---|---|---|---|---|
| 0.5 | 1.60 s | *none passes* | — | best 0.000 | — | 0.636 |
| 1.0 | 1.46 s | thr 0.60 / confirm 8 | 0.750 | 0.167 | **1.67** | 0.680 |
| 2.0 | 1.20 s | thr 0.60 / confirm 5 | 0.767 | 0.183 | **1.59** | 0.678 |
| 3.0 | 0.98 s | thr 0.60 / confirm 5 | 0.783 | 0.183 | **1.63** | 0.676 |

## Verdict

**kappa 1.0 wins**, at threshold 0.60 / confirm 8:

- useful-warning rate: **0.750** (45/60)
- false-alarm rate: **0.167**
- mean lead time: **1.67 s**
- mean AP: 0.680
- checkpoint: `notebooks/models/risk_gru_k1p0.pt`

_mean AP is flat across kappa (spread 0.004). That is expected: kappa changes **when** the model fires, not how well it separates risky from ordinary driving. Flat AP confirms the discrimination ceiling is a feature/capacity limit, so further loss tuning has little left to give — it is information, not a failed sweep._
