# Feature ablations, re-run post-fix -- scored at the committed operating point

Checkpoint per group: `notebooks/models/risk_gru_abl_<group>.pt` (already retrained post-fix, see `eval/train_log_abl_*.txt`). Scored here on `data/features/eval` (the frozen eval split, `eval/split_freeze.json`) at the committed operating point threshold=0.6, confirm=8, with the SAME columns zeroed at eval time as at train time for that arm.

Committed full-feature model for comparison: `notebooks/models/risk_gru_k1p0.pt` -- useful 0.750, FA 0.167, mean AP 0.680, mean lead 1.67s (`eval/operating_point_sweep_k1p0.md`).

| group | columns zeroed | useful-warning | false-alarm | mean lead (s) | mean AP |
|---|---|---|---|---|---|
| ego | 4 | 0.683 (41/60) | 0.150 | 1.68 | 0.689 |
| corridor | 13 | 0.583 (35/60) | 0.100 | 1.35 | 0.654 |
| collision_prob | 6 | 0.717 (43/60) | 0.267 | 1.65 | 0.659 |
| ttc | 12 | 0.833 (50/60) | 0.383 | 1.95 | 0.650 |
