# Attempts to raise the headline number (2026-09-28/29)

> **Correction (2026-09-29).** The first version of this document compared
> each candidate's useful-warning as printed by `sweep_operating_point.py` —
> which reports the **legacy** (first-crossing) definition — against the
> baseline's 0.612, which is the **v2** (episode-based) figure. That made
> every candidate look like it had "collapsed" (0.29–0.41 vs 0.612). Under
> one definition the gap disappears: at the same cell (0.70/c8), legacy
> useful-warning is baseline **0.243** vs 0.25–0.31 for the candidates, and
> under v2 each candidate's best passing cell sits at 0.553–0.621 vs the
> baseline's 0.612. All tables below are now from
> `scripts/rescore_candidates_both_defs.py` →
> [`../../eval/candidates_both_definitions.md`](../../eval/candidates_both_definitions.md),
> which scores every run with the same `evaluate()`, the same grid, both
> definitions, and the same selection rule. The conclusion (keep the
> baseline) survives; the strength of it does not. None of these candidates
> is *worse* beyond noise. None is *better* beyond noise either.

Per `docs/handoff/JUDGES_PREP_PLAN.md`'s one-read rule, `eval_v2` is read
**at most once more, ever**, for whichever single config is actually locked
in. Every candidate below was scored on `data/features/train_val_v2` (or its
`_seq2seq` counterpart, same clips) only. **The final `eval_v2` read has not
been spent** on any of them.

**Baseline** (locked, held-out): `risk_gru_k1p0_v2_selfix_s1234.pt`,
useful-warning **0.650** (39/60, v2), false-alarm **0.167**, mean lead
**1.51 s**, mean AP **0.646** — see [`final_eval_read.md`](final_eval_read.md).
**Baseline on validation** (what every candidate is compared against):
useful_v2 0.612 (legacy 0.243), FA 0.196, v2 lead 1.83 s, mean AP 0.630 at
threshold 0.70 / confirm 8.

**Selection rule**, applied identically to every run: FA ≤ 0.20, v2 mean lead
≥ 1.0 s, zero gross-premature; among passing cells, highest useful_v2.

**How big a difference means anything:** one validation positive is 1/103 ≈
0.010. Seed-to-seed spread for the *same* recipe at 0.70/c8 had sd 0.159
(`ensemble_calibration_results.md` §2). Every candidate here is one seed, so
a gap under ~0.1 on useful_v2 is not evidence either way. Mean AP is tighter
across seeds (0.58–0.68), so it is the better single-seed signal — and it
did not move meaningfully for any candidate.

---

## 1. Model capacity — no gain

Per `docs/handoff/FUTURE_WORK.md`'s "More model capacity". Trained on
`train_core_v2` / `train_val_v2`, same hyperparameters as the baseline except
`--hidden`/`--layers`, seed 1234.

| config | best epoch | mean AP | @0.70/c8: useful_v2 / legacy / FA | best passing cell | useful_v2 | FA | lead_v2 |
|---|---|---|---|---|---|---|---|
| baseline (hidden 64, layers 1) | 8 | **0.630** | 0.612 / 0.243 / 0.196 | 0.70 / c8 | **0.612** | 0.196 | 1.83 s |
| hidden 128, layers 2 | 4 | 0.601 | 0.427 / 0.282 / 0.078 | 0.60 / c8 | 0.553 | 0.147 | 1.48 s |
| hidden 256, layers 2 | 8 | 0.641 | 0.388 / 0.252 / 0.069 | 0.60 / c3 | 0.573 | 0.186 | 1.28 s |

Both larger heads land 0.04–0.06 below the baseline on useful_v2 at their
own best passing cell (4–6 validation clips) with shorter lead, and mean AP
moves by −0.03 / +0.01. That is inside single-seed noise in both directions.
The larger heads are also less trigger-happy at a given threshold (FA 0.07
vs 0.20 at 0.70/c8), which is why the shared cell flatters the baseline and
why comparing each at its own best cell is the fair read.

**Verdict:** capacity is not the lever. No evidence that a bigger head helps
on ~1,160 training clips, and weak evidence of shorter lead. Keep
`--hidden 64 --layers 1`. Not a "clear regression" — the first version of
this document overstated that.

Raw artifacts: `eval/train_log_h128_v2_s1234.txt`,
`eval/train_log_h256_v2_s1234.txt`,
`eval/operating_point_sweep_h128_v2_s1234.md`,
`eval/operating_point_sweep_h256_v2_s1234.md` (legacy definition only).

---

## 2. Seq2Seq-LSTM forecaster features — a tie on coverage, shorter lead

Revisits `docs/status/pivot_open_questions.md` §1's open question ("the
Seq2Seq-LSTM 'beats Kalman' claim doesn't transfer to the system that
ships"). This is the first time it was tested end to end through the GRU
risk head rather than left as an open question.

**Step 0 — truncation check first.** Before spending GPU time on full
re-extraction, re-scored both forecasters' ADE/FDE on 8- and 9-point
truncated histories (matching `TrackBuffer.ready()`'s real gate, vs. the
full 10-point histories `mine_trajectories.py` always mined). The LSTM's
4 s-horizon edge survived truncation essentially unchanged (ADE 56.29–56.33
across 8/9/10-pt histories vs. Kalman's 60.62–61.32), so the padded-history
concern specifically was cleared — full results in
`eval/forecast_table_hist8.md` / `_hist9.md`, against a freshly-regenerated
full-history baseline in `eval/forecast_table.md` (the previously-committed
table was stale relative to `kalman_cv.py`'s calibrated `MEAS_VAR`/
`PROCESS_VAR` constants; archived as `eval/forecast_table_PRE_KALMAN_RECAL.md`).

**Step 1 — re-extracted the full v2 pool (1,485 clips) with `forecaster:
seq2seq`** (`configs/seq2seq_eval.yaml`), against the same frozen v2 split
membership (`eval/split_freeze_v2.json`) as the committed Kalman-based
features, so the comparison is on identical clips:
`data/features/{eval_v2,train_val_v2,train_core_v2}_seq2seq`. Measured
throughput on this machine (RTX 3050 Laptop GPU): ~23.5 s/clip — about 3x
the ~8.4 s/clip this project's docs previously measured elsewhere. All
1,485 clips extracted with 0 failures.

**Step 2 — trained and scored a GRU on the LSTM-based features**, identical
hyperparameters to the baseline (`--kappa 1.0 --pre-alert-weight 0.5
--pos-weight 1.0 --hidden 64 --layers 1 --seed 1234`):

| features | mean AP | @0.70/c8: useful_v2 / legacy / FA | best passing cell | useful_v2 | legacy | FA | lead_v2 |
|---|---|---|---|---|---|---|---|
| Kalman (baseline) | 0.630 | 0.612 / 0.243 / 0.196 | 0.70 / c8 | **0.612** | 0.243 | 0.196 | **1.83 s** |
| LSTM | 0.622 | 0.417 / 0.311 / 0.069 | 0.60 / c8 | **0.621** | 0.408 | 0.176 | 1.27 s |

At each run's own best passing cell the two are a tie on useful_v2 (0.621 vs
0.612, one clip), the LSTM run has slightly lower FA and higher legacy
useful-warning, and the baseline warns **0.56 s earlier**. Mean AP is flat
(0.622 vs 0.630).

**A plausible mechanism for the shorter lead** (not tested in isolation):
the LSTM only beats Kalman on trajectory error at the 4 s horizon (ADE 56.33
vs 60.62); at 1–2 s Kalman is clearly better (ADE@1s 12.94 vs 18.85, ~31%
lower). `collision_prob`/`ttc_pred` come from the forecaster's rollout, so
the swap changes exactly the features that drive alerting. The LSTM's
log-variance head was also never calibrated the way Kalman's std was
(`kalman_cv.py`'s `RMSE(k)/std(k) ≈ 1`), which would affect
`collision_prob` directly.

**Verdict:** answers `pivot_open_questions.md` §1. The LSTM's trajectory-error
win **does not turn into a risk-detection win**: same coverage, same
discrimination, less lead, plus a trained model and ~3× slower extraction.
`configs/default.yaml`'s `forecaster: kalman_cv` default is justified by
that comparison — not "actively vindicated", which the first version of
this document claimed on the mismatched numbers.

Raw artifacts: `configs/seq2seq_eval.yaml`,
`eval/train_log_k1p0_v2_seq2seq_s1234.txt`,
`eval/risk_gru_history_k1p0_v2_seq2seq_s1234.json`,
`eval/operating_point_sweep_k1p0_v2_seq2seq_val.md` (legacy definition only),
`notebooks/models/risk_gru_k1p0_v2_seq2seq_s1234.pt`.

---

## 3. Net conclusion

Five improvements tried (kappa sweep, ensemble, calibration, capacity, LSTM
forecaster), all scored on validation only. **None beat the baseline beyond
noise, and none was clearly worse on coverage either** — the honest summary
is "the number is stable under every lever we had," not "every lever made it
worse." The strongest single signal is lead time: every alternative that
matched coverage did so with less lead.

**The current committed checkpoint (`risk_gru_k1p0_v2_selfix_s1234.pt`)
remains the headline result.** The single remaining `eval_v2` read has not
been spent and should be held for a genuinely promising candidate, not any
of the above.

**Lesson for any future comparison:** `sweep_operating_point.py` prints the
legacy definition only. Compare runs with
`scripts/rescore_candidates_both_defs.py` (or `evaluate()`'s `_v2` keys),
never by reading the sweep's "useful" column against a v2 number.
