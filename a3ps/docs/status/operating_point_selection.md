# Operating-point selection: a documented near-miss (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document.

**Status: the documented rule has zero survivors (Section 1). By explicit
instruction, this document proceeds anyway with a NEAR-MISS configuration —
seed 1234 at threshold 0.70 / confirm 8 — for the dashboard rebuild and
sanity check (Sections 3–4). This is not the rule succeeding. Every place
this configuration's numbers appear below states the shortfall plainly.**

## 1. Full three-seed grid, v2 definition

All three seeds (1234/1235/1236, `risk_gru_k1p0_v2_selfix_s*.pt`,
pre_alert_weight=0.5), swept threshold × confirm on `data/features/train_val_v2`
only, under the corrected (v2) useful-warning definition.

### Seed 1234 (mean AP 0.630)

| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature | n_gross_premature |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.942 | 97 | 0.667 | 2.47 | 0.612 | 1 |
| 0.60 | 3 | 0.796 | 82 | 0.490 | 2.18 | 0.544 | 1 |
| 0.70 | 3 | 0.699 | 72 | 0.304 | 1.75 | 0.417 | 0 |
| 0.80 | 3 | 0.515 | 53 | 0.147 | 1.31 | 0.223 | 0 |
| 0.90 | 3 | 0.252 | 26 | 0.029 | 0.90 | 0.087 | 0 |
| 0.50 | 5 | 0.903 | 93 | 0.627 | 2.56 | 0.612 | 1 |
| 0.60 | 5 | 0.786 | 81 | 0.461 | 2.16 | 0.534 | 1 |
| 0.70 | 5 | 0.689 | 71 | 0.275 | 1.76 | 0.398 | 0 |
| 0.80 | 5 | 0.495 | 51 | 0.088 | 1.28 | 0.214 | 0 |
| 0.90 | 5 | 0.194 | 20 | 0.029 | 0.96 | 0.078 | 0 |
| 0.50 | 8 | 0.903 | 93 | 0.569 | 2.55 | 0.612 | 1 |
| 0.60 | 8 | 0.748 | 77 | 0.373 | 2.23 | 0.515 | 1 |
| **0.70** | **8** | **0.612** | 63 | **0.196** | **1.83** | 0.379 | **0** |
| 0.80 | 8 | 0.447 | 46 | 0.088 | 1.39 | 0.204 | 0 |
| 0.90 | 8 | 0.136 | 14 | 0.020 | 1.17 | 0.058 | 0 |

### Seed 1235 (mean AP 0.590)

| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature | n_gross_premature |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.816 | 84 | 0.480 | 1.59 | 0.485 | 1 |
| 0.60 | 3 | 0.728 | 75 | 0.363 | 1.29 | 0.379 | 0 |
| **0.70** | **3** | 0.573 | 59 | **0.206** | 1.25 | 0.243 | 0 |
| 0.80 | 3 | 0.447 | 46 | 0.108 | 1.00 | 0.126 | 0 |
| 0.90 | 3 | 0.301 | 31 | 0.039 | 0.76 | 0.049 | 0 |
| 0.50 | 5 | 0.757 | 78 | 0.402 | 1.66 | 0.437 | 0 |
| 0.60 | 5 | 0.709 | 73 | 0.265 | 1.31 | 0.369 | 0 |
| **0.70** | **5** | **0.524** | 54 | **0.196** | **1.25** | 0.223 | **0** |
| 0.80 | 5 | 0.417 | 43 | 0.098 | 1.01 | 0.126 | 0 |
| 0.90 | 5 | 0.243 | 25 | 0.029 | 0.84 | 0.039 | 0 |
| 0.50 | 8 | 0.728 | 75 | 0.314 | 1.67 | 0.437 | 0 |
| 0.60 | 8 | 0.621 | 64 | 0.206 | 1.35 | 0.311 | 0 |
| 0.70 | 8 | 0.495 | 51 | 0.137 | 1.27 | 0.204 | 0 |
| 0.80 | 8 | 0.398 | 41 | 0.078 | 1.00 | 0.107 | 0 |
| 0.90 | 8 | 0.194 | 20 | 0.020 | 0.90 | 0.029 | 0 |

### Seed 1236 (mean AP 0.648)

| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature | n_gross_premature |
|---|---|---|---|---|---|---|---|
| 0.50 | 3 | 0.961 | 99 | 0.784 | 3.00 | 0.757 | 3 |
| 0.60 | 3 | 0.961 | 99 | 0.686 | 2.70 | 0.718 | 2 |
| 0.70 | 3 | 0.874 | 90 | 0.588 | 2.52 | 0.631 | 0 |
| 0.80 | 3 | 0.738 | 76 | 0.353 | 2.26 | 0.476 | 0 |
| **0.90** | **3** | **0.544** | 56 | **0.147** | **1.35** | 0.262 | **0** |
| 0.50 | 5 | 0.961 | 99 | 0.755 | 3.00 | 0.757 | 3 |
| 0.60 | 5 | 0.951 | 98 | 0.647 | 2.71 | 0.709 | 2 |
| 0.70 | 5 | 0.874 | 90 | 0.520 | 2.52 | 0.631 | 0 |
| 0.80 | 5 | 0.709 | 73 | 0.324 | 2.33 | 0.466 | 0 |
| 0.90 | 5 | 0.476 | 49 | 0.118 | 1.41 | 0.262 | 0 |
| 0.50 | 8 | 0.961 | 99 | 0.706 | 3.00 | 0.748 | 3 |
| 0.60 | 8 | 0.922 | 95 | 0.618 | 2.76 | 0.709 | 2 |
| 0.70 | 8 | 0.864 | 89 | 0.431 | 2.54 | 0.621 | 0 |
| 0.80 | 8 | 0.680 | 70 | 0.284 | 2.38 | 0.456 | 0 |
| 0.90 | 8 | 0.427 | 44 | 0.108 | 1.49 | 0.223 | 0 |

(Bolded rows are each seed's best cell under the *reduced* rule below — not
survivors of the full rule, shown only to illustrate how close each seed came.)

## The rule, applied exactly as written

`FA ≤ 0.20`, `useful_warning_rate_v2 ≥ 0.70`, `lead_vs_event_v2 ≥ 1.0 s`, gate
on `n_gross_premature == 0` (not `frac_premature` — established in
`metric_fix_results.md` and `selection_fix_and_tradeoff.md` as not itself a
defect signal), then maximise mean lead among survivors, tie-break on mean AP.

**Zero survivors, across all 45 cells and all three seeds.** No `(seed,
threshold, confirm)` combination satisfies all four gates simultaneously.

## How close it got

The blocking gate is `useful_warning_rate_v2 ≥ 0.70`. Restricting to cells
that pass every *other* gate (`FA ≤ 0.20`, `lead ≥ 1.0 s`, `n_gross_premature
== 0`) and finding the best `useful_warning_rate_v2` among those:

| seed | best cell | useful_v2 | FA | lead_vs_event_v2 | gap to 0.70 |
|---|---|---|---|---|---|
| **1234** | thr 0.70 / confirm 8 | **0.612** | 0.196 | 1.83 | **0.088** |
| 1235 | thr 0.70 / confirm 5 | 0.524 | 0.196 | 1.25 | 0.176 |
| 1236 | thr 0.90 / confirm 3 | 0.544 | 0.147 | 1.35 | 0.156 |

`n_gross_premature` is never the binding constraint — every seed has multiple
cells with `n_gross_premature == 0` (it only turns nonzero at the loosest
thresholds, thr ≤ 0.60, where FA is already far over target and those cells
are excluded on the FA gate first). The rule is failing on useful-warning
coverage at the FA-compliant end of the grid, not on gross prematurity.

Seed 1234 comes closest (0.612 vs. the 0.70 floor, an 0.088 gap), and it is
the closest by a real margin over seeds 1235/1236, not by a marginal amount —
its next-nearest competitor is 0.156 short. This is *some* evidence toward a
consistent ordering rather than one lucky cell, since seed 1234 is also ahead
at most nearby thresholds in its own grid (thr 0.70/confirm 3 → 0.699, thr
0.70/confirm 5 → 0.689 — both also close to 0.70 and both still short).

## 2. Proceeding on a documented near-miss, by explicit instruction

Zero survivors was reported and the decision was handed up rather than made
here. The instruction received back: proceed with **seed 1234 at threshold
0.70 / confirm 8** for the dashboard rebuild and sanity check, with the
shortfall recorded plainly rather than treated as met.

**`useful_warning_rate_v2 = 0.612` against the rule's `≥ 0.70` floor — a gap
of 0.088, not closed.** This configuration does **not** satisfy the
documented selection rule. It is being carried forward as a demonstration
and interim configuration, not as a validated result.

### Why the 0.70 floor itself is untrusted, not just the miss

The `≥ 0.70` useful-warning floor comes from `KAPPA_RETRAIN.md`'s original
decision rule, written **before**:

- the partition fix (Step 6 — `data/nexar/index.csv`'s split was redrawn
  after `label_window_audit.md` found the original eval split held every
  clip with a window over 2.97 s, biasing every prior number toward
  easier positives),
- the useful-warning metric fix (Step 8 — the legacy first-crossing
  definition could score a clip that warned continuously through the whole
  actionable window identically to one that never warned at all), and
- the checkpoint-selection fallback fix (Step 9 — the training loop could
  silently keep a chance-level epoch when no epoch met `fa_target`).

The 0.70 floor was never independently re-validated as achievable, or even as
the right number, under any of those three corrected conditions — it is a
number carried over from a measurement regime three fixes removed from this
one. **Missing it now is not the same claim as failing a floor that was set
under fair, current conditions and validated as reachable.** It may still be
the right floor; it has simply never been checked against a fair regime, and
that absence of validation is itself part of why this is being treated as a
near-miss worth investigating rather than a hard failure.

## 3. Multi-seed check at the selected cell (thr 0.70 / confirm 8)

The 0.088 gap is smaller than the cross-seed spread already observed at
comparable operating points in `selection_fix_and_tradeoff.md` (population sd
0.10–0.12 on `useful_warning_rate_v2` across seeds at fixed hyperparameters).
A single seed cannot distinguish "0.612 is representative" from "0.612 was a
lucky draw" when the measurement noise is comparable to the gap being
measured. Two more seeds (1237, 1238) were trained with **identical
hyperparameters** to seed 1234's run (kappa 1.0, pre-alert-weight 0.5,
pos-weight 1.0, fa-target 0.20, epochs 30, patience 8, batch-size 8, hidden
64/layers 1, `train_core_v2` → `train_val_v2`) and scored at this exact cell.

| seed | best epoch | fa_target_met | fallback_used | val mean AP | useful_v2 @ thr 0.70/c8 | FA | lead_vs_event_v2 |
|---|---|---|---|---|---|---|---|
| 1234 | 4 | False | False | 0.630 | 0.612 | 0.196 | 1.83 |
| 1235 | 13 | False | False | 0.590 | 0.495 | 0.137 | 1.27 |
| 1236 | 6 | False | True | 0.648 | 0.864 | 0.431 | 2.54 |
| 1237 | 4 | False | False | 0.580 | 0.388 | 0.078 | 1.13 |
| **1238** | **1** | **True** | **False** | **0.490** | **0.107** | 0.029 | 1.01 |

**All five: mean 0.493, range 0.107–0.864, population sd 0.250.**

### Seed 1238 needs a separate flag: it appears to be a chance-level
checkpoint the Step 9 fix does not catch

Seed 1238's selected epoch (epoch 1) has **mean AP 0.490 — at or below chance
(0.502) on this validation set.** This is the exact failure pattern Step 9
fixed for seed 1236 (an under-trained epoch mistaken for a good one) — except
here it is **not** caught by that fix, because epoch 1 happened to meet
`fa_target` on the *primary* selection rule (FA 0.127 ≤ 0.20 at the training
loop's threshold 0.5/confirm 3 scoring point). The near-chance fallback
(`docs/status/selection_fix_and_tradeoff.md` §1) only activates when **no**
epoch ever meets `fa_target` — it was never designed to catch a compliant
epoch that is itself near-chance, because meeting the FA gate was treated as
sufficient evidence of a legitimate result. Seed 1238 shows that assumption
does not always hold. **This is flagged, not fixed, here** — fixing it is out
of this task's scope and would need its own investigation (the same kind of
"stop and ask" moment as Steps 8 and 9's mid-task corrections), not a
same-task patch.

Excluding seed 1238 as an apparent-degenerate outlier rather than a
representative draw:

**Four seeds (1234/1235/1236/1237): mean 0.590, range 0.388–0.864, population
sd 0.177.**

Both statistics are reported because the honest answer is that seed 1238's
status is itself ambiguous — a near-chance checkpoint could be a real,
if unlucky, draw from the same training-noise distribution as the others, not
necessarily a bug to be excluded by fiat. Excluding it is a judgment call, not
an established fact.

### What this means for interpreting the near-miss

The 4-seed mean (0.590, excluding the apparent outlier) sits **below**
0.612 — the single-seed figure that produced the 0.088 gap is *not* an
unlucky low draw; if anything it is slightly above the more representative
mean. Combined with the full 5-seed sample's much lower mean (0.493) and very
wide spread (sd 0.250, more than double the earlier 4-seed estimate this
document opened with), **the single-seed 0.612 understates how much
uncertainty actually surrounds this cell.** A useful-warning figure anywhere
from roughly 0.11 to 0.86 has been observed at the identical hyperparameters
and identical operating point, differing only by training seed. **Step 11's
single eval read at this configuration should be read against that spread,
not against 0.612 as if it were a stable number.**

## 4. Dashboard rebuild

Per instruction, the seed-1234 checkpoint is used regardless of what the
multi-seed check shows, since it is already trained and available and is the
configuration explicitly authorized to proceed. Rebuilt with
`scripts/build_dashboard_demo.py --checkpoint
notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt --threshold 0.70 --confirm
8`, against the demo clip pool already used before: `--clips auto --n 6`
(picked 621, 488, 1004, 690, 1042, 1054) and the six named clips
(621, 488, 1004, 690, 1085, 1261, all present with local video). Both runs
completed with no errors; `scripts/update_manifest.py --require-video`
rebuilt the dropdown afterward.

**`build_dashboard_demo.py` and the dashboard's MODEL COMPARISON panel now
show both verdict definitions.** Previously the panel (`dashboard/app.js`'s
`renderCompare()`) rendered only the legacy learned-head verdict. Added:
a `verdict_v2` (and `window_covered_v2` boolean) field to each clip's
`risk.json`, computed with `find_alert_episodes()` exactly as
`train_risk_head.py`'s `evaluate()` does — a clip is `useful` under v2 if any
confirmed episode's active interval intersects `[alert_t, event_t]`, not
only the first crossing. The comparison panel gained a `↳ v2 (episode-based)`
row underneath the legacy verdict, and the console summary prints
`learned[legacy]` and `learned[v2]` side by side. The v2 row has no single
fire time to show (the definition asks whether the window is covered at all,
not when the first episode began), so its "when" column shows "window
covered" / "window not covered" in words rather than a timestamp.

### ⚠️ This demo pool overlaps the seed-1234 model's own training set

**Checked before writing the table below, because this project has caught
exactly this kind of leakage repeatedly.** The demo pool (`--features
data/features/eval`, the default) is the **v1** eval-split directory — a
different directory from `data/features/eval_v2`, so the hard rule ("do not
read `data/features/eval_v2`") is not violated. But by **clip identity**,
checked against `eval/split_freeze_v2.json`:

| clip | v2 membership |
|---|---|
| 621 | `train_val_v2` (validation only — not trained on) |
| 488, 1004, 690, 1085, 1261, 1054 | **`train_core_v2` — the model was TRAINED on these** |
| 1042 | `eval_v2` (the current held-out split, by identity — see below) |

**Six of the eight demo clips are inside seed 1234's own training set.**
Their `useful`/`clean` verdicts in the table below reflect the model having
seen these exact clips during training, not generalisation. This is exactly
why the task frames this set as illustrative rather than evidentiary, and it
is stated here as plainly as possible: **do not read anything in Section 4 as
evidence of model quality.**

Clip 1042 is a separate, smaller note: it is classified `eval_v2` by identity
in the current freeze, even though its features here were read from the v1
`data/features/eval/` directory, not `data/features/eval_v2/` — the hard
rule's literal subject. No file named `eval_v2` was opened. Flagged anyway,
because a technically-compliant read of a held-out clip's identity through a
side door is exactly the kind of thing worth surfacing rather than leaving
implicit.

### Demo-clip table (illustrative only — see the warning above)

| clip | label | v2 split membership | legacy verdict | v2 verdict | window covered (v2) | fire time | lead (event − fire) |
|---|---|---|---|---|---|---|---|
| 621 | pos | train_val_v2 | useful | useful | True | 22.80 s | 2.90 s |
| 488 | pos | **train_core_v2 (trained on)** | useful | useful | True | 20.43 s | 3.17 s |
| 1004 | pos | **train_core_v2 (trained on)** | useful | useful | True | 21.03 s | 0.77 s |
| 690 | pos | **train_core_v2 (trained on)** | useful | useful | True | 19.03 s | 2.10 s |
| 1085 | neg | **train_core_v2 (trained on)** | clean | clean | — | — | — |
| 1261 | neg | **train_core_v2 (trained on)** | clean | clean | — | — | — |
| 1042 | neg | eval_v2 (by identity; read via v1 dir) | clean | clean | — | — | — |
| 1054 | neg | **train_core_v2 (trained on)** | false alarm | false_alarm | — | — | — |

Legacy and v2 agree on every demo clip shown — none of them happen to
illustrate the clip-1013-style disagreement this whole metric fix was built
around, which is itself a reminder that a demo set this small (and this
overlapped with training) is not where that distinction would reliably show
up. `early_firing_diagnosis.md` and `metric_fix_results.md` remain the actual
evidence for the legacy/v2 gap, not this table.

## 5. Sanity check before the eval read

1. **Checkpoint exists and loads:**
   `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt` — confirmed, loaded
   successfully by both the sweep in Section 1/3 and the dashboard build
   above (`RiskGRU.load()` printed `best epoch 4, trained kappa=1.0
   pre_alert_weight=0.5` with no error).
2. **Operating point fully specified:** threshold **0.70**, confirm **8**.
3. **Dashboard runs end-to-end with no errors:** both builds (`--clips auto
   --n 6` and the six named clips) completed with zero skipped clips and zero
   exceptions; `update_manifest.py --require-video` rebuilt the dropdown
   afterward with no error. Served locally (`scripts/serve_dashboard.py`) and
   verified directly: `index.html`, `app.js`, `clips/manifest.json`, and
   clip 621's `risk.json` / `events.json` / `raw.mp4` all returned HTTP 200,
   and `risk.json` parses as valid JSON containing the new `verdict_v2` /
   `window_covered_v2` fields.

`data/features/eval_v2` has not been read at any point in this task.
