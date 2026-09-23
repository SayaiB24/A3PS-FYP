# Operating-point selection: the rule has zero survivors (2026-09-23)

`data/features/eval_v2` was **not read** to produce this document. **No
operating point is selected. No dashboard was rebuilt.** Per the task's hard
rule, this document stops after Section 1 and reports rather than relaxing
any gate.

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

## What this document does not do

Per the task's hard rule: **zero survivors means this document does not
proceed to Sections 2–4.** No configuration is selected, no runner-up is
named as a selection, `build_dashboard_demo.py` was not run,
`data/features/eval_v2` was not read. Reported here for you to decide the
next step, rather than relaxing a gate, picking the closest miss as a de
facto selection, or otherwise deciding on your behalf.
