# Judges-prep plan (written 2026-09-27, revised 2026-09-29)

Written after a fresh audit of the working tree on top of everything
`docs/REPORT_evaluation_audit.md` already found. Nothing here contradicts that
report; this document turns it into a one-week execution plan. Priority, set
explicitly: **raise the headline number** and **live demo quality**, in that
order, without reopening the four methodology bugs the project already spent
eleven steps fixing.

**Where this stands (2026-09-29, Day 3 of 7).** The headline-number attempts
(Part 2) are finished: nothing beat the baseline beyond noise, so the
headline stays. The rest of the week is **evidence for Q&A, the paper's open
placeholders, and the demo**. Part 3 is the plan from here; Part 1 and Part 2
are the record.

**Current locked state, for reference:** `risk_gru_k1p0_v2_selfix_s1234.pt`,
threshold 0.70 / confirm 8, `eval_v2` (120 clips, held out): useful-warning
**0.650** (39/60, v2 definition) / **0.317** (19/60, legacy), false-alarm
**0.167** (10/60), mean lead **1.51 s**, mean AP **0.646**. One read, one
seed, already spent (`docs/status/final_eval_read.md`).

**The one rule that overrides everything below:** `eval_v2` gets read **at
most once more**, and only for a single config that has already been locked
on `train_val_v2`. Every experiment before that is scored on `train_val_v2`
only. Reading the test set repeatedly, on different candidates, is
mechanically the same mistake that produced the original 0.750 leak
(`docs/status/eval_leakage_audit.md`). **Recommendation for this week: do not
spend it.** No candidate earned it (Part 2), and an unspent read costs
nothing.

---

## Part 1 — Bugs found, and their status

Ordered by how much damage they'd do if a judge (or you, mid-Q&A) hit them.

| # | issue | status |
|---|---|---|
| 1.1 | `INDEX.md` said there was no quotable number | ✅ fixed Day 1 |
| 1.2 | Hardcoded `MEAN_ALERT_LEAD_S = 3.49` | ✅ fixed, commit `3a4b349` |
| 1.3 | Ablation reruns "uncommitted" | ✅ was already committed; retracted |
| 1.4 | `dashboard_v2` uncommitted and unreviewed | ✅ committed (`49923ce`); ⬜ manual QA still open (3.4) |
| 1.5 | Seq2Seq-LSTM "beats Kalman" doesn't transfer | ✅ tested end to end (Part 2) |
| 1.6 | BEV calibration fails plausibility | ⚠️ known limit; demo mitigation in 3.4 |
| 1.7 | `status/results.md` describes the superseded system | ⬜ Day 6 |
| 1.8 | Paper `.docx` drafts in repo root | ✅ `-3` is current, gitignored |
| **1.9** | **Demo clip list in this plan was the contaminated set** | ✅ corrected here (3.4) |
| **1.10** | **Handoff docs still quoted 0.750 as current** | ✅ fixed 2026-09-29 |
| **1.11** | **Part 2 compared legacy against v2 numbers** | ✅ re-scored 2026-09-29 |
| **1.12** | **`build_dashboard_demo.py` defaults rebuild the contaminated demo** | ⬜ Day 3 (3.1) |
| **1.13** | **Paper has stale forecasting numbers and fillable placeholders** | ⬜ Day 4 (3.3) |

### 1.1 `docs/INDEX.md` contradicted the project's own final result — ✅ fixed

The "Quote a headline number" row said there was no quotable held-out number
after the single `eval_v2` read had already happened. Now points at
`REPORT_evaluation_audit.md` and quotes 0.650 / FA 0.167.

### 1.2 Hardcoded wrong constant — ✅ fixed in `3a4b349`

`MEAN_ALERT_LEAD_S = 3.49` in `compare_kappa.py` and `train_risk_head.py` was
measured over the old 65-clip hand-truncated pool, not the real positive
population (mean 1.60 s). Commit `3a4b349` replaced it with
`mean_alert_lead_s()` computed from the loaded clips
(`a3ps/risk/anticipation_loss.py`, fallback 1.60 s, tested). Diagnostic
output only; no scored metric changed. The only remaining `3.49` is the
old-system report text in `scripts/eval_anticipation.py:99,757`, which
describes the superseded split and is correct in that context.

### 1.3 Ablation reruns uncommitted — retracted

Written from `pivot_open_questions.md` §4.3's caveat without checking git.
The four `risk_gru_abl_*.pt` checkpoints and `eval/ablation_rerun_2026-09-22/`
are tracked, and the `ablations.md` banner was fixed in `1f60a04`.

Remaining caveat for Q&A: every ablation and the capacity run in
`ablations.md` was trained and scored against the v1 `eval` split under the
biased protocol. Relative ranking at best, never comparable with 0.650. The
paper therefore has an open `[PENDING]` for ablations; 3.2 closes it.

### 1.4 `dashboard_v2` — committed, not yet clicked through

Committed in `49923ce` (`dashboard_v2/`, `scripts/dashboard_v2_data.py`,
`scripts/serve_dashboard_v2.py`; `tests/test_dashboard_v2_data.py` passes).
The data layer has tests, but nobody has driven the UI end to end. It is the
best demo asset and the least-audited code. QA is in 3.4.

### 1.5 Seq2Seq-LSTM "beats Kalman" — ✅ answered by an end-to-end test

The LSTM's ADE/FDE win at 4 s was measured on fixed 10-point histories and
never pushed through the risk head. Both gaps are now closed
(`docs/status/headline_number_attempts.md` §2):

- **Padded histories:** re-scored at 8 and 9 points (what `TrackBuffer.ready()`
  actually gates on). The 4 s edge survives unchanged.
- **End to end:** re-extracted all 1,485 clips with `forecaster: seq2seq` and
  trained the same GRU. At each run's best passing cell on `train_val_v2`:
  useful_v2 **0.621 vs 0.612** (a tie), FA 0.176 vs 0.196, but lead **1.27 s
  vs 1.83 s**. Mean AP 0.622 vs 0.630.

**The Q&A answer:** "The LSTM wins on 4-second trajectory error, but Kalman is
about 31% more accurate at 1 s, which is where our alerts live. Pushed
through the full risk pipeline, the LSTM matched coverage but warned half a
second later, so Kalman stays the default." Do **not** say the LSTM features
"collapsed" the number. An earlier draft said that on mismatched definitions
(1.11).

### 1.6 BEV/ground-plane calibration — known limit, demo mitigation only

The ground plane is one default trapezoid, never calibrated per clip, and no
`ground.yaml` override exists. `verify_bev.py` marks 6 of 8 full-pipeline
clips `SUSPECT`.

**New (2026-09-29):** `verify_bev.py` was also run on the six clean demo clips
(3.4). All six print `SUSPECT`, but for a different reason: their
`events.json` are slim demo overlays with **no `centroid_bev` at all**, so the
check has nothing to test. `dashboard_v2` fills those positions by projecting
image foot-points through the same uncalibrated trapezoid
(`_fill_ground()` in `scripts/dashboard_v2_data.py`, tagged `"projected"`).
So the simulator pane on every demo clip is a geometric illustration, not a
measurement. That is fine if you say so up front. Don't fix calibration this
week (out of scope).

### 1.7 `status/results.md` still describes the superseded system

Flagged in `INDEX.md` §4, but a file literally named "results" still opens
on 150+ lines of pre-pivot numbers. Day 6: trim it to a pointer.

### 1.8 Paper `.docx` drafts — ✅ resolved

`A3PS_paper-3.docx` is the current draft (quotes 0.650 / 0.167); `-2` is the
July pre-pivot draft. Paper drafts stay local and are never pushed:
`A3PS_paper*.docx` is in the root `.gitignore`.

### 1.9 This plan's demo clip list was the contaminated set — ✅ corrected

The first version of §3.1 and §3.2 said to demo `621, 488, 1004, 690, 1085,
1261` ("chosen for the sharpest old-vs-new contrast per `TODO.md`"). That
list predates the decontamination in
`docs/status/instability_and_checkpoint_audit.md` §3, which found **6 of 8
clips in the old pool were in seed 1234's own `train_core_v2`**. Of the six
named here, only 621 is clean. Two of them (1004, 1085) are also BEV
`SUSPECT`. The clean set is **621, 206, 630, 932, 1426, 1564**
(`train_val_v2`, never trained on). See 3.4.

The dashboard makes this easy to get wrong: `dashboard/clips/manifest.json`
lists the stale clips in the **same** "Phase IV — learned vs threshold
system" group, with the same kind of label ("488 · pos · learned useful / old
too early"), and neither dashboard shows a clip's split. A judge picking 488
would see a success on a training clip.

### 1.10 The handoff front door still quoted the superseded headline — ✅ fixed 2026-09-29

`handoff/README.md` and `handoff/NEXT_STEPS.md` presented `risk_gru_k1p0.pt`,
threshold 0.60, useful-warning 0.750 as the current state. `NEXT_STEPS.md` §1
also told the reader to train with `--val-features data/features/eval`, the
exact leak the audit found (the guard now refuses it, but the doc still
said to do it). `INDEX.md` marked all of `handoff/` "current ✅", and
`status/TODO.md` still named capacity as the next lever. All four were
rewritten or corrected on 2026-09-29.

### 1.11 Part 2 compared legacy numbers against a v2 baseline — ✅ re-scored

`sweep_operating_point.py` prints the **legacy** useful-warning. The first
write-up of Part 2 read that column for each candidate (0.29–0.41) and
compared it with the baseline's **v2** 0.612, concluding that every
candidate "collapsed". At the same cell (0.70/c8) the baseline's own legacy
figure is **0.243**, lower than every candidate's. Re-scored everything on one
footing with `scripts/rescore_candidates_both_defs.py` →
`eval/candidates_both_definitions.md`. Conclusion unchanged (keep the
baseline); strength of it corrected (ties within noise, not collapses). See
Part 2.

**Rule from now on:** never read the "useful" column of an
`operating_point_sweep_*.md` against a v2 number.

### 1.12 `build_dashboard_demo.py` defaults rebuild the contaminated demo

`--checkpoint` defaults to `notebooks/models/risk_gru_k1p0.pt` (the biased
v1 model), `--features` to `data/features/eval` (the v1 pool the
decontamination moved away from), and `DEFAULT_THRESHOLD = 0.60` (the
superseded operating point). Anyone running the rebuild command the old
`handoff/README.md` gave (`--clips auto --n 6`) regenerates the contaminated
demo. Fix in 3.1.

### 1.13 Paper draft: stale numbers and placeholders now fillable

`A3PS_paper-3.docx` (local only), checked 2026-09-29:

- **Forecasting paragraph is pre-recalibration.** It says Kalman is "slightly
  more accurate up to 2 s" and quotes Kalman 4 s ADE/FDE 61.27 / 122.55. The
  regenerated `eval/forecast_table.md` has Kalman **12.94** vs LSTM 18.85 ADE@1s
  (~31% lower), 26.41 vs 32.68 ADE@2s, and at 4 s ADE 60.62 vs 56.33, FDE
  **130.36** vs 99.74. The paper's forecast table needs the same update.
- **`[VERIFY: LSTM ADE@4s and FDE@2s both 56.33]`** — genuine coincidence,
  not a transcription error. Different code paths (`ade_fde_at()`: mean over
  20 steps vs error at step 10), and they diverge under truncation (56.29 vs
  56.24 at 8 points, `eval/forecast_table_hist8.md`). Remove the flag.
- **`[GPU RESULT]`, `[PENDING: re-time perception on GPU]`** — done:
  `results/latency_gpu_summary.md`. RTX 3050 Laptop (4 GB), fp16, 15 dev
  clips, 8,170 frames: fused perception + tracking **61.9 ms/frame** (YOLO
  34.7 + BoT-SORT/CMC ≈ 27.2 on CPU), vs 81.65 ms on CPU (1.32×). End to end
  with the GRU head ≈ **81.9 ms/frame, 12.2 fps**. Still not camera rate, so
  keep "we do not claim real-time operation at 30 fps". Update the
  Limitations line "latency for perception was measured only on a CPU".
- **`[FILL IN GPU model and VRAM]`** (extraction machine) — **careful.** The
  paper says extraction ran at 8.4 s/clip; this RTX 3050 measured ~23.5
  s/clip (`headline_number_attempts.md` §2). Those are probably different
  machines. Fill in whichever GPU actually did the 8.4 s/clip extraction,
  not the 3050, unless you confirm it was the same.
- **Section VII forecaster comparison** only reports ADE/FDE. Add one
  sentence with the end-to-end result (1.5 above). It is the strongest
  evidence the paper has for keeping Kalman.
- **`[PENDING: feature-group ablations]`** — 3.2.
- **`[INSERT LLM before/after example]`** — 3.3.
- **`[PENDING: re-score threshold engine on eval_v2 windows]`** and the
  "No comparable baseline yet" limitation — partly addressed by 3.2's simple
  baselines on `train_val_v2`; the full re-score stays future work.

---

## Part 2 — Try to raise the headline number (done: nothing beat the baseline)

Tried on `train_val_v2` only; `eval_v2` untouched. Numbers are each run's
best cell passing FA ≤ 0.20, v2 lead ≥ 1.0 s, zero gross-premature
(`eval/candidates_both_definitions.md`):

| run | mean AP | best cell | useful_v2 | FA | lead_v2 |
|---|---|---|---|---|---|
| **baseline** (h64, l1, Kalman) | 0.630 | 0.70 / c8 | **0.612** | 0.196 | **1.83 s** |
| hidden 128, layers 2 | 0.601 | 0.60 / c8 | 0.553 | 0.147 | 1.48 s |
| hidden 256, layers 2 | 0.641 | 0.60 / c3 | 0.573 | 0.186 | 1.28 s |
| LSTM-forecaster features | 0.622 | 0.60 / c8 | 0.621 | 0.176 | 1.27 s |

One validation clip is 0.010; seed sd for the same recipe was 0.159. All
four are the same model to within noise on coverage and AP. Every
alternative warns later. Together with the earlier kappa sweep (AP flat
within 0.004), five-seed ensemble (+0.019, CI straddles zero) and
temperature calibration (−0.019), that is **five levers, none moved the
number**. Full write-up: `docs/status/headline_number_attempts.md`.

**Longer feature window (`--window-s 20`) — not attempted,** and now
deprioritised further: the loss gives zero signal outside `[alert_t,
event_t]` (`REPORT_evaluation_audit.md` §5.3), and 13 s already covers ~11 s
of run-up before a ~1.6 s window. A longer window changes what the model can
see, not what it is asked to predict. See `FUTURE_WORK.md`.

**Lock:** the current checkpoint stays the headline. The last `eval_v2` read
is unspent.

---

## Part 3 — The plan from here (Days 3–7)

### 3.1 Day 3 (today, 09-29): commit and close the demo trap (~2 h)

1. **Review and commit** this session's edits and the untracked results
   they cite: `docs/status/headline_number_attempts.md`,
   `eval/candidates_both_definitions.md`,
   `scripts/rescore_candidates_both_defs.py`, `configs/seq2seq_eval.yaml`,
   `eval/forecast_table*.md`, the h128/h256/seq2seq checkpoints, histories,
   logs and sweeps, `figures/`, `results/`, `scripts/benchmark_latency_gpu.py`,
   and the doc edits (`INDEX.md`, `handoff/*`, `status/TODO.md`,
   `REPORT_evaluation_audit.md`, `new-architecture.md`). Paper `.docx` files
   stay local.
2. **Fix `build_dashboard_demo.py` defaults (1.12)**: `--checkpoint` →
   `notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt`, `--features` →
   `data/features/train_val_v2`, `DEFAULT_THRESHOLD` → 0.70. ~10 min.
3. **Separate the stale clips in the dropdown (1.9)**: in
   `scripts/update_manifest.py`, put `488, 1004, 690, 1085, 1261, 1054, 1042`
   in their own group, e.g. "Pre-decontamination (training-set clips) — not
   for demo", then re-run it with `--require-video`. Edit the script, not
   `manifest.json`: the script regenerates the file. ~20 min.
4. **Start the ablation runs in the background** (3.2). They are CPU-only and
   take about an hour total.

### 3.2 Day 3–4: feature ablations and simple baselines on `train_val_v2`

These answer the two questions a judge is most likely to ask about the
model: "which features matter?" and "compared to what?" Both are cheap,
neither touches `eval_v2`, and both close paper placeholders.

**Ablations** (~4 min training each, same recipe as the baseline):

```powershell
foreach ($g in "ego","collision_prob","ttc","corridor") {
  python scripts/train_risk_head.py `
      --features data/features/train_core_v2 --val-features data/features/train_val_v2 `
      --kappa 1.0 --pre-alert-weight 0.5 --hidden 64 --layers 1 --fa-target 0.20 `
      --epochs 30 --patience 8 --batch-size 8 --seed 1234 --ablate $g `
      --out notebooks/models/risk_gru_abl_${g}_v2_s1234.pt `
      --history-json eval/risk_gru_history_abl_${g}_v2_s1234.json *>&1 `
      | Tee-Object eval/train_log_abl_${g}_v2_s1234.txt
}
```

Score them with `scripts/rescore_candidates_both_defs.py` (add the four rows
to `CANDIDATES`), **not** by reading the sweep's useful column (1.11).

**Read them by mean AP first.** It is the more stable single-seed signal.
Useful-warning moves by ±0.16 between seeds of the same recipe, so a one-seed
ablation cannot resolve small useful-warning effects. `ablations.md` used
±0.022 as the mean-AP noise level, and its v1 post-fix rerun found every
group inside roughly that band (Δ +0.009 to −0.030). Expect the same, and
only call a group "important" if removing it drops mean AP clearly beyond
~0.03; otherwise report "no single group is necessary; the model uses them
redundantly", which is itself a finding.
Write the result to `docs/status/ablations_v2.md` and put that table in the
paper in place of the `[PENDING]`.

**Simple baselines** (~2 h, new small script): score, through the same
`evaluate()` machinery and grid, (a) the single feature `max_prob` (frame-level
max collision probability) thresholded with confirm-N, and (b) a per-frame
logistic regression on the same 112-d features, fit on `train_core_v2`. This
tells you whether the GRU adds anything beyond its best input and a linear
model. Either outcome is reportable: "the GRU beats both by X" or "most of the
signal is in `collision_prob`". It also gives the "No comparable baseline
yet" limitation a partial answer without re-running the threshold engine.

### 3.3 Day 4: paper fixes (no compute)

Work through 1.13 in order: forecasting paragraph and table, remove the
56.33 `[VERIFY]`, fill the GPU latency rows and fix the Limitations sentence,
add the end-to-end LSTM sentence to Section VII, drop in the ablation table,
and produce the LLM before/after example with
`a3ps/explain/llm_client_permissive_test.py` (~30 min, needs the Groq key).
Resolve the three reference `[VERIFY]`s. Don't guess the extraction GPU.

### 3.4 Day 5: click through `dashboard_v2` end to end

Nobody has done this manually yet (1.4). Start it:

```powershell
python scripts/serve_dashboard_v2.py --port 8010
```

**Demo clips: 621, 206, 630, 932, 1426, 1564 only** (all `train_val_v2`, never
trained on; `instability_and_checkpoint_audit.md` §3):

| clip | label | verdict (legacy / v2) | fire time | lead | use it to show |
|---|---|---|---|---|---|
| 621 | pos | useful / useful | 22.80 s | 2.90 s | the best-case warning |
| 206 | pos | useful / useful | 18.73 s | 2.00 s | a typical useful warning |
| 932 | pos | useful / useful | 19.83 s | 0.20 s | a warning too late to act on |
| 630 | pos | miss / missed | — | — | an honest failure case |
| 1426 | neg | clean / clean | — | — | no false alarm |
| 1564 | neg | clean / clean | — | — | no false alarm |

Check, against real data, not just "does it load":

- **Live replay:** for each of the six, confirm video + labels + masks +
  trails + predicted paths + ego corridor render, the simulator pane tracks
  the same clock, and the alert/brake banners fire in both panes at the same
  instant. Watch the simulator pane specifically: its positions are
  **projected through the uncalibrated default ground plane** (1.6), so
  decide per clip whether it looks plausible enough to show.
- `⏮ alert` / `alert ⏭` (or `p`/`n`) jump to 1.5 s before the previous/next
  alert.
- **Compare runs:** tick multiple runs and confirm the "different splits"
  warning appears when mixing an `eval`-scored run with an `eval_v2`-scored
  one.
- **History** and **Drill-down** render without a torch hiccup; test
  `--no-checkpoints` as the fast-boot path for a mid-session restart.
- Every chart's "table" button shows the underlying data.
- **Nice to have if time allows:** a split badge in the live-replay header
  (`train_val_v2` / `train_core_v2` / `eval_v2`), so nobody, including you,
  can mistake a training clip for evidence.

Budget a real afternoon. Fix what breaks.

### 3.5 Day 6: rehearse, record, slides, Q&A

- **Pick the demo path** from the table above and rehearse it twice. Include
  630 (the miss). Showing a failure on purpose is more convincing than six
  successes.
- **Record a backup screen capture** of the rehearsed path. If the live
  machine misbehaves, play the recording instead of debugging in front of the
  judges.
- **Slides:** `figures/fig_operating_point.*` (trade-off curve, locked point
  highlighted) and `figures/fig_qualitative_example.*` are ready; captions
  are in `figures/FIGURE_NOTES.md`. Say that clip 879's window (3.03 s) is
  about twice the median.
- **Docs cleanup (1.7):** trim `docs/status/results.md` to a two-line pointer
  at `docs/REPORT_evaluation_audit.md`. Don't delete it; git keeps the content.
- **Drill the Q&A list below.**

### 3.6 Day 7 (10-03): buffer

Full `pytest` run (194 passing as of 2026-09-29), dashboard smoke test on the
actual demo machine, and one timed dry run of the whole presentation.

---

## Part 4 — Q&A prep

Memorise the shape of `REPORT_evaluation_audit.md` §7 ("Viva-ready answer"),
not the words. Then have a one-to-two sentence answer for each of these:

| likely question | answer to have ready |
|---|---|
| What is the actual performance? | 0.650 useful-warning (39/60) at FA 0.167, lead 1.51 s, mean AP 0.646, on 120 held-out clips read once. Legacy definition: 0.317. Always pair useful-warning with FA. |
| Why trust it? | "We found our own headline had been selected on the test set, root-caused three more issues alongside it, fixed all four, and this number is what survived." |
| Did you try to improve it? | Five levers: kappa, ensemble, calibration, capacity, LSTM forecaster. None moved it beyond noise, all on validation only, and the test set was never used to pick among them. |
| Why is lead time only 1.5 s? | Nexar's annotated window averages 1.6 s and the loss gives no signal before it, so 1.5 s is what the labels ask for. Whether a different objective could warn earlier is untested, and that's the top future-work item. |
| Is 1.5 s enough for a driver? | Have one sourced figure ready. Typical brake reaction for an alert driver is usually quoted around 1.5 s (e.g. Green, 2000), and road-design standards use 2.5 s (AASHTO). **Check both citations before quoting.** Honest answer: it's borderline, and that's why lead time is reported as a missed target. |
| Why Kalman and not the LSTM? | See 1.5: better where alerts live (1–2 s), and end to end the LSTM tied on coverage but warned 0.56 s later. |
| Why a GRU and not a transformer / end-to-end video model? | ~1,160 training clips; a bigger GRU didn't help (Part 2), so a bigger architecture is unlikely to. The interpretable feature pipeline is also a design goal: every alert traces back to logged geometric cues. |
| How does this compare to published work / the Kaggle leaderboard? | Not directly comparable: our 0.646 is on a local held-out split, not the competition test set. Look up the Nexar challenge's top public mAP before the day, and name the Kaggle submission as the way to get a comparable number. |
| What is "agentic" about it? | The paper expands the name as "Agentic and Explainable Pipeline". The honest framing: a closed perceive → forecast → assess → decide (SAFE/ALERT/BRAKE) → explain loop that logs every decision with its inputs. It is not an LLM agent. Be ready to concede the name is aspirational if pushed. |
| Can it run in real time? | Perception + tracking 61.9 ms/frame on an RTX 3050 Laptop; about 12 fps end to end. The risk head is 0.25 ms. Not 30 fps, and the paper doesn't claim it. |
| Which features matter? | From 3.2's ablations, once run. Until then: "In the v1 post-fix rerun, removing any one group (ego, corridor, collision probability, TTC) moved mean AP by at most 0.03, inside that run's noise level, so no single group was carrying the model. Those runs were selected under the flawed protocol, so we re-ran them." Don't cite the pre-fix `abl_ttc` AP 0.465; that run was invalid. |
| Wasn't the metric changed after you saw the results? | Yes, and the paper says so. The justification was committed before any code changed, v2 is a strict superset of legacy (0 of 45 violations), and legacy is always reported beside it. |
| How seed-dependent is it? | Very, on useful-warning (validation range 0.107–0.864 across five seeds at this operating point); much less on mean AP (0.58–0.68). It is one seed, and the paper says so. |
| Were any held-out clips ever seen? | 12 of the 120 `eval_v2` clips were in the v1 eval split, which influenced v1-era hyperparameter choices. Stated as a limitation; probably small, not quantified. |
| Why do the simulator positions look off? | The ground plane is an uncalibrated default; the simulator is an illustration of the pipeline's tracks, not a metric reconstruction. Per-clip calibration is future work. |

---

## Day-by-day summary

| Day | Date | Focus | Output | Status |
|---|---|---|---|---|
| 1 | 09-27 | Bug fixes (1.1–1.3, 1.8) | INDEX fixed, constant fixed, docx resolved, `dashboard_v2` committed | ✅ |
| 2 | 09-28 | Capacity sweep | h128 / h256 trained and swept | ✅ |
| 3 | 09-29 | LSTM end to end; audit fixes (1.9–1.11); commit; demo trap (3.1); start ablations | Corrected docs, both-definition re-score | 🔄 in progress |
| 4 | 09-30 | Ablations + simple baselines (3.2); paper fixes (3.3) | `ablations_v2.md`, baseline table, paper placeholders closed | ⬜ |
| 5 | 10-01 | Dashboard QA on the clean six (3.4) | Bug list fixed; demo clips vetted | ⬜ |
| 6 | 10-02 | Rehearse, backup recording, slides, `results.md` trim, Q&A (3.5) | Rehearsed demo + recording | ⬜ |
| 7 | 10-03 | Buffer (3.6) | `pytest` green, smoke test, dry run | ⬜ |

---

## What not to do under deadline pressure

- Don't read `eval_v2` this week. Nothing earned it.
- Don't quote a validation number (Part 2, ablations, baselines) as if it were
  held out.
- Don't compare an `operating_point_sweep_*.md` "useful" column (legacy)
  against a v2 number (1.11).
- Don't say the LSTM "collapsed" the number. Say it tied on coverage and warned
  later (1.5).
- Don't demo any clip outside 621 / 206 / 630 / 932 / 1426 / 1564. The other
  Phase IV clips in the dropdown were in the training set (1.9).
- Don't present the simulator pane's positions as calibrated (1.6).
- Don't put "RTX 3050" in the paper's extraction sentence unless that was the
  extraction machine (1.13).
