# Judges-prep plan (2026-09-27, one week out)

Written after a fresh audit of the working tree on top of everything
`docs/REPORT_evaluation_audit.md` already found. Nothing here contradicts that
report; this document turns it into a one-week execution plan. Priority, set
explicitly: **raise the headline number** and **live demo quality**, in that
order, without reopening the four methodology bugs the project already spent
eleven steps fixing.

**Current locked state, for reference:** `risk_gru_k1p0_v2_selfix_s1234.pt`,
threshold 0.70 / confirm 8, `eval_v2` (120 clips, held out): useful-warning
**0.650** (39/60, v2 definition) / **0.317** (19/60, legacy), false-alarm
**0.167** (10/60), mean lead **1.51 s**, mean AP **0.646**. One read, one
seed, already spent (`docs/status/final_eval_read.md`).

**The one rule that overrides everything below:** `eval_v2` gets read **at
most once more**, at the very end, for whichever single config is locked in
on Day 4. Every experiment before that is scored on `train_val_v2` only. This
is not caution for its own sake — reading the test set more than once,
repeatedly, on different candidates, is mechanically the same mistake that
produced the original 0.750 leak (`docs/status/eval_leakage_audit.md`). Breaking
this rule under deadline pressure would be worse than presenting 0.65 as-is.

---

## Part 1 — Bugs found this session

Ordered by how much damage they'd do if a judge (or you, mid-Q&A) hit them.

### 1.1 `docs/INDEX.md` contradicts the project's own final result

**Where:** [`docs/INDEX.md:30`](../INDEX.md#L30), the "Quote a headline
number" row.

**What's wrong:** it says *"Start here. There is currently no quotable
held-out number. The splits were redrawn on 2026-09-23 and the new eval split
has not been scored..."* and points at `status/repartition_results.md`. That
was true when INDEX.md was last touched (17:25, 2026-09-23), but the single
`eval_v2` read happened later that same day (commit `294f38f`, 20:10) and
produced exactly the number this document is missing. INDEX.md is the
project's own "start here" page — if anyone, including a judge, opens it
first, it tells them the opposite of the truth.

**Fix:** point that row at `docs/REPORT_evaluation_audit.md` (or
`status/final_eval_read.md` §1) and quote 0.650 / FA 0.167 directly. ~5
minutes.

### 1.2 Hardcoded wrong constant in two places

**Where:** [`scripts/compare_kappa.py:43`](../../scripts/compare_kappa.py#L43)
and [`scripts/train_risk_head.py:800`](../../scripts/train_risk_head.py#L800):

```python
MEAN_ALERT_LEAD_S = 3.49        # measured over the 65 local positives
```

**What's wrong:** `docs/status/label_window_audit.md` §5 already established
this was measured over the old, hand-truncated 65-clip pool, not the real
745-positive population (mean **1.60 s**). It feeds the "requested lead"
figure printed in training logs and in `eval/kappa_comparison.md`. It does
not affect any scored metric — only a printed/logged number — but it's
exactly the kind of thing a sharp question ("what lead is the loss actually
asking for?") will catch, and the honest answer is already computed
elsewhere (`repartition_results.md` §6: ~0.67 s at kappa 1.0).

**Fix:** replace the literal with the real pooled mean (or compute it from
the loaded feature set at call time). ~10 minutes, touches two files.

### 1.3 ~~Ablation reruns uncommitted, banner stale~~ — already resolved; INDEX.md was the stale part

**Correction (Day 1):** this item was written from
`pivot_open_questions.md` §4.3's caveat without checking git. The four
`risk_gru_abl_*.pt` checkpoints and `eval/ablation_rerun_2026-09-22/` are
tracked, and the `ablations.md` banner was fixed in commit `1f60a04`
(2026-09-22). What *was* still stale was `docs/INDEX.md`, which kept
describing the banner as unfixed and `eval_v2` as unscored. Fixed on Day 1.

Remaining caveat worth remembering for Q&A: every ablation and the capacity
run in `ablations.md` was trained/scored against the v1 `eval` split under
the biased protocol, so they give a relative ranking at best and must not be
compared with the 0.650 `eval_v2` number.

### 1.4 `dashboard_v2` — ~1,100 lines of Python + ~unknown lines of JS, entirely new, entirely uncommitted, entirely unreviewed by a human

**Where:** `dashboard_v2/` (5 JS files, CSS, HTML), `scripts/dashboard_v2_data.py`
(823 lines), `scripts/serve_dashboard_v2.py` (181 lines). `tests/test_dashboard_v2_data.py`
exists and passes (100 lines, part of the 193 green tests), so the data
layer has *some* coverage — but nothing has exercised the UI end to end.

**What's wrong:** this is your best demo asset (per the priority you set —
live demo quality) and the least-audited code in the repository. It isn't
referenced anywhere in `docs/INDEX.md` either, so there's no map from "start
here" to "this is what you show the judges."

**Fix:** commit it (Day 1), then manually click through every view (Day
4–5, Part 3 below). Add one line to `docs/INDEX.md` pointing at it.

### 1.5 The Seq2Seq-LSTM "beats Kalman" claim doesn't transfer to the system that ships

**Where:** `docs/status/pivot_open_questions.md` §1 (already investigated,
not by me this session, but worth restating because it's a live risk for
Q&A).

**What's wrong:** the LSTM's ADE/FDE win at 4 s (56.33 vs 61.27 px) was
measured on `mine_trajectories.py`'s fixed 10-point, fully-contiguous
histories. In production, a track becomes forecastable at 8 points
(`TrackBuffer.ready()`), at which point `Seq2SeqForecaster.predict()`
left-pads by repeating the oldest point — a degraded input never present in
the 289-window validation set. The LSTM was also never run through feature
extraction into the actual GRU risk head; the pipeline still defaults to
Kalman-CV in `configs/default.yaml`, unchanged since the first commit.

**Fix — do not attempt to fully resolve this in a week.** If the paper or
slides currently state "LSTM beats Kalman" as a flat claim, soften it to
what's actually true: *"beats Kalman on trajectory error under full
histories; not yet validated end-to-end through the risk pipeline or under
the padded-history condition real tracks forecast from."* Costs zero
compute, prevents a claim that doesn't survive a follow-up question.

### 1.6 BEV/ground-plane calibration fails plausibility on 6 of 8 real pipeline clips

**Where:** `scripts/verify_bev.py`, run against every real `events.json` in
`dashboard/clips/` (documented in `pivot_open_questions.md` §3): `1004`,
`1042`, `1054`, `1085`, `dev01`, `dev05` all come back `SUSPECT`; only
`02134_pipe` and `dev10` pass.

**What's wrong:** the ground plane is one fixed default trapezoid
(`configs/default.yaml`), never calibrated per clip, and no `ground.yaml`
override exists anywhere in the repo. The `dashboard_v2` simulator's 3D
positions for any clip outside the 6 curated demo clips are therefore a
geometric guess, not a calibrated measurement. This is a known, correctly
scoped limitation (`FUTURE_WORK.md` — "affects no headline metric") — the
risk is purely a live-demo one: if a judge asks to see an arbitrary clip
rather than the prepared six, the simulator pane may look visibly wrong.

**Fix:** don't fix the calibration (out of scope for a week). Do decide, before
the demo, which clips you will show, and avoid driving the simulator pane on
an uncurated clip live.

### 1.7 Doc sprawl: two documents actively describe a superseded system as current

**Where:** `docs/status/results.md` (header states it reflects the state
"immediately before the risk engine / agentic decision layer was
implemented" — i.e. pre-Phase IV) and `docs/design/metrics.md` §1–8 (flagged
in-file, but only if you scroll past the old material first).

**What's wrong:** already flagged honestly in `docs/INDEX.md` §4, so this
isn't a new discovery — but it means there is currently no single document
you can hand someone without a caveat attached. For a live Q&A, that caveat
is one beat too many.

**Fix:** Day 6 below — trim `results.md` to a two-line pointer at
`docs/REPORT_evaluation_audit.md`, rather than leaving 150+ lines of
pre-pivot numbers at the top of a file literally named "results."

### 1.8 Two untracked `.docx` files in the repo root, canonical version unclear

**Where:** `A3PS_paper-2.docx`, `A3PS_paper-3.docx` (repo root, both
untracked).

**Fix:** confirm which is the current draft, commit or move the other out
of the repo. Two minutes, but worth doing before anyone else opens the
wrong one.

---

## Part 2 — Try to raise the headline number

Per `docs/handoff/FUTURE_WORK.md` (Tier 1: "cheap, and likely to move a
reported number") and `NEXT_STEPS.md` §1: mean AP was **flat within 0.004**
across the entire kappa sweep (0.5 → 3.0), which rules out loss weighting as
the lever and points at model capacity or features instead. Capacity is the
cheapest thing nobody has tried yet.

### 2.1 Capacity sweep (Day 2, ~30–45 min compute total)

Baseline for comparison, **on `train_val_v2`, not `eval_v2`**: useful_v2
**0.612**, FA **0.196**, lead **1.83 s**, mean AP **0.630** at 0.70/confirm 8
(seed 1234, current committed config).

```powershell
python scripts/train_risk_head.py `
    --features data/features/train_core_v2 --val-features data/features/train_val_v2 `
    --kappa 1.0 --hidden 128 --layers 2 --fa-target 0.20 `
    --epochs 40 --patience 8 --batch-size 8 --seed 1234 `
    --out notebooks/models/risk_gru_h128_v2_s1234.pt `
    --history-json eval/risk_gru_history_h128_v2_s1234.json *>&1 `
    | Tee-Object eval/train_log_h128_v2_s1234.txt

python scripts/sweep_operating_point.py `
    --checkpoint notebooks/models/risk_gru_h128_v2_s1234.pt `
    --features data/features/train_val_v2 `
    --thresholds 0.5,0.6,0.7,0.8 --confirms 3,5,8 `
    --out eval/operating_point_sweep_h128_v2_s1234.md
```

Repeat with `--hidden 256` and, for whichever size looks best, repeat again
at seeds 1235/1236 (or reuse the four other seeds already on file) — **a
single seed is not evidence here.** The seed spread at this exact operating
point on validation ranged **0.107–0.864** across five seeds
(`docs/REPORT_evaluation_audit.md` §6). A capacity change has to clear that
spread, not just beat one baseline seed, before you trust it.

**Watch for:** overfitting. 1,365 (now ~1,160 under `train_core_v2`)
training clips against a bigger model may just move the best epoch earlier;
judge by best-epoch validation numbers, not final-epoch ones. The committed
model already peaks around epoch 8 of 30.

**Done when:** either a config clears mean AP **and** useful_v2 beyond the
seed spread at FA ≤ 0.20, lead ≥ 1.0 s, zero gross-premature (the project's
own selection rule) — or two sizes both fail to move mean AP, which is
itself a usable finding ("capacity isn't the ceiling either — write that
down and keep the current checkpoint").

### 2.2 Stretch, only if 2.1 shows a real signal by Day 3 (optional, GPU ~4h)

Longer feature window (`--window-s 20 --tail-s 1`,
`FUTURE_WORK.md` Tier 2). Only worth the GPU re-extraction if capacity alone
already moved the number — otherwise this is too expensive for a one-week
budget with an uncertain payoff.

### 2.3 Lock and read once (Day 4)

Pick the single best config from 2.1 (or 2.2). If none clears the current
checkpoint by more than noise, **the current checkpoint stays the headline**
— 0.650/0.167 is already a real, honestly-earned held-out number, and that
is a fine outcome for this step.

```powershell
python scripts/final_eval_read.py `
    --checkpoint notebooks/models/<chosen>.pt `
    --features data/features/eval_v2 `
    --threshold <chosen> --confirm <chosen> `
    --out-json eval/final_eval_read_v2_<tag>.json `
    --final-eval-report
```

`final_eval_read.py` refuses to overwrite its own output, so this is
naturally a one-shot — don't work around that refusal by renaming the output
and trying again with a different config. If you need to compare a second
candidate, that candidate wasn't actually locked; go back to 2.1 and decide
on validation only.

---

## Part 3 — Demo quality

### 3.1 Click through `dashboard_v2` end to end (Day 4–5)

Nobody has done this manually yet (1.4 above). Start it:

```powershell
python scripts/serve_dashboard_v2.py --port 8010
```

Check, against real data, not just "does it load":

- **Live replay:** pick a clip from the dropdown, confirm video + labels +
  masks + trails + predicted paths + ego corridor render, confirm the
  simulator pane tracks the same clock and the alert/brake banners fire in
  both panes at the same instant. Use one of the 6 curated Phase IV clips
  first (`621, 488, 1004, 690, 1085, 1261` — chosen for the sharpest
  old-vs-new contrast per `docs/status/TODO.md`).
- `⏮ alert` / `alert ⏭` (or `p`/`n`) jump correctly to 1.5 s before the
  previous/next alert.
- **Compare runs:** tick multiple runs, confirm the "different splits"
  warning actually appears when it should (mixing an `eval`-scored and an
  `eval_v2`-scored run is exactly the case that should trigger it, given
  everything in Part 1 of `REPORT_evaluation_audit.md`).
- **History** and **Drill-down** tabs render without a torch dependency
  hiccup; test `--no-checkpoints` separately as the fast-boot path in case
  the live demo machine needs a quick restart mid-session.
- Every chart's "table" button actually shows the underlying data.

Fix whatever breaks. This is exploratory QA, not a scripted test — budget a
real afternoon, not twenty minutes.

### 3.2 Pick and rehearse the demo path (Day 5)

Decide the exact clip(s) and view order you'll drive live, and avoid
steering the simulator pane onto an uncurated clip (1.6 above) unless a
judge specifically asks — in which case, know in advance that a `SUSPECT`
BEV plausibility clip is possible and be ready to say why, rather than being
surprised by it live.

---

## Part 4 — Story and docs cleanup (Day 6)

- Trim `docs/status/results.md` to a two-line pointer: *"Superseded — see
  `docs/REPORT_evaluation_audit.md` for the current, held-out result."*
  Don't delete it (git history keeps the old content), just stop it from
  being the first thing anyone reads under a file named "results."
- Add one line to `docs/INDEX.md` for `dashboard_v2` / `serve_dashboard_v2.py`
  so there's a documented path from the index to the demo.
- Prep `docs/REPORT_evaluation_audit.md` §7 ("Viva-ready answer") as literal
  talking points. It already answers "what is A3PS's actual performance and
  why should I trust that number" in one paragraph — memorize the shape of
  it, not the exact words.
- One sentence worth having ready regardless of stated priority, because
  it's free and disarms the hardest question a judge can ask: *"We found our
  own headline number had been selected on the test set, root-caused three
  more issues alongside it, fixed all four, and this number is what
  survived."*

---

## Day-by-day summary

| Day | Focus | Output |
|---|---|---|
| 1 | Bug fixes (Part 1, items 1.1–1.3, 1.8) | INDEX.md corrected, constant fixed, ablation reruns committed, docx resolved, `dashboard_v2` committed, `pytest` green |
| 2 | Capacity sweep, hidden 128/256 (2.1) | Validation-only comparison table, multi-seed |
| 3 | Multi-seed confirmation / optional stretch (2.2) | Either a validated new config or a written "capacity didn't move it either" finding |
| 4 | Lock config, one `eval_v2` read (2.3); start dashboard QA (3.1) | Final headline number; dashboard bug list |
| 5 | Finish dashboard QA (3.1), rehearse demo path (3.2) | Working, rehearsed live demo |
| 6 | Docs cleanup, talking points (Part 4) | One current results doc, INDEX.md updated, viva answer prepped |
| 7 | Buffer | Full `pytest` run, dashboard smoke test, dry-run presentation |

---

## What not to do under deadline pressure

- Don't read `eval_v2` more than once for the final config (see the rule at
  the top).
- Don't quote a capacity-sweep or window-length result measured on
  `train_val_v2` as if it were the held-out number — it's a validation
  estimate until the single final read happens.
- Don't claim the Seq2Seq-LSTM "wins" without the caveat in 1.5 if asked
  directly — the honest, softened version is still a good answer.
- Don't drive the live simulator pane on a clip outside the curated six
  unless you're prepared for a `SUSPECT` BEV plausibility result (1.6).
