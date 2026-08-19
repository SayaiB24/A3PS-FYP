# Master documentation index

Every document in this project: what it is for, when to read it, and — critically
— **whether it is current**. Several large docs predate the Phase IV pivot and
still describe the superseded threshold system as if it were live. §4 lists them
explicitly.

- **Continuing the project?** → [`handoff/README.md`](handoff/README.md)
- **Running it for the first time?** → [`runbooks/execution.md`](runbooks/execution.md)
- **Writing the thesis?** → §3 below, and mind the currency warnings.

---

## 1. "I want to…" → read this

| I want to… | Read | Notes |
|---|---|---|
| Get the project running from nothing | [`runbooks/execution.md`](runbooks/execution.md) — quick start | 3 paths: tests only, dashboard, full reproduction |
| Know what to do next | [`handoff/NEXT_STEPS.md`](handoff/NEXT_STEPS.md) | Priority-ordered, with a done/remaining table |
| Reproduce a committed number | [`handoff/REPRODUCE_BY_HAND.md`](handoff/REPRODUCE_BY_HAND.md) | Every command + the exact config that produced it |
| Improve the model | [`handoff/KAPPA_RETRAIN.md`](handoff/KAPPA_RETRAIN.md) | The one prioritised improvement, standalone |
| Understand a metric, or read a result honestly | [`design/metrics.md`](design/metrics.md) | §9–10 = current system; §1–8 = old |
| Write up problems solved / lessons | [`handoff/CHALLENGES.md`](handoff/CHALLENGES.md) | 19 entries, thesis-oriented |
| Pick optional future work | [`handoff/FUTURE_WORK.md`](handoff/FUTURE_WORK.md) | Tiered by value, each with its risk |
| See the current project status | [`status/TODO.md`](status/TODO.md) | "Status at a glance" at the top |
| Understand how a module works | [`design/logic_pipeline.md`](design/logic_pipeline.md) | Phases I–III current; its Phase 4 is superseded |
| Answer a tricky implementation question | [`design/QnA.md`](design/QnA.md) | Q&A form, mostly Phases I–III |
| Run the GPU-side dataset/extraction work | [`runbooks/GPU_PHASE4.md`](runbooks/GPU_PHASE4.md) | Current GPU runbook |
| Set up the two-laptop workflow | [`runbooks/GPU_HANDOFF.md`](runbooks/GPU_HANDOFF.md) | Env setup current; its task list is pre-pivot |
| Find measured numbers from the old system | [`status/results.md`](status/results.md) | ⚠️ pre-pivot throughout |
| Read the long-form project narrative | [`design/explanation.md`](design/explanation.md) | ⚠️ pre-pivot throughout |

---

## 2. The four folders

```
docs/
  handoff/    ← START HERE to continue the project. Written for working
              without an assistant. All current.
  runbooks/   ← Things you execute, step by step.
  design/     ← How and why it works. Reference, not front-to-back.
  status/     ← Where the project stands, and measured results.
```

Generated evaluation output is **not** in `docs/` — it lives in
[`../eval/`](../eval/), because scripts read and write those files at fixed
paths. See §5.

### `handoff/` — all current ✅

| doc | purpose |
|---|---|
| [`README.md`](handoff/README.md) | Index, current state, the 3 things most likely to trip you up |
| [`NEXT_STEPS.md`](handoff/NEXT_STEPS.md) | Priority-ordered next actions; done/remaining table |
| [`REPRODUCE_BY_HAND.md`](handoff/REPRODUCE_BY_HAND.md) | End-to-end reproduction, every command and config value |
| [`KAPPA_RETRAIN.md`](handoff/KAPPA_RETRAIN.md) | The top-priority improvement, standalone runbook |
| [`CHALLENGES.md`](handoff/CHALLENGES.md) | 19 problems: symptom → diagnosis → fix → what generalises |
| [`FUTURE_WORK.md`](handoff/FUTURE_WORK.md) | Optional improvements, tiered, with risks |

### `runbooks/`

| doc | currency |
|---|---|
| [`execution.md`](runbooks/execution.md) | ✅ Quick start current. §4–13 describe the **old** pipeline (labelled in the doc). |
| [`GPU_PHASE4.md`](runbooks/GPU_PHASE4.md) | ✅ Current GPU runbook — dataset, rate decision, extraction. Steps 1–3 marked done. |
| [`GPU_HANDOFF.md`](runbooks/GPU_HANDOFF.md) | ⚠️ Mixed. Env/CUDA setup (§2) is current and still referenced. Its task list is pre-pivot. |

### `design/`

| doc | currency |
|---|---|
| [`metrics.md`](design/metrics.md) | ✅ **§9–10 = current system.** §1–8 = old system, flagged at the top of the file. |
| [`logic_pipeline.md`](design/logic_pipeline.md) | ⚠️ Phases I–III current. Its "Phase 4" is the superseded threshold engine. |
| [`QnA.md`](design/QnA.md) | ⚠️ Mostly Phases I–III, still useful. No Phase IV content. |
| [`explanation.md`](design/explanation.md) | ⚠️ **Pre-pivot throughout.** Longest doc; historical narrative. |
| `logic_pipeline.pdf` | PDF companion to `logic_pipeline.md`. |

### `status/`

| doc | currency |
|---|---|
| [`TODO.md`](status/TODO.md) | ✅ Header + Part B current. Part A is a pre-pivot record, marked as such. |
| [`results.md`](status/results.md) | ⚠️ **Pre-pivot throughout.** Every number is the old threshold system. |

---

## 3. Reading orders

**New to the project (≈1 h).**
[`../README.md`](../README.md) → [`runbooks/execution.md`](runbooks/execution.md) quick start (run Path A) →
[`handoff/README.md`](handoff/README.md) → [`design/logic_pipeline.md`](design/logic_pipeline.md) skim.

**Continuing the work.**
[`handoff/README.md`](handoff/README.md) → [`handoff/NEXT_STEPS.md`](handoff/NEXT_STEPS.md) →
[`handoff/KAPPA_RETRAIN.md`](handoff/KAPPA_RETRAIN.md). Keep
[`handoff/REPRODUCE_BY_HAND.md`](handoff/REPRODUCE_BY_HAND.md) open for commands.

**Writing the thesis.**
[`handoff/CHALLENGES.md`](handoff/CHALLENGES.md) (the methodology story) →
[`design/metrics.md`](design/metrics.md) §9–10 (what each number means and how to
read it honestly) → `../eval/operating_point_sweep.md` and `../eval/train_log.txt`
(the numbers themselves) → [`handoff/NEXT_STEPS.md`](handoff/NEXT_STEPS.md) §3
(what may and may not be claimed).
**Do not** take numbers from `status/results.md` or `design/explanation.md`
without checking §4 first.

**Debugging a bad result.**
[`design/metrics.md`](design/metrics.md) "Failure cases" section →
[`handoff/REPRODUCE_BY_HAND.md`](handoff/REPRODUCE_BY_HAND.md) §7 ("if your
numbers differ") → [`handoff/CHALLENGES.md`](handoff/CHALLENGES.md) (the trap may
already be documented).

---

## 4. ⚠️ Docs that describe the superseded system

Phase IV replaced the hand-tuned threshold risk formula with a learned temporal
head. These documents were written before that and **do not mention it at all**,
so every performance number in them is the old system's:

| doc | approx. size | what to do |
|---|---|---|
| [`design/explanation.md`](design/explanation.md) | ~1,760 lines | Read as history. Its findings about Phases I–III still hold. |
| [`runbooks/GPU_HANDOFF.md`](runbooks/GPU_HANDOFF.md) | ~710 lines | Use §2 (environment/CUDA) only; ignore the task list. |
| [`status/results.md`](status/results.md) | ~680 lines | Do not quote. Current numbers are in `../eval/`. |
| [`design/logic_pipeline.md`](design/logic_pipeline.md) | ~410 lines | Phases I–III valid; skip its Phase 4 section. |
| [`design/QnA.md`](design/QnA.md) | ~310 lines | Still useful for implementation questions. |

That is ~3,900 lines of pre-pivot documentation, deliberately retained: Phases
I–III are unchanged and now act as feature extractors, and the old system is the
before/after baseline in the dashboard. **The risk is quoting a stale number, not
reading stale prose.** When in doubt, prefer:

| for | use |
|---|---|
| Current metrics and how to read them | [`design/metrics.md`](design/metrics.md) §9–10 |
| Current numbers | `../eval/operating_point_sweep.md`, `../eval/train_log.txt` |
| Current status | [`status/TODO.md`](status/TODO.md) header |

---

## 5. Generated vs hand-written

**Hand-written** — everything under `docs/`. Safe to edit.

**Generated** — everything under `../eval/` is script output and is overwritten
on the next run. Treat it as a result, not a document. Do not hand-edit.

| file | written by | current? |
|---|---|---|
| `../eval/operating_point_sweep.md` | `scripts/sweep_operating_point.py` | ✅ learned head |
| `../eval/train_log.txt`, `../eval/risk_gru_history.json` | `scripts/train_risk_head.py` | ✅ learned head |
| `../eval/assoc_rate_check.md`, `..._check2.md` | `scripts/check_assoc_rate.py` | ✅ 30/10 Hz and 30/15 Hz |
| `../eval/anticipation.md`, `..._per_clip.csv` | `scripts/eval_anticipation.py` | ⚠️ old system — regenerate (NEXT_STEPS §2) |
| `../eval/forecast_table.md` | `scripts/eval_forecast.py` | Phase III forecaster (unaffected by the pivot) |
| `../eval/paper_stats_summary.md` | `scripts/collect_paper_stats.py` | ⚠️ old system — regenerate |
| `../eval/ablation_table.md` | `scripts/run_ablations.py` | ⚠️ old system |

Two files under `../eval/` are **not** generated and must not be deleted:

| file | why it matters |
|---|---|
| `../eval/split_freeze.json` | Pins the held-out split. Everything comparable depends on it. |
| `../eval/nexar_index_gpu.csv`, `nexar_index_full.csv` | The split indices, kept outside gitignored `data/`. |

`../eval/README.md` explains why the index lives there.

---

## 6. Non-doc reference points

Sometimes the code is the clearest answer:

| question | file |
|---|---|
| What exactly is in a feature vector? | `a3ps/features/extract.py` (named feature list) |
| What does the anticipation loss optimise? | `a3ps/risk/anticipation_loss.py` (`expected_lead_time()` reports the lead it asks for) |
| Why can't the split drift? | `a3ps/common/splits.py` (module docstring) |
| What does the model architecture look like? | `a3ps/risk/temporal.py` (34,465 params, causality assertion) |
| What config produced the committed numbers? | the checkpoint itself — `torch.load(...)["extra"]` |
| What do the dashboard's panels mean? | `dashboard/app.js` (`renderCompare`, `drawCompare`) |
