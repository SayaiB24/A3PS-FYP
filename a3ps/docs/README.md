# a3ps documentation

Grouped by what you are trying to do. Generated evaluation output is **not**
here — it lives in [`../eval/`](../eval/), because scripts write and read those
files at fixed paths.

## [`runbooks/`](runbooks/) — things you execute

Follow these top to bottom; they contain copy-pasteable commands.

| doc | when to read it |
|---|---|
| [`execution.md`](runbooks/execution.md) | Running the project from scratch on one machine. Start here. |
| [`GPU_HANDOFF.md`](runbooks/GPU_HANDOFF.md) | The two-laptop workflow (CPU dev / GPU run): what to copy, env setup, which steps need a GPU. |
| [`GPU_PHASE4.md`](runbooks/GPU_PHASE4.md) | Phase IV specifically — dataset indexing, the frame-rate decision, and the feature-extraction pass. |

## [`design/`](design/) — how and why it works

Reference material. Read as needed rather than front to back.

| doc | covers |
|---|---|
| [`logic_pipeline.md`](design/logic_pipeline.md) | What each module does and why, stage by stage. Has a [PDF companion](design/logic_pipeline.pdf). |
| [`metrics.md`](design/metrics.md) | Every metric: its definition, the command that produces it, and how to read it honestly. |
| [`explanation.md`](design/explanation.md) | Long-form narrative of decisions, findings, and known gaps. The most detailed doc here. |
| [`QnA.md`](design/QnA.md) | Answers to the tricky implementation questions, in Q&A form. |

## [`status/`](status/) — where the project stands

| doc | covers |
|---|---|
| [`results.md`](status/results.md) | Measured numbers with the hardware and caveats they depend on. |
| [`TODO.md`](status/TODO.md) | Remaining work and what is already done. |

## Conventions

- **Generated vs. written.** Anything under `../eval/` is script output and gets
  overwritten; treat it as a result, not a document. Everything under `docs/` is
  hand-written and safe to edit.
- **Numbers carry their hardware.** CPU and GPU timings differ by ~20-40x in this
  project, so any latency figure states which machine produced it. Keep that
  habit when adding results.
- **Cross-references are relative paths.** These docs used to sit side by side,
  so links were bare filenames; after grouping they became relative — a runbook
  now reaches the metrics doc as `..` then `design/metrics.md`. Keep new links
  relative too, so the tree stays movable.
