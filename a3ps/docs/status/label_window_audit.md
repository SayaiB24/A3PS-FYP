# Where the disjoint alert-to-event window comes from (2026-09-23, READ-ONLY)

[`clean_protocol_results.md`](clean_protocol_results.md) reported that the
alert-to-event window is disjoint between splits at exactly 2.97 s: every
training positive below it, every eval positive above it. This document traces
one number through every layer to find where that disjointness is created.

**Answer: it is created before any of this project's code runs, in
`data/nexar/labels.xlsx` — a hand-built 355-row subset whose positive rows are
sorted by window length descending and truncated at the 65 longest in the
dataset.** `dev` and `eval` were drawn from that subset, so they absorbed all 65.
No code in this repo transforms the times; every layer downstream is faithful.

My earlier guess — "two annotation sources with different conventions" — was
wrong, and is retracted. There is one annotation source and its numbers are
never altered. The problem is *which rows were kept*, not what they say.

No relabelling, retraining, re-extraction, or script edits were performed. Nothing
was written except this file.

## 1. The layer-by-layer trace

Ten eval positives and ten training positives, read at every layer. Times are
`event_time_s` / `alert_time_s` in seconds.

| clip | split | raw `train.csv` | `labels.xlsx` | `index.csv` | `.npz` meta | window |
|---|---|---|---|---|---|---|
| 900 | eval | 21.177/17.942 | 21.177/17.942 | 21.177/17.942 | 21.177/17.942 | 3.235 |
| 795 | eval | 20.706/16.493 | 20.706/16.493 | 20.706/16.493 | 20.706/16.493 | 4.213 |
| 927 | eval | 20.033/16.400 | 20.033/16.400 | 20.033/16.400 | 20.033/16.400 | 3.633 |
| 858 | eval | 20.033/16.633 | 20.033/16.633 | 20.033/16.633 | 20.033/16.633 | 3.400 |
| 600 | eval | 19.400/15.933 | 19.400/15.933 | 19.400/15.933 | 19.400/15.933 | 3.467 |
| 702 | eval | 20.745/17.641 | 20.745/17.641 | 20.745/17.641 | 20.745/17.641 | 3.104 |
| 819 | eval | 19.520/15.449 | 19.520/15.449 | 19.520/15.449 | 19.520/15.449 | 4.071 |
| 68 | eval | 19.900/16.733 | 19.900/16.733 | 19.900/16.733 | 19.900/16.733 | 3.167 |
| 109 | eval | 20.133/16.167 | 20.133/16.167 | 20.133/16.167 | 20.133/16.167 | 3.966 |
| 847 | eval | 19.667/15.812 | 19.667/15.812 | 19.667/15.812 | 19.667/15.812 | 3.855 |
| 822 | train_risk | 19.500/18.633 | **absent** | 19.500/18.633 | 19.500/18.633 | 0.867 |
| 208 | train_risk | 19.800/19.233 | **absent** | 19.800/19.233 | 19.800/19.233 | 0.567 |
| 72 | train_risk | 20.101/20.068 | **absent** | 20.101/20.068 | 20.101/20.068 | 0.033 |
| 128 | train_risk | 19.267/18.333 | **absent** | 19.267/18.333 | 19.267/18.333 | 0.934 |
| 205 | train_risk | 21.146/18.196 | **absent** | 21.146/18.196 | 21.146/18.196 | 2.950 |
| 408 | train_risk | 19.953/17.551 | **absent** | 19.953/17.551 | 19.953/17.551 | 2.402 |
| 948 | train_val | 20.287/19.219 | **absent** | 20.287/19.219 | 20.287/19.219 | 1.068 |
| 457 | train_risk | 19.600/19.367 | **absent** | 19.600/19.367 | 19.600/19.367 | 0.233 |
| 471 | train_risk | 19.833/18.267 | **absent** | 19.833/18.267 | 19.833/18.267 | 1.566 |
| 841 | train_risk | 19.000/16.167 | **absent** | 19.000/16.167 | 19.000/16.167 | 2.833 |

The values never change. The only thing that differs between the two blocks is
the `labels.xlsx` column: **every eval positive is present in it; no training
positive is.**

This is not a sample effect. Checked across **all 1,485 extracted clips**
(`eval` 120 + `train_all` 1,365), comparing raw `train.csv` → `index.csv` →
`.npz` meta:

```
event_time_s mismatches: 0
alert_time_s mismatches: 0
label mismatches       : 0
```

**Layer verdict: the disjointness appears at none of §B, §C or §D.**
`prepare_nexar.py` copies the columns through `_to_seconds` (a `round(float,3)`,
`prepare_nexar.py:91-97`) with no derivation, offset or fallback.
`extract_features.py` stores absolute video times unchanged — clip 584's `.npz`
carries `event_time_s: 20.233, alert_time_s: 15.767` and a `t` axis running
8.233→21.133 s in absolute time, not re-based or clamped. Both feature
directories were written by the same configuration (`window_s 13.0`,
`tail_s 1.0`, `rate_hz 10.0`, `schema_version 1`, `feature_dim 112`,
`n_failed 0`), so the "eval was extracted first" note in
`REPRODUCE_BY_HAND.md` §3 did not produce a version skew. And there is only one
consuming path: `evaluate()` and `batch_anticipation_loss()` both read
`c["alert_time_s"]` / `c["event_time_s"]` off the same dicts built by
`load_clips()` (`train_risk_head.py:67-75`), so §D is ruled out too.

This is **§A** — but not in the sense §A anticipated. The disjointness is not in
the true raw labels.

## 2. §A — the selection was rank-ordered, not random

### The true raw distribution is continuous

`D:\Coding\FYP\Nexar-Dataset\train.csv`, 1,500 rows, columns
`id, time_of_event, time_of_alert, target`. 750 positives, **all 750 timed**
(zero missing times — see §2.3). Window = `time_of_event − time_of_alert`:

| window | count |
|---|---|
| 0.0–0.5 s | 47 |
| 0.5–1.0 s | 142 |
| 1.0–1.5 s | 208 |
| 1.5–2.0 s | 150 |
| 2.0–2.5 s | 89 |
| 2.5–3.0 s | 51 |
| 3.0–3.5 s | 34 |
| 3.5–4.0 s | 19 |
| 4.0–4.5 s | 10 |

Mean 1.60 s, median 1.43 s, range 0.033–4.466 s. **Continuous, unimodal, no gap
at 2.97 s.** So 2.97 s is not a ceiling in the source data — it is an artifact
introduced downstream.

**Exactly 65 raw positives have a window > 2.97 s.**

### `labels.xlsx` is those 65, sorted

`data/nexar/labels.xlsx` (355 rows: 65 positives + 290 negatives) is the table
`prepare_nexar.py` actually consumed for the original download, and
`data/nexar/videos/` holds exactly the matching 355 `.mp4` files.

Its positive rows are **sorted by window descending** — verified
programmatically, `all(g[i] >= g[i+1])` is `True` — and truncated:

| position | clip | event | alert | window |
|---|---|---|---|---|
| 1st | 00584 | 20.233 | 15.767 | 4.466 |
| 2nd | 00650 | 18.867 | 14.600 | 4.267 |
| 3rd | 00795 | 20.706 | 16.493 | 4.213 |
| … | | | | |
| 64th | 00638 | 17.573 | 14.601 | 2.972 |
| 65th | 00696 | 20.211 | 17.240 | 2.971 |

The 65 positives in `labels.xlsx` **are exactly the top 65 by window length
across the whole 750-positive pool** (set equality, verified). The cut is
4 milliseconds wide: 65th largest = 2.971 s, 66th = 2.967 s.

No script in this repository writes `labels.xlsx`; it is referenced only as an
input (`prepare_nexar.py`, and the runbooks that say to drop it in
`data/nexar/`). It was produced by hand — the sort order is what an "pick the
clearest examples to start with" step looks like in a spreadsheet.

### `assign_splits` shuffles, but the pool was already filtered

`prepare_nexar.py::assign_splits` (lines 247-278) takes `dev` by shortest
duration, then `random.Random(1234).shuffle`s the remainder for `eval`. That
shuffle is genuinely random — and irrelevant. It shuffled a pool in which
**every** available positive already had a window above 2.97 s, so all 65 went
to `dev` (5) and `eval` (60). A random draw from a pre-filtered pool is still
a filtered sample.

The later full index (`eval/nexar_index_full.csv`, committed 2026-08-18 in
`3189ef5`, 1,500 rows) brought in the remaining 685 positives — **all of them
below 2.967 s, because the 65 above it were already spoken for.**

| | timed positives | mean | min | max |
|---|---|---|---|---|
| inside the 355-clip subset | 65 | 3.49 s | 2.971 | 4.466 |
| outside it (the training pool) | 685 | 1.42 s | 0.033 | **2.967** |

### Did eval absorb the long-window clips? Yes — all of them.

**Training-pool positives with a window over 2.97 s: zero.** The widest window
available to training is 2.967 s. The dataset contains 63 positives with
windows over 3.0 s and every one sits in `dev` or `eval`.

So the answer to "does the source contain enough long-window clips to support a
2–6 s lead target" is: the source contains 63 clips over 3 s, and the split
gave 100% of them to the held-out set.

### 2.3 The 685-vs-582 count is not a discrepancy

`train_risk` showing 685 positives but 582 "timed" was my own bookkeeping, not a
data problem: 103 of those 685 are now `train_val` after the Step 4 carve.
582 + 103 = 685. Confirmed from the index:

```
train_risk label=1 timed=True : 582
train_val  label=1 timed=True : 103
eval       label=1 timed=True :  60
dev        label=1 timed=True :   5     total 750
```

**Every positive in the dataset is timed.** There is no untimed subset, so the
timed/untimed distinction cannot be correlated with window length.

## 3. Cost of a fix: CPU, not GPU

**`event_time_s` is consistent across every layer for all 1,485 extracted
clips** (§1, zero mismatches). Feature windows are placed around `event_time_s`
(`window_start_s`/`window_end_s` in each `.npz` meta, e.g. clip 584:
20.233 − 12.0 = 8.233 ✓), so with `event_time_s` unchanged, **every existing
window is correctly placed and the extracted features stay valid.**

Any fix that only re-partitions clips is therefore **CPU-only**: it is a file
copy plus a retrain of the order of 5 minutes per run, exactly like the Step 4
carve. The ~3.5 h GPU re-extraction is *not* required.

One caveat: features exist for 1,485 of the 1,500 labelled clips. The 15 `dev`
clips have no `.npz`. A re-split that needs them would need a short extraction
run for 15 clips — minutes, not hours.

## 4. Can the freeze survive a fix?

**No. Any fix that makes `eval` representative necessarily redraws it.**

`eval`'s 60 positives are 60 of the only 65 clips in the entire dataset with a
window over 2.97 s. The property that makes `eval` unrepresentative *is* its
membership. There is no transformation of the times that fixes it — the times
are correct — so the only lever is which clips are in it.

Concretely:
- **Keeping the freeze** keeps a test set whose positives are the easiest 8.7%
  (65/750) of the dataset by anticipation window, and keeps training blind to
  every clip over 2.967 s.
- **Redrawing** invalidates every number ever measured on `eval`, including the
  clean-protocol number from Step 4, and the 3.7 GB cached pipeline output in
  `eval/anticipation/` that the freeze note says it matches.

The freeze did its job perfectly — it prevented drift in a split that was
already skewed when it was frozen. The skew predates the freeze by one manual
spreadsheet step.

## 5. Lead-time hypothesis: **supports**, on structural grounds

> Hypothesis: the anticipation loss, keyed to `alert_time_s`, has only ever
> asked for ~1.5 s of lead, so the 2–6 s target was never learnable — meaning
> "this feature set supports about 1.7 s of warning" blames the features for a
> label problem.

### The loss cannot ask for more lead than the clip's own window

`alert_weights()` (`a3ps/risk/anticipation_loss.py:104-110`):

```python
inside = (t >= float(alert_t)) & (t <= float(event_t))
u = ((t - float(alert_t)) / span).clamp(0.0, 1.0)
return torch.exp(float(kappa) * (u - 1.0)) * inside.to(t.dtype)
```

Every frame outside `[alert_t, event_t]` is multiplied by zero. The loss gives
**no positive signal whatsoever** for firing earlier than `alert_time_s`. Since
no training clip has a window above 2.967 s, and the mean is 1.41 s, the
objective could never request more than ~1.4 s of lead on average and never more
than 2.967 s on any single clip. **A 2–6 s lead target was not learnable from
this training pool, irrespective of features, capacity or kappa.**

### The documented "requested lead" was computed on the wrong population

`scripts/compare_kappa.py:43` sets `MEAN_ALERT_LEAD_S = 3.49`, commented
"measured over the 65 local positives" — i.e. the `dev`/`eval` population, the
only one available when it was written. `train_risk_head.py:536` hardcodes the
same 3.49. That is the number behind the "requested lead" column in
`eval/kappa_comparison.md`. Recomputed on the population actually trained on:

| kappa | requested lead, as documented (3.49 s window) | on the real training pool (1.41 s) |
|---|---|---|
| 0.5 | 1.60 s | **0.65 s** |
| 1.0 | 1.46 s | **0.59 s** |
| 2.0 | 1.20 s | **0.48 s** |
| 3.0 | 0.98 s | **0.40 s** |

The loss was asking for roughly **0.6 s**, not the documented 1.46 s.

### The measured leads sit at the training windows, not eval's

Every mean lead this project has measured — 1.59, 1.62, 1.67 s — sits just above
the training-window mean (1.41 s train_core / 1.50 s train_val) and nowhere near
`eval`'s own 3.51 s mean, despite `eval` clips *allowing* up to 4.47 s. If the
features were the ceiling, leads would not cluster this precisely on the
training population's window statistics.

### Verdict: supports, but the per-clip question cannot be settled

The structural argument above is conclusive on its own terms: the objective
provably could not request more than the training windows allow. What I
**cannot** answer from existing artifacts is the per-clip question — whether
lead scales with each eval clip's own window, and what the largest lead is on
the widest-window eval clips.

**No existing artifact emits per-clip lead.** `evaluate()`
(`train_risk_head.py:99-158`) builds exactly the needed per-clip rows —
`rows.append({"clip": c, "first_alert_t": fa, ...})` at line 106-108 — and then
discards them, returning only aggregates. `eval/final_eval_read_k1p0_clean.md`
and every sweep file contain only aggregate rows.
`eval/anticipation_per_clip.csv` is from the superseded threshold system, not
the GRU.

I did not retrain or re-score to obtain it. **Smallest change that would
produce it:** have `evaluate()` optionally return its existing `rows` list, and
have `sweep_operating_point.py` write `clip_id, label, event_time_s,
alert_time_s, window, first_alert_t, lead` to a CSV. Roughly ten lines, no
retraining, one forward pass over the already-trained checkpoint.

So: **supports**, decisively at the objective level; the per-clip confirmation
is one small, cheap change away and is not manufactured here.

## 6. Options

Nothing below is implemented. Each is stated with its cost, what it invalidates,
and what it does not fix.

### Option 1 — Change nothing, document the limitation
- **Cost:** none.
- **Invalidates:** nothing.
- **Does not fix:** the headline stays measured on the easiest 8.7% of
  positives, the 2–6 s lead target stays unreachable, and "the features support
  ~1.7 s" stays a claim about the labels rather than the features. Any external
  reader who checks the window distribution will find this.

### Option 2 — Redraw dev/eval as a stratified sample of the full 750
- **Cost:** CPU-only. New freeze, feature copies, one retrain (~5 min/run),
  plus a ~15-clip extraction if `dev` is kept. Under an hour.
- **Invalidates:** **every number ever measured on `eval`** — the Step 4 clean
  number, `kappa_comparison.md`, `ablations.md`, `phase4_comparison.md`, and the
  3.7 GB cached `eval/anticipation/` output the freeze was pinned to. The old
  and new headline both become unquotable.
- **Does not fix:** the dataset still only has 63 positives over 3 s, so a
  representative eval set makes the 2–6 s target *visibly* unreachable rather
  than making it reachable. It also breaks the freeze's whole guarantee, which
  should not be done casually or more than once.

### Option 3 — Keep eval frozen; report stratified by window length
- **Cost:** CPU-only, and mostly reporting. Needs the per-clip lead dump from §5.
- **Invalidates:** nothing; it adds a breakdown beside existing numbers.
- **Does not fix:** training still never sees a window over 2.967 s, so the
  model still cannot learn long leads. It makes the bias legible without
  removing it — the cheapest honest option.

### Option 4 — Restate the lead-time target against the data that exists
- **Cost:** documentation only, plus recomputing the "requested lead" column
  with the real training-window mean (§5).
- **Invalidates:** the "requested lead" column of `eval/kappa_comparison.md` and
  the 2–6 s target in `docs/design/metrics.md`, both of which are currently
  stated against a population the model never trained on.
- **Does not fix:** any measured number. It corrects what the numbers are
  compared *against* — arguably the single most misleading thing in the current
  docs, since it is what turns a label artifact into an apparent feature failure.

### Option 5 — Rebalance the training pool by moving long-window clips into it
- **Cost:** CPU-only if drawn from already-extracted clips.
- **Invalidates:** the freeze, since the only long-window clips are in
  `dev`/`eval`. Same blast radius as Option 2.
- **Does not fix:** taking long-window clips *out* of `eval` to give to training
  makes `eval` smaller and still non-representative unless it is redrawn anyway.
  Strictly worse than Option 2; listed to be ruled out explicitly.

---

**Recommended reading order for whoever decides:** §2 (the mechanism), §4 (the
freeze cannot survive a fix), then Options 3 and 4, which are the only two that
invalidate nothing.
