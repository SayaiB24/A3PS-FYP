#!/usr/bin/env python
"""Redraw eval/train_val/train_core, stratified by alert-to-event window.

    python scripts/repartition_splits.py --seed 1234

Why the old partition had to go
-------------------------------
``data/nexar/labels.xlsx`` -- the hand-built 355-row table ``prepare_nexar.py``
consumed for the original download -- has its positive rows sorted by
alert-to-event window descending and truncated at the 65 longest clips in the
dataset. ``dev`` and ``eval`` were drawn from that subset, so they absorbed all
65, and the training pool was left with nothing above 2.967 s (65th largest
window 2.971 s, 66th 2.967 s -- a 4 ms cut). The held-out split was the most
anticipatable slice of the data *by construction*, and every number ever
measured on it was measured on an easier population than the model trains on.
See ``docs/status/label_window_audit.md``.

Nothing in the pipeline transforms the times -- raw ``train.csv``,
``index.csv`` and the ``.npz`` meta agree exactly on all 1,485 extracted clips
-- so only the partition is at fault, and only the partition is redrawn here.
No features are re-extracted: ``event_time_s`` is unchanged, so every existing
feature window is still correctly placed.

Stratification
--------------
Positives are binned into **deciles of window length** over the pool and each
split draws proportionally from every bin, so all three splits span the full
0.03-4.47 s range rather than one of them taking a tail. Equal-count deciles
give 74-75 clips per bin and exactly 6 eval positives per bin -- fine enough to
control the shape, coarse enough that no bin is thin. Negatives carry no window,
so they are stratified by label alone.

v1 (``eval/split_freeze.json``) is left on disk untouched and is still consulted
by the held-out guard, because ``data/features/eval/`` still exists and scoring
it would still be scoring a test set.
"""

import argparse
import csv
import json
import os
import random
import shutil
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.splits import (  # noqa: E402
    FREEZE_PATH_V2,
    LEGACY_FREEZE_PATH,
    SCHEMA_VERSION,
    _sort_key,
    load_freeze,
    normalize_clip_id,
    save_freeze,
    verify_freeze,
)

SOURCE_DIRS = (os.path.join("data", "features", "eval"),
               os.path.join("data", "features", "train_all"))


def read_index(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames)


def quota_per_bin(bin_sizes, target):
    """Largest-remainder apportionment of ``target`` across bins.

    Done globally rather than by rounding each bin independently: per-bin
    rounding overshoots, and patching the overshoot afterwards has to take the
    excess from *somewhere*, which silently strips whichever bin was processed
    last -- the top window decile, i.e. exactly the clips this repartition
    exists to spread evenly.
    """
    total = sum(bin_sizes)
    exact = [n * target / total for n in bin_sizes]
    take = [int(x) for x in exact]
    short = target - sum(take)
    order = sorted(range(len(bin_sizes)), key=lambda i: (-(exact[i] - take[i]), i))
    for i in order[:short]:
        take[i] += 1
    # A bin can never give more than it holds.
    for i, n in enumerate(bin_sizes):
        take[i] = min(take[i], n)
    return take


def describe(name, windows):
    if not windows:
        return f"| {name} | 0 | — | — | — | — |"
    return ("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        name, len(windows), st.mean(windows), st.median(windows),
        min(windows), max(windows)))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default=os.path.join("data", "nexar", "index.csv"))
    p.add_argument("--freeze-out", default=FREEZE_PATH_V2)
    p.add_argument("--eval-pos", type=int, default=60)
    p.add_argument("--eval-neg", type=int, default=60)
    p.add_argument("--val-pos", type=int, default=103)
    p.add_argument("--val-neg", type=int, default=102)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--suffix", default="_v2")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows, fieldnames = read_index(args.index)
    by_id = {normalize_clip_id(r["clip_id"]): r for r in rows}

    paths = [r["path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise SystemExit("index.csv has clips sharing a source video; this "
                         "script only splits at clip level.")

    # Pool = clips that already have extracted features, and where they live.
    home = {}
    for d in SOURCE_DIRS:
        for f in os.listdir(d):
            if f.endswith(".npz"):
                home[normalize_clip_id(f[:-4])] = d
    pool = sorted(home, key=_sort_key)
    missing = [c for c in pool if c not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} feature clip(s) absent from the index")

    def window(c):
        r = by_id[c]
        if not (r["event_time_s"] and r["alert_time_s"]):
            return None
        return float(r["event_time_s"]) - float(r["alert_time_s"])

    pos = [c for c in pool if int(by_id[c]["label"]) == 1]
    neg = [c for c in pool if int(by_id[c]["label"]) == 0]
    untimed = [c for c in pos if window(c) is None]
    if untimed:
        raise SystemExit(
            f"{len(untimed)} positive(s) have no alert/event time: {untimed[:10]}. "
            "A clip with no window cannot be scored for useful-warning, so the "
            "stratification premise does not hold -- stop and decide where they go.")

    print(f"pool: {len(pool)} clips with features "
          f"({len(pos)} pos / {len(neg)} neg); every positive is timed.")
    print(f"out of scope: 15 dev clips with no extracted features.\n")

    rng = random.Random(args.seed)
    pos_sorted = sorted(pos, key=lambda c: (window(c), _sort_key(c)))
    n = len(pos_sorted)
    assign = {}

    # --- positives: proportional draw from each window decile -------------
    bins, edges = [], []
    for b in range(args.bins):
        lo, hi = b * n // args.bins, (b + 1) * n // args.bins
        chunk = list(pos_sorted[lo:hi])
        edges.append((window(chunk[0]), window(chunk[-1])))
        rng.shuffle(chunk)
        bins.append(chunk)

    sizes = [len(c) for c in bins]
    eval_q = quota_per_bin(sizes, args.eval_pos)
    # train_val is apportioned over what each bin has LEFT, so its quota can
    # always be met without reaching back into eval's share.
    val_q = quota_per_bin([s - e for s, e in zip(sizes, eval_q)], args.val_pos)

    picked = {"eval": [], "train_val": [], "train_core": []}
    for b, chunk in enumerate(bins):
        picked["eval"] += chunk[:eval_q[b]]
        picked["train_val"] += chunk[eval_q[b]:eval_q[b] + val_q[b]]
        picked["train_core"] += chunk[eval_q[b] + val_q[b]:]

    for split, target in (("eval", args.eval_pos), ("train_val", args.val_pos)):
        if len(picked[split]) != target:
            raise SystemExit(
                f"{split}: apportionment produced {len(picked[split])} positives, "
                f"expected {target}")

    # --- negatives: no window, so label-only stratification ---------------
    negs = sorted(neg, key=_sort_key)
    rng.shuffle(negs)
    picked_neg = {"eval": negs[:args.eval_neg],
                  "train_val": negs[args.eval_neg:args.eval_neg + args.val_neg],
                  "train_core": negs[args.eval_neg + args.val_neg:]}

    for s in ("eval", "train_val", "train_core"):
        for c in picked[s] + picked_neg[s]:
            assign[c] = s

    assert len(assign) == len(pool), (len(assign), len(pool))

    # --- report ------------------------------------------------------------
    print("window distribution by split (positives only)\n")
    print("| split | n | mean | median | min | max |")
    print("|---|---|---|---|---|---|")
    print(describe("pool", [window(c) for c in pos]))
    for s in ("train_core", "train_val", "eval"):
        print(describe(s, [window(c) for c in picked[s]]))

    print("\nper-decile counts (bin edges are window seconds)\n")
    hdr = "| decile | range | pool | train_core | train_val | eval |"
    print(hdr + "\n" + "|---|---|---|---|---|---|")
    for b in range(args.bins):
        lo, hi = b * n // args.bins, (b + 1) * n // args.bins
        chunk = set(pos_sorted[lo:hi])
        cnt = {s: sum(1 for c in picked[s] if c in chunk)
               for s in ("train_core", "train_val", "eval")}
        print(f"| {b+1} | {edges[b][0]:.2f}-{edges[b][1]:.2f} | {len(chunk)} "
              f"| {cnt['train_core']} | {cnt['train_val']} | {cnt['eval']} |")

    ranges = {s: (min(window(c) for c in picked[s]),
                  max(window(c) for c in picked[s]))
              for s in ("train_core", "train_val", "eval")}
    pool_rng = (min(window(c) for c in pos), max(window(c) for c in pos))
    print(f"\npool range {pool_rng[0]:.3f}-{pool_rng[1]:.3f} s")
    for s, (a, b) in ranges.items():
        print(f"  {s:11s} {a:.3f}-{b:.3f} s")
    overlap = all(a <= pool_rng[0] * 3 + 0.5 and b >= pool_rng[1] * 0.6
                  for a, b in ranges.values())
    print(f"\nall three splits span the pool's range: {overlap}")
    print("counts: " + ", ".join(
        f"{s} {sum(1 for v in assign.values() if v == s)} "
        f"({len(picked[s])} pos / {len(picked_neg[s])} neg)"
        for s in ("train_core", "train_val", "eval")))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # --- materialise --------------------------------------------------------
    for s in ("train_core", "train_val", "eval"):
        out_dir = os.path.join("data", "features", s + args.suffix)
        os.makedirs(out_dir, exist_ok=True)
        ids = picked[s] + picked_neg[s]
        for c in ids:
            shutil.copy2(os.path.join(home[c], c + ".npz"),
                         os.path.join(out_dir, c + ".npz"))
        print(f"copied {len(ids)} .npz -> {out_dir}")

    for c, s in assign.items():
        by_id[c]["split"] = s
    with open(args.index, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {args.index}")

    # --- freeze v2 ----------------------------------------------------------
    legacy = load_freeze(LEGACY_FREEZE_PATH) or {"clips": {}}
    clips = {}
    for c, s in assign.items():
        if s in ("eval", "train_val"):
            clips[c] = {"split": s, "label": int(by_id[c]["label"])}
    # dev keeps its v1 membership: those 15 clips have no extracted features and
    # are out of scope for this repartition.
    for c, m in legacy["clips"].items():
        if m["split"] == "dev":
            clips[c] = dict(m)
    counts = {}
    for m in clips.values():
        d = counts.setdefault(m["split"], {"pos": 0, "neg": 0})
        d["pos" if m["label"] == 1 else "neg"] += 1
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "v2, frozen 2026-09-23. Supersedes eval/split_freeze.json, which is "
            "kept on disk and still guarded. v1's dev/eval came from "
            "labels.xlsx, a hand-built subset sorted by alert-to-event window "
            "descending and cut at the 65 longest clips in the dataset, so the "
            "held-out split held every window over 2.97 s and training none. v2 "
            "redraws eval/train_val/train_core over the 1,485 clips with "
            f"extracted features, stratified by window decile, seed {args.seed}. "
            "dev is carried over unchanged (no features, out of scope). See "
            "docs/status/label_window_audit.md and "
            "docs/status/repartition_results.md."),
        "counts": counts,
        "clips": dict(sorted(clips.items(), key=lambda kv: _sort_key(kv[0]))),
    }
    save_freeze(freeze, args.freeze_out)
    print(f"wrote {args.freeze_out}: {json.dumps(counts, sort_keys=True)}")

    # Verify against the whole index, not just the pool: dev's 15 clips are
    # carried over into v2 and are in the index, they simply have no features.
    ok, problems = verify_freeze(rows, freeze)
    if not ok:
        raise SystemExit("verify_freeze FAILED:\n  " + "\n  ".join(problems[:20]))
    print("verify_freeze: OK against the live index.")


if __name__ == "__main__":
    main()
