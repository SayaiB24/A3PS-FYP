#!/usr/bin/env python
"""Carve a frozen ``train_val`` split out of ``train_all``, on disk and in the freeze.

    python scripts/carve_train_val.py --frac 0.15 --seed 1234

Why this exists
---------------
Every GRU checkpoint in this project was early-stopped and best-epoch-selected
against ``data/features/eval`` -- the frozen held-out split -- and the operating
point, kappa and capacity calls were all made on eval-scored numbers too
(``docs/status/eval_leakage_audit.md``). Selection needs a validation set that
is *not* the test set, and none existed: ``train_all`` was used whole.

This script carves one, and does it at the *file* level. The ``.npz`` features
for every ``train_all`` clip already exist, so no re-extraction is needed --
``train_core`` and ``train_val`` are just two disjoint copies of that directory,
and ``train_all`` itself is left exactly as it was so older results stay
reproducible.

The membership is written into ``eval/split_freeze.json`` alongside ``dev`` and
``eval`` so it inherits the same drift protection: once frozen, a clip cannot
silently move between the validation set and the training set on a later
data-prep run. ``data/nexar/index.csv`` is updated to match, because
``verify_freeze()`` compares the freeze against the live index and a freeze the
index contradicts is a guard that fails open.
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.splits import (  # noqa: E402
    DEFAULT_FREEZE_PATH,
    FROZEN_SPLITS,
    SCHEMA_VERSION,
    _sort_key,
    load_freeze,
    normalize_clip_id,
    save_freeze,
    verify_freeze,
)

SPLIT_NAME = "train_val"


def read_index(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames)


def write_index(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clip_ids_in(feature_dir):
    return {os.path.splitext(f)[0] for f in os.listdir(feature_dir)
            if f.endswith(".npz")}


def dir_size(feature_dir, names):
    return sum(os.path.getsize(os.path.join(feature_dir, n + ".npz")) for n in names)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default=os.path.join("data", "nexar", "index.csv"))
    p.add_argument("--freeze", default=DEFAULT_FREEZE_PATH)
    p.add_argument("--train-all", default=os.path.join("data", "features", "train_all"))
    p.add_argument("--val-out", default=os.path.join("data", "features", "train_val"))
    p.add_argument("--core-out", default=os.path.join("data", "features", "train_core"))
    p.add_argument("--frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true",
                   help="Report the carve and the disk cost, write nothing.")
    args = p.parse_args()

    if SPLIT_NAME not in FROZEN_SPLITS:
        raise SystemExit(
            f"{SPLIT_NAME!r} is not in a3ps.common.splits.FROZEN_SPLITS. Add it "
            "there first, or the carved split gets no drift protection.")

    rows, fieldnames = read_index(args.index)
    by_id = {normalize_clip_id(r["clip_id"]): r for r in rows}

    # One clip per source video, or the split has to be done at video level so
    # near-duplicate frames cannot straddle it. Checked, never assumed.
    paths = [r["path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise SystemExit(
            "index.csv has clips sharing a source video path. This script only "
            "splits at clip level; group by `path` before splitting.")
    print(f"{len(rows)} index rows, {len(set(paths))} distinct source videos "
          "-> one clip per video, so a clip-level split is a video-level split.")

    pool = sorted(clip_ids_in(args.train_all), key=_sort_key)
    unknown = [c for c in pool if c not in by_id]
    if unknown:
        raise SystemExit(f"{len(unknown)} train_all clip(s) absent from the index: "
                         f"{unknown[:10]}")

    freeze = load_freeze(args.freeze)
    if freeze is None:
        raise SystemExit(f"no freeze at {args.freeze}")
    already = {c for c, m in freeze["clips"].items() if m["split"] == SPLIT_NAME}
    if already:
        raise SystemExit(
            f"{len(already)} clip(s) are already frozen as {SPLIT_NAME}. The split "
            "exists; re-carving it would move clips between validation and "
            "training, which is exactly what the freeze prevents.")
    held_out = {c for c, m in freeze["clips"].items() if m["split"] in ("dev", "eval")}
    bleed = sorted(set(pool) & held_out, key=_sort_key)
    if bleed:
        raise SystemExit(f"{len(bleed)} train_all clip(s) are frozen as dev/eval: "
                         f"{bleed[:10]}")

    pos = [c for c in pool if int(by_id[c]["label"]) == 1]
    neg = [c for c in pool if int(by_id[c]["label"]) == 0]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_vp = int(round(len(pos) * args.frac))
    n_vn = int(round(len(neg) * args.frac))
    val = sorted(pos[:n_vp] + neg[:n_vn], key=_sort_key)
    core = sorted(pos[n_vp:] + neg[n_vn:], key=_sort_key)

    print(f"\ntrain_all {len(pool)} clips ({len(pos)} pos / {len(neg)} neg)")
    print(f"  -> {SPLIT_NAME}  {len(val)} clips ({n_vp} pos / {n_vn} neg), "
          f"seed {args.seed}, frac {args.frac}")
    print(f"  -> train_core   {len(core)} clips "
          f"({len(pos) - n_vp} pos / {len(neg) - n_vn} neg)")

    assert not (set(val) & set(core))
    assert set(val) | set(core) == set(pool)

    cost = dir_size(args.train_all, pool)
    print(f"\ndisk cost: copying both directories duplicates {cost / 1e6:.1f} MB "
          f"of .npz ({cost / 1e6:.1f} MB added, train_all kept). Copying rather "
          "than linking, since it is small enough not to matter.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    for out_dir, names in ((args.val_out, val), (args.core_out, core)):
        os.makedirs(out_dir, exist_ok=True)
        for name in names:
            shutil.copy2(os.path.join(args.train_all, name + ".npz"),
                         os.path.join(out_dir, name + ".npz"))
        print(f"copied {len(names)} .npz -> {out_dir}")

    for cid in val:
        by_id[cid]["split"] = SPLIT_NAME
    write_index(args.index, rows, fieldnames)
    print(f"updated {args.index}: {len(val)} row(s) now split={SPLIT_NAME}")

    # Extend the freeze in place. dev/eval entries are copied through untouched
    # so verify_freeze() keeps passing for the splits that already had numbers
    # measured against them.
    clips = dict(freeze["clips"])
    for cid in val:
        clips[cid] = {"split": SPLIT_NAME, "label": int(by_id[cid]["label"])}
    counts = {}
    for meta in clips.values():
        c = counts.setdefault(meta["split"], {"pos": 0, "neg": 0})
        c["pos" if meta["label"] == 1 else "neg"] += 1
    note = freeze.get("note", "")
    freeze_out = {
        "schema_version": SCHEMA_VERSION,
        "note": note + (
            f" | {SPLIT_NAME} added 2026-09-23: {len(val)} clips carved from "
            f"train_all at frac={args.frac}, seed={args.seed}, stratified by "
            "label, to give model/hyperparameter selection a validation set "
            "that is not the eval split (see docs/status/eval_leakage_audit.md)."),
        "counts": counts,
        "clips": dict(sorted(clips.items(), key=lambda kv: _sort_key(kv[0]))),
    }
    save_freeze(freeze_out, args.freeze)
    print(f"updated {args.freeze}: counts {json.dumps(counts, sort_keys=True)}")

    ok, problems = verify_freeze(rows, freeze_out)
    if not ok:
        raise SystemExit("verify_freeze FAILED after the carve:\n  " +
                         "\n  ".join(problems[:20]))
    print("verify_freeze: OK for every frozen split.")


if __name__ == "__main__":
    main()
