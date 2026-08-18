#!/usr/bin/env python
"""Freeze / verify the held-out dev+eval split membership.

The eval split is drawn by a size-dependent shuffle in
``scripts/prepare_nexar.py`` (see ``a3ps/common/splits.py`` for the full
explanation), so it must be pinned to a file BEFORE the clip pool grows.
Once pinned, ``prepare_nexar.py --freeze eval/split_freeze.json`` reproduces the
same dev/eval membership regardless of how many new clips were downloaded, and
new clips can only join ``train_traj``.

Write the freeze from the current index (do this once, now):

    python scripts/freeze_split.py write --index data/nexar/index2.csv

Check that an index still agrees with the freeze (cheap; safe to run anywhere,
and worth running before any scoring pass):

    python scripts/freeze_split.py verify --index data/nexar/index2.csv

Show what the freeze contains:

    python scripts/freeze_split.py show
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.splits import (  # noqa: E402
    DEFAULT_FREEZE_PATH,
    build_freeze,
    load_freeze,
    save_freeze,
    verify_freeze,
)


def read_index(path):
    """Read an index CSV into split-assignment records (clip_id/label/split)."""
    if not os.path.isfile(path):
        raise SystemExit("index not found: {}".format(path))
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            label = int(float(r.get("label")))
        except (TypeError, ValueError):
            label = 0
        out.append({
            "clip_id": r.get("clip_id", ""),
            "label": label,
            "split": r.get("split", "") or "",
        })
    return out


def cmd_write(args):
    records = read_index(args.index)
    if os.path.isfile(args.freeze) and not args.force:
        raise SystemExit(
            "{} already exists. Overwriting it re-draws the held-out set, which "
            "is exactly what the freeze exists to prevent. Pass --force only if "
            "you intend to invalidate every number measured against the current "
            "split.".format(args.freeze))

    freeze = build_freeze(records, note=args.note)
    if not freeze["clips"]:
        raise SystemExit(
            "no dev/eval rows in {} -- nothing to freeze. Run "
            "scripts/prepare_nexar.py first.".format(args.index))

    save_freeze(freeze, args.freeze)
    print("wrote {}".format(args.freeze))
    for split, c in sorted(freeze["counts"].items()):
        print("  {:11s} {:3d} clips ({} pos / {} neg)".format(
            split, c["pos"] + c["neg"], c["pos"], c["neg"]))
    print("  {} clip ids pinned in total".format(len(freeze["clips"])))
    print("\nCommit this file. From now on pass --freeze {} to "
          "scripts/prepare_nexar.py.".format(args.freeze))


def cmd_verify(args):
    freeze = load_freeze(args.freeze)
    if freeze is None:
        raise SystemExit(
            "no freeze at {}. Create it with:\n"
            "  python scripts/freeze_split.py write --index {}".format(
                args.freeze, args.index))
    records = read_index(args.index)
    ok, problems = verify_freeze(records, freeze)
    if ok:
        print("OK: {} matches the {} pinned clips in {}".format(
            args.index, len(freeze["clips"]), args.freeze))
        return
    print("MISMATCH: {} problem(s) between {} and {}\n".format(
        len(problems), args.index, args.freeze))
    for p in problems[:40]:
        print("  - {}".format(p))
    if len(problems) > 40:
        print("  ... and {} more".format(len(problems) - 40))
    print("\nThe held-out split has drifted. Do NOT score against this index "
          "until it is reconciled -- rebuild it with\n"
          "  python scripts/prepare_nexar.py --root data/nexar --freeze {}".format(
              args.freeze))
    raise SystemExit(1)


def cmd_show(args):
    freeze = load_freeze(args.freeze)
    if freeze is None:
        raise SystemExit("no freeze at {}".format(args.freeze))
    print("{}  (schema v{})".format(args.freeze, freeze.get("schema_version")))
    print(freeze.get("note", ""))
    print()
    for split, c in sorted(freeze.get("counts", {}).items()):
        print("  {:11s} {:3d} clips ({} pos / {} neg)".format(
            split, c["pos"] + c["neg"], c["pos"], c["neg"]))
    print("  {} clip ids pinned".format(len(freeze.get("clips", {}))))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freeze", default=DEFAULT_FREEZE_PATH,
                   help="Freeze file path (default: %(default)s). Lives outside "
                        "the gitignored data/ tree so it travels with the repo.")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="Snapshot the current dev/eval membership.")
    w.add_argument("--index", default="data/nexar/index2.csv")
    w.add_argument("--note", default="", help="Optional provenance note.")
    w.add_argument("--force", action="store_true",
                   help="Overwrite an existing freeze (invalidates prior numbers).")
    w.set_defaults(func=cmd_write)

    v = sub.add_parser("verify", help="Check an index still matches the freeze.")
    v.add_argument("--index", default="data/nexar/index2.csv")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("show", help="Print the freeze contents.")
    s.set_defaults(func=cmd_show)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
