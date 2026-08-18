#!/usr/bin/env python
"""Assign positives left unused by a split freeze to a training split.

``scripts/prepare_nexar.py --freeze ...`` deliberately leaves every positive
absent from the freeze with a BLANK split rather than guessing where it belongs
(see ``a3ps/common/splits.py``: positives are scarce and belong to a deliberate
decision). This script makes that decision explicit and auditable: every
currently-unused positive goes to one named split (``train_risk`` by default),
in place, and it reports exactly how many it touched.

    python scripts/assign_unused_positives.py --index data/nexar/index.csv

Safe to run more than once -- it only ever touches rows whose split is still
blank, so re-running after prepare_nexar.py adds more clips picks up just the
new ones.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="train_risk",
                   help="Split name to assign unused positives to.")
    args = p.parse_args()

    with open(args.index, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.index}: no rows")

    n = 0
    for r in rows:
        if not (r.get("split") or "") and int(float(r.get("label") or 0)) == 1:
            r["split"] = args.split
            n += 1

    with open(args.index, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"assigned {n} unused positive(s) to split='{args.split}' in {args.index}")


if __name__ == "__main__":
    main()
