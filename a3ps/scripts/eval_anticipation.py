#!/usr/bin/env python
"""Evaluate anticipation: mTTA + AP + false-alarm rate on held-out Nexar clips.

    python scripts/eval_anticipation.py --index data/nexar/index.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> None:
    p = argparse.ArgumentParser(
        description="mTTA / AP / false-alarm rate on held-out Nexar clips."
    )
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="eval", choices=["dev", "demo", "eval"])
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    # TODO: run the pipeline on each eval clip; for positives compare first
    # alert time vs annotated event time (mean Time-To-Accident); compute AP
    # over clip-level detection and false-alarm rate over negatives.
    raise NotImplementedError("eval_anticipation is a stub.")


if __name__ == "__main__":
    main()
