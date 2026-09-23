#!/usr/bin/env python
"""Step 11: ONE scoring pass of a fixed checkpoint at a fixed operating point.

Exists because ``sweep_operating_point.py`` prints only the legacy-definition
columns, and the single permitted eval_v2 read has to yield BOTH definitions
from one pass. It calls the same ``evaluate()`` and the same held-out guard; no
sweep, no alternate cells. Results are written to disk BEFORE anything is
printed so a display problem cannot force a second read.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a3ps.common.splits import DEFAULT_FREEZE_PATH  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402

from train_risk_head import evaluate, guard_held_out, load_clips  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--confirm", type=int, required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--split-freeze", default=DEFAULT_FREEZE_PATH)
    p.add_argument("--final-eval-report", action="store_true")
    args = p.parse_args()
    guard_held_out(args.features, "--features", args.split_freeze,
                   args.final_eval_report, "--final-eval-report")
    if os.path.exists(args.out_json):
        raise SystemExit(f"{args.out_json} exists; refusing to overwrite/re-read")
    model, extra = RiskGRU.load(args.checkpoint)
    clips = load_clips(args.features)
    v = evaluate(model, clips, args.threshold, confirm=args.confirm, dump_rows=True)
    v["AP"] = {str(k): x for k, x in v["AP"].items()}
    v.update(checkpoint=args.checkpoint, features=args.features,
             threshold=args.threshold, confirm=args.confirm,
             n_pos=sum(c["label"] for c in clips))
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(v, fh, indent=1, default=str)
    print(json.dumps({k: x for k, x in v.items() if k != "rows"}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
