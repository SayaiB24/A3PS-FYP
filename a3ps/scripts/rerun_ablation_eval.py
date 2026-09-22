#!/usr/bin/env python
"""One-off: score the (post-fix) feature-ablation checkpoints at the committed
operating point (threshold 0.60, confirm 8), on the frozen eval split, with the
SAME feature columns zeroed at eval time as were zeroed during that arm's
training.

This is not part of the regular pipeline -- it exists to answer investigation 4
of the Phase 4 pivot audit (see A3PS_Pipeline_Logic.md request / eval/ablation_rerun_2026-09-22/README.md).
`scripts/sweep_operating_point.py` cannot be reused as-is here: it scores raw
eval features, but an ablated checkpoint was trained (and its own validation
numbers were produced) with those same columns zeroed on BOTH sides. Feeding it
intact eval features would test it on inputs unlike anything it saw in training.

Output goes to eval/ablation_rerun_2026-09-22/, a new directory, so the
existing (invalid, pre-fix) eval/*_abl_*.{json,txt,md} artifacts are left
untouched and distinguishable from this rerun.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a3ps.risk.temporal import RiskGRU  # noqa: E402
from train_risk_head import apply_ablation, evaluate, load_clips, resolve_ablation, _fmt  # noqa: E402

FEATURES = "data/features/eval"
THRESHOLD = 0.60
CONFIRM = 8
OUT_DIR = "eval/ablation_rerun_2026-09-22"

GROUPS = {
    "ego": "risk_gru_abl_ego.pt",
    "corridor": "risk_gru_abl_corridor.pt",
    "collision_prob": "risk_gru_abl_collprob.pt",
    "ttc": "risk_gru_abl_ttc.pt",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base_clips = load_clips(FEATURES)
    n_pos = sum(1 for c in base_clips if c["label"] == 1)
    print(f"scoring {len(base_clips)} eval clips ({n_pos} pos) from {FEATURES} "
          f"at threshold={THRESHOLD} confirm={CONFIRM}\n")

    rows = []
    for group, ckpt_name in GROUPS.items():
        ckpt = os.path.join("notebooks/models", ckpt_name)
        model, extra = RiskGRU.load(ckpt)
        cols, names = resolve_ablation(group)
        clips = load_clips(FEATURES)  # fresh copy so ablations don't stack
        apply_ablation(clips, cols)
        v = evaluate(model, clips, THRESHOLD, confirm=CONFIRM)
        row = {
            "group": group,
            "checkpoint": ckpt,
            "ablated_columns": names,
            "trained_ablate_field": extra.get("ablate"),
            "trained_best_epoch": extra.get("best_epoch"),
            **v,
        }
        rows.append(row)
        print(f"{group:15s} useful {_fmt(v['useful_warning_rate'])} "
              f"({v['n_useful']}/{v['n_timed']}, {v['n_too_early']} early)  "
              f"FA {_fmt(v['false_alarm_rate'])}  "
              f"lead {_fmt(v['mean_lead_s'], 2)}s  mean AP {_fmt(v['mean_AP'])}")

    with open(os.path.join(OUT_DIR, "ablation_rerun_op_point.json"), "w", encoding="utf-8") as fh:
        json.dump({"threshold": THRESHOLD, "confirm": CONFIRM, "features": FEATURES,
                    "rows": rows}, fh, indent=2, default=lambda o: o if o == o else None)

    lines = [
        "# Feature ablations, re-run post-fix -- scored at the committed operating point",
        "",
        f"Checkpoint per group: `notebooks/models/risk_gru_abl_<group>.pt` (already "
        "retrained post-fix, see `eval/train_log_abl_*.txt`). Scored here on "
        f"`{FEATURES}` (the frozen eval split, `eval/split_freeze.json`) at the "
        f"committed operating point threshold={THRESHOLD}, confirm={CONFIRM}, "
        "with the SAME columns zeroed at eval time as at train time for that arm.",
        "",
        "Committed full-feature model for comparison: `notebooks/models/risk_gru_k1p0.pt` "
        "-- useful 0.750, FA 0.167, mean AP 0.680, mean lead 1.67s "
        "(`eval/operating_point_sweep_k1p0.md`).",
        "",
        "| group | columns zeroed | useful-warning | false-alarm | mean lead (s) | mean AP |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {len(r['ablated_columns'])} "
            f"| {_fmt(r['useful_warning_rate'])} ({r['n_useful']}/{r['n_timed']}) "
            f"| {_fmt(r['false_alarm_rate'])} "
            f"| {_fmt(r['mean_lead_s'], 2)} "
            f"| {_fmt(r['mean_AP'])} |")
    lines.append("")

    out_md = os.path.join(OUT_DIR, "ablation_rerun_op_point.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
