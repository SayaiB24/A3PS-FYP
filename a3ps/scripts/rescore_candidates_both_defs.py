#!/usr/bin/env python
"""Re-score the 2026-09-28 headline-number candidates under BOTH useful-warning definitions.

`docs/status/headline_number_attempts.md` originally compared each candidate's
useful-warning from `sweep_operating_point.py` -- which reports the LEGACY
(first-crossing) definition -- against the baseline's 0.612, which is the V2
(episode-based) figure. This script puts every run on one footing: same grid,
same `evaluate()`, both definitions side by side, and the project's own
selection rule (FA <= 0.20, v2 lead >= 1.0 s, zero gross-premature) applied
identically to each.

Inference only, over cached features, on train_val_v2 (or its _seq2seq
counterpart for the LSTM-feature run). eval_v2 is never read: `guard_held_out`
refuses it without --final-eval-report, which this script never passes.

    python scripts/rescore_candidates_both_defs.py

Writes eval/candidates_both_definitions.md.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from a3ps.risk.temporal import RiskGRU  # noqa: E402
from train_risk_head import load_clips, evaluate, guard_held_out  # noqa: E402

SPLIT_FREEZE = os.path.join(ROOT, "eval", "split_freeze_v2.json")
MODELS = os.path.join(ROOT, "notebooks", "models")
FEAT = os.path.join(ROOT, "data", "features")

CANDIDATES = [
    ("baseline (h64, l1, Kalman)", "risk_gru_k1p0_v2_selfix_s1234.pt", "train_val_v2"),
    ("hidden 128, layers 2", "risk_gru_h128_v2_s1234.pt", "train_val_v2"),
    ("hidden 256, layers 2", "risk_gru_h256_v2_s1234.pt", "train_val_v2"),
    ("LSTM-forecaster features", "risk_gru_k1p0_v2_seq2seq_s1234.pt", "train_val_v2_seq2seq"),
]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
CONFIRMS = [3, 5, 8]
LOCKED = (0.70, 8)
FA_TARGET, MIN_LEAD = 0.20, 1.0


def passes(v):
    lead = v.get("mean_lead_vs_event_v2")
    return (v["false_alarm_rate"] <= FA_TARGET and lead is not None and lead >= MIN_LEAD
            and v["n_gross_premature"] == 0)


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    out = ["# Headline-number candidates, both useful-warning definitions",
           "",
           "Written by `scripts/rescore_candidates_both_defs.py`. Inference only on "
           "`train_val_v2` (103 pos / 102 neg; the LSTM run on its `_seq2seq` twin, same "
           "clips). `eval_v2` not read. `useful_v2` = episode-based (the headline "
           "definition); `useful_legacy` = first-crossing (what `sweep_operating_point.py` "
           "reports). Selection rule: FA ≤ 0.20, v2 lead ≥ 1.0 s, zero gross-premature.",
           ""]
    summary = ["## Summary", "",
               "| run | mean AP | @0.70/c8 useful_v2 | @0.70/c8 useful_legacy | @0.70/c8 FA "
               "| best passing cell (by useful_v2) | useful_v2 | useful_legacy | FA | lead_v2 (s) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for name, ckpt, feat in CANDIDATES:
        fdir = os.path.join(FEAT, feat)
        guard_held_out(fdir, "--features", SPLIT_FREEZE, False, "--final-eval-report")
        model, _ = RiskGRU.load(os.path.join(MODELS, ckpt))
        clips = load_clips(fdir)
        grid = {}
        for c in CONFIRMS:
            for t in THRESHOLDS:
                grid[(t, c)] = evaluate(model, clips, t, confirm=c)
        locked = grid[LOCKED]
        ok = [(k, v) for k, v in grid.items() if passes(v)]
        best = max(ok, key=lambda kv: (kv[1]["useful_warning_rate_v2"],
                                       -kv[1]["false_alarm_rate"])) if ok else None
        ap = locked["mean_AP"]
        if best:
            (bt, bc), bv = best
            summary.append(
                f"| {name} | {ap:.3f} | {locked['useful_warning_rate_v2']:.3f} | "
                f"{locked['useful_warning_rate']:.3f} | {locked['false_alarm_rate']:.3f} | "
                f"{bt:.2f} / c{bc} | **{bv['useful_warning_rate_v2']:.3f}** | "
                f"{bv['useful_warning_rate']:.3f} | {bv['false_alarm_rate']:.3f} | "
                f"{fmt(bv.get('mean_lead_vs_event_v2'), 2)} |")
        else:
            summary.append(
                f"| {name} | {ap:.3f} | {locked['useful_warning_rate_v2']:.3f} | "
                f"{locked['useful_warning_rate']:.3f} | {locked['false_alarm_rate']:.3f} | "
                f"none pass | — | — | — | — |")
        print(summary[-1])
        detail += [f"## {name}", "",
                   f"`notebooks/models/{ckpt}` on `data/features/{feat}` — mean AP {ap:.3f}", "",
                   "| thr | confirm | useful_v2 | useful_legacy | FA | lead_v2 (s) | frac_premature "
                   "| gross-premature | passes rule |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for c in CONFIRMS:
            for t in THRESHOLDS:
                v = grid[(t, c)]
                detail.append(
                    f"| {t:.2f} | {c} | {v['useful_warning_rate_v2']:.3f} | "
                    f"{v['useful_warning_rate']:.3f} | {v['false_alarm_rate']:.3f} | "
                    f"{fmt(v.get('mean_lead_vs_event_v2'), 2)} | {fmt(v.get('frac_premature'))} | "
                    f"{v['n_gross_premature']} | {'✅' if passes(v) else ''} |")
        detail.append("")
    path = os.path.join(ROOT, "eval", "candidates_both_definitions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out + summary + [""] + detail))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
