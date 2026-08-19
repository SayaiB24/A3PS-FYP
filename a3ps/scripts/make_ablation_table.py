#!/usr/bin/env python
"""Table II — ablation of the LEARNED risk head (Phase IV).

The pre-Phase IV ablation (``scripts/run_ablations.py``) varies ``forecaster`` /
``dynamic_threshold`` / ``ema_smoothing`` / ``forecast_space`` -- none of which
exist in the learned head. This builds the equivalent table for the system the
paper actually claims: feature-group ablations plus capacity, each evaluated at
its own best FA-compliant operating point so no arm is penalised for a decision
boundary tuned on a different model.

    python scripts/make_ablation_table.py --out eval/ablation_table_phase4.md

Every arm is trained with identical hyperparameters and seed, and ablations zero
feature columns rather than removing them, so input width, architecture and
parameter count are unchanged. A difference is therefore attributable to the
missing information, not to a different-sized model. The capacity arm is the one
exception and is labelled as such.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a3ps.risk.temporal import RiskGRU, count_parameters  # noqa: E402

from train_risk_head import _fmt, evaluate, load_clips  # noqa: E402

# (checkpoint tag, table label, what it tests)
ARMS = [
    ("k1p0", "**Full model** (all features, hidden 64)", "reference"),
    ("abl_ego", "− ego motion", "the feature class Phase IV added"),
    ("abl_collprob", "− Phase III collision probability", "the old system's output as a feature"),
    ("abl_ttc", "− looming / TTC", "time-to-collision cues"),
    ("abl_corridor", "− ego-corridor geometry", "where the actor is relative to our path"),
    ("cap_h128", "hidden 128, 2 layers", "capacity (not a feature ablation)"),
]


def best_row(model, clips, thresholds, confirms, fa_target, min_lead):
    """Best FA-compliant operating point for one checkpoint, else best-FA row."""
    rows = [{"threshold": t, "confirm": c, **evaluate(model, clips, t, confirm=c)}
            for c in confirms for t in thresholds]
    ok = [r for r in rows
          if r["false_alarm_rate"] == r["false_alarm_rate"]
          and r["false_alarm_rate"] <= fa_target
          and r["mean_lead_s"] == r["mean_lead_s"]
          and r["mean_lead_s"] >= min_lead]
    if ok:
        return max(ok, key=lambda r: r["useful_warning_rate"]), True
    return min(rows, key=lambda r: r["false_alarm_rate"]), False


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="notebooks/models")
    p.add_argument("--features", default="data/features/eval")
    p.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    p.add_argument("--confirms", default="3,5,8")
    p.add_argument("--fa-target", type=float, default=0.20)
    p.add_argument("--min-lead", type=float, default=1.0)
    p.add_argument("--out", default="eval/ablation_table_phase4.md")
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    confirms = [int(x) for x in args.confirms.split(",") if x.strip()]

    clips = load_clips(args.features)
    if not clips:
        raise SystemExit(f"no features in {args.features}")
    n_pos = sum(1 for c in clips if c["label"] == 1)
    print(f"scoring {len(clips)} clips ({n_pos} pos) from {args.features}\n")

    results, ref = [], None
    for tag, label, tests in ARMS:
        path = os.path.join(args.models_dir, f"risk_gru_{tag}.pt")
        if not os.path.isfile(path):
            print(f"  ! missing {path} -- skipped")
            continue
        model, extra = RiskGRU.load(path)
        row, compliant = best_row(model, clips, thresholds, confirms,
                                  args.fa_target, args.min_lead)
        rec = {"tag": tag, "label": label, "tests": tests, "row": row,
               "compliant": compliant, "params": count_parameters(model),
               "ablate": extra.get("ablate")}
        if tag == "k1p0":
            ref = rec
        results.append(rec)
        print(f"  {label:44s} useful {_fmt(row['useful_warning_rate'])} "
              f"FA {_fmt(row['false_alarm_rate'])} lead "
              f"{_fmt(row['mean_lead_s'], 2)}s AP {_fmt(row['mean_AP'])}"
              f"{'' if compliant else '  (no FA-compliant row)'}")

    if not results:
        raise SystemExit("no checkpoints found")

    def delta(rec, key):
        if ref is None or rec is ref:
            return "—"
        d = rec["row"][key] - ref["row"][key]
        return f"{d:+.3f}"

    lines = [
        "# Table II — ablation of the learned risk head (Phase IV)",
        "",
        f"- scored on `{args.features}` — {len(clips)} clips, {n_pos} positive "
        "(frozen held-out split)",
        "- every arm: identical hyperparameters and seed (kappa 1.0, "
        "pre_alert_weight 0.5, 30 epochs max, patience 8, seed 1234)",
        f"- each arm reported at **its own** best operating point meeting "
        f"FA ≤ {args.fa_target:.2f} and lead ≥ {args.min_lead:.1f} s, so no arm is "
        "penalised for a threshold tuned on a different model",
        "- ablations **zero** feature columns rather than removing them, so input "
        "width, architecture and parameter count are unchanged; a difference is "
        "attributable to the missing information, not to a different-sized model",
        "",
        "| variant | tests | thr/conf | useful-warning | Δ | FA | mean lead | mean AP | Δ AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in results:
        r = rec["row"]
        flag = "" if rec["compliant"] else " ⚠️"
        lines.append(
            f"| {rec['label']} | {rec['tests']} "
            f"| {r['threshold']:.2f}/{r['confirm']} "
            f"| {_fmt(r['useful_warning_rate'])}{flag} | {delta(rec, 'useful_warning_rate')} "
            f"| {_fmt(r['false_alarm_rate'])} | {_fmt(r['mean_lead_s'], 2)} "
            f"| {_fmt(r['mean_AP'])} | {delta(rec, 'mean_AP')} |")

    lines += ["", "## How to read this", ""]
    if ref is not None:
        drops = [(rec["label"], rec["row"]["mean_AP"] - ref["row"]["mean_AP"])
                 for rec in results if rec is not ref and rec["tag"] != "cap_h128"]
        drops.sort(key=lambda kv: kv[1])
        if drops:
            worst, wd = drops[0]
            best, bd = drops[-1]
            lines += [
                f"- **Most load-bearing feature group: {worst}** — removing it costs "
                f"{abs(wd):.3f} mean AP.",
                f"- **Least: {best}** ({bd:+.3f} mean AP). A near-zero or positive "
                "delta means that group is redundant given the others, which is a "
                "result worth reporting rather than hiding — it says the model does "
                "not need it, not that the feature is wrong.",
                "",
            ]
    lines += [
        "- **`mean AP` is the honest comparison across arms.** It is threshold-free, "
        "so it cannot be moved by an arm happening to suit a particular decision "
        "boundary. Read useful-warning alongside its FA, never alone.",
        "- **⚠️ marks an arm with no FA-compliant operating point at all** — its row "
        "is the lowest-FA row available and its useful-warning rate is therefore not "
        "comparable to the others.",
        "- The capacity arm changes the parameter count and is **not** a feature "
        "ablation; it is included because the kappa sweep pointed at capacity rather "
        "than the loss as the remaining lever "
        "(see `kappa_comparison.md`).",
        "",
        "| arm | params |",
        "|---|---|",
    ]
    for rec in results:
        lines.append(f"| {rec['label']} | {rec['params']:,} |")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
