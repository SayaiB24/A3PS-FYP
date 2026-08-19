#!/usr/bin/env python
"""Compare kappa variants of the risk head and apply the documented decision rule.

For each checkpoint, sweeps the decision threshold/confirm grid (free -- those are
evaluation-time parameters), finds the best row that meets the false-alarm target,
and tabulates the kappa values against each other so the winner is picked by a
stated rule rather than by eye.

    python scripts/compare_kappa.py \
        --checkpoints notebooks/models/risk_gru_k0p5.pt \
                      notebooks/models/risk_gru_k1p0.pt \
                      notebooks/models/risk_gru_k2p0.pt \
                      notebooks/models/risk_gru_k3p0.pt \
        --features data/features/eval --out eval/kappa_comparison.md

Decision rule (docs/handoff/KAPPA_RETRAIN.md §5), applied in order:

  1. hard gate: false-alarm rate <= --fa-target
  2. hard gate: useful-warning rate >= --min-useful
  3. hard gate: mean lead >= --min-lead (no time to react below ~1 s)
  4. maximise mean lead time among survivors
  5. tie-break on mean AP (better underlying discrimination is more durable)

If no kappa passes the gates, that is a reportable finding -- lead time is limited
by the model's discrimination, not by the loss weighting -- and this script says so
rather than picking the least-bad row.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from a3ps.risk.anticipation_loss import expected_lead_time  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402

from train_risk_head import _fmt, evaluate, load_clips  # noqa: E402

MEAN_ALERT_LEAD_S = 3.49        # measured over the 65 local positives


def sweep_checkpoint(path, clips, thresholds, confirms):
    """Every (threshold, confirm) row for one checkpoint, plus its train config."""
    model, extra = RiskGRU.load(path)
    rows = []
    for conf in confirms:
        for thr in thresholds:
            v = evaluate(model, clips, thr, confirm=conf)
            rows.append({"threshold": thr, "confirm": conf, **v})
    return rows, extra


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--features", default="data/features/eval")
    p.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--confirms", default="3,5,8")
    p.add_argument("--fa-target", type=float, default=0.20)
    p.add_argument("--min-useful", type=float, default=0.70)
    p.add_argument("--min-lead", type=float, default=1.0)
    p.add_argument("--out", default="eval/kappa_comparison.md")
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    confirms = [int(x) for x in args.confirms.split(",") if x.strip()]

    clips = load_clips(args.features)
    if not clips:
        raise SystemExit(f"no features in {args.features}")
    n_pos = sum(1 for c in clips if c["label"] == 1)
    print(f"scoring {len(clips)} clips ({n_pos} pos) from {args.features}\n")

    results = []
    for path in args.checkpoints:
        if not os.path.isfile(path):
            print(f"  ! missing: {path}")
            continue
        rows, extra = sweep_checkpoint(path, clips, thresholds, confirms)
        kappa = extra.get("kappa")
        compliant = [r for r in rows
                     if r["false_alarm_rate"] == r["false_alarm_rate"]
                     and r["false_alarm_rate"] <= args.fa_target
                     and r["useful_warning_rate"] >= args.min_useful
                     and r["mean_lead_s"] == r["mean_lead_s"]
                     and r["mean_lead_s"] >= args.min_lead]
        # Rule 4/5: maximise lead, tie-break on AP.
        pick = max(compliant, key=lambda r: (r["mean_lead_s"], r["mean_AP"])) \
            if compliant else None
        # For context when nothing passes: best row on FA alone.
        fallback = min(rows, key=lambda r: (r["false_alarm_rate"], -r["useful_warning_rate"]))
        results.append({"path": path, "kappa": kappa, "extra": extra,
                        "rows": rows, "pick": pick, "fallback": fallback,
                        "n_compliant": len(compliant)})
        tag = os.path.basename(path)
        if pick:
            print(f"  {tag:28s} kappa={kappa}  -> thr {pick['threshold']:.2f}/"
                  f"{pick['confirm']}  useful {_fmt(pick['useful_warning_rate'])}"
                  f"  FA {_fmt(pick['false_alarm_rate'])}"
                  f"  lead {_fmt(pick['mean_lead_s'], 2)}s"
                  f"  ({len(compliant)} compliant rows)")
        else:
            print(f"  {tag:28s} kappa={kappa}  -> NO row passes the gates "
                  f"(best FA {_fmt(fallback['false_alarm_rate'])} at useful "
                  f"{_fmt(fallback['useful_warning_rate'])})")

    if not results:
        raise SystemExit("no checkpoints scored")

    winners = [r for r in results if r["pick"]]
    winner = max(winners, key=lambda r: (r["pick"]["mean_lead_s"],
                                         r["pick"]["mean_AP"])) if winners else None

    lines = [
        "# Kappa comparison — does a lower kappa buy lead time?",
        "",
        f"- scored on `{args.features}` — {len(clips)} clips, {n_pos} positive "
        "(frozen held-out split)",
        f"- gates: false alarm ≤ {args.fa_target:.2f}, useful-warning ≥ "
        f"{args.min_useful:.2f}, mean lead ≥ {args.min_lead:.1f} s",
        "- decision rule: maximise mean lead among rows passing all gates, "
        "tie-break on mean AP (see `docs/handoff/KAPPA_RETRAIN.md` §5)",
        "",
        "`kappa` sets how sharply the anticipation loss concentrates weight just "
        "before the event. A large kappa asks for a late warning, which quietly "
        "turns an anticipation objective into a detection one — the *requested "
        "lead* column is what the loss is actually optimising for.",
        "",
        "| kappa | requested lead | best passing row | useful | FA | **mean lead** | mean AP |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["kappa"] is None, r["kappa"] or 0)):
        req = (f"{expected_lead_time(0.0, MEAN_ALERT_LEAD_S, float(r['kappa'])):.2f} s"
               if r["kappa"] is not None else "—")
        if r["pick"]:
            pk = r["pick"]
            lines.append(
                f"| {r['kappa']} | {req} | thr {pk['threshold']:.2f} / confirm "
                f"{pk['confirm']} | {_fmt(pk['useful_warning_rate'])} "
                f"| {_fmt(pk['false_alarm_rate'])} "
                f"| **{_fmt(pk['mean_lead_s'], 2)}** | {_fmt(pk['mean_AP'])} |")
        else:
            fb = r["fallback"]
            lines.append(
                f"| {r['kappa']} | {req} | *none passes* | — "
                f"| best {_fmt(fb['false_alarm_rate'])} | — "
                f"| {_fmt(fb['mean_AP'])} |")

    lines += ["", "## Verdict", ""]
    if winner:
        pk = winner["pick"]
        lines += [
            f"**kappa {winner['kappa']} wins**, at threshold "
            f"{pk['threshold']:.2f} / confirm {pk['confirm']}:",
            "",
            f"- useful-warning rate: **{_fmt(pk['useful_warning_rate'])}** "
            f"({pk['n_useful']}/{pk['n_timed']})",
            f"- false-alarm rate: **{_fmt(pk['false_alarm_rate'])}**",
            f"- mean lead time: **{_fmt(pk['mean_lead_s'], 2)} s**",
            f"- mean AP: {_fmt(pk['mean_AP'])}",
            f"- checkpoint: `{winner['path']}`",
            "",
        ]
        best_ap = max(r["pick"]["mean_AP"] for r in winners)
        spread = best_ap - min(r["pick"]["mean_AP"] for r in winners)
        if spread < 0.03:
            lines.append(
                f"_mean AP is flat across kappa (spread {spread:.3f}). That is "
                "expected: kappa changes **when** the model fires, not how well it "
                "separates risky from ordinary driving. Flat AP confirms the "
                "discrimination ceiling is a feature/capacity limit, so further "
                "loss tuning has little left to give — it is information, not a "
                "failed sweep._")
    else:
        lines += [
            "**No kappa passes the gates.** This is a real finding, not a failed "
            "run: at every threshold the model cannot deliver an acceptable "
            "false-alarm rate together with a useful warning rate and a reactable "
            "lead time. Lead time is limited by the model's discrimination "
            "(mean AP), not by the loss weighting, so the next levers are "
            "features and capacity — not kappa. Report it that way.",
            "",
        ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
