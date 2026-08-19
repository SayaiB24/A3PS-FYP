#!/usr/bin/env python
"""Sweep the DECISION threshold and confirm-frames of a trained risk head.

Why this is separate from training
----------------------------------
``threshold`` and ``confirm`` are evaluation-time parameters: they turn the
head's per-frame probability into a discrete alert, and they do not affect the
weights at all. So the whole trade-off curve between catching events and
crying wolf can be explored on an ALREADY-TRAINED checkpoint, for the cost of
one forward pass per clip (~a second for the 120-clip eval split) instead of a
~20-minute retrain per point.

Sweep these FIRST, before touching anything that requires retraining
(``--pre-alert-weight``, ``--pos-weight``, ``--kappa``). The first run of the
head scored a useful-warning rate of 0.833 at a false-alarm rate of 0.583 --
and false-alarm rate, not useful-warning rate, is the metric that decides
whether a result is publishable (``docs/design/metrics.md`` targets <= 0.20).
Raising the threshold or requiring more consecutive frames is the cheapest way
to buy FA back, and it costs lead time; this script shows you exactly what that
exchange rate is so the operating point is a deliberate choice.

    python scripts/sweep_operating_point.py \
        --checkpoint notebooks/models/risk_gru_v1.pt \
        --features data/features/eval \
        --thresholds 0.5,0.6,0.7,0.8,0.9 --confirms 3,5,8 \
        --out eval/operating_point_sweep.md

Note ``mean AP`` is identical at every row: it ranks clips by peak probability
and never applies the threshold. That is the point of including it -- it is the
model's threshold-free discrimination, so it tells you the ceiling that no
choice of operating point can exceed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from a3ps.risk.temporal import RiskGRU  # noqa: E402

from train_risk_head import evaluate, load_clips, _fmt  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="notebooks/models/risk_gru_v1.pt")
    p.add_argument("--features", default="data/features/eval",
                   help="Directory of .npz features to score (the held-out split).")
    p.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9",
                   help="Comma-separated probability thresholds to try.")
    p.add_argument("--confirms", default="3,5,8",
                   help="Comma-separated consecutive-frame requirements to try.")
    p.add_argument("--fa-target", type=float, default=0.20,
                   help="False-alarm rate to flag as acceptable (default 0.20, "
                        "per docs/design/metrics.md).")
    p.add_argument("--out", default="eval/operating_point_sweep.md")
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    confirms = [int(x) for x in args.confirms.split(",") if x.strip()]

    model, extra = RiskGRU.load(args.checkpoint)
    clips = load_clips(args.features)
    if not clips:
        raise SystemExit(f"no .npz features in {args.features}")

    n_pos = sum(1 for c in clips if c["label"] == 1)
    print(f"checkpoint {args.checkpoint} (best epoch "
          f"{extra.get('best_epoch')}, trained with "
          f"pre_alert_weight={extra.get('pre_alert_weight')}, "
          f"kappa={extra.get('kappa')})")
    print(f"scoring {len(clips)} clips ({n_pos} pos) from {args.features}\n")

    rows = []
    for conf in confirms:
        for thr in thresholds:
            v = evaluate(model, clips, thr, confirm=conf)
            rows.append({"threshold": thr, "confirm": conf, **v})
            print(f"  thr={thr:.2f} confirm={conf}  "
                  f"useful {_fmt(v['useful_warning_rate'])} "
                  f"({v['n_useful']}/{v['n_timed']}, {v['n_too_early']} early)  "
                  f"FA {_fmt(v['false_alarm_rate'])}  "
                  f"lead {_fmt(v['mean_lead_s'], 2)}s", flush=True)

    lines = [
        "# Operating-point sweep (decision threshold x confirm frames)",
        "",
        f"- checkpoint: `{args.checkpoint}` (best epoch "
        f"{extra.get('best_epoch')})",
        f"- scored on: `{args.features}` — {len(clips)} clips, {n_pos} positive",
        f"- trained with: `pre_alert_weight={extra.get('pre_alert_weight')}`, "
        f"`kappa={extra.get('kappa')}`",
        "",
        "These two parameters are applied **after** the model runs, so every row "
        "below is the same weights at a different decision boundary — no "
        "retraining involved.",
        "",
        "| threshold | confirm | useful | useful n | too early | **false alarm** "
        "| mean lead (s) | mean AP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = " ✅" if r["false_alarm_rate"] <= args.fa_target else ""
        lines.append(
            f"| {r['threshold']:.2f} | {r['confirm']} "
            f"| {_fmt(r['useful_warning_rate'])} "
            f"| {r['n_useful']}/{r['n_timed']} | {r['n_too_early']} "
            f"| **{_fmt(r['false_alarm_rate'])}**{flag} "
            f"| {_fmt(r['mean_lead_s'], 2)} | {_fmt(r['mean_AP'])} |")

    ok = [r for r in rows if r["false_alarm_rate"] <= args.fa_target]
    best_ok = max(ok, key=lambda r: r["useful_warning_rate"]) if ok else None

    lines += [
        "",
        "## How to read this",
        "",
        f"- **`mean AP` is constant across rows** ({_fmt(rows[0]['mean_AP'])}). "
        "It ranks clips by peak probability and ignores the threshold entirely, "
        "so it is the model's threshold-free discrimination — the ceiling no "
        "operating point can beat. Improving it needs better features or "
        "training, not a different threshold.",
        f"- **False-alarm rate is the gatekeeper** — target ≤ {args.fa_target:.2f}. "
        "A high useful-warning rate at a high FA is not a result: a system that "
        "alerts on most negatives will also 'catch' most positives.",
        "- **Lead time is the price.** Raising the threshold or confirm count "
        "buys FA back by alerting later. Below ~1 s of lead there is no time to "
        "react, so a row that fixes FA by collapsing lead has not solved "
        "anything.",
        "- **`too early`** counts positives alerted before the dataset's own "
        "`time_of_alert`. Those are not credited as useful, by design.",
        "",
    ]
    if best_ok is not None:
        lines.append(
            f"**Best row meeting FA ≤ {args.fa_target:.2f}:** threshold "
            f"{best_ok['threshold']:.2f}, confirm {best_ok['confirm']} → useful "
            f"{_fmt(best_ok['useful_warning_rate'])}, FA "
            f"{_fmt(best_ok['false_alarm_rate'])}, lead "
            f"{_fmt(best_ok['mean_lead_s'], 2)} s.")
    else:
        lines.append(
            f"**No row reaches FA ≤ {args.fa_target:.2f}.** The decision boundary "
            "alone cannot fix this: the model's separation between risky and "
            "ordinary driving is not sharp enough at any threshold. Next levers "
            "are retraining ones — raise `--pre-alert-weight` (penalise firing "
            "before `time_of_alert`), lower `--pos-weight` (weight negatives "
            "more heavily), or improve the features — not a further threshold "
            "sweep.")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
