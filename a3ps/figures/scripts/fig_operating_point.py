#!/usr/bin/env python
"""Figure A: operating-point (theta/c) trade-off for the locked checkpoint.

Scores the LOCKED checkpoint (seed 1234, notebooks/models/risk_gru_k1p0_v2_selfix_s1234.pt)
on data/features/train_val_v2 ONLY, at the same theta x confirm grid used for
the paper's Section VI-C v2-vs-legacy check (thresholds {0.5,0.6,0.7,0.8,0.9},
confirms {3,5,8} -- 15 cells for this seed; 45 across all three audited seeds).

Reuses evaluate() from scripts/train_risk_head.py unmodified -- no metric code,
config, split file or checkpoint is changed. Inference only, no training.

    python figures/scripts/fig_operating_point.py

Writes figures/fig_operating_point.{png,pdf,svg}.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from a3ps.risk.temporal import RiskGRU  # noqa: E402
from train_risk_head import load_clips, evaluate, guard_held_out  # noqa: E402

CHECKPOINT = os.path.join(ROOT, "notebooks", "models", "risk_gru_k1p0_v2_selfix_s1234.pt")
FEATURES = os.path.join(ROOT, "data", "features", "train_val_v2")
SPLIT_FREEZE = os.path.join(ROOT, "eval", "split_freeze_v2.json")

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
CONFIRMS = [3, 5, 8]
LOCKED = (0.70, 8)
FA_TARGET = 0.20

OUT_BASE = os.path.join(ROOT, "figures", "fig_operating_point")


def score_grid():
    guard_held_out(FEATURES, "--features", SPLIT_FREEZE, False, "--final-eval-report")
    model, extra = RiskGRU.load(CHECKPOINT)
    clips = load_clips(FEATURES)
    n_pos = sum(1 for c in clips if c["label"] == 1)
    rows = []
    for confirm in CONFIRMS:
        for thr in THRESHOLDS:
            v = evaluate(model, clips, thr, confirm=confirm)
            rows.append({
                "threshold": thr, "confirm": confirm,
                "useful_v2": v["useful_warning_rate_v2"],
                "fa": v["false_alarm_rate"],
            })
    return rows, len(clips), n_pos, extra


def main():
    rows, n_clips, n_pos, extra = score_grid()

    locked_row = next(r for r in rows if (r["threshold"], r["confirm"]) == LOCKED)
    print(f"checkpoint: {CHECKPOINT}")
    print(f"scored on: {FEATURES} -- {n_clips} clips, {n_pos} positive")
    print(f"sanity check @ theta=0.70, c=8: useful_v2={locked_row['useful_v2']:.4f} "
          f"(expect 0.612), FA={locked_row['fa']:.4f} (expect 0.196)")
    if not (abs(locked_row["useful_v2"] - 0.612) < 0.001 and abs(locked_row["fa"] - 0.196) < 0.001):
        raise SystemExit("SANITY CHECK FAILED -- locked point does not match the paper's "
                          "reported 0.612 / 0.196. Stopping without plotting; do not adjust "
                          "anything to force a match.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    INK = "#222222"
    BLUE = "#2F5D8A"
    GREY = "#6E6E6E"

    MARKERS = {3: "o", 5: "s", 8: "^"}
    SHADES = {3: "#9DB8D2", 5: "#5A85AC", 8: BLUE}

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for confirm in CONFIRMS:
        pts = sorted((r for r in rows if r["confirm"] == confirm), key=lambda r: r["threshold"])
        xs = [r["fa"] for r in pts]
        ys = [r["useful_v2"] for r in pts]
        ax.plot(xs, ys, color=SHADES[confirm], marker=MARKERS[confirm], markersize=3.2,
                linewidth=0.9, label=f"$c={confirm}$", zorder=2)

    ax.axvline(FA_TARGET, color=GREY, linestyle="--", linewidth=0.7, zorder=1)
    ax.text(FA_TARGET + 0.012, 0.03, "FA target", color=GREY, fontsize=6, rotation=90,
            va="bottom", ha="left")

    ax.scatter([locked_row["fa"]], [locked_row["useful_v2"]], s=38, facecolor=BLUE,
               edgecolor=INK, linewidth=0.6, zorder=3)
    ax.annotate(r"$\theta=0.70,\ c=8$", (locked_row["fa"], locked_row["useful_v2"]),
                xytext=(6, -9), textcoords="offset points", fontsize=6.5, color=INK)

    all_fa = [r["fa"] for r in rows]
    all_useful = [r["useful_v2"] for r in rows]
    x_hi = max(0.35, min(1.0, max(all_fa) * 1.15))
    y_lo = max(0.0, min(all_useful) * 0.85)
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(y_lo, 1.0)

    ax.set_xlabel("false-alarm rate", fontsize=7.5)
    ax.set_ylabel(r"useful warning ($p_t \geq \theta$, v2)", fontsize=7.5)
    ax.tick_params(labelsize=7)

    leg = ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.4,
                     borderaxespad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(OUT_BASE + ext, **kw)
    plt.close(fig)

    print(f"wrote {OUT_BASE}.png/.pdf/.svg")

    notes = {
        "checkpoint": CHECKPOINT.replace("\\", "/"),
        "features": FEATURES.replace("\\", "/"),
        "n_clips": n_clips, "n_pos": n_pos,
        "grid_thresholds": THRESHOLDS, "grid_confirms": CONFIRMS,
        "sanity_check": {"threshold": 0.70, "confirm": 8,
                          "useful_warning_v2": locked_row["useful_v2"],
                          "false_alarm_rate": locked_row["fa"],
                          "expected_useful_warning_v2": 0.612,
                          "expected_false_alarm_rate": 0.196},
        "rows": rows,
    }
    with open(os.path.join(ROOT, "figures", "fig_operating_point_data.json"), "w",
              encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)


if __name__ == "__main__":
    main()
