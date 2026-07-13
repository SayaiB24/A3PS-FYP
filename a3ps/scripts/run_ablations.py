#!/usr/bin/env python
"""Ablation sweep: mTTA per config variant -> eval/ablation_table.md.

Runs ``scripts/eval_anticipation.py --run`` once per row in ``VARIANTS``
below, each time with a *temporary* override of ``configs/default.yaml``
(the original file is restored immediately after each run, even on failure).
Each variant flips one or more of the three ablation switches wired into the
pipeline for exactly this purpose (see ``a3ps/risk/decision.py``'s
``dynamic_threshold`` and ``scripts/run_pipeline.py``'s ``forecaster`` /
``ema_smoothing`` config keys):

    forecaster:        kalman_cv | seq2seq
    dynamic_threshold: true | false   (context-aware threshold lowering)
    ema_smoothing:     true | false   (cross-frame EMA on collision_prob)

Each variant's clips go to their own ``eval/ablation/<slug>/`` directory (so
variants never overwrite each other's cached events.json, and interrupted
sweeps can resume variant-by-variant -- eval_anticipation.py's own --run
already skips clips with an existing events.json).

    python scripts/run_ablations.py --index data/nexar/index.csv --split eval

VARIANTS below is Table II from the paper draft, in its exact row order:
Kalman-CV, Seq2Seq-LSTM, Static threshold, Dynamic threshold, EMA off, EMA on.
"Dynamic threshold" and "EMA on" are configurationally IDENTICAL to the
Kalman-CV baseline row (dynamic_threshold and ema_smoothing both default to
True) -- so instead of burning GPU time re-processing all 120 clips a second
and third time for the same config, those two rows are marked with a "reuse"
pointer and just copy the Kalman-CV row's already-computed numbers. Only 4 of
the 6 rows trigger an actual pipeline run.
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for eval_anticipation

import yaml  # noqa: E402

import eval_anticipation as ea  # noqa: E402

# ---------------------------------------------------------------------------
# Table II, exact row order. Each variant is either:
#   {"label": ..., "overrides": {...}}   -- runs eval_anticipation.py --run
#                                            with these overrides on top of
#                                            the ORIGINAL configs/default.yaml
#   {"label": ..., "reuse": "<label>"}   -- configurationally identical to an
#                                            earlier row; copies its summary
#                                            instead of re-running the pipeline
# Written to eval/ablation_table.md in this exact order.
# ---------------------------------------------------------------------------
VARIANTS = [
    {"label": "Kalman-CV", "overrides": {}},
    {"label": "Seq2Seq-LSTM", "overrides": {"forecaster": "seq2seq"}},
    {"label": "Static threshold", "overrides": {"dynamic_threshold": False}},
    {"label": "Dynamic threshold", "reuse": "Kalman-CV"},
    {"label": "EMA off", "overrides": {"ema_smoothing": False}},
    {"label": "EMA on", "reuse": "Kalman-CV"},
]


def _slug(label: str) -> str:
    keep = (c if c.isalnum() else "_" for c in label.lower())
    s = "".join(keep)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:60] or "variant"


def _write_config(base_config: dict, overrides: dict, path: str) -> None:
    merged = copy.deepcopy(base_config)
    merged.update(overrides)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, sort_keys=False)


def run_variant(label, overrides, config_path, original_text, original_config,
                index_path, split, clips_root, videos_root):
    """Apply one variant's config override, run eval_anticipation --run, restore.

    Returns the variant's A3PS summary dict (from eval_anticipation.compute_metrics),
    or None if nothing was processed (e.g. all videos missing).
    """
    slug = _slug(label)
    clips_dir = os.path.join(clips_root, slug)
    out_md = os.path.join(clips_root, f"{slug}.md")
    out_csv = os.path.join(clips_root, f"{slug}.csv")

    _write_config(original_config, overrides, config_path)
    try:
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "eval_anticipation.py"),
            "--index", index_path, "--split", split, "--run",
            "--config", config_path,
            "--clips-dir", clips_dir, "--out", out_md, "--per-clip-csv", out_csv,
        ]
        if videos_root:
            cmd += ["--videos-root", videos_root]
        print(f"\n=== variant: {label} (overrides={overrides or '{}'}) ===")
        subprocess.run(cmd, check=True)
    finally:
        # Restore the original file after EVERY variant, success or failure.
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(original_text)

    return _summarize_variant_csv(out_csv)


def _summarize_variant_csv(csv_path):
    """Read a variant's per-clip CSV back and compute the A3PS summary.

    Reuses eval_anticipation.compute_metrics on the a3ps_first_alert_t column
    so the ablation numbers are computed by the exact same tested logic as the
    main anticipation report, rather than re-implementing or regex-parsing
    the markdown table.
    """
    import csv as _csv

    if not os.path.isfile(csv_path):
        return None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            label = 1 if r["label"] == "pos" else (0 if r["label"] == "neg" else None)
            if label is None:
                continue
            rows.append({
                "clip_id": r["clip_id"],
                "label": label,
                "event_time_s": ea._to_float(r["event_time_s"]),
                "first_alert_t": ea._to_float(r["a3ps_first_alert_t"]),
                "peak_prob": ea._to_float(r["peak_prob"]) or 0.0,
                "processed": r["processed"] == "1",
            })
    if not rows:
        return None
    summary, _ = ea.compute_metrics(rows)
    return summary if summary["n_processed"] else None


def build_ablation_table(results, split):
    """``results``: list of (label, summary_or_None, reused_from_or_None)."""
    lines = [
        f"# Ablation study (split: {split})",
        "",
        "One row per config variant (see scripts/run_ablations.py VARIANTS for "
        "the exact overrides). Metrics are A3PS's own detection rate / "
        "false-alarm rate / mTTA under that variant's config -- not compared "
        "against the reactive baseline (see eval/anticipation.md for that). "
        "Rows marked \"(= <row>)\" are configurationally identical to an "
        "earlier row and reuse its numbers rather than re-running the pipeline.",
        "",
        "| variant | detect | false-alarm | mTTA (s) | n |",
        "|---|---|---|---|---|",
    ]
    for label, summary, reused_from in results:
        display_label = f"{label} (= {reused_from})" if reused_from else label
        if summary is None:
            lines.append(f"| {display_label} | n/a | n/a | n/a | 0 |")
            continue
        lines.append(
            f"| {display_label} | {ea._fmt(summary['detection_recall'])} "
            f"| {ea._fmt(summary['false_alarm_rate'])} "
            f"| {ea._fmt(summary['mTTA_s'], 2)} | {summary['n_tta']} |")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Run eval_anticipation.py once per ablation config variant.")
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="eval", choices=["dev", "demo", "eval"])
    p.add_argument("--config", default="configs/default.yaml",
                   help="Config file to temporarily override per variant (restored after each run).")
    p.add_argument("--clips-root", default="eval/ablation",
                   help="Per-variant processed clips + per-variant report/csv go under here.")
    p.add_argument("--videos-root", default=None)
    p.add_argument("--out", default="eval/ablation_table.md")
    args = p.parse_args()

    if not os.path.isfile(args.config):
        print(f"No config at {args.config}.")
        return
    if not os.path.isfile(args.index):
        print(f"No manifest at {args.index}. Run scripts/prepare_nexar.py first.")
        return

    with open(args.config, "r", encoding="utf-8") as fh:
        original_text = fh.read()
    original_config = yaml.safe_load(original_text) or {}

    # Backup file on disk too, in case this process is killed mid-run.
    backup_path = args.config + ".ablation_backup"
    shutil.copyfile(args.config, backup_path)

    results = []
    summary_by_label = {}
    try:
        for variant in VARIANTS:
            label = variant["label"]
            reuse = variant.get("reuse")
            if reuse is not None:
                summary = summary_by_label.get(reuse)
                print(f"\n=== variant: {label} (reusing '{reuse}' -- "
                      f"identical config, no re-run) ===")
            else:
                summary = run_variant(
                    label, variant.get("overrides", {}), args.config,
                    original_text, original_config,
                    args.index, args.split, args.clips_root, args.videos_root)
            summary_by_label[label] = summary
            results.append((label, summary, reuse))
            if summary is None:
                print(f"  ! variant '{label}': no processed clips (check videos/config).")
            else:
                print(f"  detect={ea._fmt(summary['detection_recall'])} "
                      f"false_alarm={ea._fmt(summary['false_alarm_rate'])} "
                      f"mTTA={ea._fmt(summary['mTTA_s'], 2)}s (n={summary['n_tta']})")
    finally:
        # Final safety net: guarantee the original config survives even if a
        # variant raised before its own restore, or the loop was interrupted.
        with open(args.config, "w", encoding="utf-8") as fh:
            fh.write(original_text)
        if os.path.isfile(backup_path):
            os.remove(backup_path)

    table = build_ablation_table(results, args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(table)

    print(f"\nwrote {args.out}")
    print(f"config restored: {args.config}")


if __name__ == "__main__":
    main()
