#!/usr/bin/env python
"""Evaluate anticipation: mTTA + AP + false-alarm rate on Nexar clips.

The heavy perception (YOLO + tracking) and the light metric computation are
decoupled so you can iterate on metrics without a GPU:

  * The pipeline writes a full ClipResult (meta + frames + events) to
    ``<clips-dir>/<clip_id>/events.json`` for each clip.
  * This script reads those processed results and scores them against the
    manifest labels (``event_time_s`` for positives).

On the GPU laptop, add ``--run`` to process any clip whose ``events.json`` is
missing before scoring:

    python scripts/eval_anticipation.py --index data/nexar/index.csv --split eval --run

Without ``--run`` it scores whatever is already processed (no GPU needed) --
this is how the metric logic is verified locally.

Metrics (clip level):
  * detection recall -- fraction of positives that raised an alert BEFORE the
    annotated event time.
  * mTTA (s)         -- mean Time-To-Accident over correctly-anticipated
    positives: mean(event_time - first_alert_time).
  * false-alarm rate -- fraction of negatives that raised any alert.
  * AP               -- average precision ranking clips by their peak
    collision probability (positive vs negative).

An "alert" is any event whose type is in ``--alert-types`` (default
ALERT,VIRTUAL_BRAKE). Writes eval/anticipation_report.md.
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for run_pipeline hooks

from a3ps.common.schema import ClipResult  # noqa: E402

DEFAULT_ALERT_TYPES = ("ALERT", "VIRTUAL_BRAKE")


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def load_manifest(index_path, split):
    """Read index.csv rows for one split -> list of dicts (clip_id/path/label/...)."""
    with open(index_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if split and (r.get("split") or "") != split:
            continue
        out.append({
            "clip_id": r.get("clip_id", ""),
            "path": r.get("path", ""),
            "label": _to_int(r.get("label")),
            "event_time_s": _to_float(r.get("event_time_s")),
        })
    return out


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# per-clip summary from a processed ClipResult
# ---------------------------------------------------------------------------

def summarize_clip(clip_result, alert_types):
    """Reduce a ClipResult to (first_alert_t, peak_prob) for scoring.

    first_alert_t: earliest event time whose type is an alert type (or None).
    peak_prob:     max smoothed collision probability over all tracks/frames
                   (the clip-level detection score used for AP; 0.0 if none).
    """
    alert_ts = [e.t for e in clip_result.events if e.type in alert_types]
    first_alert_t = min(alert_ts) if alert_ts else None

    peak = 0.0
    for fr in clip_result.frames:
        for tr in fr.tracks:
            p = getattr(tr.prediction, "collision_prob", None) if tr.prediction else None
            if p is not None and p > peak:
                peak = float(p)
    return first_alert_t, peak


def clip_events_path(clips_dir, clip_id):
    return os.path.join(clips_dir, str(clip_id), "events.json")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def average_precision(scores, labels):
    """Average precision over clip scores (1=positive, 0=negative).

    Exact area under the precision-recall curve. Ties are broken by input
    order; NaN if there are no positives.
    """
    pos = sum(1 for y in labels if y == 1)
    if pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def compute_metrics(rows):
    """Aggregate per-clip rows into anticipation metrics.

    Each row: {clip_id, label, event_time_s, first_alert_t, peak_prob,
    processed}. Only processed rows contribute. Returns (summary, detail_rows).
    """
    used = [r for r in rows if r["processed"]]
    pos = [r for r in used if r["label"] == 1]
    neg = [r for r in used if r["label"] == 0]

    ttas = []
    detected = 0
    for r in pos:
        te = r["event_time_s"]
        fa = r["first_alert_t"]
        anticipated = fa is not None and (te is None or fa <= te)
        r["anticipated"] = anticipated
        if anticipated:
            detected += 1
            if te is not None:
                ttas.append(te - fa)

    false_alarms = 0
    for r in neg:
        fa = r["first_alert_t"] is not None
        r["false_alarm"] = fa
        if fa:
            false_alarms += 1

    scores = [r["peak_prob"] for r in used]
    labels = [1 if r["label"] == 1 else 0 for r in used]

    summary = {
        "n_processed": len(used),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "detection_recall": (detected / len(pos)) if pos else float("nan"),
        "mTTA_s": (sum(ttas) / len(ttas)) if ttas else float("nan"),
        "n_tta": len(ttas),
        "false_alarm_rate": (false_alarms / len(neg)) if neg else float("nan"),
        "AP": average_precision(scores, labels),
    }
    return summary, used


# ---------------------------------------------------------------------------
# optional: run the pipeline for missing clips (GPU path)
# ---------------------------------------------------------------------------

def process_clip(video_path, out_dir, config):
    """Run the full pipeline (forecaster + risk) on one clip -> events.json."""
    from run_pipeline import make_forecaster_hook, make_risk_hook
    from a3ps.pipeline import Pipeline

    pipeline = Pipeline(
        config,
        forecaster=make_forecaster_hook(config),
        risk_engine=make_risk_hook(config),
        explainer=None,
    )
    return pipeline.run(video_path, out_dir)


def resolve_video(row, videos_root):
    """Resolve a manifest row's video path relative to the index directory."""
    rel = row["path"] or ""
    return os.path.normpath(os.path.join(videos_root, *rel.split("/")))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def build_report(summary, detail, split):
    lines = [f"# Anticipation eval (split: {split})", ""]
    lines += [
        f"- clips scored: **{summary['n_processed']}** "
        f"({summary['n_positive']} positive, {summary['n_negative']} negative)",
        f"- detection recall (alert before event): **{_fmt(summary['detection_recall'])}**",
        f"- mTTA: **{_fmt(summary['mTTA_s'], 2)} s** (over {summary['n_tta']} anticipated positives)",
        f"- false-alarm rate (negatives): **{_fmt(summary['false_alarm_rate'])}**",
        f"- AP (peak-prob ranking): **{_fmt(summary['AP'])}**",
        "",
        "| clip_id | label | event_t | first_alert_t | peak_prob | outcome |",
        "|---------|-------|---------|---------------|-----------|---------|",
    ]
    for r in sorted(detail, key=lambda r: (r["label"] != 1, str(r["clip_id"]))):
        if r["label"] == 1:
            outcome = "anticipated" if r.get("anticipated") else "MISS"
        else:
            outcome = "FALSE ALARM" if r.get("false_alarm") else "clean"
        lines.append(
            f"| {r['clip_id']} | {'pos' if r['label'] == 1 else 'neg'} "
            f"| {_fmt(r['event_time_s'], 2)} | {_fmt(r['first_alert_t'], 2)} "
            f"| {_fmt(r['peak_prob'])} | {outcome} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="mTTA / AP / false-alarm rate on Nexar clips.")
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="eval", choices=["dev", "demo", "eval"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--clips-dir", default="eval/anticipation",
                   help="Where processed <clip_id>/events.json live (and go with --run).")
    p.add_argument("--videos-root", default=None,
                   help="Root for resolving manifest paths (default: index dir).")
    p.add_argument("--run", action="store_true",
                   help="Run the pipeline for clips missing events.json (needs GPU/model).")
    p.add_argument("--alert-types", default=",".join(DEFAULT_ALERT_TYPES),
                   help="Comma-separated event types that count as an alert.")
    p.add_argument("--out", default="eval/anticipation_report.md")
    args = p.parse_args()

    if not os.path.isfile(args.index):
        print(f"No manifest at {args.index}. Run scripts/prepare_nexar.py first.")
        return

    alert_types = tuple(t.strip() for t in args.alert_types.split(",") if t.strip())
    videos_root = args.videos_root or os.path.dirname(os.path.abspath(args.index))
    manifest = load_manifest(args.index, args.split)
    if not manifest:
        print(f"No clips with split={args.split} in {args.index}.")
        return

    config = None
    if args.run:
        from a3ps.pipeline import load_config
        config = load_config(args.config)

    rows = []
    n_missing = 0
    for m in manifest:
        ev_path = clip_events_path(args.clips_dir, m["clip_id"])

        if not os.path.isfile(ev_path) and args.run:
            video = resolve_video(m, videos_root)
            if os.path.isfile(video):
                print(f"  running pipeline: {m['clip_id']} ...")
                process_clip(video, os.path.dirname(ev_path), config)
            else:
                print(f"  ! video not found for {m['clip_id']}: {video}")

        row = {**m, "first_alert_t": None, "peak_prob": 0.0, "processed": False}
        if os.path.isfile(ev_path):
            clip = ClipResult.load_json(ev_path)
            row["first_alert_t"], row["peak_prob"] = summarize_clip(clip, alert_types)
            row["processed"] = True
        else:
            n_missing += 1
        rows.append(row)

    summary, detail = compute_metrics(rows)
    if summary["n_processed"] == 0:
        print(f"No processed clips found under {args.clips_dir}. "
              f"Re-run with --run (on the GPU laptop) to process them.")
        return
    if n_missing:
        print(f"note: {n_missing}/{len(manifest)} clips not processed "
              f"(scored the {summary['n_processed']} available). Use --run to fill in.")

    report = build_report(summary, detail, args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)

    print()
    print(f"detection recall : {_fmt(summary['detection_recall'])}")
    print(f"mTTA             : {_fmt(summary['mTTA_s'], 2)} s "
          f"(n={summary['n_tta']})")
    print(f"false-alarm rate : {_fmt(summary['false_alarm_rate'])}")
    print(f"AP               : {_fmt(summary['AP'])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
