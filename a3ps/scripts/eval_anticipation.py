#!/usr/bin/env python
"""Evaluate anticipation: A3PS vs a naive reactive-proximity baseline.

Runs the full pipeline (no rendering) over every clip in a Nexar split, flags a
clip when any ALERT / VIRTUAL_BRAKE fires, and reports, for BOTH methods:

  * detection rate  -- fraction of positives that alerted BEFORE the annotated
    event time (true anticipation).
  * false-alarm rate -- fraction of negatives that raised any alert.
  * mTTA (s)        -- mean Time-To-Accident over anticipated positives:
    mean(event_time - first_alert_time).

The two methods:

  * **A3PS** -- the pipeline's forecast + risk decision engine (the ALERT /
    VIRTUAL_BRAKE events already in events.json).
  * **Reactive-proximity baseline** -- a naive ADAS that "brakes" the first
    frame ANY actor comes physically close to the ego corridor. This mirrors
    the dashboard's reactive-ADAS marker exactly (BEV centroid within 2.0 m of
    the BEV corridor, else image centroid within 8% of frame height of the
    image corridor). No forecasting -- it can only react once the actor is
    already close, so its mTTA is the "how much later a dumb system would have
    reacted" reference A3PS is measured against.

Heavy perception (YOLO + tracking) and the light metric computation are
decoupled: the pipeline writes ``<clips-dir>/<clip_id>/events.json`` per clip
(no video, ``render=False``), and this script scores those. Add ``--run`` to
process any clip whose events.json is missing (needs GPU/model):

    python scripts/eval_anticipation.py --index data/nexar/index.csv --split eval --run

Without ``--run`` it scores whatever is already processed (no GPU needed).

Writes ``eval/anticipation.md`` (the two-method comparison) and
``eval/anticipation_per_clip.csv`` (one row per clip, both methods).

Threshold sweep (Step 7.1, ``--sweep``): once clips are processed, each clip's
per-frame max collision_prob is cached to ``eval/cache/<clip_id>.csv`` (a
one-time extraction from the already-loaded ClipResult -- no extra GPU work).
Precision/recall are then recomputed directly from that cache for every
``base_threshold`` in 0.40..0.90 (step 0.05), entirely in-memory, so sweeping
11 thresholds costs nothing beyond the one pipeline pass already paid for.
Writes ``eval/pr_curve_data.csv`` (columns: threshold, precision, recall) and
``eval/pr_curve.png``:

    python scripts/eval_anticipation.py --index data/nexar/index.csv --split eval --sweep
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for run_pipeline hooks

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.risk.collision import point_in_dilated_polygon  # noqa: E402

DEFAULT_ALERT_TYPES = ("ALERT", "VIRTUAL_BRAKE")

# Reactive-proximity baseline thresholds -- kept identical to the dashboard's
# reactive-ADAS marker (dashboard/app.js: computeReactiveMarkers):
#   BEV: centroid within 2.0 m of the BEV corridor polygon,
#   img: centroid within 8% of frame height of the image corridor polygon.
REACTIVE_BEV_THRESH_M = 2.0
REACTIVE_IMG_FRAC = 0.08

# "Useful warning" window, in seconds before the annotated collision.
#
# Why this exists: mTTA rewards warning EARLY without bound, so a method that
# alarms in the first frame of every clip scores a huge mTTA and an unbeatable
# anticipation gain while telling the driver nothing. The reactive-proximity
# baseline does exactly that -- on the Nexar eval clips it fires within 2 s of
# clip start on ~76% of positives, roughly 18 s before impact. A warning is only
# actionable in a band: late enough to be about THIS hazard, early enough to
# brake. Outside that band it is either noise or a surprise.
#
# Lower bound: below ~0.5 s there is no time to react at all.
# Upper bound: beyond ~6 s the warning cannot be attributed to the event.
USEFUL_WINDOW_LO_S = 0.5
USEFUL_WINDOW_HI_S = 6.0


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
    """Reduce a ClipResult to (first_alert_t, peak_prob) for A3PS scoring.

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


def reactive_first_alert(clip_result, frame_height,
                         bev_thresh=REACTIVE_BEV_THRESH_M, img_frac=REACTIVE_IMG_FRAC):
    """First time ANY actor is close to the ego corridor (naive reactive ADAS).

    Mirrors the dashboard's reactive-ADAS marker: per frame, per track, prefer
    BEV metres (centroid_bev within ``bev_thresh`` m of corridor_poly_bev),
    else fall back to image pixels (centroid_img within ``img_frac`` * frame
    height of corridor_poly_img). Returns the earliest such frame time, or None
    if no actor ever comes close. Independent of A3PS's events -- it only reads
    tracked positions + the ego corridor, so it is a fair standalone baseline.
    """
    px_thresh = float(img_frac) * float(frame_height or 0)
    for fr in clip_result.frames:
        ego = fr.ego or {}
        bev_poly = ego.get("corridor_poly_bev")
        img_poly = ego.get("corridor_poly_img")
        for tr in fr.tracks:
            if tr.centroid_bev and bev_poly:
                if point_in_dilated_polygon(tr.centroid_bev, bev_poly, bev_thresh):
                    return fr.t
            elif img_poly and tr.centroid_img and px_thresh > 0:
                if point_in_dilated_polygon(tr.centroid_img, img_poly, px_thresh):
                    return fr.t
    return None


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


def matched_subset_metrics(rows):
    """mTTA for BOTH methods, restricted to positives BOTH correctly anticipate.

    Comparing each method's mTTA over its own (differently-sized) true-positive
    set is confounded by recall: a method with lower recall/higher false-alarm
    rate can show an inflated mTTA simply because it only "counts" an easier or
    smaller subset of clips. Restricting both methods to the SAME matched
    subset removes that bias, so a "+X.X s earlier" claim is defensible.

    ``rows`` are the master per-clip rows (clip_id, label, event_time_s,
    a3ps_first_alert_t, reactive_first_alert_t, processed). Returns None if no
    positive clip is anticipated by both methods (nothing to compare).
    """
    a3ps_ttas, reactive_ttas = [], []
    for r in rows:
        if not r.get("processed") or r["label"] != 1:
            continue
        te = r["event_time_s"]
        if te is None:
            continue
        a_fa = r["a3ps_first_alert_t"]
        r_fa = r["reactive_first_alert_t"]
        a_ok = a_fa is not None and a_fa <= te
        r_ok = r_fa is not None and r_fa <= te
        if a_ok and r_ok:
            a3ps_ttas.append(te - a_fa)
            reactive_ttas.append(te - r_fa)

    if not a3ps_ttas:
        return None
    a3ps_mtta = sum(a3ps_ttas) / len(a3ps_ttas)
    reactive_mtta = sum(reactive_ttas) / len(reactive_ttas)
    return {
        "n_matched": len(a3ps_ttas),
        "a3ps_mtta_s": a3ps_mtta,
        "reactive_mtta_s": reactive_mtta,
        "gain_s": a3ps_mtta - reactive_mtta,
    }


# ---------------------------------------------------------------------------
# threshold sweep (Step 7.1): cache per-frame max prob once, sweep in-memory
# ---------------------------------------------------------------------------

# base_threshold sweep range: 0.40, 0.45, ..., 0.90 (11 points). round() avoids
# float-step drift (0.4 + 0.05*11 != exactly 0.95 in binary floating point).
THRESHOLD_SWEEP = [round(0.40 + 0.05 * i, 2) for i in range(11)]


def cache_frame_probs(clip_result, clip_id, cache_dir, force=False):
    """Extract each frame's max collision_prob (over all tracks) to a small
    per-clip CSV cache under ``cache_dir`` -- computed ONCE from a ClipResult
    already loaded for the main scoring pass, so this adds no extra parsing of
    the (much larger) events.json, and repeated ``--sweep`` runs skip clips
    that are already cached (pass ``force=True`` to rebuild).

    This is what lets the threshold sweep avoid re-running the GPU pipeline:
    every candidate ``base_threshold`` is just a comparison against these
    already-extracted numbers, not a new forecast/risk pass.
    """
    path = os.path.join(cache_dir, f"{clip_id}.csv")
    if os.path.isfile(path) and not force:
        return path
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "max_prob"])
        for fr in clip_result.frames:
            max_p = 0.0
            for tr in fr.tracks:
                p = getattr(tr.prediction, "collision_prob", None) if tr.prediction else None
                if p is not None and p > max_p:
                    max_p = float(p)
            w.writerow([f"{fr.t:.3f}", f"{max_p:.4f}"])
    return path


def load_frame_prob_cache(cache_dir, clip_id):
    """Read one clip's cached (t, max_prob) list back, or None if not cached."""
    path = os.path.join(cache_dir, f"{clip_id}.csv")
    if not os.path.isfile(path):
        return None
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append((float(r["t"]), float(r["max_prob"])))
    return out


def sweep_thresholds(rows, cache_dir, thresholds=THRESHOLD_SWEEP):
    """Precision/recall at each candidate ``base_threshold``, from the cache only.

    For a positive clip, "detected" at threshold T means some cached frame at
    or before ``event_time_s`` has max_prob >= T (mirrors the "anticipated"
    semantics used everywhere else in this file). For a negative clip,
    "flagged" at T means ANY cached frame has max_prob >= T (a false alarm).
    This intentionally ignores the decision engine's frames_to_confirm/cooldown
    debounce -- the sweep is measuring the pure probability decision boundary,
    holding the rest of the pipeline fixed, which is what a PR-curve-over-
    threshold is meant to isolate.

    ``rows`` are the master per-clip rows (need clip_id/label/event_time_s/
    processed). Returns a list of dicts in ``thresholds`` order:
    {threshold, precision, recall, tp, fp, fn}.
    """
    caches = {}
    for r in rows:
        if not r.get("processed"):
            continue
        cache = load_frame_prob_cache(cache_dir, r["clip_id"])
        if cache is not None:
            caches[r["clip_id"]] = cache

    sweep = []
    for thr in thresholds:
        tp = fp = fn = 0
        for r in rows:
            cache = caches.get(r["clip_id"])
            if cache is None:
                continue
            if r["label"] == 1:
                te = r["event_time_s"]
                hit = any(p >= thr and (te is None or t <= te) for t, p in cache)
                if hit:
                    tp += 1
                else:
                    fn += 1
            elif r["label"] == 0:
                if any(p >= thr for _t, p in cache):
                    fp += 1
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        sweep.append({"threshold": thr, "precision": precision, "recall": recall,
                       "tp": tp, "fp": fp, "fn": fn})
    return sweep


def write_pr_csv(path, sweep):
    """threshold,precision,recall -- one row per swept threshold."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "precision", "recall"])
        for row in sweep:
            w.writerow([
                f"{row['threshold']:.2f}",
                "" if math.isnan(row["precision"]) else f"{row['precision']:.4f}",
                "" if math.isnan(row["recall"]) else f"{row['recall']:.4f}",
            ])


def render_pr_curve(path_png, sweep):
    """Precision-vs-recall curve, each point annotated with its threshold."""
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt

    pts = [(row["recall"], row["precision"], row["threshold"]) for row in sweep
           if not math.isnan(row["recall"]) and not math.isnan(row["precision"])]
    pts.sort(key=lambda p: p[0])   # ascending recall -> a left-to-right curve

    fig, ax = plt.subplots(figsize=(6, 6))
    if pts:
        rs = [p[0] for p in pts]
        ps = [p[1] for p in pts]
        ax.plot(rs, ps, "-o", color="#1f77b4")
        for r, prec, t in pts:
            ax.annotate(f"{t:.2f}", (r, prec), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
    else:
        ax.text(0.5, 0.5, "no valid (precision, recall) points",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("A3PS precision-recall sweep over base_threshold (0.40-0.90)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def compute_metrics(rows, useful_lo=USEFUL_WINDOW_LO_S, useful_hi=USEFUL_WINDOW_HI_S):
    """Aggregate per-clip rows into anticipation metrics.

    Each row: {clip_id, label, event_time_s, first_alert_t, peak_prob,
    processed}. Only processed rows contribute. Returns (summary, detail_rows).
    Works for either method -- pass rows whose ``first_alert_t`` is the A3PS
    alert time or the reactive-baseline alert time.

    ``useful_lo``/``useful_hi`` bound the actionable warning window used for
    ``useful_warning_rate`` (see USEFUL_WINDOW_LO_S).
    """
    used = [r for r in rows if r["processed"]]
    pos = [r for r in used if r["label"] == 1]
    neg = [r for r in used if r["label"] == 0]

    ttas = []
    detected = 0
    useful = 0
    n_timed = 0            # positives with a known event time (useful-rate denominator)
    for r in pos:
        te = r["event_time_s"]
        fa = r["first_alert_t"]
        anticipated = fa is not None and (te is None or fa <= te)
        r["anticipated"] = anticipated
        if te is not None:
            n_timed += 1
        r["useful"] = False
        if anticipated:
            detected += 1
            if te is not None:
                tta = te - fa
                ttas.append(tta)
                # Actionable only inside the window -- see USEFUL_WINDOW_*.
                if useful_lo <= tta <= useful_hi:
                    r["useful"] = True
                    useful += 1

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
        # Fraction of ALL timed positives (not just anticipated ones) warned
        # inside the actionable window: a miss and an unusably-early warning
        # both count against it, which is the point.
        "useful_warning_rate": (useful / n_timed) if n_timed else float("nan"),
        "n_useful": useful,
        "n_timed": n_timed,
        "useful_window": (useful_lo, useful_hi),
        "AP": average_precision(scores, labels),
    }
    return summary, used


# ---------------------------------------------------------------------------
# optional: run the pipeline for missing clips (GPU path, no rendering)
# ---------------------------------------------------------------------------

def process_clip(video_path, out_dir, config):
    """Run the full pipeline (forecaster + risk, NO rendering) -> events.json."""
    from run_pipeline import make_forecaster_hook, make_risk_hook
    from a3ps.pipeline import Pipeline

    pipeline = Pipeline(
        config,
        forecaster=make_forecaster_hook(config),
        risk_engine=make_risk_hook(config),
        explainer=None,
    )
    # render=False: skip annotated.mp4 / raw.mp4 -- events.json is all we score.
    return pipeline.run(video_path, out_dir, render=False)


def resolve_video(row, videos_root):
    """Resolve a manifest row's video path relative to the index directory."""
    rel = row["path"] or ""
    return os.path.normpath(os.path.join(videos_root, *rel.split("/")))


# ---------------------------------------------------------------------------
# report + csv
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _outcome(label, first_alert_t, event_time_s):
    """Human-readable per-clip outcome for one method."""
    flagged = first_alert_t is not None
    if label == 1:
        anticipated = flagged and (event_time_s is None or first_alert_t <= event_time_s)
        return "anticipated" if anticipated else ("late" if flagged else "MISS")
    return "FALSE_ALARM" if flagged else "clean"


def _tta(label, first_alert_t, event_time_s):
    """Time-to-accident for a correctly-anticipated positive, else None."""
    if label == 1 and first_alert_t is not None and event_time_s is not None \
            and first_alert_t <= event_time_s:
        return event_time_s - first_alert_t
    return None


def build_report(a3ps, reactive, split, thresholds, matched=None):
    """Markdown report comparing A3PS vs the reactive-proximity baseline.

    ``matched`` (optional, from :func:`matched_subset_metrics`) adds a
    same-subset mTTA comparison -- the only comparison a "+X.X s earlier"
    anticipation-gain claim should be substantiated with, since the raw
    per-method mTTA rows above are computed over each method's own,
    differently-sized true-positive set (confounded by recall/false-alarm
    differences, not just timing).
    """
    lines = [
        f"# Anticipation eval (split: {split})",
        "",
        f"- clips scored: **{a3ps['n_processed']}** "
        f"({a3ps['n_positive']} positive, {a3ps['n_negative']} negative)",
        f"- thresholds (A3PS): base **{thresholds['base']}**, "
        f"alert **{thresholds['alert']}** (base - alert_margin), "
        f"floor **{thresholds['floor']}**",
        f"- reactive baseline: BEV < {REACTIVE_BEV_THRESH_M} m "
        f"(else img < {int(REACTIVE_IMG_FRAC * 100)}% of frame height) to the ego corridor",
        "",
        f"| method | detection rate | false-alarm rate | mTTA (s) | "
        f"useful warning rate ({_fmt(a3ps['useful_window'][0], 1)}-"
        f"{_fmt(a3ps['useful_window'][1], 1)} s) |",
        "|---|---|---|---|---|",
        f"| **A3PS (proactive)** | {_fmt(a3ps['detection_recall'])} "
        f"| {_fmt(a3ps['false_alarm_rate'])} "
        f"| {_fmt(a3ps['mTTA_s'], 2)} (n={a3ps['n_tta']}) "
        f"| {_fmt(a3ps['useful_warning_rate'])} "
        f"({a3ps['n_useful']}/{a3ps['n_timed']}) |",
        f"| Reactive-proximity (baseline) | {_fmt(reactive['detection_recall'])} "
        f"| {_fmt(reactive['false_alarm_rate'])} "
        f"| {_fmt(reactive['mTTA_s'], 2)} (n={reactive['n_tta']}) "
        f"| {_fmt(reactive['useful_warning_rate'])} "
        f"({reactive['n_useful']}/{reactive['n_timed']}) |",
        "",
        "_The mTTA row above is averaged over each method's own true-positive "
        "set (different n, different clips) -- do NOT read the difference "
        "between these two mTTA values as an anticipation-gain claim. See the "
        "matched-subset comparison below for that._",
        "",
        "_**Read the useful-warning-rate column, not mTTA, to judge whether "
        "warnings are actionable.** mTTA rewards warning early without bound, "
        "so a method that alarms on the first frame of every clip maximises it "
        "while telling the driver nothing -- which is exactly what the reactive "
        "baseline does here. The useful warning rate instead counts positives "
        "warned inside a window that is late enough to concern this hazard and "
        "early enough to brake; both misses and unusably-early alarms count "
        "against it._",
        "",
    ]

    if matched is not None:
        gain = matched["gain_s"]
        earlier = "earlier" if gain >= 0 else "later"
        lines += [
            "## Matched-subset mTTA (fair, same-clips comparison)",
            "",
            f"Over the **{matched['n_matched']}** positive clip(s) BOTH methods "
            "correctly anticipate (removes the recall/false-alarm-driven bias "
            "of comparing mTTA across each method's own, differently-sized "
            "true-positive set):",
            "",
            f"- A3PS mTTA: **{_fmt(matched['a3ps_mtta_s'], 2)} s**",
            f"- Reactive-baseline mTTA: **{_fmt(matched['reactive_mtta_s'], 2)} s**",
            f"- **Anticipation gain: A3PS is {_fmt(abs(gain), 2)} s {earlier}** "
            "than the reactive baseline on this matched subset.",
            "",
        ]
    else:
        lines += [
            "## Matched-subset mTTA (fair, same-clips comparison)",
            "",
            "_No positive clip is correctly anticipated by both methods -- "
            "matched-subset mTTA is not computable, and no anticipation-gain "
            "claim can be substantiated from this run._",
            "",
        ]

    lines.append(f"_A3PS AP (peak-prob ranking): {_fmt(a3ps['AP'])}_")
    lines.append("")
    return "\n".join(lines)


def write_per_clip_csv(path, rows):
    """One row per clip: label, event time, and both methods' alert/tta/outcome."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "clip_id", "label", "processed", "event_time_s",
        "a3ps_first_alert_t", "a3ps_tta_s", "a3ps_flagged", "a3ps_outcome",
        "reactive_first_alert_t", "reactive_tta_s", "reactive_flagged", "reactive_outcome",
        "peak_prob",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            lbl = r["label"]
            a_fa, r_fa, te = r["a3ps_first_alert_t"], r["reactive_first_alert_t"], r["event_time_s"]
            a_tta, r_tta = _tta(lbl, a_fa, te), _tta(lbl, r_fa, te)
            w.writerow([
                r["clip_id"],
                "pos" if lbl == 1 else ("neg" if lbl == 0 else ""),
                int(bool(r["processed"])),
                "" if te is None else f"{te:.3f}",
                "" if a_fa is None else f"{a_fa:.3f}",
                "" if a_tta is None else f"{a_tta:.3f}",
                int(a_fa is not None),
                _outcome(lbl, a_fa, te) if r["processed"] else "unprocessed",
                "" if r_fa is None else f"{r_fa:.3f}",
                "" if r_tta is None else f"{r_tta:.3f}",
                int(r_fa is not None),
                _outcome(lbl, r_fa, te) if r["processed"] else "unprocessed",
                f"{r['peak_prob']:.4f}",
            ])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="A3PS vs reactive-proximity anticipation eval.")
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="eval", choices=["dev", "demo", "eval"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--clips-dir", default="eval/anticipation",
                   help="Where processed <clip_id>/events.json live (and go with --run).")
    p.add_argument("--videos-root", default=None,
                   help="Root for resolving manifest paths (default: index dir).")
    p.add_argument("--run", action="store_true",
                   help="Run the pipeline (no rendering) for clips missing events.json.")
    p.add_argument("--alert-types", default=",".join(DEFAULT_ALERT_TYPES),
                   help="Comma-separated event types that count as an alert.")
    p.add_argument("--useful-lo", type=float, default=USEFUL_WINDOW_LO_S,
                   help="Lower bound (s before impact) of the actionable "
                        "warning window used for useful_warning_rate.")
    p.add_argument("--useful-hi", type=float, default=USEFUL_WINDOW_HI_S,
                   help="Upper bound (s before impact) of that window.")
    p.add_argument("--out", default="eval/anticipation.md")
    p.add_argument("--per-clip-csv", default="eval/anticipation_per_clip.csv")
    p.add_argument("--sweep", action="store_true",
                   help="Also cache per-frame max collision_prob and sweep "
                        "base_threshold (0.40-0.90 step 0.05) for a PR curve, "
                        "without re-running the GPU pipeline.")
    p.add_argument("--cache-dir", default="eval/cache",
                   help="Per-clip per-frame max-prob cache (built once, reused across sweeps).")
    p.add_argument("--force-cache", action="store_true",
                   help="Rebuild the per-frame cache even if it already exists for a clip.")
    p.add_argument("--pr-csv", default="eval/pr_curve_data.csv")
    p.add_argument("--pr-png", default="eval/pr_curve.png")
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

    rows = []          # master rows (both methods' alert times per clip)
    n_missing = 0
    for m in manifest:
        ev_path = clip_events_path(args.clips_dir, m["clip_id"])

        if not os.path.isfile(ev_path) and args.run:
            video = resolve_video(m, videos_root)
            if os.path.isfile(video):
                print(f"  running pipeline (no render): {m['clip_id']} ...")
                process_clip(video, os.path.dirname(ev_path), config)
            else:
                print(f"  ! video not found for {m['clip_id']}: {video}")

        row = {**m, "a3ps_first_alert_t": None, "reactive_first_alert_t": None,
               "peak_prob": 0.0, "processed": False}
        if os.path.isfile(ev_path):
            clip = ClipResult.load_json(ev_path)
            frame_h = int(clip.meta.get("height") or 720)
            row["a3ps_first_alert_t"], row["peak_prob"] = summarize_clip(clip, alert_types)
            row["reactive_first_alert_t"] = reactive_first_alert(clip, frame_h)
            row["processed"] = True
            if args.sweep:
                # One-time extraction from the ClipResult already in memory --
                # no re-parse of events.json, no GPU work.
                cache_frame_probs(clip, m["clip_id"], args.cache_dir, force=args.force_cache)
        else:
            n_missing += 1
        rows.append(row)

    # Score each method with the shared compute_metrics (differ only in first_alert_t).
    def _method_rows(key):
        return [{"clip_id": r["clip_id"], "label": r["label"],
                 "event_time_s": r["event_time_s"], "first_alert_t": r[key],
                 "peak_prob": r["peak_prob"], "processed": r["processed"]}
                for r in rows]

    a3ps_summary, _ = compute_metrics(_method_rows("a3ps_first_alert_t"),
                                      args.useful_lo, args.useful_hi)
    reactive_summary, _ = compute_metrics(_method_rows("reactive_first_alert_t"),
                                          args.useful_lo, args.useful_hi)

    if a3ps_summary["n_processed"] == 0:
        print(f"No processed clips found under {args.clips_dir}. "
              f"Re-run with --run (on the GPU laptop) to process them.")
        return
    if n_missing:
        print(f"note: {n_missing}/{len(manifest)} clips not processed "
              f"(scored the {a3ps_summary['n_processed']} available). Use --run to fill in.")

    thresholds = {"base": 0.75, "alert": 0.60, "floor": 0.45}
    if config is not None:
        base = float(config.get("base_threshold", 0.75))
        margin = float(config.get("alert_margin", 0.15))
        thresholds = {"base": base, "alert": round(base - margin, 4),
                      "floor": float(config.get("threshold_floor", 0.45))}

    matched = matched_subset_metrics(rows)

    report = build_report(a3ps_summary, reactive_summary, args.split, thresholds, matched)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    write_per_clip_csv(args.per_clip_csv, rows)

    print()
    print(f"{'':22s} {'detect':>8s} {'false-alarm':>12s} {'mTTA(s)':>9s} {'useful':>8s}")
    print(f"{'A3PS (proactive)':22s} {_fmt(a3ps_summary['detection_recall']):>8s} "
          f"{_fmt(a3ps_summary['false_alarm_rate']):>12s} "
          f"{_fmt(a3ps_summary['mTTA_s'], 2):>9s} "
          f"{_fmt(a3ps_summary['useful_warning_rate']):>8s}")
    print(f"{'reactive baseline':22s} {_fmt(reactive_summary['detection_recall']):>8s} "
          f"{_fmt(reactive_summary['false_alarm_rate']):>12s} "
          f"{_fmt(reactive_summary['mTTA_s'], 2):>9s} "
          f"{_fmt(reactive_summary['useful_warning_rate']):>8s}")
    if matched is not None:
        print(f"\nmatched-subset (n={matched['n_matched']}): "
              f"A3PS {_fmt(matched['a3ps_mtta_s'], 2)}s vs reactive "
              f"{_fmt(matched['reactive_mtta_s'], 2)}s "
              f"(gain {_fmt(matched['gain_s'], 2)}s)")
    else:
        print("\nmatched-subset: no positive anticipated by both methods -- n/a")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.per_clip_csv}")

    if args.sweep:
        sweep = sweep_thresholds(rows, args.cache_dir, THRESHOLD_SWEEP)
        write_pr_csv(args.pr_csv, sweep)
        render_pr_curve(args.pr_png, sweep)
        print(f"\nthreshold sweep ({THRESHOLD_SWEEP[0]}-{THRESHOLD_SWEEP[-1]}, "
              f"step 0.05):")
        for row in sweep:
            print(f"  thr={row['threshold']:.2f}  "
                  f"precision={_fmt(row['precision'])}  recall={_fmt(row['recall'])}  "
                  f"(tp={row['tp']} fp={row['fp']} fn={row['fn']})")
        print(f"\nwrote {args.pr_csv}")
        print(f"wrote {args.pr_png}")


if __name__ == "__main__":
    main()
