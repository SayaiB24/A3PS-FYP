#!/usr/bin/env python
"""Evaluate anticipation: A3PS vs a naive reactive-proximity baseline.

Runs the full pipeline (no rendering) over every clip in a Nexar split, flags a
clip when any ALERT / VIRTUAL_BRAKE fires, and reports, for BOTH methods:

  * detection rate  -- fraction of positives that alerted BEFORE the annotated
    event time (true anticipation).
  * false-alarm rate -- fraction of negatives that raised any alert.
  * mTTA (s)        -- mean Time-To-Accident over anticipated positives:
    mean(event_time - first_alert_time).
  * useful warning rate -- fraction of positives warned inside the dataset's OWN
    actionable window ``[alert_time_s, event_time_s]``. See USEFUL_SLACK_S: this
    is keyed to Nexar's ``time_of_alert`` annotation, not to a window we chose.
  * official-style Nexar AP -- clip ranking using only the frames available
    500 / 1000 / 1500 ms before the annotated event, plus their mean. See
    :func:`nexar_cutoff_ap`.

Every run first checks the index against ``eval/split_freeze.json`` and refuses
to score a drifted held-out split (``--allow-split-drift`` to override). See
``a3ps/common/splits.py`` for why that guard exists.

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

    python scripts/eval_anticipation.py --index eval/nexar_index_gpu.csv --split eval --run

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

    python scripts/eval_anticipation.py --index eval/nexar_index_gpu.csv --split eval --sweep
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for run_pipeline hooks

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.common.splits import (  # noqa: E402
    DEFAULT_FREEZE_PATH,
    load_freeze,
    verify_freeze,
)
from a3ps.risk.collision import point_in_dilated_polygon  # noqa: E402

DEFAULT_ALERT_TYPES = ("ALERT", "VIRTUAL_BRAKE")

# Reactive-proximity baseline thresholds -- kept identical to the dashboard's
# reactive-ADAS marker (dashboard/app.js: computeReactiveMarkers):
#   BEV: centroid within 2.0 m of the BEV corridor polygon,
#   img: centroid within 8% of frame height of the image corridor polygon.
REACTIVE_BEV_THRESH_M = 2.0
REACTIVE_IMG_FRAC = 0.08

# "Useful warning" window -- now keyed to the dataset's OWN annotation.
#
# Why the metric exists at all: mTTA rewards warning EARLY without bound, so a
# method that alarms in the first frame of every clip scores a huge mTTA and an
# unbeatable anticipation gain while telling the driver nothing. The
# reactive-proximity baseline does exactly that -- on the Nexar eval clips it
# fires within 2 s of clip start on ~76% of positives, roughly 18 s before
# impact. A warning is only actionable in a band: late enough to be about THIS
# hazard, early enough to brake.
#
# Why it changed: the band used to be a pair of constants WE picked (0.5 s to
# 6.0 s before impact). Nexar already annotates the band's lower edge itself --
# `time_of_alert`, the ground-truth earliest actionable moment -- and carries it
# per clip in `alert_time_s`. Measured over the 65 local positives, the real
# lead (`time_of_event - time_of_alert`) is 2.97-4.47 s (mean 3.49, sd 0.40), so
# the invented 6.0 s ceiling was ~2.5 s too generous and the invented 0.5 s
# floor was unrelated to anything in the data.
#
# The window is therefore now PER CLIP:
#
#     useful  <=>  alert_time_s - slack  <=  first_alert_t  <=  event_time_s
#
# Firing before `alert_time_s` is not credit-worthy: by the dataset's own
# annotation there was nothing actionable to see yet, so an earlier alarm is
# either luck or a standing false alarm. Firing after `event_time_s` is too late
# by definition. `slack` (default 0) grants grace for annotation jitter without
# reintroducing a made-up window; set it explicitly if you want that grace, and
# say so in the writeup.
USEFUL_SLACK_S = 0.0

# Official-style Nexar anticipation cutoffs: the clip is scored using only the
# frames available up to N ms BEFORE the annotated event, then ranked by AP.
NEXAR_CUTOFFS_S = (0.5, 1.0, 1.5)


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
            # Nexar's ground-truth "earliest actionable moment". Present for
            # positives, blank for negatives -- see USEFUL_SLACK_S.
            "alert_time_s": _to_float(r.get("alert_time_s")),
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


def frame_prob_timeline(clip_result):
    """Per-frame max collision probability over all tracks -> [(t, max_prob)].

    The single source of truth for "what score did the model hold at time t".
    Used by the per-frame cache (threshold sweep) and by the Nexar cutoff AP, so
    the two can never drift apart in how they reduce a frame to one number.
    """
    out = []
    for fr in clip_result.frames:
        max_p = 0.0
        for tr in fr.tracks:
            p = getattr(tr.prediction, "collision_prob", None) if tr.prediction else None
            if p is not None and p > max_p:
                max_p = float(p)
        out.append((float(fr.t), max_p))
    return out


def score_before(timeline, cutoff_t):
    """Max prob over frames at or before ``cutoff_t`` (None => whole clip).

    This is the clip-level score under a truncated observation window: what the
    model would have output if the video had been cut at ``cutoff_t``. Returns
    0.0 when no frame qualifies (a cutoff earlier than the first frame).
    """
    best = 0.0
    for t, p in timeline:
        if cutoff_t is not None and t > cutoff_t:
            break
        if p > best:
            best = p
    return best


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

    Area under the precision-recall curve, computed over TIE GROUPS: all clips
    sharing a score are consumed together before precision/recall are read off.
    NaN if there are no positives.

    Why tie handling is not a detail here. This pipeline's clip score is a max
    over a saturating probability, so scores pile up on exactly 1.0 -- on the
    120-clip eval split, 101 clips (57 positive, 44 negative) tie at 1.0000 and
    the whole split takes only 7 distinct values. The earlier implementation
    walked a plain descending sort and broke ties by input order. Python's sort
    is stable, so the tied block kept CSV order -- and the index lists all 60
    positives before all 60 negatives, which ranked every tied positive above
    every tied negative. That reported AP 0.978. The identical scores with the
    rows reversed give 0.371, and the mean over 200 random row orders is 0.579.
    The number was measuring the index's sort order, not the model.

    Consuming ties as a group removes the dependence on input order entirely and
    yields the expected value over random tie orderings, which is the only
    defensible reading of a tied ranking.
    """
    n = min(len(scores), len(labels))
    pos = sum(1 for y in labels[:n] if y == 1)
    if pos == 0:
        return float("nan")

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    i = 0
    while i < n:
        tie_score = scores[order[i]]
        j = i
        while j < n and scores[order[j]] == tie_score:
            if labels[order[j]] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def nexar_cutoff_ap(rows, cutoffs=NEXAR_CUTOFFS_S, truncate_negatives=False):
    """Official-style Nexar AP: rank clips using only pre-event observations.

    For each cutoff ``tau`` the clip's score is the max collision probability
    the model held over the frames it would have seen if the video had been cut
    ``tau`` seconds before the annotated event. AP is then computed over all
    scored clips at that cutoff, and the three cutoffs are averaged into
    ``mean_AP`` -- the headline number the Nexar task reports.

    Each row needs: ``label``, ``event_time_s``, ``timeline`` (from
    :func:`frame_prob_timeline`), ``processed``.

    Negatives have no ``event_time_s``, so there is nothing to truncate them at.
    By default they are scored over their FULL clip, which is what the official
    setup does (a negative video is just a video) and which is the conservative
    choice for us: negatives get more frames than positives and therefore more
    opportunity to produce a high max, so any AP reported here is a lower bound
    on what a length-matched comparison would give. Pass
    ``truncate_negatives=True`` to instead cut every negative at
    ``median(event_time_s) - tau`` over the positives, which equalises observed
    duration at the cost of departing from the official protocol; report which
    one you used.
    """
    used = [r for r in rows if r.get("processed") and r.get("timeline")]
    pos_events = [r["event_time_s"] for r in used
                  if r["label"] == 1 and r["event_time_s"] is not None]
    median_event = None
    if pos_events:
        ordered = sorted(pos_events)
        mid = len(ordered) // 2
        median_event = (ordered[mid] if len(ordered) % 2
                        else 0.5 * (ordered[mid - 1] + ordered[mid]))

    per_cutoff = {}
    for tau in cutoffs:
        scores, labels = [], []
        for r in used:
            if r["label"] == 1:
                te = r["event_time_s"]
                if te is None:
                    continue          # a positive with no event time is unscorable
                cutoff = te - tau
            else:
                cutoff = (median_event - tau
                          if (truncate_negatives and median_event is not None)
                          else None)
            scores.append(score_before(r["timeline"], cutoff))
            labels.append(1 if r["label"] == 1 else 0)
        per_cutoff[tau] = {
            "AP": average_precision(scores, labels),
            "n": len(scores),
            "n_positive": sum(labels),
        }

    aps = [v["AP"] for v in per_cutoff.values()
           if not (isinstance(v["AP"], float) and math.isnan(v["AP"]))]
    return {
        "per_cutoff": per_cutoff,
        "mean_AP": (sum(aps) / len(aps)) if aps else float("nan"),
        "truncate_negatives": bool(truncate_negatives),
        "negative_cutoff_s": (median_event if truncate_negatives else None),
    }


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
        for t, max_p in frame_prob_timeline(clip_result):
            w.writerow([f"{t:.3f}", f"{max_p:.4f}"])
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


def compute_metrics(rows, useful_slack=USEFUL_SLACK_S):
    """Aggregate per-clip rows into anticipation metrics.

    Each row: {clip_id, label, event_time_s, alert_time_s, first_alert_t,
    peak_prob, processed}. Only processed rows contribute. Returns (summary,
    detail_rows). Works for either method -- pass rows whose ``first_alert_t`` is
    the A3PS alert time or the reactive-baseline alert time.

    ``useful_warning_rate`` is keyed to the dataset's per-clip ``alert_time_s``
    (see USEFUL_SLACK_S): a warning counts as useful iff it lands in
    ``[alert_time_s - useful_slack, event_time_s]``. Its denominator is every
    positive that carries BOTH times, so a miss, an alarm that arrives after
    impact, and an alarm that fired before anything was there to see all count
    against it equally.
    """
    used = [r for r in rows if r["processed"]]
    pos = [r for r in used if r["label"] == 1]
    neg = [r for r in used if r["label"] == 0]

    ttas = []
    detected = 0
    useful = 0
    early = 0             # fired before alert_time_s: not actionable, not credit
    n_timed = 0           # positives with BOTH event and alert times (useful denominator)
    for r in pos:
        te = r["event_time_s"]
        ta = r.get("alert_time_s")
        fa = r["first_alert_t"]
        anticipated = fa is not None and (te is None or fa <= te)
        r["anticipated"] = anticipated
        scorable = te is not None and ta is not None
        if scorable:
            n_timed += 1
        r["useful"] = False
        r["too_early"] = False
        if anticipated:
            detected += 1
            if te is not None:
                ttas.append(te - fa)
            if scorable:
                if fa < ta - useful_slack:
                    # Warned before the ground-truth earliest actionable moment.
                    r["too_early"] = True
                    early += 1
                else:
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
        # Fraction of ALL scorable positives (not just anticipated ones) warned
        # inside [alert_time_s - slack, event_time_s]: a miss, a too-late alarm
        # and a fired-before-anything-was-there alarm all count against it.
        "useful_warning_rate": (useful / n_timed) if n_timed else float("nan"),
        "n_useful": useful,
        "n_too_early": early,
        "n_timed": n_timed,
        "useful_slack_s": float(useful_slack),
        "useful_window_def": (
            "[alert_time_s - {:.2f}s, event_time_s] per clip "
            "(ground truth, not a chosen window)".format(float(useful_slack))
        ),
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


def _nexar_ap_section(nexar):
    """Markdown block for the official-style pre-event AP cutoffs."""
    if nexar is None:
        return []
    neg_note = (
        "negatives truncated at median(event_time) - cutoff = "
        "{:.2f} s (length-matched, NOT the official protocol)".format(
            nexar["negative_cutoff_s"])
        if nexar["truncate_negatives"] else
        "negatives scored over their full clip (official protocol; gives "
        "negatives more frames than positives, so this AP is a lower bound)"
    )
    lines = [
        "## Official-style Nexar AP at pre-event cutoffs",
        "",
        "Each clip is ranked by the highest collision probability the model "
        "held using ONLY the frames it would have seen had the video been cut "
        "the stated interval before the annotated event. This is the "
        "dataset's own anticipation protocol, independent of our alert "
        "thresholds and decision state machine.",
        "",
        "| cutoff before event | AP | clips scored |",
        "|---|---|---|",
    ]
    for tau in sorted(nexar["per_cutoff"]):
        v = nexar["per_cutoff"][tau]
        lines.append("| {:.0f} ms | {} | {} ({} pos) |".format(
            tau * 1000, _fmt(v["AP"]), v["n"], v["n_positive"]))
    lines += [
        "",
        f"**mean AP over the three cutoffs: {_fmt(nexar['mean_AP'])}**",
        "",
        f"_{neg_note}._",
        "",
    ]
    return lines


def build_report(a3ps, reactive, split, thresholds, matched=None, nexar=None):
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
        "| method | detection rate | false-alarm rate | mTTA (s) | "
        "useful warning rate | fired too early |",
        "|---|---|---|---|---|---|",
        f"| **A3PS (proactive)** | {_fmt(a3ps['detection_recall'])} "
        f"| {_fmt(a3ps['false_alarm_rate'])} "
        f"| {_fmt(a3ps['mTTA_s'], 2)} (n={a3ps['n_tta']}) "
        f"| {_fmt(a3ps['useful_warning_rate'])} "
        f"({a3ps['n_useful']}/{a3ps['n_timed']}) "
        f"| {a3ps['n_too_early']}/{a3ps['n_timed']} |",
        f"| Reactive-proximity (baseline) | {_fmt(reactive['detection_recall'])} "
        f"| {_fmt(reactive['false_alarm_rate'])} "
        f"| {_fmt(reactive['mTTA_s'], 2)} (n={reactive['n_tta']}) "
        f"| {_fmt(reactive['useful_warning_rate'])} "
        f"({reactive['n_useful']}/{reactive['n_timed']}) "
        f"| {reactive['n_too_early']}/{reactive['n_timed']} |",
        "",
        f"_Useful-warning window: **{a3ps['useful_window_def']}**. "
        "`time_of_alert` is Nexar's own annotation of the earliest actionable "
        "moment, so this window is the dataset's, not ours. The measured lead "
        "(`time_of_event - time_of_alert`) is 2.97-4.47 s over the 65 local "
        "positives (mean 3.49, sd 0.40)._",
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
        "warned inside the dataset's own actionable window; misses, too-late "
        "alarms and alarms fired before `time_of_alert` all count against it. "
        "The 'fired too early' column isolates that last failure mode._",
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

    lines += _nexar_ap_section(nexar)

    lines.append(f"_A3PS AP (whole-clip peak-prob ranking): {_fmt(a3ps['AP'])} "
                 "-- uses every frame including post-event ones, so it is NOT "
                 "an anticipation number; compare the cutoff APs above instead._")
    lines.append("")
    return "\n".join(lines)


def write_per_clip_csv(path, rows):
    """One row per clip: label, event time, and both methods' alert/tta/outcome."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "clip_id", "label", "processed", "event_time_s", "alert_time_s",
        "a3ps_first_alert_t", "a3ps_tta_s", "a3ps_flagged", "a3ps_outcome",
        "a3ps_useful", "a3ps_too_early",
        "reactive_first_alert_t", "reactive_tta_s", "reactive_flagged", "reactive_outcome",
        "reactive_useful", "reactive_too_early",
        "peak_prob",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            lbl = r["label"]
            a_fa, r_fa, te = r["a3ps_first_alert_t"], r["reactive_first_alert_t"], r["event_time_s"]
            a_tta, r_tta = _tta(lbl, a_fa, te), _tta(lbl, r_fa, te)
            ta = r.get("alert_time_s")
            w.writerow([
                r["clip_id"],
                "pos" if lbl == 1 else ("neg" if lbl == 0 else ""),
                int(bool(r["processed"])),
                "" if te is None else f"{te:.3f}",
                "" if ta is None else f"{ta:.3f}",
                "" if a_fa is None else f"{a_fa:.3f}",
                "" if a_tta is None else f"{a_tta:.3f}",
                int(a_fa is not None),
                _outcome(lbl, a_fa, te) if r["processed"] else "unprocessed",
                int(bool(r.get("a3ps_useful"))),
                int(bool(r.get("a3ps_too_early"))),
                "" if r_fa is None else f"{r_fa:.3f}",
                "" if r_tta is None else f"{r_tta:.3f}",
                int(r_fa is not None),
                _outcome(lbl, r_fa, te) if r["processed"] else "unprocessed",
                int(bool(r.get("reactive_useful"))),
                int(bool(r.get("reactive_too_early"))),
                f"{r['peak_prob']:.4f}",
            ])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _check_split_freeze(index_path, freeze):
    """Verify the WHOLE index (all splits) against the freeze. -> (ok, problems).

    Reads the index directly rather than reusing ``load_manifest``, because the
    freeze covers dev and eval together and the drift that matters most is a
    clip having moved between them.
    """
    with open(index_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    records = [{"clip_id": r.get("clip_id", ""),
                "label": _to_int(r.get("label")) or 0,
                "split": r.get("split", "") or ""} for r in rows]
    return verify_freeze(records, freeze)


def main():
    p = argparse.ArgumentParser(
        description="A3PS vs reactive-proximity anticipation eval.")
    # Default to the tracked GPU-laptop index, NOT data/nexar/index.csv: the two
    # encode different eval splits that overlap on only 69 of 120 clips, and the
    # cached output under eval/anticipation/ was produced against this one.
    # See eval/README.md.
    p.add_argument("--index", default="eval/nexar_index_gpu.csv")
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
    p.add_argument("--useful-slack", type=float, default=USEFUL_SLACK_S,
                   help="Seconds of grace before the dataset's alert_time_s "
                        "that still count as a useful warning (default "
                        "%(default)s = pure ground truth). Raising this loosens "
                        "the metric -- say so if you do.")
    p.add_argument("--truncate-negatives", action="store_true",
                   help="For the Nexar cutoff AP, also truncate negatives (at "
                        "median positive event time - cutoff) so both classes "
                        "are observed for a comparable duration. Departs from "
                        "the official protocol; report which you used.")
    p.add_argument("--freeze", default=DEFAULT_FREEZE_PATH,
                   help="Split-freeze file to check the index against before "
                        "scoring (default: %(default)s). Pass '' to skip.")
    p.add_argument("--allow-split-drift", action="store_true",
                   help="Score anyway when the index disagrees with the freeze. "
                        "Only for deliberate experiments -- the resulting "
                        "numbers are not comparable to any previous run.")
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

    # Pre-flight: the held-out split must be the one the freeze pinned, or every
    # number below is incomparable to previous runs (see a3ps/common/splits.py).
    freeze = load_freeze(args.freeze) if args.freeze else None
    if freeze is not None:
        ok, problems = _check_split_freeze(args.index, freeze)
        if not ok:
            print(f"SPLIT DRIFT: {len(problems)} disagreement(s) between "
                  f"{args.index} and {args.freeze}:")
            for msg in problems[:10]:
                print(f"  - {msg}")
            if len(problems) > 10:
                print(f"  ... and {len(problems) - 10} more")
            if not args.allow_split_drift:
                print("\nRefusing to score against a drifted split. Run\n"
                      f"  python scripts/freeze_split.py verify --index {args.index}\n"
                      "to see the full report, or pass --allow-split-drift if the "
                      "change is deliberate (the numbers will not be comparable).")
                return
            print("\n--allow-split-drift set: scoring anyway. These numbers are "
                  "NOT comparable to any previous run.\n")
    elif args.freeze:
        print(f"note: no split freeze at {args.freeze} -- split membership is "
              "unverified. Create one with scripts/freeze_split.py write.")

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
               "peak_prob": 0.0, "processed": False, "timeline": None}
        if os.path.isfile(ev_path):
            clip = ClipResult.load_json(ev_path)
            frame_h = int(clip.meta.get("height") or 720)
            row["a3ps_first_alert_t"], row["peak_prob"] = summarize_clip(clip, alert_types)
            row["reactive_first_alert_t"] = reactive_first_alert(clip, frame_h)
            # Kept in memory for the Nexar cutoff AP: extracted once here from
            # the ClipResult already loaded, so the cutoff metric costs no extra
            # parse of the (much larger) events.json.
            row["timeline"] = frame_prob_timeline(clip)
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
                 "event_time_s": r["event_time_s"],
                 "alert_time_s": r["alert_time_s"], "first_alert_t": r[key],
                 "peak_prob": r["peak_prob"], "processed": r["processed"]}
                for r in rows]

    a3ps_rows = _method_rows("a3ps_first_alert_t")
    reactive_rows = _method_rows("reactive_first_alert_t")
    a3ps_summary, a3ps_detail = compute_metrics(a3ps_rows, args.useful_slack)
    reactive_summary, reactive_detail = compute_metrics(reactive_rows, args.useful_slack)

    # Fold the per-clip usefulness verdicts back onto the master rows for the CSV.
    for src, prefix in ((a3ps_detail, "a3ps"), (reactive_detail, "reactive")):
        by_id = {r["clip_id"]: r for r in src}
        for r in rows:
            d = by_id.get(r["clip_id"])
            r[f"{prefix}_useful"] = bool(d.get("useful")) if d else False
            r[f"{prefix}_too_early"] = bool(d.get("too_early")) if d else False

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
    nexar = nexar_cutoff_ap(rows, NEXAR_CUTOFFS_S,
                            truncate_negatives=args.truncate_negatives)

    report = build_report(a3ps_summary, reactive_summary, args.split, thresholds,
                          matched, nexar)
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
    print(f"\nuseful-warning window: {a3ps_summary['useful_window_def']}")
    print(f"  A3PS fired before alert_time_s on "
          f"{a3ps_summary['n_too_early']}/{a3ps_summary['n_timed']} positives; "
          f"reactive on {reactive_summary['n_too_early']}/"
          f"{reactive_summary['n_timed']}")

    print("\nofficial-style Nexar AP (pre-event cutoffs):")
    for tau in sorted(nexar["per_cutoff"]):
        v = nexar["per_cutoff"][tau]
        print(f"  -{tau * 1000:.0f} ms  AP={_fmt(v['AP'])}  "
              f"(n={v['n']}, {v['n_positive']} pos)")
    print(f"  mean AP = {_fmt(nexar['mean_AP'])}"
          f"{'  [negatives truncated]' if nexar['truncate_negatives'] else ''}")

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
