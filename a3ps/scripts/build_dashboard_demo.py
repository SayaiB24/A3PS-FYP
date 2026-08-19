#!/usr/bin/env python
"""Build dashboard demo data: learned risk head vs the old threshold system.

What this produces
------------------
For each chosen clip, a small ``dashboard/clips/<clip_id>/risk.json`` holding
BOTH systems' per-frame risk curves on the same clip, plus the ground-truth
timing markers, so the dashboard can show the Phase IV improvement instead of
only asserting it:

* ``learned``   -- the trained GRU's per-frame risk probability, and the alert it
  fires at the committed operating point (threshold 0.70, confirm 5).
* ``threshold`` -- the OLD system's per-frame max collision probability and every
  ALERT/VIRTUAL_BRAKE it fired, read from the cached ``eval/anticipation``
  output. This is a real previous run, not a re-simulation.
* ``markers``   -- Nexar's own ``time_of_alert`` (earliest actionable moment) and
  ``time_of_event`` (impact), which is what makes an alert judgeable as early,
  useful, or late.

**No GPU is required.** Both curves come from artefacts already on disk: the
learned curve from the extracted ``.npz`` features, the old curve from cached
``events.json``. Perception is never re-run. The clip's video is copied as-is.

    python scripts/build_dashboard_demo.py --clips auto --n 6
    python scripts/build_dashboard_demo.py --clips 237,311,1284

Clips must have all three of: extracted features, cached old-system output, and
a local video file. ``--clips auto`` picks that intersection, preferring
positives whose old system fired absurdly early (the clearest contrast) plus a
couple of negatives (where the old system's false alarms show up).
"""

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a3ps.common.schema import Event  # noqa: E402
from a3ps.explain.templates import explain as template_explain  # noqa: E402
from a3ps.risk.anticipation_loss import first_alert_time  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402

# The committed operating point (see eval/operating_point_sweep.md). 0.70/5 is
# the best row meeting the false-alarm target of <= 0.20: FA 0.167, useful 0.717,
# mean lead 1.58 s. 0.60/8 is the documented alternative (better detection and
# lead, FA 0.217 -- slightly over target).
DEFAULT_THRESHOLD = 0.70
DEFAULT_CONFIRM = 5

ALERT_TYPES = ("ALERT", "VIRTUAL_BRAKE")


def load_index(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {str(int(r["clip_id"])): r for r in csv.DictReader(fh)}


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def resolve_video(cid, row, videos_root):
    """Find a clip's video across the two laptops' differing layouts.

    ``eval/nexar_index_full.csv`` was built on the GPU laptop, so its ``path``
    column points at that machine's dataset tree
    (``../../../../Nexar-Dataset/train/01924.mp4``) and will not resolve on the
    CPU laptop, where the same clips live in ``data/nexar/videos/00237.mp4``.
    Try the index's own path first, then the local flat layout with the id
    zero-padded to the usual 5 digits. Returns None when nothing exists.
    """
    candidates = []
    if row.get("path"):
        candidates.append(os.path.join(videos_root, *row["path"].split("/")))
    for width in (5, 0):
        name = f"{int(cid):0{width}d}.mp4" if width else f"{cid}.mp4"
        candidates.append(os.path.join(videos_root, "videos", name))
        candidates.append(os.path.join("data", "nexar", "videos", name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def old_system_curve(events_path):
    """Per-frame max collision prob + fired events from a cached ClipResult."""
    with open(events_path, encoding="utf-8") as fh:
        j = json.load(fh)
    curve = []
    for fr in j["frames"]:
        best = 0.0
        for tr in fr["tracks"]:
            pred = tr.get("prediction") or {}
            p = pred.get("collision_prob")
            if p is not None and p > best:
                best = float(p)
        curve.append([round(float(fr["t"]), 3), round(best, 4)])
    events = [
        {"t": round(float(e["t"]), 3), "type": e["type"],
         "prob": e.get("collision_prob"), "actor_cls": e.get("actor_cls"),
         "explanation": e.get("explanation_template")}
        for e in j["events"] if e["type"] in ALERT_TYPES
    ]
    # Dominant actor class at each frame, for the learned model's explanation.
    return curve, events, j


def dominant_actor_at(clip_json, t):
    """The track with the highest collision prob near time ``t`` (or None).

    Used only to name an actor in the learned model's explanation text. The
    learned head itself consumes pooled features and does not attribute risk to
    a specific track, so this is presentation, not a model output -- and the
    explanation wording says so.
    """
    best, best_dt = None, 1e9
    for fr in clip_json["frames"]:
        dt = abs(float(fr["t"]) - t)
        if dt > 0.5 or dt > best_dt:
            continue
        for tr in fr["tracks"]:
            pred = tr.get("prediction") or {}
            p = pred.get("collision_prob") or 0.0
            if best is None or p > best.get("_p", -1):
                best = {"cls": tr.get("cls"), "id": tr.get("id"),
                        "ttc_s": pred.get("ttc_s"), "_p": p}
                best_dt = dt
    return best


def learned_explanation(prob, alert_t, event_t, actor, threshold):
    """One-line explanation for the learned head's alert, via the shared template.

    Reuses ``a3ps.explain.templates.explain`` so the dashboard's wording matches
    the rest of the system, then appends the timing judgement that only the
    ground truth can supply.
    """
    ev = Event(
        event_id=1, frame_idx=0, t=float(alert_t), type="ALERT",
        actor_id=int((actor or {}).get("id") or 0),
        actor_cls=str((actor or {}).get("cls") or "vehicle"),
        collision_prob=float(prob), threshold=float(threshold),
        ttc_s=(actor or {}).get("ttc_s"),
    )
    base = template_explain(ev, None, {"flags": []})
    if event_t is None:
        return base
    lead = event_t - alert_t
    return f"{base} Warned {lead:.2f} s before impact."


def build_clip(cid, row, model, args):
    feat_path = os.path.join(args.features, f"{cid}.npz")
    cached = os.path.join(args.cached_dir, cid, "events.json")
    if not os.path.isfile(feat_path) or not os.path.isfile(cached):
        return None, "missing features or cached output"

    with np.load(feat_path, allow_pickle=False) as d:
        X = torch.from_numpy(d["X"])
        t = torch.from_numpy(d["t"])
        meta = json.loads(str(d["meta"]))

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).flatten()
    learned_curve = [[round(float(a), 3), round(float(b), 4)]
                     for a, b in zip(t.tolist(), probs.tolist())]
    fa = first_alert_time(probs, t, args.threshold, args.confirm)

    event_t = _f(row.get("event_time_s"))
    alert_t = _f(row.get("alert_time_s"))
    label = int(float(row.get("label") or 0))

    old_curve, old_events, clip_json = old_system_curve(cached)

    learned_event = None
    if fa is not None:
        idx = min(range(len(learned_curve)), key=lambda i: abs(learned_curve[i][0] - fa))
        actor = dominant_actor_at(clip_json, fa)
        learned_event = {
            "t": round(float(fa), 3),
            "type": "ALERT",
            "prob": learned_curve[idx][1],
            "actor_cls": (actor or {}).get("cls"),
            "explanation": learned_explanation(
                learned_curve[idx][1], fa, event_t, actor, args.threshold),
        }

    def verdict():
        if label == 0:
            return "false alarm" if fa is not None else "clean"
        if fa is None:
            return "miss"
        if event_t is not None and fa > event_t:
            return "too late"
        if alert_t is not None and fa < alert_t:
            return "too early"
        return "useful"

    def old_verdict():
        ts = [e["t"] for e in old_events]
        if label == 0:
            return "false alarm" if ts else "clean"
        if not ts:
            return "miss"
        first = min(ts)
        if event_t is not None and first > event_t:
            return "too late"
        if alert_t is not None and first < alert_t:
            return "too early"
        return "useful"

    out = {
        "clip_id": cid,
        "label": label,
        "duration_s": _f(row.get("duration_s")),
        "fps": _f(row.get("fps")),
        "markers": {"time_of_alert": alert_t, "time_of_event": event_t},
        "operating_point": {"threshold": args.threshold, "confirm": args.confirm,
                            "rate_hz": meta.get("rate_hz")},
        "window": {"start_s": meta.get("window_start_s"),
                   "end_s": meta.get("window_end_s")},
        "learned": {
            "name": "Learned risk head (GRU, Phase IV)",
            "curve": learned_curve,
            "event": learned_event,
            "verdict": verdict(),
            "checkpoint": os.path.basename(args.checkpoint),
        },
        "threshold_system": {
            "name": "Threshold risk system (pre-Phase IV)",
            "curve": old_curve,
            "events": old_events,
            "n_events": len(old_events),
            "first_alert_t": min([e["t"] for e in old_events], default=None),
            "verdict": old_verdict(),
        },
    }
    return out, None


def pick_auto(index, args, n):
    """Clips having features + cached output + a local video, most contrastive first."""
    feats = {os.path.splitext(f)[0] for f in os.listdir(args.features)
             if f.endswith(".npz")}
    cached = {d for d in os.listdir(args.cached_dir)
              if os.path.isdir(os.path.join(args.cached_dir, d))}
    usable = []
    for cid in feats & cached:
        row = index.get(cid)
        if row is None:
            continue
        if resolve_video(cid, row, args.videos_root) is None:
            continue
        usable.append(cid)

    def prematurity(cid):
        """How absurdly early the OLD system fired -- the clearest contrast."""
        try:
            _, ev, _ = old_system_curve(os.path.join(args.cached_dir, cid, "events.json"))
        except Exception:                                    # noqa: BLE001
            return -1.0
        te = _f(index[cid].get("event_time_s"))
        if not ev or te is None:
            return -1.0
        return te - min(e["t"] for e in ev)

    pos = sorted([c for c in usable if index[c]["label"] == "1"],
                 key=prematurity, reverse=True)
    neg = sorted([c for c in usable if index[c]["label"] == "0"], key=int)
    n_neg = max(1, n // 3)
    return pos[:n - n_neg] + neg[:n_neg]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="notebooks/models/risk_gru_v1.pt")
    p.add_argument("--features", default="data/features/eval")
    p.add_argument("--cached-dir", default="eval/anticipation",
                   help="Cached OLD-system output (<clip>/events.json).")
    p.add_argument("--index", default="eval/nexar_index_full.csv")
    p.add_argument("--videos-root", default="data/nexar")
    p.add_argument("--out-dir", default="dashboard/clips")
    p.add_argument("--clips", default="auto",
                   help="Comma-separated clip ids, or 'auto'.")
    p.add_argument("--n", type=int, default=6, help="How many clips when auto.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--confirm", type=int, default=DEFAULT_CONFIRM)
    p.add_argument("--no-video", action="store_true",
                   help="Skip copying the clip video (curves only).")
    args = p.parse_args()

    index = load_index(args.index)
    model, extra = RiskGRU.load(args.checkpoint)
    print(f"checkpoint {args.checkpoint} (best epoch {extra.get('best_epoch')}, "
          f"trained kappa={extra.get('kappa')} "
          f"pre_alert_weight={extra.get('pre_alert_weight')})")
    print(f"operating point: threshold {args.threshold} confirm {args.confirm}\n")

    cids = (pick_auto(index, args, args.n) if args.clips == "auto"
            else [str(int(c)) for c in args.clips.split(",") if c.strip()])

    built = []
    for cid in cids:
        row = index.get(cid)
        if row is None:
            print(f"  {cid}: not in index -- skipped")
            continue
        data, err = build_clip(cid, row, model, args)
        if data is None:
            print(f"  {cid}: {err} -- skipped")
            continue

        dst = os.path.join(args.out_dir, cid)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, "risk.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

        if not args.no_video:
            src = resolve_video(cid, row, args.videos_root)
            out_vid = os.path.join(dst, "raw.mp4")
            if src and not os.path.isfile(out_vid):
                shutil.copyfile(src, out_vid)
            elif src is None:
                print(f"     ! no video found for {cid}; curves only")

        lr, th = data["learned"], data["threshold_system"]
        print(f"  {cid} ({'pos' if data['label'] else 'neg'}): "
              f"learned {lr['verdict']:11s} "
              f"@{(lr['event'] or {}).get('t', float('nan')):>6} s  |  "
              f"old {th['verdict']:11s} @{th['first_alert_t']} s "
              f"({th['n_events']} events)")
        built.append(cid)

    # Merge into the manifest, newest demo clips first, without dropping others.
    man_path = os.path.join(args.out_dir, "manifest.json")
    existing = []
    if os.path.isfile(man_path):
        with open(man_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    merged = built + [c for c in existing if c not in built]
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh)

    print(f"\nbuilt {len(built)} clip(s); manifest now lists {len(merged)}")
    print(f"wrote {man_path}")


if __name__ == "__main__":
    main()
