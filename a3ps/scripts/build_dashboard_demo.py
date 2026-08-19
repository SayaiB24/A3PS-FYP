#!/usr/bin/env python
"""Build dashboard demo data: learned risk head vs the old threshold system.

What this produces
------------------
For each chosen clip, a small ``dashboard/clips/<clip_id>/risk.json`` holding
BOTH systems' per-frame risk curves on the same clip, plus the ground-truth
timing markers, so the dashboard can show the Phase IV improvement instead of
only asserting it:

* ``learned``   -- the trained GRU's per-frame risk probability, and the alert it
  fires at the committed operating point (threshold 0.60, confirm 8).
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

# The committed operating point, from the kappa sweep (eval/kappa_comparison.md
# and eval/operating_point_sweep_k1p0.md). The kappa=1.0 checkpoint at
# threshold 0.60 / confirm 8 is the best row meeting the <= 0.20 false-alarm
# target: FA 0.167, useful 0.750, mean lead 1.67 s.
#
# It replaced the earlier kappa=3.0 model at 0.70/5 (FA 0.167, useful 0.717,
# lead 1.58 s) -- strictly better on useful-warning and lead at identical FA.
DEFAULT_THRESHOLD = 0.60
DEFAULT_CONFIRM = 8

# Second decision stage: VIRTUAL_BRAKE. The learned head emits a probability, so a
# two-stage decision is just a second, higher threshold -- which is what the old
# DecisionEngine did with its SAFE -> ALERT -> BRAKE state machine, and what the
# Phase IV head was missing.
#
# 0.80 is not arbitrary: from eval/operating_point_sweep_k1p0.md at confirm 8 the
# false-alarm rate is 0.167 at threshold 0.60 but **0.017** at 0.80 -- one false
# intervention in 60 negatives. Braking is a far more consequential action than a
# warning, so it should demand roughly an order of magnitude fewer false
# positives. The cost is recall: 0.300 vs 0.750 useful-warning, i.e. the brake
# stage fires on well under half the events the alert stage catches. That is the
# intended asymmetry -- warn readily, intervene rarely.
DEFAULT_BRAKE_THRESHOLD = 0.80
DEFAULT_BRAKE_CONFIRM = 8

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


def learned_explanation(prob, alert_t, event_t, actor, threshold, kind="ALERT"):
    """One-line explanation for a learned-head event, via the shared template.

    Reuses ``a3ps.explain.templates.explain`` so the dashboard's wording matches
    the rest of the system, then appends the timing judgement that only the
    ground truth can supply.
    """
    ev = Event(
        event_id=1, frame_idx=0, t=float(alert_t), type=kind,
        actor_id=int((actor or {}).get("id") or 0),
        actor_cls=str((actor or {}).get("cls") or "vehicle"),
        collision_prob=float(prob), threshold=float(threshold),
        ttc_s=(actor or {}).get("ttc_s"),
    )
    base = template_explain(ev, None, {"flags": []})
    if event_t is None:
        return base
    lead = event_t - alert_t
    verb = "Intervened" if kind == "VIRTUAL_BRAKE" else "Warned"
    return f"{base} {verb} {lead:.2f} s before impact."


def two_stage_decision(probs, t, args):
    """(alert_t, brake_t) from one probability track -- the learned head's decision.

    Reuses :func:`first_alert_time` for both stages, so the alert semantics are
    identical to the ones every metric is computed with; the brake stage is the
    same rule at a higher threshold.

    Brake is clamped to never precede alert. With a fixed confirm count a higher
    threshold is almost always crossed later, but "N consecutive frames above X"
    is not strictly monotonic in X on a spiky curve, and a brake shown before its
    own warning would be nonsense.
    """
    alert_t = first_alert_time(probs, t, args.threshold, args.confirm)
    brake_t = first_alert_time(probs, t, args.brake_threshold, args.brake_confirm)
    if brake_t is not None and alert_t is not None and brake_t < alert_t:
        brake_t = alert_t
    if brake_t is not None and alert_t is None:
        # Can only happen if the alert stage never confirmed; treat the brake as
        # its own warning rather than dropping the more severe event.
        alert_t = brake_t
    return alert_t, brake_t


def risk_level_for(prob, threshold):
    """Map a learned frame-level probability to the dashboard's three risk bands.

    Note this is a **scene-level** judgement applied to every actor in the frame.
    The learned head pools features across actors and emits one probability per
    frame, so it genuinely cannot say *which* actor is dangerous — colouring all
    actors by the frame's risk is the honest visualisation of what the model
    actually outputs. Per-actor attribution needs the per-actor head described in
    docs/handoff/FUTURE_WORK.md.
    """
    if prob >= threshold:
        return "danger"
    if prob >= 0.6 * threshold:
        return "caution"
    return "safe"


def build_slim_overlay(clip_json, learned_curve, args, cid, learned_events=None):
    """A small events.json that still drives masks/trails/predictions/corridor.

    The cached ClipResult is 20-50 MB per clip: it carries BEV fields the image-
    space overlay never reads, 40-point mask polygons, full float precision and
    indent=2. The renderer only needs frame_idx/t, the ego corridor, and per
    track: id, cls, bbox, mask_poly, history_img, prediction.mean_img and
    collision_prob. Keeping exactly those, decimating to ``--overlay-hz``,
    simplifying masks and rounding to integers gets the same picture in a few MB.

    Actor colour (``risk_level``) comes from the LEARNED head's frame-level
    probability, so the overlay reflects the Phase IV model rather than the old
    threshold system. ``collision_prob`` is left as the cached per-actor
    geometric estimate, which is what the Top-threats panel ranks by.
    """
    from a3ps.perception.segmenter import simplify_polygon

    def learned_at(t):
        if not learned_curve:
            return None
        best, bp = 1e9, None
        for lt, lp in learned_curve:
            d = abs(lt - t)
            if d < best:
                best, bp = d, lp
        return bp if best <= 0.5 else None

    period = 1.0 / float(args.overlay_hz) if args.overlay_hz else 0.0
    frames_out, next_t = [], None
    for fr in clip_json["frames"]:
        t = float(fr["t"])
        if period and next_t is not None and t < next_t - 1e-9:
            continue
        next_t = t + period

        p = learned_at(t)
        level = risk_level_for(p, args.threshold) if p is not None else "safe"

        tracks = []
        for tr in fr["tracks"]:
            pred = tr.get("prediction") or {}
            mask = tr.get("mask_poly")
            if mask:
                mask = simplify_polygon(mask, args.overlay_mask_points)
            out = {
                "id": tr["id"], "cls": tr["cls"],
                "bbox": [round(float(v)) for v in tr["bbox"]],
                "risk_level": level,
            }
            if mask is not None and len(mask):
                out["mask_poly"] = [[round(float(x)), round(float(y))] for x, y in mask]
            if tr.get("history_img"):
                out["history_img"] = [[round(float(x)), round(float(y))]
                                      for x, y in tr["history_img"]]
            mean_img = pred.get("mean_img")
            if mean_img or pred.get("collision_prob") is not None:
                out["prediction"] = {}
                if mean_img:
                    out["prediction"]["mean_img"] = [
                        [round(float(x)), round(float(y))] for x, y in mean_img]
                if pred.get("collision_prob") is not None:
                    out["prediction"]["collision_prob"] = round(
                        float(pred["collision_prob"]), 3)
            tracks.append(out)

        ego = fr.get("ego") or {}
        frames_out.append({
            "frame_idx": fr["frame_idx"], "t": round(t, 3), "tracks": tracks,
            "ego": {"corridor_poly_img": ego.get("corridor_poly_img")},
            "context": {"learned_risk": None if p is None else round(p, 4)},
        })

    events_out = []
    for _n, _ev in enumerate(learned_events or (), start=1):
        # Surface the learned head's ALERT and VIRTUAL_BRAKE in the dashboard
        # event log, so the escalation and its explanation are visible there and
        # not only in the compare panel. The brake event is also what drives the
        # red border flash, vignette and banner in app.js.
        events_out.append({
            "event_id": _n,
            "frame_idx": 0,
            "t": _ev["t"],
            "type": _ev["type"],
            "actor_id": 0,
            "actor_cls": _ev.get("actor_cls") or "scene",
            "collision_prob": _ev["prob"],
            "threshold": (args.brake_threshold if _ev["type"] == "VIRTUAL_BRAKE"
                          else args.threshold),
            "ttc_s": _ev.get("ttc_s"),
            "explanation_template": _ev["explanation"],
            "explanation_llm": None,
        })

    meta = dict(clip_json.get("meta") or {})
    meta["overlay_note"] = (
        "Slim overlay built by scripts/build_dashboard_demo.py. Geometry "
        "(masks/trails/predicted paths/corridor) is the shared perception+"
        "tracking output. risk_level is the LEARNED head's frame-level risk "
        "applied scene-wide -- it is not per-actor attribution. "
        "prediction.collision_prob is the cached per-actor geometric estimate."
    )
    meta["overlay_hz"] = args.overlay_hz
    return {"meta": meta, "frames": frames_out, "events": events_out}


def build_clip(cid, row, model, args):
    feat_path = os.path.join(args.features, f"{cid}.npz")
    cached = os.path.join(args.cached_dir, cid, "events.json")
    if not os.path.isfile(feat_path) or not os.path.isfile(cached):
        return None, "missing features or cached output", None

    with np.load(feat_path, allow_pickle=False) as d:
        X = torch.from_numpy(d["X"])
        t = torch.from_numpy(d["t"])
        meta = json.loads(str(d["meta"]))

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).flatten()
    learned_curve = [[round(float(a), 3), round(float(b), 4)]
                     for a, b in zip(t.tolist(), probs.tolist())]
    fa, brake_t = two_stage_decision(probs, t, args)

    event_t = _f(row.get("event_time_s"))
    alert_t = _f(row.get("alert_time_s"))
    label = int(float(row.get("label") or 0))

    old_curve, old_events, clip_json = old_system_curve(cached)

    def at_curve(when, thr, kind):
        """Build one learned-head event at time ``when``."""
        idx = min(range(len(learned_curve)),
                  key=lambda i: abs(learned_curve[i][0] - when))
        actor = dominant_actor_at(clip_json, when)
        ev = {
            "t": round(float(when), 3),
            "type": kind,
            "prob": learned_curve[idx][1],
            "actor_cls": (actor or {}).get("cls"),
            "ttc_s": (actor or {}).get("ttc_s"),
            "explanation": learned_explanation(
                learned_curve[idx][1], when, event_t, actor, thr, kind=kind),
        }
        return ev

    learned_event = at_curve(fa, args.threshold, "ALERT") if fa is not None else None
    learned_brake = (at_curve(brake_t, args.brake_threshold, "VIRTUAL_BRAKE")
                     if brake_t is not None else None)

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
                            "brake_threshold": args.brake_threshold,
                            "brake_confirm": args.brake_confirm,
                            "rate_hz": meta.get("rate_hz")},
        "window": {"start_s": meta.get("window_start_s"),
                   "end_s": meta.get("window_end_s")},
        "learned": {
            "name": "Learned risk head (GRU, Phase IV)",
            "curve": learned_curve,
            "event": learned_event,
            "brake_event": learned_brake,
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
    overlay = (None if args.no_overlay
               else build_slim_overlay(clip_json, learned_curve, args, cid,
                                       [e for e in (learned_event, learned_brake)
                                        if e is not None]))
    return out, None, overlay


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
    p.add_argument("--checkpoint", default="notebooks/models/risk_gru_k1p0.pt")
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
    p.add_argument("--brake-threshold", type=float, default=DEFAULT_BRAKE_THRESHOLD,
                   help="Second-stage VIRTUAL_BRAKE threshold (default "
                        "%(default)s -- false-alarm rate 0.017 vs 0.167 at the "
                        "alert threshold; see the module docstring).")
    p.add_argument("--brake-confirm", type=int, default=DEFAULT_BRAKE_CONFIRM)
    p.add_argument("--overlay-hz", type=float, default=15.0,
                   help="Frame rate for the slim overlay events.json (default "
                        "%(default)s). Lower = smaller file, choppier overlay.")
    p.add_argument("--overlay-mask-points", type=int, default=16,
                   help="Max points per segmentation polygon in the overlay.")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip the slim events.json (risk curves only, no "
                        "masks/trails/predictions).")
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

    built, built_meta = [], []
    for cid in cids:
        row = index.get(cid)
        if row is None:
            print(f"  {cid}: not in index -- skipped")
            continue
        data, err, overlay = build_clip(cid, row, model, args)
        if data is None:
            print(f"  {cid}: {err} -- skipped")
            continue

        dst = os.path.join(args.out_dir, cid)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, "risk.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

        if overlay is not None:
            ov_path = os.path.join(dst, "events.json")
            with open(ov_path, "w", encoding="utf-8") as fh:
                json.dump(overlay, fh, separators=(",", ":"))
            ov_mb = os.path.getsize(ov_path) / 1e6

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
              f"({th['n_events']} events)"
              + (f"  brake @{lr['brake_event']['t']}s" if lr.get("brake_event") else "")
              + (f"  overlay {ov_mb:.1f} MB" if overlay is not None else ""))
        built.append(cid)
        built_meta.append({"id": cid, "label": data["label"],
                           "brake": bool(lr.get("brake_event"))})

    # The manifest is owned by scripts/update_manifest.py, which writes the
    # grouped/labelled form the dropdown needs. Writing a plain id list here
    # would silently downgrade it and lose the grouping, so prompt instead.
    print(f"\nbuilt {len(built)} clip(s): {', '.join(built)}")

    n_pos = sum(1 for c in built_meta if c.get("label") == 1)
    n_brake = sum(1 for c in built_meta if c.get("brake"))
    if n_pos:
        print(f"  virtual brake engaged on {n_brake}/{n_pos} positive(s). The brake "
              f"stage is a higher threshold ({args.brake_threshold}) and fires on a "
              f"minority of events by design -- useful-warning recall is ~0.30 there "
              f"vs ~0.75 at the alert threshold, bought with ~10x fewer false "
              f"interventions. 'not reached' is the expected common case.")

    print("\nNow rebuild the dropdown (this script does not touch the manifest):")
    print("  python scripts/update_manifest.py --require-video")


if __name__ == "__main__":
    main()
