#!/usr/bin/env python
"""Build a RICH dummy clip so the full dashboard (console + timeline + banner)
can be developed before the real risk engine (Phase 4) exists.

Produces a mock-matching scenario keyed entirely off the real schema, so the
dashboard flips to live data with no changes — just point manifest.json at a
real clip.

Scenario (over the clip's length):
  * person ID-4  — walks toward the ego path; collision_prob ramps 0.10 -> 0.85,
    crossing 0.65 (ALERT) then 0.78 (VIRTUAL_BRAKE). safe -> caution -> danger.
  * car ID-12    — steady ~0.34 (caution).
  * car ID-7     — steady ~0.12 (safe).
Events carry TTC, threshold (0.65, crosswalk), template + LLM narrative.

    python scripts/make_demo_clip.py --video data/dev_clips/02134.mp4 \
        --out dashboard/clips/fake_demo/

Acceptance: the written events.json loads via ClipResult.from_dict.
"""

import argparse
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import (  # noqa: E402
    ClipResult, Event, FrameRecord, Prediction, TrackState,
)
from a3ps.explain.templates import explain as template_explain

PREDICT_HZ = 5.0
DT = 1.0 / PREDICT_HZ
HORIZON_S = 4.0
PRED_STEPS = int(round(HORIZON_S / DT))    # 20
HIST_PTS = 10                              # 2 s @ 5 Hz
THRESHOLD = 0.65                           # crosswalk-lowered danger threshold
ALERT_AT = 0.65
BRAKE_AT = 0.78


def _lerp(a, b, u):
    return a + (b - a) * max(0.0, min(1.0, u))


def _risk(prob):
    if prob >= THRESHOLD:
        return "danger"
    if prob >= 0.30:
        return "caution"
    return "safe"


# --- per-track position (image px) and probability as functions of frac -----

def car7_pos(W, H, frac):
    return [0.86 * W + 6 * math.sin(frac * 6), 0.42 * H]


def car12_pos(W, H, frac):
    return [0.70 * W - 0.06 * W * frac, 0.55 * H + 0.04 * H * frac]


def person4_pos(W, H, frac):
    u = (frac - 0.30) / 0.70                       # local progress once present
    return [_lerp(0.30 * W, 0.50 * W, u), _lerp(0.45 * H, 0.92 * H, u)]


def person4_prob(frac):
    if frac < 0.40:
        return 0.10
    if frac > 0.72:
        return 0.85
    return _lerp(0.10, 0.85, (frac - 0.40) / (0.72 - 0.40))


def car12_prob(frac):
    return min(0.34, 0.15 + 0.4 * frac)


def _bbox_from_center(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _rect_poly(b):
    x1, y1, x2, y2 = b
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _prediction(pos_fn, W, H, frac, fps, prob, ttc):
    cx, cy = pos_fn(W, H, frac)
    dfrac = 1.0 / max(1, fps)                       # ~one frame back
    px, py = pos_fn(W, H, max(0.0, frac - dfrac))
    vx, vy = (cx - px) / DT, (cy - py) / DT         # px per second
    mean_img = [[cx + vx * DT * k, cy + vy * DT * k] for k in range(1, PRED_STEPS + 1)]
    std = [[round(1.0 + 0.25 * k, 2), round(1.0 + 0.25 * k, 2)]
           for k in range(1, PRED_STEPS + 1)]
    return Prediction(horizon_s=HORIZON_S, dt=DT, mean_img=mean_img,
                      std_bev=std, collision_prob=round(prob, 3),
                      ttc_s=(round(ttc, 2) if ttc is not None else None))


def _history(pos_fn, W, H, frac, fps):
    """Last HIST_PTS foot points at 5 Hz (oldest first, most recent last)."""
    pts = []
    for k in range(HIST_PTS - 1, -1, -1):
        fk = max(0.0, frac - k * DT)
        pts.append([round(v, 1) for v in pos_fn(W, H, fk)])
    return pts


def build_clip(video_name, fps, W, H, frame_count, ego):
    frames, events = [], []
    fired_alert = fired_brake = False

    for i in range(frame_count):
        t = i / fps
        frac = i / max(1, frame_count - 1)
        tracks = []

        # car ID-7 (low)
        c7 = car7_pos(W, H, frac)
        b7 = _bbox_from_center(c7[0], c7[1], 0.05 * W, 0.06 * H)
        tracks.append(TrackState(
            id=7, cls="car", bbox=[round(v, 1) for v in b7],
            centroid_img=[round(c7[0], 1), round(b7[3], 1)],
            mask_poly=[[round(x, 1), round(y, 1)] for x, y in _rect_poly(b7)],
            history_img=_history(car7_pos, W, H, frac, fps),
            prediction=_prediction(car7_pos, W, H, frac, fps, 0.12, None),
            risk_level=_risk(0.12)))

        # car ID-12 (caution)
        p12 = car12_prob(frac)
        c12 = car12_pos(W, H, frac)
        b12 = _bbox_from_center(c12[0], c12[1], 0.12 * W, 0.15 * H)
        tracks.append(TrackState(
            id=12, cls="car", bbox=[round(v, 1) for v in b12],
            centroid_img=[round(c12[0], 1), round(b12[3], 1)],
            mask_poly=[[round(x, 1), round(y, 1)] for x, y in _rect_poly(b12)],
            history_img=_history(car12_pos, W, H, frac, fps),
            prediction=_prediction(car12_pos, W, H, frac, fps, p12,
                                   round(4.0 * (1 - p12), 2)),
            risk_level=_risk(p12)))

        # person ID-4 (danger, only present after frac 0.30)
        if frac >= 0.30:
            p4 = person4_prob(frac)
            c4 = person4_pos(W, H, frac)
            b4 = _bbox_from_center(c4[0], c4[1], 0.05 * W, 0.20 * H)
            ttc4 = round(max(0.3, 4.0 * (1 - p4)), 2)
            tracks.append(TrackState(
                id=4, cls="person", bbox=[round(v, 1) for v in b4],
                centroid_img=[round(c4[0], 1), round(b4[3], 1)],
                mask_poly=[[round(x, 1), round(y, 1)] for x, y in _rect_poly(b4)],
                history_img=_history(person4_pos, W, H, frac, fps),
                prediction=_prediction(person4_pos, W, H, frac, fps, p4, ttc4),
                risk_level=_risk(p4)))

            if not fired_alert and p4 >= ALERT_AT:
                fired_alert = True
                ev = Event(event_id=len(events) + 1, frame_idx=i, t=round(t, 3),
                           type="ALERT", actor_id=4, actor_cls="person",
                           collision_prob=round(p4, 3), threshold=THRESHOLD,
                           ttc_s=ttc4)
                ev.explanation_template = template_explain(
                    ev, {"flags": ["crosswalk_ahead"]})
                events.append(ev)
            if not fired_brake and p4 >= BRAKE_AT:
                fired_brake = True
                ev = Event(event_id=len(events) + 1, frame_idx=i, t=round(t, 3),
                           type="VIRTUAL_BRAKE", actor_id=4, actor_cls="person",
                           collision_prob=round(p4, 3), threshold=THRESHOLD,
                           ttc_s=ttc4)
                ev.explanation_template = template_explain(
                    ev, {"flags": ["crosswalk_ahead"]})
                ev.explanation_llm = (
                    f"Virtual braking engaged: pedestrian ID-4 is crossing the "
                    f"ego path from the left and will intersect it in ~{ttc4:.1f} s. "
                    f"Predicted collision probability {p4:.2f}, above the "
                    f"{THRESHOLD:.2f} crosswalk threshold. Recommend immediate brake.")
                events.append(ev)

        frames.append(FrameRecord(frame_idx=i, t=round(t, 3), tracks=tracks,
                                  ego=ego,
                                  context={"flags": ["crosswalk_ahead"],
                                           "active_threshold": THRESHOLD}))

    meta = {
        "clip_id": os.path.splitext(video_name)[0],
        "source_video": "raw.mp4", "fps": round(fps, 3),
        "width": W, "height": H, "n_frames": frame_count,
        "config": {"model": "yolov8s-seg", "forecaster": "kalman_cv",
                   "base_threshold": 0.75, "horizon_s": HORIZON_S, "demo": True},
    }
    return ClipResult(meta=meta, frames=frames, events=events)


def main():
    p = argparse.ArgumentParser(description="Build a rich dummy demo clip.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not os.path.isfile(args.video):
        p.error(f"video not found: {args.video}")

    import cv2
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw.mp4")
    shutil.copyfile(args.video, raw)

    cap = cv2.VideoCapture(raw)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(fps * 20)
    cap.release()

    ego = {"corridor_poly_img": [[0.39 * W, 1.0 * H], [0.61 * W, 1.0 * H],
                                 [0.52 * W, 0.60 * H], [0.48 * W, 0.60 * H]],
           "speed_note": "unknown"}
    ego["corridor_poly_img"] = [[round(x, 1), round(y, 1)]
                                for x, y in ego["corridor_poly_img"]]

    clip = build_clip(os.path.basename(args.video), fps, W, H, n, ego)
    clip.save_json(os.path.join(args.out, "events.json"))
    import json
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(clip.meta, fh, indent=2)

    assert isinstance(ClipResult.load_json(os.path.join(args.out, "events.json")),
                      ClipResult)
    print(f"wrote {n} frames, {len(clip.events)} events -> {args.out}")
    print(f"events: {[(e.type, e.t, e.collision_prob) for e in clip.events]}")


if __name__ == "__main__":
    main()
