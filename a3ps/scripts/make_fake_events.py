#!/usr/bin/env python
"""Synthesize a plausible events.json (and copy the raw video) for a clip.

Given a real video, this copies it to ``<out>/raw.mp4``, probes its
fps/size/frame-count with OpenCV, and fabricates two tracks + one event so the
dashboard has something to render on Day 1 — before the real pipeline exists.

    python scripts/make_fake_events.py --video clip.mp4 --out dashboard/clips/clip/

Acceptance: the written events.json loads cleanly via ``ClipResult.from_dict``.
"""

import argparse
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import (  # noqa: E402
    ClipResult,
    Event,
    FrameRecord,
    Prediction,
    TrackState,
)
from a3ps.explain.templates import explain as template_explain

PREDICT_HZ = 5.0          # prediction / history sample rate
HORIZON_S = 4.0
HISTORY_S = 2.0
BASE_THRESHOLD = 0.75
CROSSWALK_THRESHOLD = 0.65


def probe_video(path):
    """Return (fps, width, height, frame_count) using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        # Some containers don't report a count; fall back to ~5 s.
        frame_count = int(round(fps * 5))
    return float(fps), width, height, frame_count


def _rect_poly(x1, y1, x2, y2):
    """A bbox as a 4-point polygon (simple rectangular mask stand-in)."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _straight_prediction(cx, cy, vx, vy, n_steps, dt, collision_prob=None, ttc_s=None):
    """Straight-line future path in image space with growing per-step std."""
    mean_img = []
    std_bev = []
    for k in range(1, n_steps + 1):
        tk = k * dt
        mean_img.append([cx + vx * tk, cy + vy * tk])
        s = 0.3 + 0.25 * k          # uncertainty grows with horizon step
        std_bev.append([round(s, 3), round(s, 3)])
    mean_bev = [[round(1.5 + 0.1 * k, 3), round(10.0 - 0.2 * k, 3)]
                for k in range(1, n_steps + 1)]
    return Prediction(
        horizon_s=HORIZON_S,
        dt=dt,
        mean_img=mean_img,
        mean_bev=mean_bev,
        std_bev=std_bev,
        collision_prob=collision_prob,
        ttc_s=ttc_s,
    )


def _person_collision_prob(frac):
    """Ramp 0.1 -> 0.85 across the middle third of the clip.

    ``frac`` is progress through the clip in [0, 1). Flat 0.1 before the middle
    third, linear ramp within it, flat 0.85 after.
    """
    lo, hi = 1.0 / 3.0, 2.0 / 3.0
    if frac <= lo:
        return 0.10
    if frac >= hi:
        return 0.85
    return 0.10 + (0.85 - 0.10) * (frac - lo) / (hi - lo)


def build_clip(video_name, fps, width, height, frame_count):
    dt = 1.0 / PREDICT_HZ
    pred_steps = int(round(HORIZON_S / dt))          # 20 future points
    hist_len = int(round(HISTORY_S * PREDICT_HZ))     # 10 history points
    hist_stride = max(1, int(round(fps / PREDICT_HZ)))

    frames = []
    events = []
    fired = False

    for i in range(frame_count):
        t = i / fps
        frac = i / max(1, frame_count - 1)

        # --- Track 1: car crossing left -> right along the mid-height band ---
        car_cx = 40 + (width - 80) * frac
        car_cy = height * 0.55 + math.sin(frac * math.pi * 2) * height * 0.03
        car_w, car_h = width * 0.12, height * 0.16
        car_bbox = [car_cx - car_w / 2, car_cy - car_h / 2,
                    car_cx + car_w / 2, car_cy + car_h / 2]
        car_hist = [[car_cx - k * hist_stride * (width / max(1, frame_count)),
                     car_cy + math.sin((frac - k * 0.02) * math.pi * 2) * height * 0.03]
                    for k in range(hist_len, 0, -1)]
        car_track = TrackState(
            id=1,
            cls="car",
            bbox=[round(v, 1) for v in car_bbox],
            centroid_img=[round(car_cx, 1), round(car_cy, 1)],
            centroid_bev=[round(2.0 + frac * 8, 2), 12.0],
            velocity_bev=[round((width - 80) / max(1e-6, frame_count / fps) / 5.0, 3), 0.0],
            mask_poly=[[round(x, 1), round(y, 1)] for x, y in _rect_poly(*car_bbox)],
            history_img=[[round(x, 1), round(y, 1)] for x, y in car_hist],
            prediction=_straight_prediction(
                car_cx, car_cy, vx=(width * 0.02), vy=0.0,
                n_steps=pred_steps, dt=dt, collision_prob=0.15, ttc_s=None),
            risk_level="safe",
        )

        # --- Track 2: person walking toward frame bottom-center ---
        person_cx = width * 0.5 + math.sin(frac * math.pi) * width * 0.05
        person_cy = height * 0.35 + (height * 0.55) * frac
        p_w, p_h = width * 0.045, height * 0.20
        person_bbox = [person_cx - p_w / 2, person_cy - p_h / 2,
                       person_cx + p_w / 2, person_cy + p_h / 2]
        person_hist = [[person_cx + math.sin((frac - k * 0.02) * math.pi) * width * 0.05,
                        person_cy - k * (height * 0.55 / max(1, frame_count)) * hist_stride]
                       for k in range(hist_len, 0, -1)]

        prob = round(_person_collision_prob(frac), 3)
        if prob >= BASE_THRESHOLD:
            risk = "danger"
        elif prob >= 0.4:
            risk = "caution"
        else:
            risk = "safe"
        # TTC shrinks as risk climbs; None while clearly safe.
        ttc = round(max(0.3, HORIZON_S * (1.0 - prob)), 2) if prob >= 0.4 else None

        person_track = TrackState(
            id=2,
            cls="person",
            bbox=[round(v, 1) for v in person_bbox],
            centroid_img=[round(person_cx, 1), round(person_cy, 1)],
            centroid_bev=[round(1.8, 2), round(14.2 - frac * 6, 2)],
            velocity_bev=[round(math.cos(frac * math.pi) * 1.1, 3), -0.9],
            mask_poly=[[round(x, 1), round(y, 1)] for x, y in _rect_poly(*person_bbox)],
            history_img=[[round(x, 1), round(y, 1)] for x, y in person_hist],
            prediction=_straight_prediction(
                person_cx, person_cy, vx=0.0, vy=(height * 0.03),
                n_steps=pred_steps, dt=dt, collision_prob=prob, ttc_s=ttc),
            risk_level=risk,
        )

        context = {"flags": ["crosswalk_ahead"], "active_threshold": CROSSWALK_THRESHOLD}
        ego = {
            "corridor_poly_img": [
                [round(width * 0.39, 1), height],
                [round(width * 0.61, 1), height],
                [round(width * 0.52, 1), round(height * 0.60, 1)],
                [round(width * 0.48, 1), round(height * 0.60, 1)],
            ],
            "speed_note": "unknown",
        }
        frames.append(FrameRecord(frame_idx=i, t=round(t, 3),
                                  tracks=[car_track, person_track],
                                  ego=ego, context=context))

        # Fire a single VIRTUAL_BRAKE the first time the person crosses 0.75.
        if not fired and prob >= BASE_THRESHOLD:
            fired = True
            ev = Event(
                event_id=1,
                frame_idx=i,
                t=round(t, 3),
                type="VIRTUAL_BRAKE",
                actor_id=2,
                actor_cls="person",
                collision_prob=prob,
                threshold=CROSSWALK_THRESHOLD,
                ttc_s=ttc,
                explanation_template=None,
            )
            ev.explanation_template = template_explain(ev, context)
            events.append(ev)

    meta = {
        "clip_id": os.path.splitext(video_name)[0],
        "source_video": "raw.mp4",
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "processed_at": None,          # set by real pipeline; fake data leaves null
        "config": {
            "model": "yolov8s-seg",
            "forecaster": "kalman_cv",
            "base_threshold": BASE_THRESHOLD,
            "horizon_s": HORIZON_S,
            "fake": True,
        },
    }
    return ClipResult(meta=meta, frames=frames, events=events)


def main():
    p = argparse.ArgumentParser(description="Synthesize a fake events.json for a clip.")
    p.add_argument("--video", required=True, help="Source video path.")
    p.add_argument("--out", required=True, help="Output clip directory.")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        p.error(f"video not found: {args.video}")

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.mp4")
    shutil.copyfile(args.video, raw_path)

    fps, width, height, frame_count = probe_video(raw_path)
    clip = build_clip(os.path.basename(args.video), fps, width, height, frame_count)

    events_path = os.path.join(args.out, "events.json")
    clip.save_json(events_path)

    # Sidecar meta.json (dashboard fetches this; also supplies source_video
    # so the player falls back to raw.mp4 when annotated.mp4 is absent).
    import json
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(clip.meta, fh, indent=2, ensure_ascii=False)

    # Acceptance self-check: the file must load back through the schema.
    reloaded = ClipResult.load_json(events_path)
    assert isinstance(reloaded, ClipResult)

    print(f"Copied  -> {raw_path}")
    print(f"Probed  -> fps={fps:.3f} size={width}x{height} frames={frame_count}")
    print(f"Wrote   -> {events_path} "
          f"({len(clip.frames)} frames, {len(clip.events)} events)")
    print(f"Wrote   -> {meta_path}")


if __name__ == "__main__":
    main()
