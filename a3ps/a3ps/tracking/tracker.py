"""Ultralytics BoT-SORT tracking wrapper + a per-track TrajectoryBuffer.

``Tracker(config)`` owns a SINGLE YOLO model run in tracking mode
(``model.track(..., persist=True, tracker='botsort.yaml')``). Tracking mode
already returns segmentation masks, so there is no need to also run a separate
``Segmenter`` — that would double the inference cost. Mask simplification is
reused from ``a3ps.perception.segmenter.simplify_polygon``.

Device/precision follows the segmenter: cuda + half when a GPU is available,
otherwise CPU fp32 (half is unsupported on CPU).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from a3ps.common.schema import TrackState
from a3ps.perception.segmenter import simplify_polygon

CONFIG_DEFAULTS = {
    "model_name": "yolov8s-seg.pt",
    "tracker": "botsort.yaml",
    "conf": 0.35,
    "img_size": 1280,
    "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    "mask_poly_max_points": 40,
    "process_fps": 30,
    "predict_hz": 5,
    "history_s": 2.0,
}

STALE_S = 1.0  # drop a track id not seen for this long


class TrajectoryBuffer:
    """Per-track image-space centroid history, decimated to ``predict_hz``.

    Keeps, per track id, the last ``history_s`` seconds of (t, centroid_img)
    samples resampled to ``predict_hz`` (default 5 Hz). Incoming samples are
    binned into ``1/predict_hz``-second buckets; the first sample seen in each
    bucket is stored, so a 30 fps stream is decimated to ~5 Hz. Track ids not
    seen for ``STALE_S`` seconds are pruned.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**CONFIG_DEFAULTS, **(config or {})}
        self.history_s = float(cfg["history_s"])
        self.predict_hz = float(cfg["predict_hz"])
        self.period = 1.0 / self.predict_hz
        # Target sample count over the window (e.g. 2.0 s * 5 Hz = 10).
        self.maxlen = max(1, int(round(self.history_s * self.predict_hz)))
        self.ready_min = self.history_s * self.predict_hz * 0.8

        self._buf: Dict[int, Deque[Tuple[float, List[float]]]] = {}
        self._last_bucket: Dict[int, int] = {}
        self._last_seen: Dict[int, float] = {}

    def _bucket(self, t: float) -> int:
        # +1e-9 so t exactly on a boundary (e.g. 0.2) lands in the new bucket.
        return int(math.floor(t * self.predict_hz + 1e-9))

    def push(self, frame_idx: int, t: float, tracks) -> None:
        """Ingest all tracks observed at time ``t`` (list of TrackState)."""
        for tr in tracks:
            tid = int(tr.id)
            self._last_seen[tid] = t
            bucket = self._bucket(t)
            if tid not in self._last_bucket or bucket != self._last_bucket[tid]:
                if tid not in self._buf:
                    self._buf[tid] = deque(maxlen=self.maxlen)
                c = tr.centroid_img
                self._buf[tid].append((t, [float(c[0]), float(c[1])]))
                self._last_bucket[tid] = bucket
        self._prune(t)

    def _prune(self, now: float) -> None:
        stale = [tid for tid, seen in self._last_seen.items()
                 if now - seen > STALE_S]
        for tid in stale:
            self._buf.pop(tid, None)
            self._last_bucket.pop(tid, None)
            self._last_seen.pop(tid, None)

    def history(self, track_id: int) -> List[List[float]]:
        """Centroid history for a track, most recent last."""
        return [c for (_, c) in self._buf.get(int(track_id), ())]

    def ready(self, track_id: int) -> bool:
        """True once enough history exists to forecast (>= 80% of the window)."""
        return len(self._buf.get(int(track_id), ())) >= self.ready_min

    def drop(self, track_id: int) -> None:
        tid = int(track_id)
        self._buf.pop(tid, None)
        self._last_bucket.pop(tid, None)
        self._last_seen.pop(tid, None)


class Tracker:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**CONFIG_DEFAULTS, **(config or {})}
        self.model_name = cfg["model_name"]
        self.tracker_cfg = cfg["tracker"]
        self.conf = float(cfg["conf"])
        self.img_size = int(cfg["img_size"])
        self.class_names = list(cfg["classes"])
        self.mask_poly_max_points = int(cfg["mask_poly_max_points"])

        from ultralytics import YOLO

        # Exactly ONE model instance for the whole pipeline.
        self.model = YOLO(self.model_name)

        import torch

        self.use_half = torch.cuda.is_available()
        self.device = "cuda" if self.use_half else "cpu"
        if self.use_half:
            self.model.to("cuda").half()

        # Map configured class names -> model class ids.
        names = self.model.names  # {id: name}
        name_to_id = {v: k for k, v in names.items()}
        self.class_ids: List[int] = [name_to_id[n] for n in self.class_names
                                     if n in name_to_id]

        self.buffer = TrajectoryBuffer(cfg)

    @staticmethod
    def _foot_point(bbox) -> List[float]:
        """Bottom-center of the bbox (the object's contact point with ground)."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2.0, float(y2)]

    def update(self, frame, frame_idx: int, t: float) -> List[TrackState]:
        """Track objects in one frame -> list[TrackState].

        Populates id, cls, bbox, centroid_img (foot point), mask_poly, and
        history_img. Leaves centroid_bev / velocity_bev / prediction as None.
        """
        # No `half=True` kwarg here: the model is already converted to half
        # precision once in __init__ (self.model.to("cuda").half()), so
        # passing it again per-call is redundant and triggers Ultralytics'
        # "half is deprecated, use quantize" warning on every frame.
        # NOTE: do NOT pass classes= to model.track(). Filtering detections by
        # class *inside* the tracker call breaks BoT-SORT's association and
        # leaves boxes.id None on nearly every frame (observed on dev01:
        # 10/540 frames got ids with the filter vs 274-327 without it). Track
        # ALL classes so association stays stable, then keep only the
        # configured target classes below.
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        masks = getattr(r, "masks", None)
        polys = masks.xy if masks is not None else [None] * len(boxes)

        out: List[TrackState] = []
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            # Class filter moved out of model.track() (see note above): keep
            # only the configured target classes, now that tracking has already
            # associated ids across the full detection set.
            if self.class_ids and cls_id not in self.class_ids:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            foot = self._foot_point(bbox)

            poly = polys[i] if i < len(polys) else None
            mask_poly = (simplify_polygon(poly, self.mask_poly_max_points)
                         if poly is not None and len(poly) else None)

            out.append(TrackState(
                id=track_id,
                cls=self.model.names.get(cls_id, str(cls_id)),
                bbox=[float(v) for v in bbox],
                centroid_img=[round(foot[0], 1), round(foot[1], 1)],
                centroid_bev=None,
                velocity_bev=None,
                mask_poly=mask_poly,
                history_img=[],
                prediction=None,
                risk_level="safe",
            ))

        # Feed this frame's tracks into the buffer (decimated to predict_hz),
        # then fill each TrackState's history_img from the buffer.
        self.buffer.push(frame_idx, t, out)
        for tr in out:
            tr.history_img = self.buffer.history(tr.id)
        return out


# ---------------------------------------------------------------------------
# __main__ demo: per-ID colored masks + "cls ID" labels -> tracking_preview.mp4
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    import os

    import yaml

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "default.yaml"
    )
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def _color_for_id(track_id: int):
    """Deterministic bright BGR color keyed by track id."""
    palette = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (255, 194, 0), (255, 56, 132),
        (207, 210, 49), (72, 249, 10), (255, 148, 0), (200, 149, 255),
    ]
    return palette[track_id % len(palette)]


def _demo(video_path: str) -> None:
    import time

    import cv2
    import numpy as np

    tracker = Tracker(_load_config())
    print(f"[demo] model={tracker.model_name} tracker={tracker.tracker_cfg} "
          f"device={tracker.device} half={tracker.use_half}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "tracking_preview.mp4"
    # Every 5th frame kept -> play back at fps/5. persist=True still tracks
    # continuity across the skipped frames because we only feed kept frames.
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / 5.0), (width, height)
    )

    times_ms = []
    idx = 0
    processed = 0
    seen_ids = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 5 == 0:
            t0 = time.perf_counter()
            tracks = tracker.update(frame, idx, idx / fps)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            processed += 1

            overlay = frame.copy()
            for tr in tracks:
                seen_ids.add(tr.id)
                color = _color_for_id(tr.id)
                if tr.mask_poly:
                    pts = np.asarray(tr.mask_poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], color)
                x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            for tr in tracks:
                color = _color_for_id(tr.id)
                x1, y1, _, _ = [int(v) for v in tr.bbox]
                label = f"{tr.cls} {tr.id}"
                cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    print(f"[demo] processed {processed} frames (every 5th of {idx})")
    print(f"[demo] unique track ids seen: {len(seen_ids)}")
    print(f"[demo] mean inference: {mean_ms:.1f} ms/frame")
    print(f"[demo] wrote {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m a3ps.tracking.tracker <video_path>")
        raise SystemExit(2)
    _demo(sys.argv[1])
