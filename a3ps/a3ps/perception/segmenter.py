"""YOLOv8-Seg wrapper producing per-frame instance segmentations.

``Segmenter(config)`` loads the Ultralytics model named in the config
(``model_name``), restricts inference to the config's ``classes`` (mapped to
the model's class ids), and returns a list of detection dicts per frame.

Device/precision: uses ``cuda`` + half precision when a GPU is available,
otherwise falls back to CPU fp32 (half is not supported on CPU).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

CONFIG_DEFAULTS = {
    "model_name": "yolov8s-seg.pt",
    "conf": 0.35,
    "img_size": 1280,
    "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    "mask_poly_max_points": 40,
}


def simplify_polygon(poly, max_points: int) -> List[List[int]]:
    """Reduce a polygon to <= ``max_points`` points via cv2.approxPolyDP.

    Shared by the segmenter and tracker so mask output stays consistent.
    """
    import cv2
    import numpy as np

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) <= max_points:
        return [[int(x), int(y)] for x, y in pts.reshape(-1, 2)]

    peri = cv2.arcLength(pts, True)
    # Increase epsilon until the point count is within budget.
    eps = 0.001 * peri
    approx = pts
    for _ in range(50):
        approx = cv2.approxPolyDP(pts, eps, True)
        if len(approx) <= max_points:
            break
        eps *= 1.4
    return [[int(x), int(y)] for x, y in approx.reshape(-1, 2)]


class Segmenter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**CONFIG_DEFAULTS, **(config or {})}
        self.model_name = cfg["model_name"]
        self.conf = float(cfg["conf"])
        self.img_size = int(cfg["img_size"])
        self.class_names = list(cfg["classes"])
        self.mask_poly_max_points = int(cfg["mask_poly_max_points"])

        from ultralytics import YOLO

        self.model = YOLO(self.model_name)

        # Device + precision selection.
        import torch

        self.use_half = torch.cuda.is_available()
        self.device = "cuda" if self.use_half else "cpu"
        if self.use_half:
            self.model.to("cuda").half()

        # Map configured class *names* to model class *ids* (COCO ids for the
        # stock yolov8 weights). Unknown names are skipped with a warning.
        names = self.model.names  # {id: name}
        name_to_id = {v: k for k, v in names.items()}
        self.class_ids: List[int] = []
        for n in self.class_names:
            if n in name_to_id:
                self.class_ids.append(name_to_id[n])
            else:
                print(f"[Segmenter] warning: class '{n}' not in model; skipping.")
        self.id_to_name = {i: names[i] for i in self.class_ids}

    def _simplify_polygon(self, poly) -> List[List[int]]:
        """Reduce a polygon to <= mask_poly_max_points via cv2.approxPolyDP."""
        return simplify_polygon(poly, self.mask_poly_max_points)

    def segment(self, frame) -> List[Dict[str, Any]]:
        """Run segmentation on one BGR frame -> list of detection dicts.

        Each dict: {cls: str, conf: float, bbox: [x1,y1,x2,y2] ints,
                    mask_poly: [[x,y], ...] simplified}. Empty list if nothing
        is detected.
        """
        # No `half=True` kwarg here: the model is already converted to half
        # precision once in __init__ (self.model.to("cuda").half()), so
        # passing it again per-call is redundant and triggers Ultralytics'
        # "half is deprecated, use quantize" warning on every frame.
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.img_size,
            classes=self.class_ids or None,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]

        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = getattr(r, "masks", None)
        polys = masks.xy if masks is not None else [None] * len(boxes)

        detections: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            poly = polys[i] if i < len(polys) else None
            mask_poly = self._simplify_polygon(poly) if poly is not None and len(poly) else None
            detections.append({
                "cls": self.model.names.get(cls_id, str(cls_id)),
                "conf": float(boxes.conf[i].item()),
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "mask_poly": mask_poly,
            })
        return detections


# ---------------------------------------------------------------------------
# __main__ demo: run on every 5th frame, write a translucent-mask preview.
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


# Deterministic per-class colors (BGR).
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (255, 194, 0), (255, 56, 132),
]


def _color_for(name: str):
    return _PALETTE[(hash(name) % len(_PALETTE))]


def _demo(video_path: str) -> None:
    import time

    import cv2
    import numpy as np

    config = _load_config()
    seg = Segmenter(config)
    print(f"[demo] model={seg.model_name} device={seg.device} "
          f"half={seg.use_half} classes={seg.class_names}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "perception_preview.mp4"
    # Preview plays back at fps/5 since we keep every 5th frame.
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / 5.0), (width, height)
    )

    times_ms = []
    idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 5 == 0:
            t0 = time.perf_counter()
            dets = seg.segment(frame)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            processed += 1

            overlay = frame.copy()
            for d in dets:
                color = _color_for(d["cls"])
                if d["mask_poly"]:
                    pts = np.asarray(d["mask_poly"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], color)
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Blend filled masks translucently over the frame.
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            for d in dets:
                color = _color_for(d["cls"])
                x1, y1, _, _ = d["bbox"]
                label = f"{d['cls']} {d['conf']:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    print(f"[demo] processed {processed} frames (every 5th of {idx})")
    print(f"[demo] mean inference: {mean_ms:.1f} ms/frame")
    print(f"[demo] wrote {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m a3ps.perception.segmenter <video_path>")
        raise SystemExit(2)
    _demo(sys.argv[1])
