"""Geometry helpers: homography (image <-> BEV), ego corridor, IoU/overlap.

These are thin, dependency-light helpers. Homography estimation uses OpenCV
when available but the pure-math helpers (IoU, point-in-polygon) do not.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

BBox = Sequence[float]      # [x1, y1, x2, y2]
Point = Sequence[float]     # [x, y]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two axis-aligned boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_centroid(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def estimate_homography(src_pts, dst_pts):
    """Estimate a 3x3 homography mapping src image points to dst (BEV) points.

    Requires at least 4 correspondences. Returns an ``np.ndarray`` (3x3).
    """
    import cv2  # local import so the module imports without OpenCV present
    import numpy as np

    H, _ = cv2.findHomography(
        np.asarray(src_pts, dtype=np.float64),
        np.asarray(dst_pts, dtype=np.float64),
    )
    return H


def apply_homography(H, pt: Point) -> Tuple[float, float]:
    """Project a single image point into BEV coordinates via homography H."""
    import numpy as np

    v = np.asarray([pt[0], pt[1], 1.0], dtype=np.float64)
    w = np.asarray(H, dtype=np.float64) @ v
    return (float(w[0] / w[2]), float(w[1] / w[2]))


def ego_corridor(
    width_m: float = 2.0,
    length_m: float = 30.0,
    forward_offset_m: float = 0.0,
) -> List[Point]:
    """Return a rectangular ego corridor polygon in BEV metres.

    The corridor is centred on x=0, extending forward (+y) from the ego.
    """
    half = width_m / 2.0
    y0 = forward_offset_m
    y1 = forward_offset_m + length_m
    return [[-half, y0], [half, y0], [half, y1], [-half, y1]]


def corridor_lateral_span(corridor: List[Point], y: float) -> Tuple[float, float]:
    """Horizontal span of the ego corridor at image row ``y``, EXTRAPOLATED.

    ``corridor`` is the image-space quad in the order ``ego_corridor``/the
    pipeline emit it: bottom-left, bottom-right, top-right, top-left.

    This is deliberately NOT a polygon containment test. The corridor quad only
    covers the lower part of the frame (down from ``top_y_frac``, ~25 m ahead),
    so a point-in-polygon test answers "is this actor inside the modelled
    corridor", and every more-distant actor -- exactly the one an anticipation
    system wants to reason about early -- falls outside it and reads as "not in
    path". The lane does not end at that row in the world; it keeps converging
    toward the vanishing point.

    So for *lateral alignment* ("is this actor left/right of my lane, at any
    distance") the two side edges are extended as lines and evaluated at any
    row, above the quad's top edge included. Use this for gating a separately
    computed longitudinal signal (e.g. scale-based time-to-contact).

    For "is this predicted point inside the corridor", keep using
    :func:`a3ps.risk.collision.point_in_dilated_polygon` -- that containment
    semantic is what the collision-probability integral wants, and this
    function is not a substitute for it.
    """
    (blx, bly), (brx, bry), (trx, try_), (tlx, tly) = corridor

    def _x_on_edge(x0, y0, x1, y1):
        if abs(y1 - y0) < 1e-9:          # horizontal edge -> no meaningful x(y)
            return x0
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0)

    xl = _x_on_edge(blx, bly, tlx, tly)
    xr = _x_on_edge(brx, bry, trx, try_)
    lo, hi = min(xl, xr), max(xl, xr)
    if hi - lo < 1.0:                    # at/beyond the vanishing point
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5, mid + 0.5
    return lo, hi


def bbox_overlaps_corridor(bbox: BBox, corridor: List[Point], pad_frac: float = 0.0) -> bool:
    """True if ``bbox`` overlaps the corridor horizontally at its base row.

    The base row (``y2``) is used because that is where the actor meets the
    ground plane. ``pad_frac`` widens the span by that fraction of its own
    width on each side, allowing for actors that are drifting toward the lane
    rather than already in it.
    """
    x1, _y1, x2, y2 = bbox
    lo, hi = corridor_lateral_span(corridor, y2)
    pad = (hi - lo) * pad_frac
    return not (x2 < lo - pad or x1 > hi + pad)


# Default ground-plane calibration (image fractions -> ground metres).
# Image trapezoid: bottom-left, bottom-right, top-right, top-left.
# Ground: lane 3.6 m wide (x in [-1.8, 1.8]) at 0 m and 25 m ahead.
DEFAULT_GROUND_PLANE = {
    "image_points_frac": [
        [0.30, 1.00], [0.70, 1.00], [0.55, 0.62], [0.45, 0.62],
    ],
    "ground_points_m": [
        [-1.8, 0.0], [1.8, 0.0], [1.8, 25.0], [-1.8, 25.0],
    ],
}


class GroundPlane:
    """Maps image <-> bird's-eye-view (BEV) ground coordinates via homography.

    Built from 4 image points and their assumed ground (metric) coordinates.
    ``img_to_bev`` projects image pixels to ground metres; ``bev_to_img`` does
    the inverse. Both accept and return a list of ``[x, y]`` points.
    """

    def __init__(self, image_points, ground_points):
        import cv2
        import numpy as np

        self.image_points = [[float(x), float(y)] for x, y in image_points]
        self.ground_points = [[float(x), float(y)] for x, y in ground_points]
        src = np.asarray(self.image_points, dtype=np.float32)
        dst = np.asarray(self.ground_points, dtype=np.float32)
        self.H_img2bev = cv2.getPerspectiveTransform(src, dst)
        self.H_bev2img = cv2.getPerspectiveTransform(dst, src)

    @classmethod
    def from_config(cls, config, width, height, override=None) -> "GroundPlane":
        """Build from config, scaling image fractions by (width, height).

        ``override`` is an optional per-clip dict merged over the config's
        ``ground_plane`` block (see ``load_ground_override``).
        """
        gp = dict(DEFAULT_GROUND_PLANE)
        gp.update((config or {}).get("ground_plane", {}) or {})
        if override:
            gp.update(override)
        image_points = [[fx * width, fy * height]
                        for fx, fy in gp["image_points_frac"]]
        return cls(image_points, gp["ground_points_m"])

    def _transform(self, points, H) -> List[List[float]]:
        import numpy as np

        if not len(points):
            return []
        import cv2

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in out]

    def img_to_bev(self, points) -> List[List[float]]:
        return self._transform(points, self.H_img2bev)

    def bev_to_img(self, points) -> List[List[float]]:
        return self._transform(points, self.H_bev2img)


def load_ground_override(path: Optional[str]):
    """Load an optional per-clip ground-plane override (YAML or JSON).

    Returns a dict (possibly with ``image_points_frac`` / ``ground_points_m``)
    or ``None`` if ``path`` is falsy or missing.
    """
    import os

    if not path or not os.path.isfile(path):
        return None
    import json

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or None
