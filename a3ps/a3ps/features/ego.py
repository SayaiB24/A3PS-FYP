"""Ego-motion features from static-scene optical flow.

Two entry points, deliberately separate:

:func:`decompose_affine` / :func:`ego_features_from_affine`
    Pure math on a 2x3 partial-affine warp. BoT-SORT *already* computes exactly
    such a warp on every frame -- ``byte_tracker.py`` calls
    ``warp = self.gmc.apply(img, results_high.xyxy)`` with
    ``gmc_method: sparseOptFlow`` (see ``configs/botsort_a3ps.yaml``) -- and then
    throws it away after using it to compensate track positions. Tapping that
    matrix gives ego motion for free, with no extra inference. These functions
    are the decoder for it, and they are unit-testable without any video.

:class:`EgoFlowEstimator`
    A standalone estimator for when the GMC warp is not available, or when
    strictly static-scene flow is wanted. Ultralytics' ``sparseOptFlow`` branch
    ignores its ``detections`` argument entirely -- ``apply_sparseoptflow`` takes
    only ``raw_frame`` -- so its keypoints include the moving actors and it
    relies on RANSAC to out-vote them. That is usually fine but it is not the
    same thing as background-only flow. This class masks detected actor boxes out
    before selecting keypoints, so the estimate is genuinely of the static scene.

Both produce the same 4-vector so the two sources are interchangeable
downstream, and both are honest about a limitation: without per-clip intrinsics
these are *proxies*, not physical units. The translation term is pixels/frame,
not m/s, and the rotation term is image-plane rotation, not vehicle yaw rate.
Feed them to a learned head, which can calibrate them against the labels; do not
present them as measured ego speed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Names of the 4 ego features, in vector order. Kept as a module constant so the
# feature schema is auditable from one place.
EGO_FEATURE_NAMES = ("ego_tx", "ego_ty", "ego_log_scale", "ego_rot")

EGO_FEATURE_DIM = len(EGO_FEATURE_NAMES)

# An identity warp decodes to an all-zero feature vector, which is also what a
# missing/failed estimate yields -- see ego_features_from_affine.
EGO_ZERO = (0.0, 0.0, 0.0, 0.0)


def decompose_affine(H: Optional[Sequence[Sequence[float]]]) -> Dict[str, float]:
    """Decompose a 2x3 partial affine into translation, uniform scale, rotation.

    ``cv2.estimateAffinePartial2D`` (what BoT-SORT's GMC uses) returns
    ``[[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]`` -- a similarity transform, so
    scale is a single number and rotation a single angle. Returns
    ``{"tx", "ty", "scale", "rot"}`` with ``rot`` in radians.

    A ``None`` or malformed input decodes to the identity (no motion) rather than
    raising: a dropped frame should read as "no evidence", not as an error, so a
    single bad frame cannot abort a 40 s clip.
    """
    ident = {"tx": 0.0, "ty": 0.0, "scale": 1.0, "rot": 0.0}
    if H is None:
        return ident
    arr = np.asarray(H, dtype=np.float64)
    if arr.shape == (3, 3):
        arr = arr[:2, :]          # accept a 3x3 homography's affine part
    if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
        return ident

    a, b = float(arr[0, 0]), float(arr[0, 1])
    c, d = float(arr[1, 0]), float(arr[1, 1])
    tx, ty = float(arr[0, 2]), float(arr[1, 2])

    # Uniform scale from the determinant; fall back to the column norm if the
    # matrix is not quite a similarity (RANSAC output occasionally is not).
    det = a * d - b * c
    scale = math.sqrt(abs(det)) if det != 0.0 else math.hypot(a, c)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    rot = math.atan2(c, a)
    if not math.isfinite(rot):
        rot = 0.0
    return {"tx": tx, "ty": ty, "scale": scale, "rot": rot}


def ego_features_from_affine(H: Optional[Sequence[Sequence[float]]],
                             width: int, height: int) -> List[float]:
    """4-vector of ego features from a 2x3 warp, scale-normalised by frame size.

    Order matches :data:`EGO_FEATURE_NAMES`:

    * ``ego_tx``        -- horizontal translation as a fraction of frame width.
      With a forward-facing dashcam this tracks yaw (panning) more than lateral
      travel, which is why it is not named "yaw": it is a proxy, and calling it
      yaw would imply intrinsics we do not have.
    * ``ego_ty``        -- vertical translation as a fraction of frame height
      (pitch / road undulation).
    * ``ego_log_scale`` -- ``log(scale)``. Positive when the static scene expands
      between frames, which is what forward motion looks like. This is the
      closest thing here to a speed proxy, and it is the term most worth having:
      it is the only one that separates "driving fast" from "stopped" without
      intrinsics.
    * ``ego_rot``       -- image-plane rotation in radians (roll / camera shake).

    ``log`` rather than raw scale so that "no motion" sits at 0.0 alongside the
    other three, and so equal proportional expansion and contraction are equal
    and opposite instead of asymmetric about 1.0.
    """
    d = decompose_affine(H)
    w = float(width) if width else 1.0
    h = float(height) if height else 1.0
    scale = d["scale"] if d["scale"] > 1e-9 else 1e-9
    return [d["tx"] / w, d["ty"] / h, math.log(scale), d["rot"]]


class EgoFlowEstimator:
    """Frame-to-frame static-scene affine from masked sparse optical flow.

    Usage is one call per frame, in order::

        est = EgoFlowEstimator()
        for frame, boxes in clip:
            feats = est.update(frame, boxes)   # 4-vector, EGO_FEATURE_NAMES order

    The first frame has no predecessor and returns :data:`EGO_ZERO`.

    ``boxes`` are actor bboxes ``[x1, y1, x2, y2]`` in full-frame pixels; the
    pixels inside them are excluded from keypoint selection so the estimate
    describes the static scene rather than the traffic. This is the difference
    from Ultralytics' ``sparseOptFlow`` GMC, which never masks (see module
    docstring). ``dilate`` grows each box slightly, because a bbox clips an
    actor's silhouette and the few pixels just outside it still move with the
    actor.

    cv2 is imported lazily inside the methods so that importing this module --
    and the pure-math helpers above -- never requires OpenCV.
    """

    def __init__(self, downscale: int = 2, max_corners: int = 400,
                 quality: float = 0.01, min_distance: int = 8,
                 dilate: float = 0.06, min_points: int = 8):
        self.downscale = max(1, int(downscale))
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.dilate = float(dilate)
        self.min_points = int(min_points)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._size = (0, 0)          # (width, height) at full resolution
        # Diagnostics -- worth logging, because a clip where the estimator keeps
        # failing (night, wipers, a near-static scene) produces all-zero ego
        # features that look identical to "stopped".
        self.n_frames = 0
        self.n_failed = 0

    # -- internals ---------------------------------------------------------

    def _actor_mask(self, gray: np.ndarray, boxes: Optional[Iterable]) -> Optional[np.ndarray]:
        """255 where keypoints may be selected, 0 over (dilated) actor boxes."""
        if not boxes:
            return None
        import cv2  # noqa: F401  (kept for symmetry / future cv2-based masking)

        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        s = self.downscale
        for box in boxes:
            try:
                x1, y1, x2, y2 = (float(v) for v in box[:4])
            except (TypeError, ValueError):
                continue
            bw, bh = (x2 - x1), (y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            px, py = bw * self.dilate, bh * self.dilate
            ix1 = max(0, int((x1 - px) / s))
            iy1 = max(0, int((y1 - py) / s))
            ix2 = min(w, int(math.ceil((x2 + px) / s)))
            iy2 = min(h, int(math.ceil((y2 + py) / s)))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 0
        # Everything masked out (a frame filled by one huge box) -> unusable.
        return mask if int(mask.max()) > 0 else None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        arr = np.asarray(frame)
        gray = (cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                if arr.ndim == 3 and arr.shape[2] == 3 else arr)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (max(1, w // self.downscale),
                                     max(1, h // self.downscale)))
        return gray

    def _detect(self, gray: np.ndarray, mask: Optional[np.ndarray]):
        import cv2

        return cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=self.max_corners,
            qualityLevel=self.quality, minDistance=self.min_distance,
            blockSize=3)

    # -- public API --------------------------------------------------------

    def reset(self) -> None:
        """Forget the previous frame (call between clips)."""
        self._prev_gray = None
        self._prev_pts = None

    def affine(self, frame: np.ndarray,
               boxes: Optional[Iterable] = None) -> Optional[np.ndarray]:
        """Estimate the 2x3 warp from the previous frame to ``frame``.

        Returns None on the first frame, or when too few points survive tracking
        (which is a real outcome on night clips and near-static scenes, not an
        error). ``self.n_failed`` counts those frames.
        """
        import cv2

        arr = np.asarray(frame)
        if arr.ndim >= 2:
            self._size = (arr.shape[1], arr.shape[0])
        gray = self._to_gray(arr)
        mask = self._actor_mask(gray, boxes)
        self.n_frames += 1

        prev_gray, prev_pts = self._prev_gray, self._prev_pts
        # Stage this frame as the next iteration's reference before any early
        # return, so one failed frame does not desynchronise the whole clip.
        self._prev_gray = gray
        self._prev_pts = self._detect(gray, mask)

        if prev_gray is None or prev_pts is None or len(prev_pts) < self.min_points:
            if prev_gray is not None:
                self.n_failed += 1
            return None
        if prev_gray.shape != gray.shape:
            # Resolution changed mid-clip; nothing to compare against.
            self.n_failed += 1
            return None

        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts.astype(np.float32), None)
        if cur_pts is None or status is None:
            self.n_failed += 1
            return None

        keep = status.reshape(-1).astype(bool)
        src = prev_pts.reshape(-1, 2)[keep]
        dst = cur_pts.reshape(-1, 2)[keep]
        if len(src) < self.min_points:
            self.n_failed += 1
            return None

        H, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if H is None or not np.all(np.isfinite(H)):
            self.n_failed += 1
            return None

        H = np.asarray(H, dtype=np.float64).copy()
        if self.downscale > 1:
            # Translation was estimated on the downscaled image; the rotation
            # and scale terms are resolution-independent, the offsets are not.
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
        return H

    def update(self, frame: np.ndarray,
               boxes: Optional[Iterable] = None) -> List[float]:
        """:meth:`affine` followed by :func:`ego_features_from_affine`.

        Returns :data:`EGO_ZERO` as a list when no estimate is available, so the
        caller always gets a fixed-width vector.
        """
        H = self.affine(frame, boxes)
        if H is None:
            return list(EGO_ZERO)
        w, h = self._size
        return ego_features_from_affine(H, w or 1, h or 1)

    def stats(self) -> Dict[str, Any]:
        """Diagnostics: how often the estimate failed over this clip."""
        return {
            "n_frames": self.n_frames,
            "n_failed": self.n_failed,
            "fail_rate": (self.n_failed / self.n_frames) if self.n_frames else 0.0,
        }
