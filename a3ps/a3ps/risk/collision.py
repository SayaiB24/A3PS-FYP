"""Collision probability of a Gaussian actor position vs the ego corridor.

The estimator is unscented: each per-step forecast Gaussian is sampled at five
sigma points (see :func:`a3ps.risk.gaussian.sigma_points_2d`) and the collision
probability is the weighted fraction of those points that fall inside the ego
corridor *dilated* by the actor's physical radius. Dilation is the Minkowski
sum of the corridor polygon with a disk of radius ``actor_radius``; a point is
inside it iff its signed distance to the polygon is ``<= actor_radius`` (inside
counts as negative distance). This is exactly the criterion
``cv2.pointPolygonTest(poly, pt, True) >= -actor_radius`` but implemented in
pure Python so this math module stays dependency-light and deterministic.

Aggregation over a trajectory yields a single ``(max_prob, ttc_s)`` pair, and
:class:`RiskSmoother` applies an exponential moving average across *frames* so a
single-frame prediction glitch cannot fire an intervention.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from a3ps.common.geometry import point_in_polygon
from a3ps.risk.gaussian import sigma_points_2d

Point = Sequence[float]

# Default actor "collision radii". BEV values are physical metres; the pixel
# values are rough stand-ins used while forecasting in image space (no ground
# plane yet -- see forecast_space="img" in the config).
DEFAULT_ACTOR_RADIUS_M = {
    "person": 0.4, "bicycle": 0.5, "motorcycle": 0.6,
    "car": 1.0, "bus": 1.5, "truck": 1.5,
}
DEFAULT_ACTOR_RADIUS_PX = {
    "person": 8.0, "bicycle": 10.0, "motorcycle": 12.0,
    "car": 20.0, "bus": 28.0, "truck": 28.0,
}


def actor_radius_for(cls: str, config: Optional[dict] = None, space: str = "bev") -> float:
    """Resolve an actor's collision radius by class from config.

    ``space`` is ``"bev"`` (metres) or ``"img"`` (pixels). Config keys
    ``actor_radius_m`` / ``actor_radius_px`` override the built-in defaults;
    unknown classes fall back to a car-sized radius.
    """
    if str(space).lower() == "img":
        key, defaults, fallback = "actor_radius_px", DEFAULT_ACTOR_RADIUS_PX, 20.0
    else:
        key, defaults, fallback = "actor_radius_m", DEFAULT_ACTOR_RADIUS_M, 1.0
    table = (config or {}).get(key) or {}
    if cls in table:
        return float(table[cls])
    if cls in defaults:
        return float(defaults[cls])
    return float(fallback)


# ---------------------------------------------------------------------------
# polygon dilation (signed-distance membership)
# ---------------------------------------------------------------------------

def _dist_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Euclidean distance from point (px, py) to segment a-b."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:                       # degenerate segment -> a == b
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def _dist_to_polygon_boundary(pt: Point, poly: List[Point]) -> float:
    """Distance from a point to the nearest edge of a closed polygon."""
    px, py = pt
    n = len(poly)
    best = float("inf")
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def point_in_dilated_polygon(pt: Point, poly: List[Point], radius: float) -> bool:
    """True if ``pt`` lies within ``radius`` of ``poly`` (Minkowski-disk dilation).

    Equivalent to a signed distance ``<= radius`` (inside is negative), i.e.
    ``cv2.pointPolygonTest(poly, pt, True) >= -radius``.
    """
    if point_in_polygon(pt, poly):
        return True
    if radius <= 0.0:
        return False
    return _dist_to_polygon_boundary(pt, poly) <= radius


# ---------------------------------------------------------------------------
# per-step and per-trajectory collision probability
# ---------------------------------------------------------------------------

def step_collision_prob(
    mean: Tuple[float, float],
    std: Tuple[float, float],
    corridor_poly: List[Point],
    actor_radius: float,
    kappa: float = 0.0,
) -> float:
    """Weighted fraction of sigma points inside the dilated corridor.

    ``mean`` / ``std`` describe one diagonal Gaussian forecast step;
    ``corridor_poly`` is the ego corridor and ``actor_radius`` dilates it by
    the actor's physical size. Returns a probability in ``[0, 1]``.
    """
    total = 0.0
    for pt, w in sigma_points_2d(mean, std, kappa=kappa):
        if point_in_dilated_polygon(pt, corridor_poly, actor_radius):
            total += w
    return total


def _moving_average(xs: Sequence[float], window: int = 3) -> List[float]:
    """Centred moving average; window is clipped at the sequence edges."""
    n = len(xs)
    if n == 0:
        return []
    half = window // 2
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = xs[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def _as_means_stds(prediction) -> Tuple[List[Point], List[Point]]:
    """Accept either a ``(means, stds)`` pair or a schema ``Prediction``.

    For a ``Prediction`` the BEV mean is preferred, falling back to the
    image-space mean; ``std_bev`` supplies the per-step std in either space.
    """
    if isinstance(prediction, tuple) and len(prediction) == 2:
        means, stds = prediction
    else:
        means = getattr(prediction, "mean_bev", None) or getattr(prediction, "mean_img", None)
        stds = getattr(prediction, "std_bev", None)
    if means is None or stds is None:
        raise TypeError("prediction must be a (means, stds) pair or a Prediction")
    return list(means), list(stds)


def trajectory_collision_prob(
    prediction,
    corridor_poly: List[Point],
    actor_radius: float,
    dt: float,
    kappa: float = 0.0,
    smooth_window: int = 3,
) -> Tuple[float, Optional[float]]:
    """Aggregate a forecast trajectory into ``(max_prob, ttc_s)``.

    Evaluates :func:`step_collision_prob` at every future step, smooths the
    per-step probabilities with a ``smooth_window``-step moving average, and
    returns:

    * ``max_prob`` -- the maximum smoothed probability over the horizon.
    * ``ttc_s``    -- the time of the first step whose smoothed probability
      exceeds 0.5; else the time of the step with the highest probability;
      else ``None`` when no step carries any collision mass.

    ``prediction`` is a ``(means, stds)`` pair (or a schema ``Prediction``);
    step ``k`` (0-indexed) is at time ``(k + 1) * dt``.
    """
    means, stds = _as_means_stds(prediction)
    n = min(len(means), len(stds))
    if n == 0:
        return 0.0, None

    probs = [
        step_collision_prob(means[k], stds[k], corridor_poly, actor_radius, kappa=kappa)
        for k in range(n)
    ]
    smoothed = _moving_average(probs, window=smooth_window)
    max_prob = max(smoothed)

    def _time(step_idx: int) -> float:
        return (step_idx + 1) * dt

    for k, p in enumerate(smoothed):
        if p > 0.5:
            return max_prob, _time(k)

    if max_prob <= 0.0:
        return max_prob, None
    argmax = max(range(n), key=lambda k: smoothed[k])
    return max_prob, _time(argmax)


# ---------------------------------------------------------------------------
# cross-frame smoothing
# ---------------------------------------------------------------------------

class RiskSmoother:
    """Per-track exponential moving average of the max collision probability.

    Blending each frame's ``max_prob`` with the running estimate means a single
    spurious spike (e.g. a one-frame forecast glitch) is damped below the
    danger threshold instead of firing an event.
    """

    def __init__(self, alpha: float = 0.4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._ema: Dict[int, float] = {}

    def update(self, track_id: int, prob: float) -> float:
        """Fold a new per-frame probability into the track's EMA; return it."""
        prev = self._ema.get(track_id)
        if prev is None:
            val = float(prob)
        else:
            val = self.alpha * float(prob) + (1.0 - self.alpha) * prev
        self._ema[track_id] = val
        return val

    def get(self, track_id: int) -> Optional[float]:
        """Current smoothed probability for a track, or ``None`` if unseen."""
        return self._ema.get(track_id)

    def reset(self, track_id: Optional[int] = None) -> None:
        """Forget one track's history, or all tracks when ``track_id`` is None."""
        if track_id is None:
            self._ema.clear()
        else:
            self._ema.pop(track_id, None)
