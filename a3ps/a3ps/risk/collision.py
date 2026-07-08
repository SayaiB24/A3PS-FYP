"""Collision probability of a Gaussian actor position vs the ego corridor.

Two estimators are provided:

* ``prob_in_axis_box`` — exact analytic probability mass of a *diagonal*
  Gaussian inside an axis-aligned rectangle (product of 1D normal CDFs).
  Easy to hand-compute, used in unit tests.

* ``sigma_point_collision_prob`` — an unscented (sigma-point) estimate that
  works for arbitrary corridor polygons and full covariances.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from a3ps.common.geometry import point_in_polygon
from a3ps.risk.gaussian import GaussianStep

Point = Sequence[float]


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def prob_in_axis_box(
    mean: Tuple[float, float],
    std: Tuple[float, float],
    box: Tuple[float, float, float, float],
) -> float:
    """P(point in axis-aligned box) for a diagonal Gaussian.

    ``box`` is (xmin, ymin, xmax, ymax). Assumes independent x, y.
    """
    mx, my = mean
    sx, sy = std
    xmin, ymin, xmax, ymax = box

    def _axis_prob(m: float, s: float, lo: float, hi: float) -> float:
        if s <= 0.0:
            return 1.0 if lo <= m <= hi else 0.0
        return _phi((hi - m) / s) - _phi((lo - m) / s)

    return _axis_prob(mx, sx, xmin, xmax) * _axis_prob(my, sy, ymin, ymax)


def sigma_points(
    mean: Tuple[float, float],
    cov: List[List[float]],
    kappa: float = 1.0,
) -> List[Tuple[Tuple[float, float], float]]:
    """Return (point, weight) pairs for a 2D unscented transform.

    Produces 2n+1 = 5 sigma points. Weights sum to 1. Uses a Cholesky-free
    closed form for the 2x2 matrix square root of (n+kappa)*cov.
    """
    n = 2
    lam = n + kappa
    # Scaled covariance whose "columns" give the sigma-point offsets.
    a = lam * cov[0][0]
    b = lam * cov[0][1]
    d = lam * cov[1][1]

    # Lower-triangular sqrt of [[a, b], [b, d]] (assumes positive definite).
    l11 = math.sqrt(max(a, 0.0))
    l21 = b / l11 if l11 > 0 else 0.0
    l22 = math.sqrt(max(d - l21 * l21, 0.0))
    cols = [(l11, l21), (0.0, l22)]  # columns of L

    mx, my = mean
    pts: List[Tuple[Tuple[float, float], float]] = [((mx, my), kappa / lam)]
    w = 0.5 / lam
    for (cx, cy) in cols:
        pts.append(((mx + cx, my + cy), w))
        pts.append(((mx - cx, my - cy), w))
    return pts


def sigma_point_collision_prob(
    g: GaussianStep,
    corridor_poly: List[Point],
    kappa: float = 1.0,
) -> float:
    """Estimate P(actor inside corridor) via weighted sigma points."""
    total = 0.0
    for pt, w in sigma_points(g.mean, g.cov, kappa=kappa):
        if point_in_polygon(pt, corridor_poly):
            total += w
    return total


def ttc_seconds(
    distance_m: float,
    closing_speed_mps: float,
) -> float:
    """Time-to-collision. Returns inf if not closing."""
    if closing_speed_mps <= 0.0:
        return float("inf")
    return distance_m / closing_speed_mps
