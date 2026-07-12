"""Convert a forecast trajectory into per-step Gaussians (mean, covariance).

Also provides the unscented (sigma-point) sampling used by the collision
estimator. We keep it simple: diagonal covariance, so a 2D Gaussian is fully
described by its mean and per-axis std ``(sx, sy)``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = Sequence[float]

SigmaPoint = Tuple[Tuple[float, float], float]   # ((x, y), weight)

# Number of dimensions for the planar unscented transform.
_N = 2


class GaussianStep:
    """A single 2D Gaussian: mean (x, y) and 2x2 covariance."""

    __slots__ = ("mean", "cov")

    def __init__(self, mean: Tuple[float, float], cov: List[List[float]]):
        self.mean = (float(mean[0]), float(mean[1]))
        self.cov = [[float(cov[0][0]), float(cov[0][1])],
                    [float(cov[1][0]), float(cov[1][1])]]


def gaussians_from_std(
    means: Sequence[Point],
    stds: Sequence[Point],
) -> List[GaussianStep]:
    """Build diagonal-covariance Gaussians from per-step means and stds."""
    if len(means) != len(stds):
        raise ValueError("means and stds must have equal length")
    out: List[GaussianStep] = []
    for (mx, my), (sx, sy) in zip(means, stds):
        cov = [[float(sx) ** 2, 0.0], [0.0, float(sy) ** 2]]
        out.append(GaussianStep((mx, my), cov))
    return out


def sigma_points_2d(
    mean: Tuple[float, float],
    std: Tuple[float, float],
    kappa: float = 0.0,
) -> List[SigmaPoint]:
    """Five UKF sigma points for a diagonal 2D Gaussian.

    Returns ``(point, weight)`` pairs: the mean, then the mean displaced by
    ``sqrt(n + kappa) * sx`` on x and ``sqrt(n + kappa) * sy`` on y (both
    directions). With the default ``kappa=0`` the spread is exactly
    ``sqrt(2)`` standard deviations, the four spread points each carry weight
    ``1 / (2(n + kappa)) = 1/4``, and the mean carries ``kappa / (n + kappa) = 0``.
    Weights always sum to 1.
    """
    mx, my = float(mean[0]), float(mean[1])
    sx, sy = abs(float(std[0])), abs(float(std[1]))
    lam = _N + kappa
    if lam <= 0.0:
        raise ValueError("n + kappa must be positive")
    spread = math.sqrt(lam)
    ox, oy = spread * sx, spread * sy
    w_mean = kappa / lam
    w_spread = 0.5 / lam
    return [
        ((mx, my), w_mean),
        ((mx + ox, my), w_spread),
        ((mx - ox, my), w_spread),
        ((mx, my + oy), w_spread),
        ((mx, my - oy), w_spread),
    ]
