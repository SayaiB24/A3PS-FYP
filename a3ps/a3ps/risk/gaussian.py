"""Convert a forecast trajectory into per-step Gaussians (mean, covariance)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Point = Sequence[float]


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
