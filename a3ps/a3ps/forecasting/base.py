"""Forecaster interface.

A forecaster maps a track's recent image-space history to a future trajectory
with per-step uncertainty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

Point = List[float]                    # [x, y]
Means = List[Point]                    # future mean points, one per step
Stds = List[Point]                     # per-step [sx, sy]


class Forecaster(ABC):
    """Predicts a future trajectory from a track's recent history."""

    @abstractmethod
    def predict(
        self,
        history: List[Point],
        dt: float,
        horizon_s: float,
    ) -> Tuple[Means, Stds]:
        """Forecast forward from ``history``.

        Args:
            history: recent points ``[[x, y], ...]``, most recent last,
                sampled at interval ``dt``.
            dt: seconds between steps (typically ``1 / predict_hz``).
            horizon_s: how far ahead to forecast.

        Returns:
            ``(means, stds)`` where ``means`` is a list of ``[x, y]`` future
            points at intervals ``dt`` (``round(horizon_s / dt)`` of them) and
            ``stds`` is the matching list of ``[sx, sy]`` per step.
        """
        raise NotImplementedError
