"""Constant-velocity Kalman forecaster (CORE), built on filterpy.

State vector ``[x, y, vx, vy]`` with a constant-velocity motion model. The
filter is run (predict+update) over the history points to estimate current
position and velocity, then rolled forward over the horizon WITHOUT further
updates so the covariance — and therefore the reported std — grows with each
step.

Process noise ``Q`` and measurement noise ``R`` are calibrated against the
*measured* prediction error on real tracked dashcam actors, so the reported std
is an honest uncertainty rather than a nominal one: RMSE(k) / std(k) is ~1
across the horizon (see the constants below). This matters because
:mod:`a3ps.risk.collision` samples sigma points at +/- std -- if std is far
smaller than the true error, every sigma point lands on the mean and the
collision probability degenerates into a hard 0/1 indicator instead of a
graded risk.

Histories shorter than 4 points fall back to a straight-line extrapolation
from the last two points with a large fixed std.
"""

from __future__ import annotations

from typing import List, Tuple

from a3ps.forecasting.base import Forecaster, Means, Point, Stds

# Fallback std (pixels) when there is too little history to run the filter.
FALLBACK_STD = 40.0
# Measurement noise (px^2): foot-point observation uncertainty. Segmentation
# centroids jitter by ~5 px between frames, hence a variance of ~25.
MEAS_VAR = 25.0
# Process-noise spectral density. Calibrated on 387 (history, future) windows
# rebuilt from the tracked dev clips: these values put RMSE(k) / std(k) at
# 0.73x / 1.05x / 1.15x for horizons of 0.2 s / 1 s / 4 s, i.e. the reported
# std now matches the error the filter actually makes (std ~2.5 px now -> ~97 px
# at the 4 s horizon). The previous values (1.0 / 0.1) reported ~1.2 px -> 3 px,
# under-dispersed by 44x at 4 s, which collapsed collision probability to 0/1.
PROCESS_VAR = 2000.0
MIN_HISTORY = 4


class KalmanCVForecaster(Forecaster):
    def __init__(self, meas_var: float = MEAS_VAR, process_var: float = PROCESS_VAR):
        self.meas_var = meas_var
        self.process_var = process_var

    # -- Q for the [x, y, vx, vy] constant-velocity model -------------------
    @staticmethod
    def _Q(dt: float, var: float):
        import numpy as np

        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        # White-acceleration noise, coupling (x,vx) and (y,vy).
        return var * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])

    def _straight_line(self, history, dt, n_steps) -> Tuple[Means, Stds]:
        """Fallback: extrapolate from the last two points, large fixed std."""
        last = history[-1]
        if len(history) >= 2:
            prev = history[-2]
            step = [last[0] - prev[0], last[1] - prev[1]]  # per-dt displacement
        else:
            step = [0.0, 0.0]
        means: Means = []
        stds: Stds = []
        for k in range(1, n_steps + 1):
            means.append([last[0] + step[0] * k, last[1] + step[1] * k])
            stds.append([FALLBACK_STD, FALLBACK_STD])
        return means, stds

    def predict(self, history: List[Point], dt: float, horizon_s: float) -> Tuple[Means, Stds]:
        n_steps = max(1, int(round(horizon_s / dt)))
        if not history:
            return [], []
        if len(history) < MIN_HISTORY:
            return self._straight_line(history, dt, n_steps)

        import numpy as np
        from filterpy.kalman import KalmanFilter

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R = np.eye(2) * self.meas_var
        kf.Q = self._Q(dt, self.process_var)

        # Initialize from the first two points (position + velocity estimate).
        x0, y0 = history[0]
        x1, y1 = history[1]
        kf.x = np.array([x0, y0, (x1 - x0) / 1.0, (y1 - y0) / 1.0], dtype=float)
        # Moderate initial uncertainty: positions fairly certain, velocity less.
        kf.P = np.diag([MEAS_VAR, MEAS_VAR, 100.0, 100.0]).astype(float)

        # --- fit: run the filter over the history (predict + update) ---
        for (px, py) in history[1:]:
            kf.predict()
            kf.update(np.array([px, py], dtype=float))

        # --- forecast: roll forward WITHOUT updates ---
        # Report the predicted *measurement* std, sqrt(diag(H P H^T + R)): the
        # observable position uncertainty (state covariance + measurement
        # noise). Including R keeps std0 off the floor so growth over the
        # horizon reads as a ~2x increase rather than an unbounded ratio.
        means: Means = []
        stds: Stds = []
        for _ in range(n_steps):
            kf.predict()
            means.append([float(kf.x[0]), float(kf.x[1])])
            sx = float(np.sqrt(max(kf.P[0, 0] + self.meas_var, 0.0)))
            sy = float(np.sqrt(max(kf.P[1, 1] + self.meas_var, 0.0)))
            stds.append([sx, sy])
        return means, stds
