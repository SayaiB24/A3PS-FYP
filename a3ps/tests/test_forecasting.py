"""Unit tests for the Kalman constant-velocity forecaster."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.forecasting.kalman_cv import KalmanCVForecaster  # noqa: E402

DT = 0.2            # 5 Hz
HORIZON = 4.0       # -> 20 steps


def test_constant_velocity_predicted_within_2px():
    # True motion: 4 px/step in x, 2 px/step in y, sampled every dt.
    vx, vy = 4.0, 2.0
    history = [[i * vx, i * vy] for i in range(12)]  # 12 exact CV points
    fc = KalmanCVForecaster()
    means, stds = fc.predict(history, DT, HORIZON)

    n = int(round(HORIZON / DT))
    assert len(means) == n and len(stds) == n

    last_i = len(history) - 1
    for k, (mx, my) in enumerate(means, start=1):
        true_x = (last_i + k) * vx
        true_y = (last_i + k) * vy
        assert abs(mx - true_x) < 2.0, f"x off at step {k}: {mx} vs {true_x}"
        assert abs(my - true_y) < 2.0, f"y off at step {k}: {my} vs {true_y}"


def test_stds_monotonically_non_decreasing():
    history = [[i * 3.0, i * 1.0] for i in range(12)]
    fc = KalmanCVForecaster()
    _, stds = fc.predict(history, DT, HORIZON)

    for axis in (0, 1):
        seq = [s[axis] for s in stds]
        for a, b in zip(seq, seq[1:]):
            assert b >= a - 1e-9, f"std decreased on axis {axis}: {a} -> {b}"
    # Sanity: it should actually grow, not stay flat.
    assert stds[-1][0] > stds[0][0]


def test_two_point_history_does_not_crash():
    fc = KalmanCVForecaster()
    means, stds = fc.predict([[0.0, 0.0], [5.0, 3.0]], DT, HORIZON)

    n = int(round(HORIZON / DT))
    assert len(means) == n and len(stds) == n
    # Straight-line extrapolation from the two points (5 px/step in x).
    assert abs(means[0][0] - 10.0) < 1e-6
    assert abs(means[1][0] - 15.0) < 1e-6
    # Large fixed fallback std.
    assert stds[0][0] >= 30.0


def test_empty_history_returns_empty():
    fc = KalmanCVForecaster()
    means, stds = fc.predict([], DT, HORIZON)
    assert means == [] and stds == []
