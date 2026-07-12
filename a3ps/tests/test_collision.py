"""Hand-computed validation of the collision decision math (Step 4.1).

Four analytic cases, each with a value worked out by hand so the test doubles
as figure material for the report ("validated decision math against analytic
cases"):

  (a) actor heading straight into the corridor -> prob -> 1, ttc matches the
      geometry within one dt;
  (b) actor moving parallel 5 m to the side     -> prob < 0.1, no ttc;
  (c) broad uncertainty centred outside          -> intermediate prob;
  (d) EMA suppresses a one-frame 0.1->0.9->0.1 spike below the danger line.

The ego corridor is ``ego_corridor(width_m=2, length_m=30)`` -> the rectangle
x in [-1, 1], y in [0, 30]. Sigma points use the default kappa=0, so the four
spread points sit sqrt(2)*std from the mean, each with weight 0.25, and the
mean carries weight 0.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.geometry import ego_corridor  # noqa: E402
from a3ps.risk.collision import (  # noqa: E402
    RiskSmoother,
    step_collision_prob,
    trajectory_collision_prob,
)

CORRIDOR = ego_corridor(width_m=2.0, length_m=30.0)   # x in [-1, 1], y in [0, 30]
DT = 0.2                                              # predict_hz = 5
PERSON_R = 0.4


def test_case_a_head_on_prob_and_ttc():
    """Actor crosses into the corridor -> prob -> 1 and ttc ~ geometry.

    Start at x=2.9, y=10 (outside, right of the corridor), closing on the ego
    path at 2 m/s (0.4 m per 0.2 s step) with tight std. It reaches the true
    corridor edge (x=1) at t = (2.9 - 1.0) / 2.0 = 0.95 s.
    """
    speed = 2.0                      # m/s toward the corridor centreline
    x0, y = 2.9, 10.0
    n_steps = 20                     # 4 s horizon at 5 Hz
    means = [(x0 - speed * DT * (k + 1), y) for k in range(n_steps)]
    stds = [(0.1, 0.1)] * n_steps

    max_prob, ttc = trajectory_collision_prob(
        (means, stds), CORRIDOR, PERSON_R, DT
    )

    assert max_prob > 0.99                       # actor ends up squarely inside
    assert ttc is not None
    t_edge = (x0 - 1.0) / speed                   # analytic corridor-entry time
    assert abs(ttc - t_edge) <= DT + 1e-9         # matches geometry within one dt


def test_case_b_parallel_offset_low_prob():
    """Actor 5 m to the side moving parallel -> negligible prob, no ttc."""
    n_steps = 20
    means = [(5.0, 5.0 + 0.4 * (k + 1)) for k in range(n_steps)]   # x fixed at 5 m
    stds = [(0.3, 0.3)] * n_steps

    max_prob, ttc = trajectory_collision_prob(
        (means, stds), CORRIDOR, PERSON_R, DT
    )

    assert max_prob < 0.1
    assert ttc is None


def test_case_c_broad_uncertainty_intermediate_prob():
    """Mean outside, broad std -> some (but not all) mass reaches the corridor.

    Mean (3, 10) is 2 m right of the dilated corridor edge (x = 1 + 0.4). With
    std 1.2, the -x sigma point lands at 3 - sqrt(2)*1.2 = 1.303, only 0.303 m
    from the edge (< 0.4 radius) -> inside; the mean (weight 0), the +x point,
    and both y points stay outside. Exactly one 0.25-weight point qualifies.
    """
    prob = step_collision_prob((3.0, 10.0), (1.2, 1.2), CORRIDOR, PERSON_R)
    assert 0.0 < prob < 1.0                       # genuinely intermediate
    assert abs(prob - 0.25) < 1e-9                # hand-computed value


def test_case_d_ema_suppresses_single_frame_spike():
    """A lone 0.1 -> 0.9 -> 0.1 spike must not cross the danger line.

    With alpha = 0.4: 0.1 -> 0.42 -> 0.292. The raw 0.9 would fire; the EMA
    peak of 0.42 stays below a 0.5 danger threshold.
    """
    s = RiskSmoother(alpha=0.4)
    e1 = s.update(1, 0.1)
    e2 = s.update(1, 0.9)
    e3 = s.update(1, 0.1)

    assert abs(e1 - 0.1) < 1e-9
    assert abs(e2 - 0.42) < 1e-9
    assert abs(e3 - 0.292) < 1e-9
    assert max(e1, e2, e3) < 0.5                  # spike suppressed below danger
