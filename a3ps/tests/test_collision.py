"""Unit tests for the risk math, with hand-computed cases."""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.geometry import ego_corridor  # noqa: E402
from a3ps.risk.collision import (  # noqa: E402
    prob_in_axis_box,
    sigma_point_collision_prob,
    sigma_points,
    ttc_seconds,
)
from a3ps.risk.gaussian import GaussianStep, gaussians_from_std  # noqa: E402


def test_prob_in_axis_box_symmetric():
    # Diagonal unit Gaussian centred in a [-1,1]^2 box.
    # Each axis: Phi(1) - Phi(-1) = 0.6826894921...
    # Joint (independent axes) = that squared.
    one_axis = 0.6826894921370859
    expected = one_axis * one_axis
    got = prob_in_axis_box((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0, 1.0, 1.0))
    assert abs(got - expected) < 1e-9


def test_prob_in_axis_box_degenerate_std():
    # Zero std -> point mass. Inside box -> 1.0, outside -> 0.0.
    inside = prob_in_axis_box((0.0, 0.0), (0.0, 0.0), (-1.0, -1.0, 1.0, 1.0))
    outside = prob_in_axis_box((5.0, 0.0), (0.0, 0.0), (-1.0, -1.0, 1.0, 1.0))
    assert inside == 1.0
    assert outside == 0.0


def test_prob_in_axis_box_full_mass_large_box():
    # A huge box captures essentially all the probability mass.
    got = prob_in_axis_box((0.0, 0.0), (1.0, 1.0), (-100.0, -100.0, 100.0, 100.0))
    assert abs(got - 1.0) < 1e-9


def test_sigma_points_weights_sum_to_one():
    pts = sigma_points((0.0, 0.0), [[1.0, 0.0], [0.0, 1.0]], kappa=1.0)
    assert len(pts) == 5
    assert abs(sum(w for _, w in pts) - 1.0) < 1e-12


def test_sigma_points_recover_mean():
    # Weighted mean of sigma points equals the distribution mean.
    mean = (3.0, -2.0)
    pts = sigma_points(mean, [[4.0, 0.0], [0.0, 1.0]], kappa=1.0)
    mx = sum(p[0] * w for p, w in pts)
    my = sum(p[1] * w for p, w in pts)
    assert abs(mx - mean[0]) < 1e-9
    assert abs(my - mean[1]) < 1e-9


def test_sigma_point_collision_prob_inside_and_outside():
    corridor = ego_corridor(width_m=2.0, length_m=30.0)
    # Tight Gaussian well inside the corridor -> all sigma points inside.
    g_in = GaussianStep((0.0, 15.0), [[0.01, 0.0], [0.0, 0.01]])
    assert abs(sigma_point_collision_prob(g_in, corridor) - 1.0) < 1e-9
    # Tight Gaussian far to the side -> no sigma points inside.
    g_out = GaussianStep((10.0, 15.0), [[0.01, 0.0], [0.0, 0.01]])
    assert sigma_point_collision_prob(g_out, corridor) == 0.0


def test_gaussians_from_std():
    gs = gaussians_from_std([[0.0, 1.0]], [[2.0, 3.0]])
    assert len(gs) == 1
    assert gs[0].mean == (0.0, 1.0)
    assert gs[0].cov == [[4.0, 0.0], [0.0, 9.0]]


def test_ttc_seconds():
    assert ttc_seconds(20.0, 10.0) == 2.0
    assert ttc_seconds(20.0, 0.0) == float("inf")
    assert ttc_seconds(20.0, -5.0) == float("inf")
