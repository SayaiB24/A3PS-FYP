# -*- coding: utf-8 -*-
"""Exact-string tests for the deterministic explanation templates (Phase 5)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import Event, TrackState  # noqa: E402
from a3ps.explain.templates import explain  # noqa: E402


def _event(etype, actor_id, cls, prob, thr, ttc):
    return Event(event_id=1, frame_idx=0, t=0.0, type=etype, actor_id=actor_id,
                 actor_cls=cls, collision_prob=prob, threshold=thr, ttc_s=ttc)


def _track(cls, vel_bev, cen_bev):
    return TrackState(id=1, cls=cls, bbox=[0, 0, 10, 10], centroid_img=[5.0, 5.0],
                      centroid_bev=cen_bev, velocity_bev=vel_bev)


def test_alert_cutting_in_with_lowered_threshold():
    # Lateral velocity toward the centreline (x=2, vx<0) -> "cutting into".
    ev = _event("ALERT", 4, "person", 0.81, 0.65, 1.2)
    tr = _track("person", vel_bev=[-1.5, -0.2], cen_bev=[2.0, 10.0])
    ctx = {"flags": ["crosswalk_ahead"], "base_threshold": 0.75}
    assert explain(ev, tr, ctx) == (
        "Collision alert: person ID-4 cutting into ego path in 1.2 s "
        "(P=0.81, threshold 0.65 — lowered due to crosswalk ahead)."
    )


def test_virtual_brake_head_on_no_context():
    # Longitudinal velocity toward ego (vy<0) -> "approaching head-on toward".
    ev = _event("VIRTUAL_BRAKE", 4, "person", 0.92, 0.75, 0.8)
    tr = _track("person", vel_bev=[0.1, -3.0], cen_bev=[0.2, 8.0])
    ctx = {"flags": [], "base_threshold": 0.75}
    assert explain(ev, tr, ctx) == (
        "Virtual braking engaged: person ID-4 approaching head-on toward "
        "ego path in 0.8 s (P=0.92, threshold 0.75)."
    )


def test_threshold_lowered_scene_event():
    ev = _event("THRESHOLD_LOWERED", -1, "scene", 0.0, 0.65, None)
    ctx = {"flags": ["crosswalk_ahead"], "base_threshold": 0.75}
    assert explain(ev, None, ctx) == (
        "Entering crosswalk ahead: risk threshold lowered to 0.65, easing speed."
    )


def test_alert_crossing_default_no_lowering():
    # Lateral velocity away from centreline (x=-1, vx<0) -> default "trajectory crossing".
    ev = _event("ALERT", 7, "car", 0.55, 0.75, 2.5)
    tr = _track("car", vel_bev=[-1.2, 0.1], cen_bev=[-1.0, 12.0])
    ctx = {"flags": [], "base_threshold": 0.75}
    assert explain(ev, tr, ctx) == (
        "Collision alert: car ID-7 trajectory crossing ego path in 2.5 s "
        "(P=0.55, threshold 0.75)."
    )


def test_virtual_brake_decelerating_dense_traffic():
    # Near-stationary in BEV (speed < 0.5) -> "decelerating within".
    ev = _event("VIRTUAL_BRAKE", 2, "person", 0.88, 0.70, 1.0)
    tr = _track("person", vel_bev=[0.1, -0.1], cen_bev=[0.5, 5.0])
    ctx = {"flags": ["dense_traffic"], "base_threshold": 0.75}
    assert explain(ev, tr, ctx) == (
        "Virtual braking engaged: person ID-2 decelerating within ego path "
        "in 1.0 s (P=0.88, threshold 0.70 — lowered due to heavy traffic)."
    )


def test_missing_velocity_falls_back_to_default_phrase():
    ev = _event("ALERT", 3, "bicycle", 0.60, 0.75, 3.0)
    tr = _track("bicycle", vel_bev=None, cen_bev=None)
    ctx = {"flags": [], "base_threshold": 0.75}
    assert explain(ev, tr, ctx) == (
        "Collision alert: bicycle ID-3 trajectory crossing ego path in 3.0 s "
        "(P=0.60, threshold 0.75)."
    )
