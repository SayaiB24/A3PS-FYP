"""Unit tests for the context-aware DecisionEngine state machine (Step 4.2)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import Prediction, TrackState  # noqa: E402
from a3ps.risk.decision import (  # noqa: E402
    DecisionEngine,
    active_context_flags,
    load_context_spans,
)

CONFIG = {"base_threshold": 0.75, "caution_threshold": 0.4}


def _track(track_id, prob, cls="person"):
    """A minimal track carrying a prediction with a given collision prob."""
    pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=[[0.0, 0.0]],
                      collision_prob=prob, ttc_s=1.0)
    return TrackState(id=track_id, cls=cls, bbox=[0, 0, 10, 10],
                      centroid_img=[5.0, 5.0], prediction=pred)


def _feed(engine, track_id, prob, n, t0=0.0, dt=0.2, flags=None):
    """Feed n frames of a constant prob for one actor; return all events."""
    events = []
    for k in range(n):
        t = t0 + k * dt
        tr = _track(track_id, prob)
        events.extend(engine.update(k, t, [tr], context_flags=flags or []))
    return events


# ---------------------------------------------------------------------------
# thresholds and context flags
# ---------------------------------------------------------------------------

def test_active_threshold_reductions_and_floor():
    eng = DecisionEngine(CONFIG)
    assert eng.active_threshold([]) == 0.75
    assert abs(eng.active_threshold(["crosswalk_ahead"]) - 0.65) < 1e-9
    assert abs(eng.active_threshold(["dense_traffic"]) - 0.70) < 1e-9
    # crosswalk (0.10) + intersection (0.10) + dense (0.05) = 0.25 -> 0.50,
    # but everything is floored at 0.45.
    assert eng.active_threshold(
        ["crosswalk_ahead", "intersection", "dense_traffic"]) == 0.50
    # An even bigger reduction would clamp at the floor.
    eng2 = DecisionEngine({"base_threshold": 0.6})
    assert eng2.active_threshold(["crosswalk_ahead", "intersection"]) == 0.45


def test_dynamic_threshold_false_ignores_context_flags():
    # Ablation switch: with dynamic_threshold off, active_threshold is flat
    # (always base_threshold) regardless of which context flags are active.
    eng = DecisionEngine({**CONFIG, "dynamic_threshold": False})
    assert eng.active_threshold([]) == 0.75
    assert eng.active_threshold(["crosswalk_ahead"]) == 0.75
    assert eng.active_threshold(
        ["crosswalk_ahead", "intersection", "dense_traffic"]) == 0.75


def test_active_context_flags_spans_and_dense_traffic():
    spans = [{"t0": 3.0, "t1": 9.0, "flag": "crosswalk_ahead"}]
    assert active_context_flags(1.0, 2, spans) == []
    assert active_context_flags(5.0, 2, spans) == ["crosswalk_ahead"]
    assert active_context_flags(9.0, 2, spans) == []          # half-open [t0, t1)
    assert active_context_flags(5.0, 8, spans) == ["crosswalk_ahead", "dense_traffic"]
    assert active_context_flags(1.0, 8, []) == ["dense_traffic"]


# ---------------------------------------------------------------------------
# per-actor state machine
# ---------------------------------------------------------------------------

def test_single_near_miss_emits_one_alert_then_one_brake():
    eng = DecisionEngine(CONFIG)
    # prob 0.80 >= threshold 0.75 -> both streaks build; after 3 frames ALERT,
    # after 3 more BRAKE, then nothing else no matter how long it stays high.
    events = _feed(eng, track_id=1, prob=0.80, n=20)
    types = [e.type for e in events]
    assert types.count("ALERT") == 1
    assert types.count("VIRTUAL_BRAKE") == 1
    assert types == ["ALERT", "VIRTUAL_BRAKE"]
    alert = next(e for e in events if e.type == "ALERT")
    brake = next(e for e in events if e.type == "VIRTUAL_BRAKE")
    assert alert.actor_id == 1 and brake.actor_id == 1
    assert brake.frame_idx > alert.frame_idx           # brake escalates after alert
    assert brake.threshold == 0.75


def test_alert_only_when_prob_between_alert_and_danger():
    eng = DecisionEngine(CONFIG)
    # 0.65 is >= alert_level (0.75 - 0.15 = 0.60) but < danger (0.75).
    events = _feed(eng, track_id=1, prob=0.65, n=20)
    types = [e.type for e in events]
    assert types == ["ALERT"]                          # never escalates to brake


def test_no_events_below_alert_level():
    eng = DecisionEngine(CONFIG)
    events = _feed(eng, track_id=1, prob=0.50, n=20)   # < alert_level 0.60
    assert events == []


def test_requires_three_consecutive_frames():
    eng = DecisionEngine(CONFIG)
    # Two high frames then a low one repeatedly -> streak never reaches 3.
    events = []
    for k in range(30):
        prob = 0.80 if (k % 3 != 2) else 0.0
        events.extend(eng.update(k, k * 0.2, [_track(1, prob)]))
    assert events == []


def test_context_lowers_threshold_and_triggers_brake_sooner():
    eng = DecisionEngine(CONFIG)
    spans = [{"t0": 0.0, "t1": 100.0, "flag": "crosswalk_ahead"}]
    # prob 0.68: below base danger 0.75 (no brake normally) but >= lowered 0.65.
    events = []
    for k in range(20):
        t = k * 0.2
        flags = active_context_flags(t, 1, spans)
        events.extend(eng.update(k, t, [_track(1, 0.68)], context_flags=flags))
    types = [e.type for e in events]
    assert "THRESHOLD_LOWERED" in types                # span began at frame 0
    assert types.count("THRESHOLD_LOWERED") == 1        # ...and only fires once
    assert types.count("VIRTUAL_BRAKE") == 1            # lowered threshold -> brake
    scene = next(e for e in events if e.type == "THRESHOLD_LOWERED")
    assert scene.actor_id == -1 and scene.threshold == 0.65


def test_risk_level_set_on_tracks():
    eng = DecisionEngine(CONFIG)
    safe, caution, danger = _track(1, 0.1), _track(2, 0.5), _track(3, 0.9)
    eng.update(0, 0.0, [safe, caution, danger])
    assert safe.risk_level == "safe"
    assert caution.risk_level == "caution"
    assert danger.risk_level == "danger"


def test_dense_traffic_flag_does_not_emit_threshold_lowered():
    # dense_traffic is derived from perception, not an annotation span, so its
    # onset must NOT emit a THRESHOLD_LOWERED event (only spans do).
    eng = DecisionEngine(CONFIG)
    tracks = [_track(i, 0.1) for i in range(8)]         # 8 tracks -> dense_traffic
    events = eng.update(0, 0.0, tracks, context_flags=["dense_traffic"])
    assert all(e.type != "THRESHOLD_LOWERED" for e in events)


def test_load_context_spans_missing_file():
    assert load_context_spans(None) == []
    assert load_context_spans("does/not/exist.json") == []
