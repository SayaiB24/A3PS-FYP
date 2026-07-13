"""Tests for the run_pipeline.py risk hook's dashboard-facing side effects.

The dashboard (app.js) reads ``frame.context.active_threshold`` every frame to
draw the threshold line and the "lowered due to X" note. That value is
computed inside DecisionEngine but must be written back onto the shared
``ctx["context"]`` dict for it to reach the serialized FrameRecord -- these
tests pin that write-through so a future refactor can't silently drop it.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from a3ps.common.schema import Prediction, TrackState  # noqa: E402

import run_pipeline as rp  # noqa: E402

CONFIG = {"base_threshold": 0.75, "caution_threshold": 0.4, "predict_hz": 5,
          "forecast_space": "img"}
CORRIDOR = [[200, 720], [440, 720], [440, 300], [200, 300]]


def _track(track_id=1, cls="person"):
    means = [[100.0, 500.0]] * 20
    stds = [[6.0, 6.0]] * 20
    pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=means, std_bev=stds)
    return TrackState(id=track_id, cls=cls, bbox=[0, 0, 10, 10],
                      centroid_img=[100.0, 500.0], prediction=pred)


def _ctx(frame_idx, t, flags):
    return {
        "frame_idx": frame_idx, "t": t,
        "ego": {"corridor_poly_img": CORRIDOR},
        "ground": None, "forecast_space": "img",
        "context": {"flags": flags},
    }


def test_active_threshold_written_back_to_context():
    hook = rp.make_risk_hook(CONFIG)
    ctx = _ctx(0, 0.0, flags=[])
    hook([_track()], ctx)
    assert ctx["context"]["active_threshold"] == 0.75


def test_active_threshold_reflects_lowering_context_flags():
    hook = rp.make_risk_hook(CONFIG)
    ctx = _ctx(0, 0.0, flags=["crosswalk_ahead"])
    hook([_track()], ctx)
    assert ctx["context"]["active_threshold"] == 0.65


def test_active_threshold_updates_across_frames_as_flags_change():
    hook = rp.make_risk_hook(CONFIG)
    ctx1 = _ctx(0, 0.0, flags=[])
    hook([_track()], ctx1)
    assert ctx1["context"]["active_threshold"] == 0.75

    ctx2 = _ctx(1, 0.2, flags=["crosswalk_ahead", "intersection"])
    hook([_track()], ctx2)
    assert ctx2["context"]["active_threshold"] == 0.55


def _track_in_corridor():
    """A track whose forecast sits inside CORRIDOR -- collision_prob > 0."""
    means = [[300.0, 500.0]] * 20
    stds = [[6.0, 6.0]] * 20
    pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=means, std_bev=stds)
    return TrackState(id=1, cls="person", bbox=[0, 0, 10, 10],
                      centroid_img=[300.0, 500.0], prediction=pred)


def test_ema_smoothing_toggle_changes_response_to_a_sudden_jump():
    """A track jumps from far-from-corridor (prob ~0) to squarely inside it
    (prob ~1) between frame 0 and frame 1, same track id both times. With
    ``ema_smoothing`` on (default), the EMA blends the jump with the prior
    (near-zero) state, so the reported prob lags below the raw value; with it
    off, the raw per-frame value is used directly, so the jump shows up in
    full immediately.
    """
    def second_frame_prob(config):
        hook = rp.make_risk_hook(config)
        hook([_track()], _ctx(0, 0.0, flags=[]))          # far -> prob ~0
        tr = _track_in_corridor()
        tr.id = 1                                          # same id -> same EMA state
        hook([tr], _ctx(1, 0.2, flags=[]))
        return tr.prediction.collision_prob

    smoothed = second_frame_prob(CONFIG)                    # ema_smoothing defaults True
    raw = second_frame_prob({**CONFIG, "ema_smoothing": False})
    assert raw > smoothed
