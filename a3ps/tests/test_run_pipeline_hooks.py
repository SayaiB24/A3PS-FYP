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
