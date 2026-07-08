"""Round-trip serialization test for the schema dataclasses."""

import json
import math
import os
import sys

# Make the package importable when run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import (  # noqa: E402
    ClipResult,
    Event,
    FrameRecord,
    Prediction,
    TrackState,
)


def _build_clip() -> ClipResult:
    prediction = Prediction(
        horizon_s=4.0,
        dt=0.2,
        mean_img=[[100.0, 200.0], [110.0, 210.0]],
        mean_bev=[[1.0, 2.0], [1.5, 2.5]],
        std_bev=[[0.1, 0.1], [0.2, 0.2]],
        collision_prob=0.82,
        ttc_s=1.7,
    )
    track = TrackState(
        id=7,
        cls="car",
        bbox=[10.0, 20.0, 110.0, 220.0],
        centroid_img=[60.0, 120.0],
        centroid_bev=[1.0, 2.0],
        velocity_bev=[0.5, 0.1],
        mask_poly=[[10.0, 20.0], [110.0, 20.0], [110.0, 220.0], [10.0, 220.0]],
        history_img=[[55.0, 115.0], [60.0, 120.0]],
        prediction=prediction,
        risk_level="danger",
    )
    frame = FrameRecord(
        frame_idx=42,
        t=1.4,
        tracks=[track],
        ego={"speed_mps": 12.0, "yaw_rate": 0.0},
        context={"scene": "urban", "raining": False},
    )
    event = Event(
        event_id=1,
        frame_idx=42,
        t=1.4,
        type="ALERT",
        actor_id=7,
        actor_cls="car",
        collision_prob=0.82,
        threshold=0.75,
        ttc_s=1.7,
        explanation_template="Car ahead crossing ego corridor; TTC 1.7s.",
        explanation_llm=None,
    )
    return ClipResult(
        meta={"clip": "demo.mp4", "fps": 30, "width": 1280, "height": 720},
        frames=[frame],
        events=[event],
    )


def test_round_trip(tmp_path):
    original = _build_clip()
    path = tmp_path / "events.json"
    original.save_json(str(path))

    reloaded = ClipResult.load_json(str(path))

    assert reloaded == original
    assert reloaded.to_dict() == original.to_dict()


def test_bev_none_serializes_as_null(tmp_path):
    clip = _build_clip()
    # Strip BEV fields from the track and its prediction.
    track = clip.frames[0].tracks[0]
    track.centroid_bev = None
    track.velocity_bev = None
    track.prediction.mean_bev = None
    track.prediction.std_bev = None

    path = tmp_path / "events.json"
    clip.save_json(str(path))

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    t = raw["frames"][0]["tracks"][0]
    assert t["centroid_bev"] is None
    assert t["velocity_bev"] is None
    assert t["prediction"]["mean_bev"] is None
    assert t["prediction"]["std_bev"] is None

    # And it must round-trip back to None (not disappear).
    reloaded = ClipResult.load_json(str(path))
    assert reloaded == clip


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "events_example.json")


def test_loads_spec_example_and_reserializes_identically(tmp_path):
    """The canonical events.json spec example must round-trip byte-for-byte
    at the dict level (load -> dataclasses -> to_dict == original dict)."""
    with open(FIXTURE, encoding="utf-8") as fh:
        original = json.load(fh)

    clip = ClipResult.from_dict(original)

    # Structural / field-level assertions on the spec example.
    assert clip.meta["clip_id"] == "ped_crossing_01"
    assert clip.meta["config"]["forecaster"] == "kalman_cv"
    ev = clip.events[0]
    assert ev.event_id == 1 and isinstance(ev.event_id, int)
    assert ev.type == "VIRTUAL_BRAKE"
    assert ev.explanation_llm is None
    tr = clip.frames[0].tracks[0]
    assert tr.risk_level == "danger"
    assert tr.centroid_bev == [1.8, 14.2]
    assert clip.frames[0].ego["speed_note"] == "unknown"
    assert clip.frames[0].context["flags"] == ["crosswalk_ahead"]

    # Full dict round-trip: re-serializing yields the same structure/values.
    assert clip.to_dict() == json.loads(json.dumps(original))

    # And a save/reload cycle preserves equality.
    path = tmp_path / "events.json"
    clip.save_json(str(path))
    assert ClipResult.load_json(str(path)) == clip
