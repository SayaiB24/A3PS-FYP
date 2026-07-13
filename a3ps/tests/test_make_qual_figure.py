"""Unit tests for make_qual_figure.py's pure logic (no video/cv2 needed).

Frame extraction + matplotlib rendering were verified manually against a real
processed clip (dashboard/clips/dev14) -- these tests cover the incident-
finding and frame-picking logic, which is what can silently pick the wrong
frames if events.json ever has an unusual shape (multiple incidents, gaps in
frame_idx, a brake at the very start/end of the clip).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest  # noqa: E402

from a3ps.common.schema import ClipResult, Event, FrameRecord, TrackState  # noqa: E402

import make_qual_figure as qf  # noqa: E402


def _frame(idx, t, actor_id=5, risk_level="safe"):
    tr = TrackState(id=actor_id, cls="car", bbox=[0, 0, 10, 10],
                    centroid_img=[5.0, 5.0], risk_level=risk_level)
    return FrameRecord(frame_idx=idx, t=t, tracks=[tr],
                       ego={"corridor_poly_img": [[0, 0], [1, 0], [1, 1], [0, 1]]})


def _event(event_id, frame_idx, t, etype, actor_id=5):
    return Event(event_id=event_id, frame_idx=frame_idx, t=t, type=etype,
                 actor_id=actor_id, actor_cls="car", collision_prob=0.8,
                 threshold=0.75, explanation_template="test explanation")


def test_find_incident_picks_matching_alert_before_brake():
    clip = ClipResult(
        frames=[_frame(i, i * 0.2) for i in range(10)],
        events=[
            _event(1, 3, 0.6, "ALERT"),
            _event(2, 6, 1.2, "VIRTUAL_BRAKE"),
        ],
    )
    alert, brake = qf.find_incident(clip)
    assert alert.frame_idx == 3
    assert brake.frame_idx == 6


def test_find_incident_ignores_other_actors():
    clip = ClipResult(
        frames=[_frame(i, i * 0.2) for i in range(10)],
        events=[
            _event(1, 2, 0.4, "ALERT", actor_id=99),       # different actor
            _event(2, 3, 0.6, "ALERT", actor_id=5),
            _event(3, 6, 1.2, "VIRTUAL_BRAKE", actor_id=5),
        ],
    )
    alert, brake = qf.find_incident(clip)
    assert alert.actor_id == 5 and alert.frame_idx == 3


def test_find_incident_picks_earliest_brake_when_several():
    clip = ClipResult(
        frames=[_frame(i, i * 0.2) for i in range(20)],
        events=[
            _event(1, 2, 0.4, "ALERT", actor_id=1),
            _event(2, 4, 0.8, "VIRTUAL_BRAKE", actor_id=1),
            _event(3, 10, 2.0, "ALERT", actor_id=2),
            _event(4, 12, 2.4, "VIRTUAL_BRAKE", actor_id=2),
        ],
    )
    alert, brake = qf.find_incident(clip)
    assert brake.actor_id == 1 and brake.frame_idx == 4


def test_find_incident_can_target_a_specific_actor():
    clip = ClipResult(
        frames=[_frame(i, i * 0.2) for i in range(20)],
        events=[
            _event(1, 2, 0.4, "ALERT", actor_id=1),
            _event(2, 4, 0.8, "VIRTUAL_BRAKE", actor_id=1),
            _event(3, 10, 2.0, "ALERT", actor_id=2),
            _event(4, 12, 2.4, "VIRTUAL_BRAKE", actor_id=2),
        ],
    )
    alert, brake = qf.find_incident(clip, actor_id=2)
    assert brake.actor_id == 2 and brake.frame_idx == 12


def test_find_incident_raises_when_no_brake():
    clip = ClipResult(frames=[_frame(0, 0.0)], events=[])
    with pytest.raises(ValueError, match="No VIRTUAL_BRAKE"):
        qf.find_incident(clip)


def test_find_incident_raises_when_brake_has_no_preceding_alert():
    clip = ClipResult(
        frames=[_frame(i, i * 0.2) for i in range(5)],
        events=[_event(1, 3, 0.6, "VIRTUAL_BRAKE")],   # no ALERT at all
    )
    with pytest.raises(ValueError, match="No ALERT event"):
        qf.find_incident(clip)


def test_pick_four_frames_uses_neighbors_by_list_position():
    clip = ClipResult(frames=[_frame(i, i * 0.2) for i in range(10)], events=[])
    alert = _event(1, 3, 0.6, "ALERT")
    brake = _event(2, 6, 1.2, "VIRTUAL_BRAKE")
    panels = qf.pick_four_frames(clip, alert, brake)
    labels = [label for label, _ in panels]
    idxs = [rec.frame_idx for _, rec in panels]
    assert labels == ["before ALERT", "SAFE -> ALERT", "VIRTUAL_BRAKE", "after BRAKE"]
    assert idxs == [2, 3, 6, 7]


def test_pick_four_frames_clamps_at_clip_boundaries():
    # ALERT on the very first frame, BRAKE on the very last -- "before" and
    # "after" must clamp instead of indexing out of range.
    clip = ClipResult(frames=[_frame(i, i * 0.2) for i in range(3)], events=[])
    alert = _event(1, 0, 0.0, "ALERT")
    brake = _event(2, 2, 0.4, "VIRTUAL_BRAKE")
    panels = qf.pick_four_frames(clip, alert, brake)
    idxs = [rec.frame_idx for _, rec in panels]
    assert idxs == [0, 0, 2, 2]   # before-ALERT clamps to 0, after-BRAKE clamps to last


def test_track_risk_level_found_and_missing():
    record = _frame(0, 0.0, actor_id=5, risk_level="danger")
    assert qf.track_risk_level(record, 5) == "danger"
    assert "not tracked" in qf.track_risk_level(record, 999)
