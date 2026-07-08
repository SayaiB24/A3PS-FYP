"""Unit tests for TrajectoryBuffer: 5 Hz decimation and stale pruning."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import TrackState  # noqa: E402
from a3ps.tracking.tracker import STALE_S, TrajectoryBuffer  # noqa: E402

CONFIG = {"history_s": 2.0, "predict_hz": 5, "process_fps": 30}


def _track(tid, x, y):
    return TrackState(id=tid, cls="car", bbox=[x - 5, y - 10, x + 5, y],
                      centroid_img=[x, y])


def test_5hz_decimation_from_30fps_stream():
    buf = TrajectoryBuffer(CONFIG)
    fps = 30.0
    # Feed exactly 1 second (30 frames) of one moving track.
    for i in range(30):
        t = i / fps
        buf.push(i, t, [_track(1, float(i), 100.0)])

    hist = buf.history(1)
    # 5 Hz over ~1 s -> ~5 samples (buckets floor(t*5) for t in [0, 0.967]).
    assert len(hist) == 5, f"expected 5 decimated points, got {len(hist)}"

    # Stored samples must be spaced ~0.2 s apart, i.e. every 6th frame at 30 fps:
    # frames 0, 6, 12, 18, 24 -> x = 0, 6, 12, 18, 24.
    xs = [p[0] for p in hist]
    assert xs == [0.0, 6.0, 12.0, 18.0, 24.0], xs
    # Most recent last.
    assert hist[-1][0] == 24.0


def test_window_and_ready_flag():
    buf = TrajectoryBuffer(CONFIG)
    fps = 30.0
    # Not ready before enough history: after 1 s we have 5 pts, need >= 8.
    for i in range(30):
        buf.push(i, i / fps, [_track(1, float(i), 0.0)])
    assert buf.ready(1) is False

    # After 2 s (60 frames) -> 10 pts == history_s*predict_hz, ready.
    for i in range(30, 60):
        buf.push(i, i / fps, [_track(1, float(i), 0.0)])
    hist = buf.history(1)
    assert len(hist) == 10          # capped at history_s * predict_hz
    assert buf.ready(1) is True

    # Window is bounded: keep feeding, length stays at maxlen (oldest dropped).
    for i in range(60, 120):
        buf.push(i, i / fps, [_track(1, float(i), 0.0)])
    assert len(buf.history(1)) == 10
    # Oldest sample has advanced past the earliest ones.
    assert buf.history(1)[0][0] > 24.0


def test_prune_stale_ids():
    buf = TrajectoryBuffer(CONFIG)
    fps = 30.0
    # Two tracks seen together for the first 0.5 s.
    for i in range(15):
        t = i / fps
        buf.push(i, t, [_track(1, float(i), 0.0), _track(2, float(i), 50.0)])
    assert buf.history(1) and buf.history(2)

    # Now only track 2 continues. Track 1 must survive until 1.0 s of absence,
    # then be pruned once now - last_seen(1) > STALE_S.
    last_t1 = 14 / fps
    # Just under the stale threshold: track 1 still present.
    t_almost = last_t1 + STALE_S - 0.01
    buf.push(100, t_almost, [_track(2, 99.0, 50.0)])
    assert buf.history(1), "track 1 pruned too early"

    # Past the threshold: track 1 pruned, track 2 remains.
    t_stale = last_t1 + STALE_S + 0.01
    buf.push(101, t_stale, [_track(2, 100.0, 50.0)])
    assert buf.history(1) == [], "stale track 1 should be pruned"
    assert buf.history(2), "active track 2 should remain"
