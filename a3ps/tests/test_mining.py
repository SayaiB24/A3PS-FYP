"""Unit tests for trajectory-mining logic (pure functions in the script)."""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import mine_trajectories as mt  # noqa: E402


def test_resample_decimates_to_buckets():
    # 30 fps stream -> one point per 0.2 s bucket.
    samples = [(i / 30.0, [float(i), 0.0]) for i in range(30)]
    buckets = mt.resample_track(samples, hz=5.0)
    # ~5 buckets over ~1 s.
    assert len(buckets) == 5
    assert sorted(buckets) == [0, 1, 2, 3, 4]


def test_contiguous_windows_respects_gaps_and_stride():
    buckets = {b: [float(b), 0.0] for b in range(35)}  # 0..34 contiguous
    wins = list(mt.contiguous_windows(buckets, win=30, stride=5))
    starts = [s for s, _ in wins]
    assert starts == [0, 5]                 # 0..29 and 5..34
    assert all(len(pts) == 30 for _, pts in wins)

    # Punch a gap so no full 30-window fits from the front.
    del buckets[15]
    wins2 = list(mt.contiguous_windows(buckets, win=30, stride=5))
    assert all(15 not in range(s, s + 30) for s, _ in wins2)


def test_is_id_switch():
    line = [[float(i), 0.0] for i in range(30)]          # steady step 1
    assert mt.is_id_switch(line) is False
    jumpy = line[:15] + [[100.0, 0.0]] + line[16:]       # one big jump
    assert mt.is_id_switch(jumpy) is True


def test_net_displacement():
    assert mt.net_displacement([[0.0, 0.0]] * 30) == 0.0
    assert abs(mt.net_displacement([[float(i), 0.0] for i in range(30)]) - 29.0) < 1e-9


def test_normalize_window_origin_and_heading():
    # Straight line moving +x, 1 unit/step, 30 points.
    pts = [[float(i), 0.0] for i in range(30)]
    hist, fut, origin, theta = mt.normalize_window(pts, hist_len=10)

    # Last history point becomes the origin.
    assert np.allclose(hist[-1], [0.0, 0.0], atol=1e-9)
    # Stored inverse transform points at the true last-history absolute point.
    assert np.allclose(origin, [9.0, 0.0])
    # Heading rotated onto +y: future has ~0 x and strictly increasing +y.
    assert np.allclose(fut[:, 0], 0.0, atol=1e-6)
    assert fut[0, 1] > 0 and fut[-1, 1] > fut[0, 1]


def test_normalize_inverse_recovers_absolute():
    pts = [[2.0 * i, 0.5 * i] for i in range(30)]         # diagonal line
    hist, fut, origin, theta = mt.normalize_window(pts, hist_len=10)
    c, s = np.cos(-theta), np.sin(-theta)
    R_inv = np.array([[c, -s], [s, c]])
    recovered = (np.vstack([hist, fut]) @ R_inv.T) + origin
    assert np.allclose(recovered, np.asarray(pts), atol=1e-6)


def _stats(thresh=1.0):
    return {
        "static_thresh": thresh, "class_mix": {},
        "short_tracks": 0, "id_switch_windows": 0,
        "static_dropped": 0, "static_kept": 0,
    }


def test_mine_clip_drops_short_tracks():
    # 3 s track (< 6 s) -> dropped, no windows.
    samples = [(i / 5.0, [float(i), 0.0]) for i in range(15)]
    tracks = {1: {"pts": samples, "cls": {"car": 15}}}
    stats = _stats()
    windows = mt.mine_clip(tracks, random.Random(1), stats)
    assert windows == []
    assert stats["short_tracks"] == 1


def test_mine_clip_keeps_about_20pct_static():
    # Long static track -> many static windows, ~20% kept.
    n = 200  # 40 s @ 5 Hz
    samples = [(i / 5.0, [10.0, 10.0]) for i in range(n)]
    tracks = {1: {"pts": samples, "cls": {"car": n}}}
    stats = _stats(thresh=1.0)
    windows = mt.mine_clip(tracks, random.Random(mt.SEED), stats)

    total_static = stats["static_kept"] + stats["static_dropped"]
    assert total_static > 20                     # plenty of windows seen
    assert len(windows) == stats["static_kept"]  # only kept ones emitted
    frac = stats["static_kept"] / total_static
    assert 0.10 <= frac <= 0.32                  # ~20% kept
