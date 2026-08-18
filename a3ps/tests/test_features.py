"""Tests for Phase IV feature extraction (a3ps/features/).

No GPU, no video, no model: every test builds a synthetic ClipResult or a
synthetic warp matrix, so the feature semantics are pinned by construction rather
than by whatever a real clip happens to contain.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import ClipResult, FrameRecord, Prediction, TrackState  # noqa: E402
from a3ps.features.ego import (  # noqa: E402
    EGO_FEATURE_DIM,
    EGO_ZERO,
    decompose_affine,
    ego_features_from_affine,
)
from a3ps.features.extract import (  # noqa: E402
    ACTOR_FEATURE_DIM,
    ACTOR_FEATURE_NAMES,
    TAU_CLIP_S,
    actor_feature_matrix,
    clip_features,
    frame_feature_dim,
    frame_feature_names,
    load_features,
    save_features,
)

W, H = 1280, 720
CORRIDOR = [[500.0, 720.0], [780.0, 720.0], [670.0, 430.0], [610.0, 430.0]]
AIDX = {n: i for i, n in enumerate(ACTOR_FEATURE_NAMES)}


def _track(tid, bbox, cls="car", prob=None, ttc=None):
    x1, y1, x2, y2 = bbox
    pred = None
    if prob is not None or ttc is not None:
        pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=[[0.0, 0.0]],
                          std_bev=[[1.0, 1.0]], collision_prob=prob, ttc_s=ttc)
    return TrackState(id=tid, cls=cls, bbox=[float(v) for v in bbox],
                      centroid_img=[(x1 + x2) / 2.0, float(y2)], prediction=pred)


def _clip(frames, fps=10.0):
    return ClipResult(
        meta={"clip_id": "t", "fps": fps, "width": W, "height": H,
              "n_frames": len(frames)},
        frames=frames, events=[])


def _approaching_clip(n=12, dt=0.1, growth=1.08):
    """One car whose bbox grows geometrically -- a pure looming sequence."""
    frames = []
    w0, h0 = 60.0, 50.0
    for i in range(n):
        s = growth ** i
        w, h = w0 * s, h0 * s
        cx, cy = 640.0, 500.0 + 6.0 * i
        bbox = [cx - w / 2, cy - h, cx + w / 2, cy]
        frames.append(FrameRecord(
            frame_idx=i, t=round(i * dt, 3),
            tracks=[_track(1, bbox, prob=min(1.0, 0.05 * i), ttc=max(0.5, 4.0 - 0.3 * i))],
            ego={"corridor_poly_img": CORRIDOR}, context=None))
    return _clip(frames)


# ---------------------------------------------------------------------------
# schema consistency -- the audit trail must not drift from the vectors
# ---------------------------------------------------------------------------

def test_feature_name_lists_match_vector_widths():
    assert len(ACTOR_FEATURE_NAMES) == ACTOR_FEATURE_DIM
    assert len(frame_feature_names()) == frame_feature_dim()
    assert len(set(frame_feature_names())) == frame_feature_dim(), "duplicate names"


def test_frame_vector_starts_with_the_ego_block():
    assert frame_feature_names()[:EGO_FEATURE_DIM] == (
        "ego_tx", "ego_ty", "ego_log_scale", "ego_rot")


# ---------------------------------------------------------------------------
# ego decomposition
# ---------------------------------------------------------------------------

def test_identity_and_missing_warp_both_read_as_no_motion():
    assert ego_features_from_affine(np.eye(2, 3), W, H) == list(EGO_ZERO)
    assert ego_features_from_affine(None, W, H) == list(EGO_ZERO)
    assert ego_features_from_affine([[float("nan")] * 3] * 2, W, H) == list(EGO_ZERO)


def test_translation_is_normalised_by_frame_size():
    f = ego_features_from_affine([[1, 0, 128], [0, 1, 72]], W, H)
    assert f[0] == pytest.approx(0.1)
    assert f[1] == pytest.approx(0.1)
    assert f[2] == pytest.approx(0.0)


def test_forward_motion_shows_up_as_positive_log_scale():
    zoom_in = ego_features_from_affine([[1.1, 0, 0], [0, 1.1, 0]], W, H)
    zoom_out = ego_features_from_affine([[1 / 1.1, 0, 0], [0, 1 / 1.1, 0]], W, H)
    assert zoom_in[2] == pytest.approx(math.log(1.1))
    # log makes equal proportional zoom equal and opposite.
    assert zoom_in[2] == pytest.approx(-zoom_out[2])


def test_rotation_recovered_in_radians():
    a = math.radians(7.0)
    H_ = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]]
    assert ego_features_from_affine(H_, W, H)[3] == pytest.approx(a)


def test_decompose_accepts_a_3x3_and_uses_its_affine_part():
    d = decompose_affine([[2, 0, 5], [0, 2, 6], [0, 0, 1]])
    assert d["scale"] == pytest.approx(2.0)
    assert (d["tx"], d["ty"]) == (5.0, 6.0)


def test_decompose_of_a_degenerate_matrix_falls_back_to_identity():
    assert decompose_affine([[0, 0, 0], [0, 0, 0]])["scale"] == 1.0


# ---------------------------------------------------------------------------
# per-actor features
# ---------------------------------------------------------------------------

def test_actor_matrix_has_one_row_per_frame_per_actor():
    clip = _approaching_clip(n=8)
    a = actor_feature_matrix(clip)
    assert a["rows"].shape == (8, ACTOR_FEATURE_DIM)
    assert a["frame_idx"].tolist() == list(range(8))
    assert set(a["track_id"].tolist()) == {1}


def test_growing_bbox_produces_positive_area_growth_and_finite_looming_ttc():
    a = actor_feature_matrix(_approaching_clip(growth=1.08))
    growth = a["rows"][:, AIDX["area_growth"]]
    tau = a["rows"][:, AIDX["tau_area"]]
    assert np.all(growth[1:-1] > 0.0), "a geometrically growing bbox must loom"
    assert np.all(tau[1:-1] < TAU_CLIP_S), "looming must give a finite TTC"


def test_static_actor_has_no_looming_and_saturated_tau():
    frames = [
        FrameRecord(frame_idx=i, t=round(i * 0.1, 3),
                    tracks=[_track(1, [600, 450, 660, 500])],
                    ego={"corridor_poly_img": CORRIDOR}, context=None)
        for i in range(8)
    ]
    a = actor_feature_matrix(_clip(frames))
    assert np.allclose(a["rows"][:, AIDX["area_growth"]], 0.0)
    assert np.allclose(a["rows"][:, AIDX["tau_area"]], TAU_CLIP_S)
    assert np.allclose(a["rows"][:, AIDX["vx"]], 0.0)
    assert np.allclose(a["rows"][:, AIDX["vy"]], 0.0)


def test_corridor_distance_is_signed_and_in_corridor_flag_agrees():
    inside = _track(1, [600, 650, 700, 700])        # foot point inside the corridor
    outside = _track(2, [80, 400, 140, 450])        # far left of it
    fr = FrameRecord(frame_idx=0, t=0.0, tracks=[inside, outside],
                     ego={"corridor_poly_img": CORRIDOR}, context=None)
    a = actor_feature_matrix(_clip([fr]))
    by_tid = {int(t): a["rows"][i] for i, t in enumerate(a["track_id"])}
    assert by_tid[1][AIDX["corridor_dist"]] < 0.0
    assert by_tid[1][AIDX["in_corridor"]] == 1.0
    assert by_tid[2][AIDX["corridor_dist"]] > 0.0
    assert by_tid[2][AIDX["in_corridor"]] == 0.0


def test_class_one_hot_is_exclusive_and_unknown_falls_through():
    fr = FrameRecord(frame_idx=0, t=0.0, tracks=[
        _track(1, [600, 650, 700, 700], cls="person"),
        _track(2, [200, 400, 260, 450], cls="train"),      # not in ACTOR_CLASSES
    ], ego={"corridor_poly_img": CORRIDOR}, context=None)
    a = actor_feature_matrix(_clip([fr]))
    by_tid = {int(t): a["rows"][i] for i, t in enumerate(a["track_id"])}
    assert by_tid[1][AIDX["cls_person"]] == 1.0
    assert by_tid[1][AIDX["cls_car"]] == 0.0
    assert by_tid[2][AIDX["cls_other"]] == 1.0
    for row in by_tid.values():
        onehot = [row[AIDX["cls_" + c]] for c in
                  ("person", "bicycle", "car", "motorcycle", "bus", "truck", "other")]
        assert sum(onehot) == 1.0


def test_missing_prediction_yields_zero_prob_and_saturated_ttc():
    fr = FrameRecord(frame_idx=0, t=0.0, tracks=[_track(1, [600, 650, 700, 700])],
                     ego={"corridor_poly_img": CORRIDOR}, context=None)
    row = actor_feature_matrix(_clip([fr]))["rows"][0]
    assert row[AIDX["collision_prob"]] == 0.0
    assert row[AIDX["ttc_pred"]] == TAU_CLIP_S


def test_derivatives_follow_each_track_not_the_frame():
    """Two tracks moving oppositely must get opposite velocities, not a blend."""
    frames = []
    for i in range(6):
        right = _track(1, [400 + 20 * i, 600, 460 + 20 * i, 650])
        left = _track(2, [900 - 20 * i, 600, 960 - 20 * i, 650])
        frames.append(FrameRecord(frame_idx=i, t=round(i * 0.1, 3),
                                  tracks=[right, left],
                                  ego={"corridor_poly_img": CORRIDOR}, context=None))
    a = actor_feature_matrix(_clip(frames))
    vx = {}
    for i, tid in enumerate(a["track_id"].tolist()):
        vx.setdefault(int(tid), []).append(float(a["rows"][i, AIDX["vx"]]))
    mid1 = vx[1][len(vx[1]) // 2]
    mid2 = vx[2][len(vx[2]) // 2]
    assert mid1 > 0 and mid2 < 0
    assert mid1 == pytest.approx(-mid2, rel=1e-5)


def test_empty_clip_returns_empty_but_correctly_shaped_arrays():
    a = actor_feature_matrix(_clip([]))
    assert a["rows"].shape == (0, ACTOR_FEATURE_DIM)


# ---------------------------------------------------------------------------
# frame pooling
# ---------------------------------------------------------------------------

def test_clip_features_shape_and_ego_absent_by_default():
    f = clip_features(_approaching_clip(n=9))
    assert f["X"].shape == (9, frame_feature_dim())
    assert f["has_ego"] is False
    assert np.allclose(f["X"][:, :EGO_FEATURE_DIM], 0.0)
    assert np.all(np.isfinite(f["X"]))


def test_supplied_ego_lands_in_the_leading_slots():
    clip = _approaching_clip(n=5)
    ego = np.tile(np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float32), (5, 1))
    f = clip_features(clip, ego=ego)
    assert f["has_ego"] is True
    assert np.allclose(f["X"][:, :EGO_FEATURE_DIM], ego)


def test_short_ego_array_is_padded_not_rejected():
    """The flow estimator yields nothing for frame 0, so T-1 rows is normal."""
    clip = _approaching_clip(n=6)
    ego = np.ones((5, EGO_FEATURE_DIM), dtype=np.float32)
    f = clip_features(clip, ego=ego)
    assert f["X"].shape[0] == 6
    assert np.allclose(f["X"][5, :EGO_FEATURE_DIM], 0.0)


def test_wrong_width_ego_is_rejected():
    with pytest.raises(ValueError, match="ego must be"):
        clip_features(_approaching_clip(n=4), ego=np.zeros((4, 2)))


def test_frames_with_no_actors_are_zeros_not_nan():
    frames = [FrameRecord(frame_idx=0, t=0.0, tracks=[],
                          ego={"corridor_poly_img": CORRIDOR}, context=None)]
    f = clip_features(_clip(frames))
    assert f["n_actors"].tolist() == [0]
    assert np.all(np.isfinite(f["X"]))
    assert np.allclose(f["X"], 0.0)


def test_top_actor_slot_carries_the_riskiest_actor_not_the_first_listed():
    """Pooling must not bury the hazard behind whichever track id came first."""
    boring = _track(1, [100, 400, 140, 440], prob=0.0)
    hazard = _track(2, [610, 660, 700, 715], prob=0.95)
    fr = FrameRecord(frame_idx=0, t=0.0, tracks=[boring, hazard],
                     ego={"corridor_poly_img": CORRIDOR}, context=None)
    f = clip_features(_clip([fr]))
    names = frame_feature_names()
    top0_prob = f["X"][0, names.index("top0_collision_prob")]
    assert top0_prob == pytest.approx(0.95)


def test_max_pool_block_is_the_elementwise_max_over_actors():
    a = _track(1, [100, 400, 140, 440], prob=0.2)
    b = _track(2, [610, 660, 700, 715], prob=0.9)
    fr = FrameRecord(frame_idx=0, t=0.0, tracks=[a, b],
                     ego={"corridor_poly_img": CORRIDOR}, context=None)
    f = clip_features(_clip([fr]))
    names = frame_feature_names()
    assert f["X"][0, names.index("max_collision_prob")] == pytest.approx(0.9)
    assert f["X"][0, names.index("max_in_corridor")] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# on-disk round trip
# ---------------------------------------------------------------------------

def test_features_round_trip_through_npz(tmp_path):
    clip = _approaching_clip(n=7)
    feats = clip_features(clip)
    path = str(tmp_path / "42.npz")
    save_features(path, feats, {"clip_id": "42", "label": 1,
                                "event_time_s": 19.0, "alert_time_s": 15.5})
    got = load_features(path)
    assert np.allclose(got["X"], feats["X"])
    assert got["meta"]["clip_id"] == "42"
    assert got["meta"]["label"] == 1
    assert got["meta"]["alert_time_s"] == 15.5
    assert got["has_ego"] is False
    assert got["meta"]["feature_names"] == list(frame_feature_names())


def test_loading_a_stale_schema_version_raises(tmp_path):
    """A silently-mixed feature layout would train a model that scores nonsense."""
    import json
    path = str(tmp_path / "old.npz")
    np.savez_compressed(
        path,
        X=np.zeros((3, frame_feature_dim()), dtype=np.float32),
        t=np.zeros((3,), dtype=np.float32),
        n_actors=np.zeros((3,), dtype=np.int32),
        meta=np.asarray(json.dumps({"schema_version": 999})))
    with pytest.raises(ValueError, match="schema_version"):
        load_features(path)


def test_loading_a_wrong_width_matrix_raises(tmp_path):
    import json
    path = str(tmp_path / "narrow.npz")
    np.savez_compressed(
        path,
        X=np.zeros((3, 5), dtype=np.float32),
        t=np.zeros((3,), dtype=np.float32),
        n_actors=np.zeros((3,), dtype=np.int32),
        meta=np.asarray(json.dumps({"schema_version": 1})))
    with pytest.raises(ValueError, match="feature width"):
        load_features(path)
