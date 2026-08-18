"""Per-frame feature vectors for the learned temporal risk head (Phase IV).

What this replaces
------------------
Phase IV used to be a hand-set threshold on a physics-derived collision
probability. The three-feature ablation showed no single hand-set threshold on
one geometric feature separates risky events from ordinary driving -- and, given
that Nexar labels near-misses as positive too, that is the expected result rather
than a failure: the task is "risky event vs. ordinary driving", and risk is a
nonlinear combination of cues. So the geometry stays, but it becomes *input* to a
small learned temporal model instead of the decision itself.

Design: two levels, one pooled
------------------------------
:func:`actor_feature_matrix` produces one row per (frame, actor) -- the raw
per-actor cues, all of them derived quantities the existing pipeline already
computes or can compute from what it stores.

:func:`clip_features` then pools those rows into ONE fixed-width vector per
frame, because the label is per-frame (an event happens at ``time_of_event``),
not per-actor. Pooling is deliberately not just a mean: a mean over a crowded
frame buries the one actor that matters. The frame vector therefore carries the
top-K actors by risk proxy *individually*, plus a max-pool over all actors, plus
a few whole-frame scalars. Per-actor sequence modelling stays possible later
(that is what :func:`actor_feature_matrix` is for); it is not needed to get a
first honest number.

Ego motion
----------
``FrameRecord.ego`` already exists in the schema with the comment "ego speed/yaw
etc." and has never been populated. When a caller supplies ego features (from
``a3ps.features.ego``) they occupy the leading :data:`EGO_FEATURE_DIM` slots.
When they are absent -- which is the case for every clip in the 3.7 GB cache
under ``eval/anticipation/``, since that pass never computed them -- those slots
are **zero-filled and the ``has_ego`` flag in the output metadata is False**.
That flag matters: zeroed ego features are indistinguishable from "stationary
vehicle", so any result computed from cache-only features is measuring a model
that was blind to ego motion, and must be reported as such.

Everything here is pure numpy over an already-computed ``ClipResult``, so it runs
on CPU with no model, no GPU and no video decode.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from a3ps.features.ego import EGO_FEATURE_DIM, EGO_FEATURE_NAMES
from a3ps.risk.collision import signed_distance_to_polygon

SCHEMA_VERSION = 1

# Actor classes, in one-hot order. Matches configs/default.yaml `classes`.
ACTOR_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

# Per-actor features, in vector order. Names are the audit trail: anything added
# here must be added to _actor_row in the same position.
ACTOR_FEATURE_NAMES: Tuple[str, ...] = (
    # position / extent, normalised by frame size and centred on the frame
    "cx", "cy", "bw", "bh", "log_area", "aspect",
    # first and second time derivatives of the foot point
    "vx", "vy", "ax", "ay",
    # looming: how fast the actor's image footprint is growing
    "area_growth", "tau_area", "tau_h",
    # what the existing forecaster/risk stage already says about this actor
    "ttc_pred", "collision_prob", "prob_trend",
    # continuous corridor geometry (the ablated feature, kept as an input)
    "corridor_dist", "corridor_dist_rate", "in_corridor",
) + tuple("cls_" + c for c in ACTOR_CLASSES) + ("cls_other",)

ACTOR_FEATURE_DIM = len(ACTOR_FEATURE_NAMES)

# How many individual actors the frame vector carries before falling back to
# pooling. 3 covers "the hazard plus two distractors" without exploding width.
TOP_K_ACTORS = 3

# Whole-frame scalars.
GLOBAL_FEATURE_NAMES = ("n_actors", "frac_in_corridor", "mean_prob", "max_prob")

# Clip values that are ratios of noisy small numbers, or the model would see
# ±1e6 spikes whenever a denominator passes through zero.
TAU_CLIP_S = 10.0
RATE_CLIP = 20.0


def frame_feature_names() -> Tuple[str, ...]:
    """Full frame-vector schema, in order. Length == :func:`frame_feature_dim`."""
    names: List[str] = list(EGO_FEATURE_NAMES)
    names += list(GLOBAL_FEATURE_NAMES)
    for k in range(TOP_K_ACTORS):
        names += [f"top{k}_{n}" for n in ACTOR_FEATURE_NAMES]
    names += [f"max_{n}" for n in ACTOR_FEATURE_NAMES]
    return tuple(names)


def frame_feature_dim() -> int:
    return (EGO_FEATURE_DIM + len(GLOBAL_FEATURE_NAMES)
            + (TOP_K_ACTORS + 1) * ACTOR_FEATURE_DIM)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _clip(v: float, lim: float) -> float:
    """Symmetric clip that also maps NaN/inf to 0.0."""
    if v is None or not math.isfinite(v):
        return 0.0
    return max(-lim, min(lim, float(v)))


def _central_diff(values: Sequence[float], times: Sequence[float]) -> List[float]:
    """d(values)/d(times), central where possible, one-sided at the ends.

    Central differences rather than backward ones because these series are
    ~10-30 Hz samples of a noisy detector: a backward difference at 30 Hz is
    almost pure detector jitter, while a centred one over the same spacing halves
    the noise for free. Runs of identical timestamps yield 0.0 rather than a
    divide-by-zero.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        dt = times[hi] - times[lo]
        out[i] = 0.0 if dt <= 0 else (values[hi] - values[lo]) / dt
    return out


def _tau(value: float, rate: float) -> float:
    """Time for ``value`` to reach zero at ``-rate``, i.e. a looming TTC.

    Only meaningful while the quantity is *growing* (rate > 0 means the actor is
    getting bigger, so closing). Returns TAU_CLIP_S when not closing, so "no
    threat" is a large number rather than a sign flip the model has to learn.
    """
    if rate is None or not math.isfinite(rate) or rate <= 1e-9:
        return TAU_CLIP_S
    if value is None or not math.isfinite(value) or value <= 0.0:
        return TAU_CLIP_S
    return min(TAU_CLIP_S, value / rate)


# ---------------------------------------------------------------------------
# per-actor rows
# ---------------------------------------------------------------------------

def _corridor_poly(frame) -> Optional[List[List[float]]]:
    ego = frame.ego or {}
    return ego.get("corridor_poly_img")


def actor_feature_matrix(clip_result) -> Dict[str, Any]:
    """Per-(frame, actor) feature rows for one clip.

    Returns ``{"rows": (N, ACTOR_FEATURE_DIM) float32, "frame_idx": (N,) int32,
    "track_id": (N,) int32, "t": (N,) float32}``.

    Derivatives are computed along each TRACK's own timeline (not across the
    frame), which is why this is a two-pass function: gather per track, then
    differentiate, then scatter back.
    """
    frames = clip_result.frames
    meta = clip_result.meta or {}
    width = float(meta.get("width") or 1280)
    height = float(meta.get("height") or 720)
    diag = math.hypot(width, height)

    # -- pass 1: gather raw measurements per track ------------------------
    per_track: Dict[int, Dict[str, List[Any]]] = {}
    for fi, fr in enumerate(frames):
        poly = _corridor_poly(fr)
        for tr in fr.tracks:
            tid = int(tr.id)
            d = per_track.setdefault(tid, {
                "fi": [], "t": [], "cx": [], "cy": [], "bw": [], "bh": [],
                "area": [], "ttc": [], "prob": [], "cdist": [], "cls": [],
            })
            x1, y1, x2, y2 = (float(v) for v in tr.bbox)
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            cen = tr.centroid_img or [0.5 * (x1 + x2), y2]
            pred = tr.prediction
            ttc = getattr(pred, "ttc_s", None) if pred else None
            prob = getattr(pred, "collision_prob", None) if pred else None
            cdist = (signed_distance_to_polygon(cen, poly)
                     if poly else float("nan"))

            d["fi"].append(fi)
            d["t"].append(float(fr.t))
            d["cx"].append(float(cen[0]))
            d["cy"].append(float(cen[1]))
            d["bw"].append(bw)
            d["bh"].append(bh)
            d["area"].append(bw * bh)
            d["ttc"].append(TAU_CLIP_S if ttc is None else float(ttc))
            d["prob"].append(0.0 if prob is None else float(prob))
            d["cdist"].append(cdist)
            d["cls"].append(str(tr.cls))

    # -- pass 2: differentiate along each track, emit rows ----------------
    rows: List[np.ndarray] = []
    out_fi: List[int] = []
    out_tid: List[int] = []
    out_t: List[float] = []

    for tid, d in per_track.items():
        t = d["t"]
        # Normalised, frame-size-independent series.
        nx = [v / width - 0.5 for v in d["cx"]]
        ny = [v / height - 0.5 for v in d["cy"]]
        nbw = [v / width for v in d["bw"]]
        nbh = [v / height for v in d["bh"]]
        # log area so growth is a proportional rate, which is what looming is.
        log_area = [math.log(max(v, 1.0) / (width * height)) for v in d["area"]]
        ndist = [(v / diag if math.isfinite(v) else 0.0) for v in d["cdist"]]

        vx, vy = _central_diff(nx, t), _central_diff(ny, t)
        ax, ay = _central_diff(vx, t), _central_diff(vy, t)
        growth = _central_diff(log_area, t)
        dh = _central_diff(nbh, t)
        prob_trend = _central_diff(d["prob"], t)
        dist_rate = _central_diff(ndist, t)

        for i in range(len(t)):
            cls = d["cls"][i]
            onehot = [0.0] * (len(ACTOR_CLASSES) + 1)
            onehot[ACTOR_CLASSES.index(cls) if cls in ACTOR_CLASSES else -1] = 1.0

            row = [
                nx[i], ny[i], nbw[i], nbh[i], log_area[i],
                _clip(nbw[i] / nbh[i] if nbh[i] > 1e-6 else 0.0, 10.0),
                _clip(vx[i], RATE_CLIP), _clip(vy[i], RATE_CLIP),
                _clip(ax[i], RATE_CLIP), _clip(ay[i], RATE_CLIP),
                _clip(growth[i], RATE_CLIP),
                # Area grows as the square of closing scale, hence the factor 2.
                _tau(2.0, growth[i]) if growth[i] > 0 else TAU_CLIP_S,
                _tau(nbh[i], dh[i]),
                min(TAU_CLIP_S, max(0.0, d["ttc"][i])),
                d["prob"][i], _clip(prob_trend[i], RATE_CLIP),
                _clip(ndist[i], 5.0), _clip(dist_rate[i], RATE_CLIP),
                1.0 if ndist[i] < 0.0 else 0.0,
            ] + onehot

            rows.append(np.asarray(row, dtype=np.float32))
            out_fi.append(d["fi"][i])
            out_tid.append(tid)
            out_t.append(t[i])

    if not rows:
        return {
            "rows": np.zeros((0, ACTOR_FEATURE_DIM), dtype=np.float32),
            "frame_idx": np.zeros((0,), dtype=np.int32),
            "track_id": np.zeros((0,), dtype=np.int32),
            "t": np.zeros((0,), dtype=np.float32),
        }

    order = np.argsort(np.asarray(out_fi, dtype=np.int64), kind="stable")
    return {
        "rows": np.vstack(rows)[order],
        "frame_idx": np.asarray(out_fi, dtype=np.int32)[order],
        "track_id": np.asarray(out_tid, dtype=np.int32)[order],
        "t": np.asarray(out_t, dtype=np.float32)[order],
    }


# ---------------------------------------------------------------------------
# frame pooling
# ---------------------------------------------------------------------------

_IDX = {n: i for i, n in enumerate(ACTOR_FEATURE_NAMES)}


def _risk_proxy(row: np.ndarray) -> float:
    """Ranking key for "which actor should the frame vector carry individually".

    Deliberately NOT the collision probability alone: that is exactly the
    saturating quantity whose ties made the old AP meaningless, and on the cached
    clips it pins to 1.0 for most actors. Combining it with corridor proximity
    and looming breaks those ties with something physical.
    """
    prob = float(row[_IDX["collision_prob"]])
    dist = float(row[_IDX["corridor_dist"]])
    loom = float(row[_IDX["area_growth"]])
    return prob + 0.5 * max(0.0, 1.0 - abs(dist)) + 0.1 * max(0.0, loom)


def clip_features(clip_result,
                  ego: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, Any]:
    """Pool per-actor rows into one fixed-width vector per frame.

    ``ego`` is an optional ``(n_frames, EGO_FEATURE_DIM)`` array from
    ``a3ps.features.ego``. When omitted the ego slots are zero-filled and
    ``has_ego`` is False -- see the module docstring on why that must be reported.

    Returns ``{"X": (T, frame_feature_dim()) float32, "t": (T,) float32,
    "has_ego": bool, "n_actors": (T,) int32}``.
    """
    actor = actor_feature_matrix(clip_result)
    frames = clip_result.frames
    T = len(frames)
    D = frame_feature_dim()

    X = np.zeros((T, D), dtype=np.float32)
    ts = np.zeros((T,), dtype=np.float32)
    n_actors = np.zeros((T,), dtype=np.int32)

    ego_arr = None
    if ego is not None:
        ego_arr = np.asarray(ego, dtype=np.float32)
        if ego_arr.ndim != 2 or ego_arr.shape[1] != EGO_FEATURE_DIM:
            raise ValueError(
                f"ego must be (n_frames, {EGO_FEATURE_DIM}), got {ego_arr.shape}")
        if ego_arr.shape[0] < T:
            # Pad rather than fail: the estimator yields nothing for frame 0.
            pad = np.zeros((T - ego_arr.shape[0], EGO_FEATURE_DIM), dtype=np.float32)
            ego_arr = np.vstack([ego_arr, pad])

    # Group actor rows by frame index (rows are already frame-sorted).
    by_frame: Dict[int, List[int]] = {}
    for i, fi in enumerate(actor["frame_idx"].tolist()):
        by_frame.setdefault(int(fi), []).append(i)

    ego_end = EGO_FEATURE_DIM
    glob_end = ego_end + len(GLOBAL_FEATURE_NAMES)

    for fi, fr in enumerate(frames):
        ts[fi] = float(fr.t)
        if ego_arr is not None:
            X[fi, :ego_end] = ego_arr[fi]

        idxs = by_frame.get(fi, [])
        n_actors[fi] = len(idxs)
        if not idxs:
            # No actors: globals stay 0 and every actor slot stays 0. That is a
            # real state (empty road), not missing data.
            continue

        rows = actor["rows"][idxs]
        probs = rows[:, _IDX["collision_prob"]]
        X[fi, ego_end:glob_end] = (
            min(len(idxs) / 10.0, 1.0),                     # n_actors, saturating
            float(np.mean(rows[:, _IDX["in_corridor"]])),
            float(np.mean(probs)),
            float(np.max(probs)),
        )

        ranked = sorted(range(len(idxs)),
                        key=lambda j: _risk_proxy(rows[j]), reverse=True)
        for k in range(TOP_K_ACTORS):
            lo = glob_end + k * ACTOR_FEATURE_DIM
            if k < len(ranked):
                X[fi, lo:lo + ACTOR_FEATURE_DIM] = rows[ranked[k]]

        lo = glob_end + TOP_K_ACTORS * ACTOR_FEATURE_DIM
        X[fi, lo:lo + ACTOR_FEATURE_DIM] = rows.max(axis=0)

    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return {"X": X, "t": ts, "has_ego": ego_arr is not None, "n_actors": n_actors}


# ---------------------------------------------------------------------------
# on-disk format
# ---------------------------------------------------------------------------

def save_features(path: str, feats: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Write one clip's features to a compressed .npz.

    Two orders of magnitude smaller than the events.json it came from (a 53 MB
    ClipResult reduces to a few hundred KB), which is what makes it practical to
    compute features on the GPU laptop and pull them back for CPU iteration.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=feats["X"].astype(np.float32),
        t=feats["t"].astype(np.float32),
        n_actors=feats["n_actors"].astype(np.int32),
        meta=np.asarray(json.dumps({
            **meta,
            "schema_version": SCHEMA_VERSION,
            "has_ego": bool(feats["has_ego"]),
            "feature_names": list(frame_feature_names()),
        })),
    )


def load_features(path: str) -> Dict[str, Any]:
    """Read a .npz written by :func:`save_features`.

    Raises on a schema-version mismatch rather than silently mixing feature
    layouts across a training set -- a stale .npz with a different column order
    would train a model that scores nonsense and never complains.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if int(meta.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: feature schema_version {meta.get('schema_version')} != "
                f"{SCHEMA_VERSION}; re-extract this clip.")
        X = z["X"]
        if X.shape[1] != frame_feature_dim():
            raise ValueError(
                f"{path}: feature width {X.shape[1]} != {frame_feature_dim()}; "
                "the feature schema changed since extraction.")
        return {"X": X, "t": z["t"], "n_actors": z["n_actors"], "meta": meta,
                "has_ego": bool(meta.get("has_ego", False))}
