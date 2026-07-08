"""Core data structures for the a3ps traffic-safety pipeline.

Every dataclass here provides ``to_dict``/``from_dict`` so the whole object
graph can be serialized to a single ``events.json`` (or full ClipResult JSON)
and reloaded losslessly. BEV (bird's-eye-view) fields are Optional and MUST
serialize as ``null`` when absent rather than being dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

Number = float
BBox = List[float]          # [x1, y1, x2, y2]
Point = List[float]         # [x, y]
Polygon = List[Point]       # [[x, y], ...]


def _round_point(p: Optional[Point]) -> Optional[Point]:
    if p is None:
        return None
    return [float(p[0]), float(p[1])]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A forecast for a single track over a fixed horizon."""

    horizon_s: float
    dt: float
    mean_img: List[Point]                     # predicted image-space centroids
    mean_bev: Optional[List[Point]] = None    # predicted BEV centroids
    std_bev: Optional[List[Point]] = None     # per-step BEV std (x, y)
    collision_prob: Optional[float] = None
    ttc_s: Optional[float] = None             # time-to-collision seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_s": self.horizon_s,
            "dt": self.dt,
            "mean_img": [[float(x), float(y)] for x, y in self.mean_img],
            "mean_bev": (
                None if self.mean_bev is None
                else [[float(x), float(y)] for x, y in self.mean_bev]
            ),
            "std_bev": (
                None if self.std_bev is None
                else [[float(x), float(y)] for x, y in self.std_bev]
            ),
            "collision_prob": self.collision_prob,
            "ttc_s": self.ttc_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prediction":
        return cls(
            horizon_s=float(d["horizon_s"]),
            dt=float(d["dt"]),
            mean_img=[[float(x), float(y)] for x, y in d.get("mean_img", [])],
            mean_bev=(
                None if d.get("mean_bev") is None
                else [[float(x), float(y)] for x, y in d["mean_bev"]]
            ),
            std_bev=(
                None if d.get("std_bev") is None
                else [[float(x), float(y)] for x, y in d["std_bev"]]
            ),
            collision_prob=d.get("collision_prob"),
            ttc_s=d.get("ttc_s"),
        )


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """State of a single tracked actor at one frame."""

    id: int
    cls: str
    bbox: BBox                                       # [x1, y1, x2, y2]
    centroid_img: Point                              # image-space centroid
    centroid_bev: Optional[Point] = None             # BEV centroid
    velocity_bev: Optional[Point] = None             # BEV velocity (m/s)
    mask_poly: Optional[Polygon] = None              # simplified seg polygon
    history_img: List[Point] = field(default_factory=list)
    prediction: Optional[Prediction] = None
    risk_level: str = "safe"                          # safe|caution|danger

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cls": self.cls,
            "bbox": [float(v) for v in self.bbox],
            "centroid_img": _round_point(self.centroid_img),
            "centroid_bev": _round_point(self.centroid_bev),
            "velocity_bev": _round_point(self.velocity_bev),
            "mask_poly": (
                None if self.mask_poly is None
                else [[float(x), float(y)] for x, y in self.mask_poly]
            ),
            "history_img": [[float(x), float(y)] for x, y in self.history_img],
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackState":
        return cls(
            id=int(d["id"]),
            cls=str(d["cls"]),
            bbox=[float(v) for v in d["bbox"]],
            centroid_img=_round_point(d["centroid_img"]),
            centroid_bev=_round_point(d.get("centroid_bev")),
            velocity_bev=_round_point(d.get("velocity_bev")),
            mask_poly=(
                None if d.get("mask_poly") is None
                else [[float(x), float(y)] for x, y in d["mask_poly"]]
            ),
            history_img=[[float(x), float(y)] for x, y in d.get("history_img", [])],
            prediction=(
                None if d.get("prediction") is None
                else Prediction.from_dict(d["prediction"])
            ),
            risk_level=str(d.get("risk_level", "safe")),
        )


# ---------------------------------------------------------------------------
# FrameRecord
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """All tracked state for a single frame."""

    frame_idx: int
    t: float                                         # timestamp seconds
    tracks: List[TrackState] = field(default_factory=list)
    ego: Optional[Dict[str, Any]] = None             # ego speed/yaw etc.
    context: Optional[Dict[str, Any]] = None         # scene context flags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t": self.t,
            "tracks": [t.to_dict() for t in self.tracks],
            "ego": self.ego,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_idx=int(d["frame_idx"]),
            t=float(d["t"]),
            tracks=[TrackState.from_dict(t) for t in d.get("tracks", [])],
            ego=d.get("ego"),
            context=d.get("context"),
        )


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """An intervention / alert event emitted by the risk decision layer."""

    event_id: int
    frame_idx: int
    t: float
    type: str                                        # ALERT | VIRTUAL_BRAKE | THRESHOLD_LOWERED
    actor_id: int
    actor_cls: str
    collision_prob: float
    threshold: float
    ttc_s: Optional[float] = None
    explanation_template: Optional[str] = None
    explanation_llm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame_idx": self.frame_idx,
            "t": self.t,
            "type": self.type,
            "actor_id": self.actor_id,
            "actor_cls": self.actor_cls,
            "collision_prob": self.collision_prob,
            "threshold": self.threshold,
            "ttc_s": self.ttc_s,
            "explanation_template": self.explanation_template,
            "explanation_llm": self.explanation_llm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            event_id=int(d["event_id"]),
            frame_idx=int(d["frame_idx"]),
            t=float(d["t"]),
            type=str(d["type"]),
            actor_id=int(d["actor_id"]),
            actor_cls=str(d["actor_cls"]),
            collision_prob=float(d["collision_prob"]),
            threshold=float(d["threshold"]),
            ttc_s=d.get("ttc_s"),
            explanation_template=d.get("explanation_template"),
            explanation_llm=d.get("explanation_llm"),
        )


# ---------------------------------------------------------------------------
# ClipResult
# ---------------------------------------------------------------------------

@dataclass
class ClipResult:
    """The full result of running the pipeline on one clip."""

    meta: Dict[str, Any] = field(default_factory=dict)
    frames: List[FrameRecord] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "frames": [f.to_dict() for f in self.frames],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClipResult":
        return cls(
            meta=d.get("meta", {}),
            frames=[FrameRecord.from_dict(f) for f in d.get("frames", [])],
            events=[Event.from_dict(e) for e in d.get("events", [])],
        )

    def save_json(self, path: str) -> None:
        """Write the ClipResult to ``path`` as UTF-8 JSON.

        Layout::

            {
              "meta":   { ... clip metadata ... },
              "frames": [ { frame_idx, t, tracks, ego, context }, ... ],
              "events": [ { event_id, frame_idx, t, type, ... }, ... ]
            }

        Optional BEV fields (``centroid_bev``, ``velocity_bev``,
        ``mean_bev``, ``std_bev``) serialize as ``null`` when absent.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "ClipResult":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
