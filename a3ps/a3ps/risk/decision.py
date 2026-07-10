"""Decision layer: context-aware thresholds + a per-actor ALERT/BRAKE state machine.

The decision engine turns per-track collision probabilities (already computed
and cross-frame smoothed upstream, stored on ``track.prediction.collision_prob``)
into a small number of intervention :class:`~a3ps.common.schema.Event` objects.

Context flags and the active threshold
--------------------------------------
The danger threshold is *lowered* under adverse context so we intervene sooner:

    active_threshold = base_threshold - sum(reductions for active flags)
                       (floored at THRESHOLD_FLOOR = 0.45)

Reductions: ``crosswalk_ahead`` 0.10, ``intersection`` 0.10, ``dense_traffic``
0.05. The flags for this project are deliberately simple and honest:

  * ``dense_traffic`` is derived from perception -- it is set when >= 8 tracks
    are active in the frame.
  * ``crosswalk_ahead`` / ``intersection`` are NOT detected from imagery. They
    come from an optional per-clip annotation file
    ``dashboard/clips/<id>/context.json`` of the form::

        {"spans": [{"t0": 3.0, "t1": 9.0, "flag": "crosswalk_ahead"}]}

    that Member D writes by watching each demo clip once. Detecting these flags
    from imagery (crosswalk/intersection classifiers) is future work.

Per-actor state machine
-----------------------
Each actor climbs ``SAFE -> ALERT -> BRAKE`` and each forward transition emits
exactly one event, so a single near-miss yields one ``ALERT`` and one
``VIRTUAL_BRAKE`` -- not one per frame:

  * ``SAFE  -> ALERT``: prob >= (active_threshold - 0.15) for 3 consecutive frames.
  * ``ALERT -> BRAKE``: prob >= active_threshold          for 3 consecutive frames.

``BRAKE`` is terminal for the incident. An actor only re-arms (back to ``SAFE``,
able to fire again) once its probability has fallen below ``caution_threshold``
*and* a 2 s per-actor cooldown has elapsed since its last event -- this stops a
jittery track from cycling out extra events.

A ``THRESHOLD_LOWERED`` event is emitted on the rising edge of an annotation
context span (crosswalk_ahead / intersection), i.e. when a span begins.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from a3ps.common.schema import Event, TrackState

# Threshold reductions per active context flag, and the hard floor.
CONTEXT_REDUCTIONS = {
    "crosswalk_ahead": 0.10,
    "intersection": 0.10,
    "dense_traffic": 0.05,
}
THRESHOLD_FLOOR = 0.45
DENSE_TRAFFIC_MIN_TRACKS = 8

# Flags sourced from the per-clip annotation file (as opposed to derived from
# perception). Their onset is what triggers a THRESHOLD_LOWERED event.
SPAN_FLAGS = ("crosswalk_ahead", "intersection")


# ---------------------------------------------------------------------------
# context flags
# ---------------------------------------------------------------------------

def load_context_spans(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load context spans from ``context.json``; return [] if absent/empty.

    Each span is ``{"t0": float, "t1": float, "flag": str}``. Malformed or
    missing files degrade gracefully to no spans (context is optional).
    """
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    spans = []
    for s in (data or {}).get("spans", []) or []:
        try:
            spans.append({
                "t0": float(s["t0"]),
                "t1": float(s["t1"]),
                "flag": str(s["flag"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return spans


def active_context_flags(
    t: float,
    n_tracks: int,
    spans: Sequence[Dict[str, Any]],
) -> List[str]:
    """Flags active at time ``t``: annotation spans covering ``t`` plus, when
    the frame is crowded, ``dense_traffic``. Order-preserving and de-duplicated.
    """
    flags: List[str] = []
    for s in spans or []:
        if s["t0"] <= t < s["t1"]:
            flags.append(s["flag"])
    if n_tracks >= DENSE_TRAFFIC_MIN_TRACKS:
        flags.append("dense_traffic")
    return list(dict.fromkeys(flags))


# ---------------------------------------------------------------------------
# per-actor state
# ---------------------------------------------------------------------------

class _ActorState:
    """Mutable per-actor state for the ALERT/BRAKE machine."""

    __slots__ = ("state", "alert_streak", "brake_streak", "last_emit_t", "last_seen_t")

    def __init__(self) -> None:
        self.state = "SAFE"          # SAFE | ALERT | BRAKE
        self.alert_streak = 0
        self.brake_streak = 0
        self.last_emit_t = -1e9
        self.last_seen_t = 0.0


class DecisionEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.base_threshold = float(config.get("base_threshold", 0.75))
        self.caution_threshold = float(config.get("caution_threshold", 0.4))
        self.alert_margin = float(config.get("alert_margin", 0.15))
        self.frames_to_confirm = int(config.get("frames_to_confirm", 3))
        self.cooldown_s = float(config.get("event_cooldown_s", 2.0))
        self.floor = float(config.get("threshold_floor", THRESHOLD_FLOOR))
        # Allow config to override reductions, else use the module defaults.
        self.reductions = dict(CONTEXT_REDUCTIONS)
        self.reductions.update(config.get("context_reductions", {}) or {})
        # Prune actor state this many seconds after it was last seen.
        self._prune_after_s = max(3.0, 2.0 * self.cooldown_s)

        self._actors: Dict[int, _ActorState] = {}
        self._prev_flags: set = set()
        self._counter = 0

    # -- thresholds ---------------------------------------------------------

    def active_threshold(self, flags: Sequence[str]) -> float:
        """base_threshold minus reductions for active flags, floored."""
        reduction = sum(self.reductions.get(f, 0.0) for f in set(flags))
        return max(self.floor, self.base_threshold - reduction)

    def risk_level(self, prob: float, active_threshold: float) -> str:
        if prob >= active_threshold:
            return "danger"
        if prob >= self.caution_threshold:
            return "caution"
        return "safe"

    # -- main entry ---------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t: float,
        tracks: List[TrackState],
        context_flags: Optional[Sequence[str]] = None,
    ) -> List[Event]:
        """Advance the decision state for one frame; return any new events.

        Sets ``track.risk_level`` on every track and emits ALERT / VIRTUAL_BRAKE
        events per the state machine, plus a THRESHOLD_LOWERED event when an
        annotation context span begins.
        """
        flags = list(dict.fromkeys(context_flags or []))
        thr = self.active_threshold(flags)
        events: List[Event] = []

        # THRESHOLD_LOWERED on the rising edge of an annotation span.
        newly_active = [
            f for f in flags
            if f in SPAN_FLAGS and f not in self._prev_flags
        ]
        if newly_active:
            events.append(self._make_scene_event(frame_idx, t, "THRESHOLD_LOWERED", thr))
        self._prev_flags = set(flags)

        alert_level = max(0.0, thr - self.alert_margin)
        for tr in tracks:
            prob = self._prob_of(tr)
            tr.risk_level = self.risk_level(prob, thr)

            st = self._actors.get(tr.id)
            if st is None:
                st = self._actors[tr.id] = _ActorState()
            st.last_seen_t = t

            # Consecutive-frame confirmation counters.
            st.alert_streak = st.alert_streak + 1 if prob >= alert_level else 0
            st.brake_streak = st.brake_streak + 1 if prob >= thr else 0

            ev = self._advance(st, tr, frame_idx, t, prob, thr)
            if ev is not None:
                events.append(ev)

        self._prune(t)
        return events

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _prob_of(tr: TrackState) -> float:
        pred = tr.prediction
        if pred is None or pred.collision_prob is None:
            return 0.0
        return float(pred.collision_prob)

    def _advance(
        self,
        st: _ActorState,
        tr: TrackState,
        frame_idx: int,
        t: float,
        prob: float,
        thr: float,
    ) -> Optional[Event]:
        # Re-arm once the incident is clearly over and the cooldown has passed.
        if st.state != "SAFE":
            if prob < self.caution_threshold and (t - st.last_emit_t) >= self.cooldown_s:
                st.state = "SAFE"
                st.alert_streak = 0
                st.brake_streak = 0

        if st.state == "SAFE":
            if st.alert_streak >= self.frames_to_confirm:
                st.state = "ALERT"
                st.last_emit_t = t
                return self._make_actor_event(tr, frame_idx, t, "ALERT", prob, thr)
        elif st.state == "ALERT":
            if st.brake_streak >= self.frames_to_confirm:
                st.state = "BRAKE"
                st.last_emit_t = t
                return self._make_actor_event(tr, frame_idx, t, "VIRTUAL_BRAKE", prob, thr)
        # BRAKE is terminal for the incident: no further events.
        return None

    def _make_actor_event(
        self, tr: TrackState, frame_idx: int, t: float, etype: str, prob: float, thr: float
    ) -> Event:
        self._counter += 1
        pred = tr.prediction
        return Event(
            event_id=self._counter,
            frame_idx=frame_idx,
            t=round(t, 3),
            type=etype,
            actor_id=tr.id,
            actor_cls=tr.cls,
            collision_prob=round(prob, 4),
            threshold=round(thr, 4),
            ttc_s=(None if pred is None else pred.ttc_s),
        )

    def _make_scene_event(
        self, frame_idx: int, t: float, etype: str, thr: float
    ) -> Event:
        """A scene-level (not actor-specific) event, e.g. THRESHOLD_LOWERED."""
        self._counter += 1
        return Event(
            event_id=self._counter,
            frame_idx=frame_idx,
            t=round(t, 3),
            type=etype,
            actor_id=-1,
            actor_cls="scene",
            collision_prob=0.0,
            threshold=round(thr, 4),
            ttc_s=None,
        )

    def _prune(self, t: float) -> None:
        stale = [aid for aid, st in self._actors.items()
                 if (t - st.last_seen_t) > self._prune_after_s]
        for aid in stale:
            del self._actors[aid]
