"""Deterministic explanation strings (CORE, the XAI layer).

Given an Event (and, for actor events, its Track + the frame context), produce a
short human-readable rationale with no LLM call. Grammar for an actor event::

    "<Action>: <actor class> ID-<id> <motion phrase> ego path in <ttc> s
     (P=<prob>, threshold <thr><threshold reason>)."

e.g.::

    "Collision alert: person ID-4 cutting into ego path in 1.2 s
     (P=0.81, threshold 0.65 — lowered due to crosswalk ahead)."

The motion phrase is inferred from the track's BEV velocity relative to the ego
corridor (which is centred on x=0, extending forward +y). The threshold reason
is appended only when the active threshold was lowered below the base threshold.
THRESHOLD_LOWERED is a scene-level event with its own sentence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from a3ps.common.schema import Event

# Verb fragment per event type (plugged into the actor grammar).
_ACTION = {
    "ALERT": "Collision alert",
    "VIRTUAL_BRAKE": "Virtual braking engaged",
}

# Human phrasing for known context flags.
_FLAG_REASON = {
    "crosswalk_ahead": "crosswalk ahead",
    "intersection": "intersection ahead",
    "dense_traffic": "heavy traffic",
    "school_zone": "school zone",
    "raining": "wet road",
    "night": "low light",
}

# Flags that lower the threshold (kept in sync with risk.decision.CONTEXT_REDUCTIONS).
_REDUCING_FLAGS = ("crosswalk_ahead", "intersection", "dense_traffic")

# BEV speed (m/s) below which an actor reads as "decelerating within" rather
# than moving in a particular direction.
_SLOW_SPEED = 0.5


def _motion_phrase(track) -> str:
    """Infer a motion phrase from the track's BEV velocity vs the corridor.

    BEV frame: +y is forward (away from ego), x is lateral (corridor centred on
    x=0). An actor approaching the ego moves in -y. Falls back to a sensible
    default when velocity is unavailable.
    """
    vel = getattr(track, "velocity_bev", None) if track is not None else None
    if not vel:
        return "trajectory crossing"
    vx, vy = float(vel[0]), float(vel[1])
    speed = math.hypot(vx, vy)
    if speed < _SLOW_SPEED:
        return "decelerating within"

    if abs(vy) >= abs(vx):
        # Longitudinal motion dominates.
        if vy < 0:
            return "approaching head-on toward"
        return "trajectory crossing"

    # Lateral motion dominates: "cutting into" if heading toward the centreline.
    cen = getattr(track, "centroid_bev", None) if track is not None else None
    x = float(cen[0]) if cen else 0.0
    if x != 0.0 and (vx * x) < 0.0:
        return "cutting into"
    return "trajectory crossing"


def _threshold_reason(event: Event, context: Optional[Dict[str, Any]]) -> str:
    """" - lowered due to <reasons>" when the threshold was reduced, else ""."""
    base = (context or {}).get("base_threshold")
    lowered = base is not None and event.threshold < float(base) - 1e-9
    if not lowered:
        return ""
    flags = (context or {}).get("flags") or []
    reasons = [_FLAG_REASON.get(f, f) for f in flags if f in _REDUCING_FLAGS]
    why = ", ".join(reasons) if reasons else "adverse context"
    return f" — lowered due to {why}"


def _ttc_clause(event: Event) -> str:
    if event.ttc_s is not None and event.ttc_s != float("inf"):
        return f"in {event.ttc_s:.1f} s"
    return "in unknown time"


def explain(
    event: Event,
    track: Any = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a deterministic one-line explanation for an event."""
    # Scene-level context event: no specific actor / TTC.
    if event.type == "THRESHOLD_LOWERED":
        flags = (context or {}).get("flags") or []
        named = next((f for f in flags if f in _REDUCING_FLAGS), None)
        flag_phrase = _FLAG_REASON.get(named, "adverse context") if named else "adverse context"
        return (
            f"Entering {flag_phrase}: risk threshold lowered to "
            f"{event.threshold:.2f}, easing speed."
        )

    action = _ACTION.get(event.type, event.type.replace("_", " ").capitalize())
    return (
        f"{action}: {event.actor_cls} ID-{event.actor_id} "
        f"{_motion_phrase(track)} ego path {_ttc_clause(event)} "
        f"(P={event.collision_prob:.2f}, threshold {event.threshold:.2f}"
        f"{_threshold_reason(event, context)})."
    )
