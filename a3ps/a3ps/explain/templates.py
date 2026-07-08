"""Deterministic explanation strings (CORE).

Given an Event, produce a short, human-readable rationale with no LLM call.
Mirrors the style in the events.json spec, e.g.:

    "Virtual braking: pedestrian ID-4 trajectory crossing ego path in 1.2 s
     (P=0.81, threshold 0.65 due to crosswalk ahead)."
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from a3ps.common.schema import Event

_ACTION = {
    "VIRTUAL_BRAKE": "Virtual braking",
    "ALERT": "Alert",
    "THRESHOLD_LOWERED": "Caution threshold lowered",
}

# Human phrasing for known context flags.
_FLAG_REASON = {
    "crosswalk_ahead": "crosswalk ahead",
    "school_zone": "school zone",
    "raining": "wet road",
    "night": "low light",
}


def _reason_clause(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""
    flags = context.get("flags") or []
    reasons = [_FLAG_REASON.get(f, f) for f in flags]
    return f" due to {', '.join(reasons)}" if reasons else ""


def explain(event: Event, context: Optional[Dict[str, Any]] = None) -> str:
    """Return a deterministic one-line explanation for an event."""
    action = _ACTION.get(event.type, event.type.replace("_", " ").capitalize())
    ttc = (
        f"{event.ttc_s:.1f} s"
        if event.ttc_s is not None and event.ttc_s != float("inf")
        else "unknown time"
    )
    return (
        f"{action}: {event.actor_cls} ID-{event.actor_id} trajectory crossing "
        f"ego path in {ttc} "
        f"(P={event.collision_prob:.2f}, threshold {event.threshold:.2f}"
        f"{_reason_clause(context)})."
    )
