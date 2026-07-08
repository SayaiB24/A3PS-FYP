"""Decision layer: thresholds, dynamic context rules, intervention events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from a3ps.common.schema import Event, TrackState


class DecisionEngine:
    def __init__(
        self,
        base_threshold: float = 0.75,
        caution_threshold: float = 0.4,
    ):
        self.base_threshold = base_threshold
        self.caution_threshold = caution_threshold
        self._counter = 0

    def _effective_threshold(self, context: Optional[Dict[str, Any]]) -> float:
        """Lower the danger threshold under adverse context (rain, night)."""
        thr = self.base_threshold
        if context:
            if context.get("raining"):
                thr -= 0.1
            if context.get("night"):
                thr -= 0.05
        return max(self.caution_threshold, thr)

    def classify(self, prob: float, context: Optional[Dict[str, Any]]) -> str:
        thr = self._effective_threshold(context)
        if prob >= thr:
            return "danger"
        if prob >= self.caution_threshold:
            return "caution"
        return "safe"

    def evaluate(
        self,
        track: TrackState,
        frame_idx: int,
        t: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Event]:
        """Return an intervention Event if the track crosses the danger threshold."""
        pred = track.prediction
        if pred is None or pred.collision_prob is None:
            return None

        prob = pred.collision_prob
        level = self.classify(prob, context)
        track.risk_level = level
        if level != "danger":
            return None

        self._counter += 1
        base_thr = self._effective_threshold(context)
        # A lowered threshold (adverse context) is itself worth surfacing.
        lowered = base_thr < self.base_threshold - 1e-9
        event_type = "VIRTUAL_BRAKE" if track.cls == "person" else "ALERT"
        if lowered and level == "caution":
            event_type = "THRESHOLD_LOWERED"
        return Event(
            event_id=self._counter,
            frame_idx=frame_idx,
            t=t,
            type=event_type,
            actor_id=track.id,
            actor_cls=track.cls,
            collision_prob=prob,
            threshold=base_thr,
            ttc_s=pred.ttc_s,
        )
