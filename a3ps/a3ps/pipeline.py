"""Orchestrator: video -> annotated.mp4 + raw.mp4 + events.json + meta.json.

``Pipeline(config)`` wires the tracker (Phase 2) and leaves three optional,
injectable hooks for later phases:

    forecaster(track, ctx)   -> Optional[Prediction]  # Phase 3 (per track)
    risk_engine(tracks, ctx) -> List[Event]           # Phase 4 (per frame; sets risk_level)
    explainer(event, ctx)    -> Optional[str]          # Phase 5

Each hook is called only if it was supplied; otherwise that stage is skipped.
The risk engine is called once per frame with the whole track list (it needs
the full frame to score context, e.g. dense traffic, and to run a per-actor
state machine) and returns the events emitted this frame.

``ctx`` is a dict carrying per-frame state (frame_idx, t, fps, config, ego,
context, buffer), so Phases 3-5 can plug in without editing this file again.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional

import yaml

from a3ps.common.geometry import GroundPlane, load_ground_override
from a3ps.common.schema import ClipResult, Event, FrameRecord
from a3ps.explain.templates import explain as template_explain
from a3ps.risk.decision import active_context_flags, load_context_spans
from a3ps.tracking.tracker import Tracker

# BGR colors keyed by risk level (green -> amber -> red).
RISK_COLORS = {
    "safe": (0, 180, 0),
    "caution": (0, 180, 255),
    "danger": (0, 0, 255),
}

# How long the red "virtual brake" border stays lit after a VIRTUAL_BRAKE event.
BRAKE_FLASH_S = 0.5

# Hook type aliases (documentation only).
Forecaster = Callable[[Any, Dict[str, Any]], Any]
RiskEngine = Callable[[List[Any], Dict[str, Any]], List[Event]]
Explainer = Callable[[Event, Dict[str, Any]], Optional[str]]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Note surfaced verbatim in every meta.json under "per_stage_ms_note" -- see
# build_per_stage_ms() below for why "perception" is reported as 0.0.
PER_STAGE_MS_NOTE = (
    "perception (YOLO detection + segmentation) and tracking (BoT-SORT "
    "association) run as a SINGLE model.track() call in this architecture "
    "(a3ps/tracking/tracker.py) -- there is no separate perception pass to "
    "time independently without doubling inference cost. All of that "
    "combined cost is attributed to 'tracking' below; 'perception' is "
    "reported as 0.0 by convention so the four-stage schema stays intact "
    "and the end-to-end sum is not double-counted."
)


def build_per_stage_ms(timings: Dict[str, float], n_frames: int) -> Dict[str, float]:
    """Accumulated per-stage ms (from Pipeline.run's ``timings`` dict) -> the
    paper's Table IV per-frame-average schema: perception, tracking,
    forecasting, risk_decision (each a float, ms/frame).

    Pure function of the accumulated totals (no cv2/YOLO dependency), so it's
    unit-testable without a real video or model. See PER_STAGE_MS_NOTE for why
    "perception" is always 0.0 here rather than a separately-measured value.
    """
    def per_frame(key: str) -> float:
        return round(timings.get(key, 0.0) / n_frames, 2) if n_frames else 0.0

    return {
        "perception": 0.0,
        "tracking": per_frame("track"),
        "forecasting": per_frame("forecast"),
        "risk_decision": per_frame("risk"),
    }


class Pipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        forecaster: Optional[Forecaster] = None,
        risk_engine: Optional[RiskEngine] = None,
        explainer: Optional[Explainer] = None,
    ):
        self.config = config
        self.tracker = Tracker(config)
        # Optional injected hooks (None => stage skipped).
        self.forecaster = forecaster
        self.risk_engine = risk_engine
        self.explainer = explainer
        self.forecast_space = config.get("forecast_space", "img")
        self.ground: Optional[GroundPlane] = None  # built in run() once size known
        self.predict_hz = float(config.get("predict_hz", 5))

    # -- ego corridor -------------------------------------------------------

    def _ego_corridor(self, width: int, height: int) -> List[List[float]]:
        c = self.config.get("ego_corridor", {})
        by = c.get("bottom_y_frac", 1.0) * height
        ty = c.get("top_y_frac", 0.60) * height
        bl = c.get("bottom_left_frac", 0.39) * width
        br = c.get("bottom_right_frac", 0.61) * width
        tl = c.get("top_left_frac", 0.48) * width
        tr = c.get("top_right_frac", 0.52) * width
        return [[round(bl, 1), round(by, 1)], [round(br, 1), round(by, 1)],
                [round(tr, 1), round(ty, 1)], [round(tl, 1), round(ty, 1)]]

    def _fill_bev(self, tracks) -> None:
        """Fill centroid_bev and velocity_bev (BEV finite diff) for every track."""
        if self.ground is None:
            return
        dt = 1.0 / self.predict_hz
        for tr in tracks:
            c = self.ground.img_to_bev([tr.centroid_img])[0]
            tr.centroid_bev = [round(c[0], 3), round(c[1], 3)]
            if tr.history_img and len(tr.history_img) >= 2:
                hb = self.ground.img_to_bev(tr.history_img[-2:])
                tr.velocity_bev = [round((hb[-1][0] - hb[-2][0]) / dt, 3),
                                   round((hb[-1][1] - hb[-2][1]) / dt, 3)]
            else:
                tr.velocity_bev = None

    # -- rendering ----------------------------------------------------------

    def _draw_frame(self, frame, record: FrameRecord, ego_poly, brake_flash: bool = False):
        import cv2
        import numpy as np

        overlay = frame.copy()
        for tr in record.tracks:
            color = RISK_COLORS.get(tr.risk_level, RISK_COLORS["safe"])
            if tr.mask_poly:
                pts = np.asarray(tr.mask_poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Ego corridor outline.
        ego_pts = np.asarray(ego_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [ego_pts], True, (0, 255, 255), 2, cv2.LINE_AA)

        for tr in record.tracks:
            color = RISK_COLORS.get(tr.risk_level, RISK_COLORS["safe"])
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Trail from history_img (past).
            if tr.history_img and len(tr.history_img) >= 2:
                trail = np.asarray(tr.history_img, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [trail], False, color, 2, cv2.LINE_AA)
            # Predicted path (future): dotted, fading with step index.
            self._draw_prediction(frame, tr, color)
            cv2.putText(frame, f"{tr.cls} {tr.id}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Red border flash: a virtual-brake intervention just fired.
        if brake_flash:
            h, w = frame.shape[:2]
            thickness = max(6, int(round(min(h, w) * 0.02)))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, "VIRTUAL BRAKE", (thickness + 8, thickness + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_prediction(frame, tr, color):
        import cv2

        pred = tr.prediction
        if pred is None or not pred.mean_img:
            return
        pts = pred.mean_img
        n = len(pts)
        h, w = frame.shape[:2]
        for i, (px, py) in enumerate(pts):
            ix, iy = int(px), int(py)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            # Fade from full color at step 0 toward faint at the horizon.
            f = 1.0 - 0.8 * (i / max(1, n - 1))
            c = tuple(int(ch * f) for ch in color)
            r = max(1, int(round(3 * f)))
            # "Dotted": draw a dot every other step.
            if i % 2 == 0:
                cv2.circle(frame, (ix, iy), r, c, -1, cv2.LINE_AA)

    # -- main loop ----------------------------------------------------------

    def run(self, video_path: str, out_dir: str, max_seconds: Optional[float] = None,
            render: bool = True) -> ClipResult:
        """Process a clip -> ClipResult (+ events.json/meta.json on disk).

        When ``render`` is False the annotated.mp4 / raw.mp4 outputs are skipped
        (no VideoWriter, no source copy) -- much faster for batch scoring over
        many clips (e.g. scripts/eval_anticipation.py). events.json and
        meta.json are still written so results can be cached and re-scored.
        """
        import cv2

        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(max_seconds * fps) if max_seconds else None

        override = load_ground_override(self.config.get("ground_plane_override"))
        self.ground = GroundPlane.from_config(self.config, width, height, override)

        ego_poly = self._ego_corridor(width, height)
        # BEV corridor too: the dashboard's reactive-ADAS marker prefers metric
        # distance over the pixel-distance fallback when this is present.
        ego = {"corridor_poly_img": ego_poly,
               "corridor_poly_bev": self.ground.img_to_bev(ego_poly),
               "speed_note": "unknown"}

        # Optional per-clip context annotation (crosswalk/intersection spans).
        context_spans = load_context_spans(os.path.join(out_dir, "context.json"))

        writer = None
        if render:
            annotated_path = os.path.join(out_dir, "annotated.mp4")
            writer = cv2.VideoWriter(
                annotated_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        result = ClipResult()
        timings = {"track": 0.0, "forecast": 0.0, "risk": 0.0, "render": 0.0}
        n_frames = 0
        last_brake_t = -1e9   # time of the most recent VIRTUAL_BRAKE, for the flash

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and idx >= max_frames:
                break
            t = idx / fps

            # --- tracking (also decimates history into the buffer) ---
            t0 = time.perf_counter()
            tracks = self.tracker.update(frame, idx, t)
            # Project every track to BEV (centroid + finite-difference velocity).
            self._fill_bev(tracks)
            timings["track"] += (time.perf_counter() - t0) * 1000.0

            flags = active_context_flags(t, len(tracks), context_spans)
            context = {"flags": flags,
                       "base_threshold": self.config.get("base_threshold", 0.75)}
            ctx = {
                "frame_idx": idx, "t": t, "fps": fps,
                "config": self.config, "ego": ego, "context": context,
                "buffer": self.tracker.buffer,
                "ground": self.ground, "forecast_space": self.forecast_space,
            }

            # --- forecasting (optional) ---
            t0 = time.perf_counter()
            if self.forecaster is not None:
                for tr in tracks:
                    pred = self.forecaster(tr, ctx)
                    if pred is not None:
                        tr.prediction = pred
            timings["forecast"] += (time.perf_counter() - t0) * 1000.0

            # --- risk + explanation (optional) ---
            # The risk engine is per-frame: it scores every track's collision
            # probability, runs the per-actor state machine, and returns the
            # events emitted this frame (also setting each track's risk_level).
            t0 = time.perf_counter()
            frame_events: List[Event] = []
            if self.risk_engine is not None:
                track_by_id = {tr.id: tr for tr in tracks}
                for event in self.risk_engine(tracks, ctx) or []:
                    # Deterministic XAI explanation is always populated here.
                    event.explanation_template = template_explain(
                        event, track_by_id.get(event.actor_id), context)
                    # Optional richer enrichment (e.g. LLM) via an injected hook.
                    if self.explainer is not None:
                        expl = self.explainer(event, ctx)
                        if expl is not None:
                            event.explanation_llm = expl
                    frame_events.append(event)
                    if event.type == "VIRTUAL_BRAKE":
                        last_brake_t = t
            timings["risk"] += (time.perf_counter() - t0) * 1000.0

            record = FrameRecord(frame_idx=idx, t=round(t, 3), tracks=tracks,
                                 ego=ego, context=context)
            result.frames.append(record)
            result.events.extend(frame_events)

            # --- render (skipped entirely when render=False) ---
            if render:
                t0 = time.perf_counter()
                brake_flash = 0.0 <= (t - last_brake_t) <= BRAKE_FLASH_S
                self._draw_frame(frame, record, ego_poly, brake_flash=brake_flash)
                writer.write(frame)
                timings["render"] += (time.perf_counter() - t0) * 1000.0

            n_frames += 1
            idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # raw.mp4 (verbatim copy of the source) + metadata.
        if render:
            shutil.copyfile(video_path, os.path.join(out_dir, "raw.mp4"))
        result.meta = self._build_meta(video_path, fps, width, height, n_frames)

        # Per-stage latency (Table IV) computed and attached BEFORE the
        # meta.json write below, so it's actually persisted to disk --
        # previously this was computed after the write and only ever lived on
        # the in-memory `result` returned to the caller, so meta.json on disk
        # never contained timing data (the root cause of it "not existing").
        result.meta["per_stage_ms"] = build_per_stage_ms(timings, n_frames)
        result.meta["per_stage_ms_note"] = PER_STAGE_MS_NOTE
        result.meta["_timings_ms_per_frame"] = {
            k: round(v / n_frames, 1) if n_frames else 0.0 for k, v in timings.items()
        }

        result.save_json(os.path.join(out_dir, "events.json"))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(result.meta, fh, indent=2, ensure_ascii=False)

        return result

    def _build_meta(self, video_path, fps, width, height, n_frames):
        return {
            "clip_id": os.path.splitext(os.path.basename(video_path))[0],
            "source_video": "raw.mp4",
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "config": {
                "model": self.config.get("model_name", "yolov8s-seg.pt"),
                "tracker": self.config.get("tracker", "botsort.yaml"),
                "forecaster": "kalman_cv" if self.forecaster else None,
                "base_threshold": self.config.get("base_threshold", 0.75),
                "horizon_s": self.config.get("horizon_s", 4.0),
            },
            "stages": {
                "forecaster": self.forecaster is not None,
                "risk_engine": self.risk_engine is not None,
                "explainer": self.explainer is not None,
            },
        }


def run_pipeline(video_path: str, out_dir: str, config_path: str,
                 max_seconds: Optional[float] = None) -> ClipResult:
    return Pipeline(load_config(config_path)).run(video_path, out_dir, max_seconds)
