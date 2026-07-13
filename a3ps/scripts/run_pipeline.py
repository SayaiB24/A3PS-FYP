#!/usr/bin/env python
"""CLI: run the a3ps pipeline on a video.

    python scripts/run_pipeline.py --video x.mp4 --out dashboard/clips/x/
    python scripts/run_pipeline.py --video x.mp4 --out /tmp/x/ --max-seconds 5

Prints a per-stage timing summary (ms/frame) at the end.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import Prediction  # noqa: E402
from a3ps.forecasting.kalman_cv import KalmanCVForecaster  # noqa: E402
from a3ps.pipeline import Pipeline, load_config  # noqa: E402
from a3ps.risk.collision import (  # noqa: E402
    RiskSmoother,
    actor_radius_for,
    trajectory_collision_prob,
)
from a3ps.risk.decision import DecisionEngine  # noqa: E402


def make_forecaster_hook(config):
    """Adapt a Forecaster (Kalman-CV or Seq2Seq) to the Pipeline hook signature.

    For each track whose buffer has enough history (buffer.ready), forecast
    from its image-space history and return a Prediction. collision_prob is
    left None until the risk stage (Phase 4).

    ``config["forecaster"]`` selects the model: ``"kalman_cv"`` (default) or
    ``"seq2seq"`` (loads ``config["seq2seq_weights"]``, default
    ``notebooks/models/seq2seq_v1.pt``) -- used by scripts/run_ablations.py to
    ablate the forecaster choice. Both implement the same
    ``predict(history, dt, horizon_s) -> (means, stds)`` interface, so nothing
    else in this hook needs to change.
    """
    predict_hz = float(config.get("predict_hz", 5))
    dt = 1.0 / predict_hz
    horizon_s = float(config.get("horizon_s", 4.0))
    forecaster_name = str(config.get("forecaster", "kalman_cv")).lower()
    if forecaster_name == "seq2seq":
        from a3ps.forecasting.seq2seq import Seq2SeqForecaster
        weights = config.get("seq2seq_weights", "notebooks/models/seq2seq_v1.pt")
        forecaster = Seq2SeqForecaster(weights_path=weights)
    else:
        forecaster = KalmanCVForecaster()

    def hook(track, ctx):
        buf = ctx["buffer"]
        if not buf.ready(track.id):
            return None
        history_img = buf.history(track.id)
        ground = ctx.get("ground")
        space = ctx.get("forecast_space", "img")

        if space == "bev" and ground is not None:
            # Forecast in metric BEV space; back-project the mean to image for
            # rendering. std_bev is now genuine metres.
            history_bev = ground.img_to_bev(history_img)
            means_bev, stds_bev = forecaster.predict(history_bev, dt, horizon_s)
            if not means_bev:
                return None
            means_img = ground.bev_to_img(means_bev)
            return Prediction(
                horizon_s=horizon_s, dt=dt,
                mean_img=means_img, mean_bev=means_bev, std_bev=stds_bev,
                collision_prob=None,
            )

        # Image-space forecast; std_bev holds pixel std until BEV is used.
        means, stds = forecaster.predict(history_img, dt, horizon_s)
        if not means:
            return None
        return Prediction(
            horizon_s=horizon_s, dt=dt,
            mean_img=means, std_bev=stds, collision_prob=None,
        )

    return hook


def make_risk_hook(config):
    """Per-frame risk engine: score collision prob, smooth, run DecisionEngine.

    For each track with a forecast we integrate the collision probability over
    the predicted trajectory against the ego corridor (dilated by the actor's
    radius), smooth it across frames with an EMA (so a one-frame glitch cannot
    fire an event), and store it on ``track.prediction``. Then the
    :class:`DecisionEngine` turns those probabilities into ALERT / VIRTUAL_BRAKE
    / THRESHOLD_LOWERED events. Returns the events emitted this frame.

    ``config["ema_smoothing"]`` (default True) toggles the cross-frame EMA --
    used by scripts/run_ablations.py to ablate smoothing. When False, each
    frame's raw ``max_prob`` is used directly as ``collision_prob``.
    """
    engine = DecisionEngine(config)
    smoother = RiskSmoother(alpha=float(config.get("risk_ema_alpha", 0.4)))
    ema_smoothing = bool(config.get("ema_smoothing", True))
    corridor_cache: dict = {}

    def _corridor(ctx):
        """Ego corridor in the active forecast space (cached; it's constant)."""
        space = ctx.get("forecast_space", "img")
        if space not in corridor_cache:
            poly_img = ctx["ego"]["corridor_poly_img"]
            ground = ctx.get("ground")
            if space == "bev" and ground is not None:
                corridor_cache[space] = ground.img_to_bev(poly_img)
            else:
                corridor_cache[space] = poly_img
        return corridor_cache[space]

    def hook(tracks, ctx):
        space = ctx.get("forecast_space", "img")
        corridor = _corridor(ctx)
        dt = 1.0 / float(config.get("predict_hz", 5))

        # Dashboard reads context.active_threshold to draw the threshold line
        # and the "lowered due to X" note -- surface it every frame.
        flags = ctx["context"].get("flags", [])
        ctx["context"]["active_threshold"] = round(float(engine.active_threshold(flags)), 4)

        for tr in tracks:
            pred = tr.prediction
            if pred is None:
                continue
            means = pred.mean_bev if space == "bev" else pred.mean_img
            stds = pred.std_bev
            if not means or not stds:
                continue
            radius = actor_radius_for(tr.cls, config, space=space)
            max_prob, ttc = trajectory_collision_prob(
                (means, stds), corridor, radius, dt)
            smoothed = smoother.update(tr.id, max_prob) if ema_smoothing else max_prob
            pred.collision_prob = round(float(smoothed), 4)
            pred.ttc_s = ttc

        return engine.update(
            ctx["frame_idx"], ctx["t"], tracks,
            context_flags=flags,
        )

    return hook


def main() -> None:
    p = argparse.ArgumentParser(description="Run the a3ps traffic-safety pipeline.")
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--out", required=True, help="Output directory for this clip.")
    p.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        help="Path to config YAML.",
    )
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Process only the first N seconds (quick tests).")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        p.error(f"video not found: {args.video}")

    config = load_config(args.config)
    # Phases 3-4 wired in (forecaster + risk decision); explainer lands in Phase 5.
    pipeline = Pipeline(
        config,
        forecaster=make_forecaster_hook(config),
        risk_engine=make_risk_hook(config),
        explainer=None,
    )

    wall0 = time.perf_counter()
    result = pipeline.run(args.video, args.out, max_seconds=args.max_seconds)
    wall = time.perf_counter() - wall0

    n = len(result.frames)
    print(f"\nProcessed {n} frames, {len(result.events)} events -> {args.out}")
    print(f"Wall time: {wall:.1f}s ({(wall * 1000 / n):.1f} ms/frame overall)"
          if n else "Wall time: n/a")

    timings = result.meta.get("_timings_ms_per_frame", {})
    if timings:
        print("\nPer-stage timing (ms/frame):")
        for stage in ("track", "forecast", "risk", "render"):
            print(f"  {stage:9s} {timings.get(stage, 0.0):8.1f}")


if __name__ == "__main__":
    main()
