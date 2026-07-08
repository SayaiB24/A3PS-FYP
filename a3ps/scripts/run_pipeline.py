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


def make_forecaster_hook(config):
    """Adapt KalmanCVForecaster to the Pipeline hook signature.

    For each track whose buffer has enough history (buffer.ready), forecast
    from its image-space history and return a Prediction. collision_prob is
    left None until the risk stage (Phase 4).
    """
    predict_hz = float(config.get("predict_hz", 5))
    dt = 1.0 / predict_hz
    horizon_s = float(config.get("horizon_s", 4.0))
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
    # Phase 3 forecaster is wired in; risk/explainer land in Phases 4-5.
    pipeline = Pipeline(
        config,
        forecaster=make_forecaster_hook(config),
        risk_engine=None,
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
