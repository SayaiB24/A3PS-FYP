#!/usr/bin/env python
"""Compare BoT-SORT association quality at two frame rates before committing.

Decimating to 10 Hz cuts the full 1,500-clip feature-extraction pass from ~9-19 h
to ~1-2 h. The risk is that BoT-SORT's association degrades: at 10 Hz a car moves
3x further between frames, IoU overlap between consecutive detections shrinks, and
one physical object can fragment into several track ids. Fragmentation is
poisonous for this pipeline specifically, because the trajectory buffer needs
~2 s of continuous history per track before it will forecast at all -- so a
fragmented track produces no features, not merely noisier ones.

This script measures that directly, on a handful of clips, at both rates:

    python scripts/check_assoc_rate.py --index data/nexar/index.csv \\
        --split eval --limit 10 --rates 30,10 --out eval/assoc_rate_check.md

GPU laptop only -- it runs real perception. Expect roughly
``limit * window_s * (sum of rates)`` frames of inference in total; at 10 clips,
13 s and rates 30+10 that is ~5.2 k frames (a few minutes on a GPU).

Reported per rate:

* ``n_tracks``          -- unique track ids seen. Inflated by fragmentation.
* ``mean_track_s``      -- mean lifetime of a track in seconds.
* ``frac_tracks_ge_2s`` -- fraction of tracks living long enough to be
  forecastable. **This is the number to read**: it is the fraction that actually
  reaches the pipeline's downstream stages.
* ``mean_actors``       -- mean simultaneously-tracked actors per frame. A big
  drop means detections are being lost, not just re-associated.
* ``switch_rate``       -- new track ids per second. Directly comparable across
  rates and clip lengths.

Decision rule: 10 Hz is safe if ``frac_tracks_ge_2s`` and ``mean_actors`` hold up
within a few percent of 30 Hz. If ``frac_tracks_ge_2s`` drops materially, use
15 Hz or 30 Hz and pay the extra GPU hours -- fragmenting the tracks would cost
more than the compute saved.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_features import load_rows, window_for  # noqa: E402

MIN_FORECASTABLE_S = 2.0        # matches configs/default.yaml history_s


def track_stats(per_frame_ids, times):
    """Reduce per-frame track-id sets into association-quality numbers."""
    first_seen, last_seen, counts = {}, {}, {}
    for ids, t in zip(per_frame_ids, times):
        for tid in ids:
            first_seen.setdefault(tid, t)
            last_seen[tid] = t
            counts[tid] = counts.get(tid, 0) + 1

    if not first_seen:
        return {"n_tracks": 0, "mean_track_s": 0.0, "frac_tracks_ge_2s": 0.0,
                "mean_actors": 0.0, "switch_rate": 0.0, "span_s": 0.0}

    lifetimes = [last_seen[tid] - first_seen[tid] for tid in first_seen]
    span = (max(times) - min(times)) if len(times) > 1 else 0.0
    return {
        "n_tracks": len(first_seen),
        "mean_track_s": sum(lifetimes) / len(lifetimes),
        "frac_tracks_ge_2s": sum(1 for v in lifetimes if v >= MIN_FORECASTABLE_S)
                             / len(lifetimes),
        "mean_actors": sum(len(s) for s in per_frame_ids) / max(len(per_frame_ids), 1),
        "switch_rate": (len(first_seen) / span) if span > 0 else 0.0,
        "span_s": span,
    }


def run_clip(pipeline, config, video, start_s, end_s, rate_hz):
    """Track one clip at one rate; return (per_frame_id_sets, times)."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Fresh tracker state per (clip, rate): ids and the GMC reference frame must
    # not leak between runs, or the second rate inherits the first's tracks.
    pipeline.tracker.buffer = type(pipeline.tracker.buffer)(config)
    if hasattr(pipeline.tracker.model, "predictor"):
        predictor = pipeline.tracker.model.predictor
        for tr in getattr(predictor, "trackers", None) or []:
            tr.reset()

    period = 1.0 / float(rate_hz) if rate_hz else 0.0
    ids_per_frame, times = [], []
    idx, next_t = 0, None
    while True:
        if not cap.grab():
            break
        t = idx / fps
        idx += 1
        if start_s is not None and t < start_s:
            continue
        if end_s is not None and t > end_s:
            break
        if period and next_t is not None and t < next_t - 1e-9:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        next_t = t + period
        tracks = pipeline.tracker.update(frame, idx, t)
        ids_per_frame.append({int(tr.id) for tr in tracks})
        times.append(t)
    cap.release()
    return ids_per_frame, times


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--split", default="eval")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--rates", default="30,10",
                   help="Comma-separated frame rates to compare (Hz).")
    p.add_argument("--window-s", type=float, default=13.0)
    p.add_argument("--tail-s", type=float, default=1.0)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--videos-root", default=None)
    p.add_argument("--out", default="eval/assoc_rate_check.md")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    if len(rates) < 2:
        raise SystemExit("--rates needs at least two values, e.g. 30,10")

    rows = load_rows(args.index, args.split)[:args.limit]
    if not rows:
        raise SystemExit(f"no clips with split={args.split} in {args.index}")

    from a3ps.pipeline import Pipeline, load_config
    config = load_config(args.config)
    # Tracking only -- no forecaster, no risk engine. This measures association.
    pipeline = Pipeline(config, forecaster=None, risk_engine=None, explainer=None)
    print(f"device: {pipeline.tracker.device}  half: {pipeline.tracker.use_half}")
    if pipeline.tracker.device != "cuda":
        print("!! running on CPU -- this will take ~30x longer than on the GPU "
              "laptop. The RATIOS between rates stay valid, the wall time does not.")

    videos_root = args.videos_root or os.path.dirname(os.path.abspath(args.index))
    per_rate = {r: [] for r in rates}
    timing = {r: 0.0 for r in rates}

    for i, row in enumerate(rows, 1):
        video = os.path.normpath(os.path.join(videos_root, *row["path"].split("/")))
        if not os.path.isfile(video):
            print(f"  [{i}/{len(rows)}] {row['clip_id']}: video not found -- skipped")
            continue
        start_s, end_s = window_for(row, args.window_s, args.tail_s)
        line = [f"  [{i}/{len(rows)}] {row['clip_id']}"]
        for r in rates:
            t0 = time.perf_counter()
            ids, times = run_clip(pipeline, config, video, start_s, end_s, r)
            timing[r] += time.perf_counter() - t0
            st = track_stats(ids, times)
            st["clip_id"] = row["clip_id"]
            per_rate[r].append(st)
            line.append(f"{r:g}Hz: {st['n_tracks']} ids, "
                        f"{st['frac_tracks_ge_2s']:.0%} >=2s")
        print("  ".join(line))

    def agg(key, r):
        vals = [s[key] for s in per_rate[r]]
        return sum(vals) / len(vals) if vals else float("nan")

    lines = [
        "# BoT-SORT association quality vs frame rate",
        "",
        f"- clips: **{len(per_rate[rates[0]])}** from split `{args.split}`",
        f"- window: {args.window_s}s (+{args.tail_s}s tail after the event)",
        f"- device: `{pipeline.tracker.device}`",
        "",
        "| rate | tracks/clip | mean track (s) | **frac >=2s** | actors/frame "
        "| switches/s | inference (s/clip) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rates:
        n = max(len(per_rate[r]), 1)
        lines.append(
            f"| {r:g} Hz | {agg('n_tracks', r):.1f} | {agg('mean_track_s', r):.2f} "
            f"| **{agg('frac_tracks_ge_2s', r):.3f}** | {agg('mean_actors', r):.2f} "
            f"| {agg('switch_rate', r):.2f} | {timing[r] / n:.1f} |")

    base, cmp_ = rates[0], rates[-1]
    b, c = agg("frac_tracks_ge_2s", base), agg("frac_tracks_ge_2s", cmp_)
    rel = ((c - b) / b * 100.0) if b else float("nan")
    ab, ac = agg("mean_actors", base), agg("mean_actors", cmp_)
    arel = ((ac - ab) / ab * 100.0) if ab else float("nan")
    speedup = (timing[base] / timing[cmp_]) if timing[cmp_] else float("nan")

    lines += [
        "",
        f"## {cmp_:g} Hz vs {base:g} Hz",
        "",
        f"- forecastable tracks (>=2 s): **{rel:+.1f}%**",
        f"- actors per frame: **{arel:+.1f}%**",
        f"- inference speed-up: **{speedup:.2f}x**",
        "",
        "`frac >=2s` is the number that decides this: the trajectory buffer needs "
        f"{MIN_FORECASTABLE_S:.0f}s of continuous history before it forecasts at "
        "all, so tracks shorter than that contribute no features rather than "
        "noisier ones.",
        "",
    ]
    if rel < -10.0 or arel < -10.0:
        lines.append(
            f"**Verdict: do NOT decimate to {cmp_:g} Hz.** Association degrades "
            "more than 10%, which costs more in lost tracks than the "
            f"{speedup:.1f}x compute saving is worth. Use an intermediate rate.")
    elif rel < -3.0 or arel < -3.0:
        lines.append(
            f"**Verdict: borderline.** {cmp_:g} Hz loses a few percent of "
            "forecastable tracks. Acceptable if the GPU hours matter; try an "
            "intermediate rate (15 Hz) first.")
    else:
        lines.append(
            f"**Verdict: {cmp_:g} Hz is safe.** Association holds within a few "
            f"percent while inference is {speedup:.1f}x faster.")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\n" + report)
    print(f"wrote {args.out}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({str(r): per_rate[r] for r in rates}, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
