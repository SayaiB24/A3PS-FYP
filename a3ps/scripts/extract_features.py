#!/usr/bin/env python
"""Build per-frame feature .npz files for the learned risk head (Phase IV).

Two modes, one output schema, so features from either source train the same head.

``--from-cache`` (CPU, no GPU, no video decode)
    Reads already-computed ``<clips-dir>/<clip_id>/events.json`` ClipResults and
    derives features from them. This is what runs on the CPU laptop against the
    120 cached eval clips. It CANNOT produce ego-motion features -- that pass
    never computed them -- so ego slots are zero-filled and ``has_ego=False`` is
    recorded in every .npz. Any model trained on these was blind to ego motion
    and must be reported that way.

        python scripts/extract_features.py --from-cache eval/anticipation \\
            --index eval/nexar_index_gpu.csv --split eval --out data/features/cache

``--from-video`` (GPU laptop)
    Runs perception + tracking + forecasting over a WINDOW of each clip and
    computes ego motion from masked static-scene optical flow alongside it.
    Windowed and decimated on purpose -- see ``--window-s`` / ``--rate-hz``: a
    full-rate pass over 1,500 clips is ~1.7 M frames, while a 13 s window at
    10 Hz is ~195 k, and the discarded frames are ones the anticipation loss
    masks out anyway.

        python scripts/extract_features.py --from-video \\
            --index data/nexar/index.csv --split train_traj \\
            --out data/features/train --window-s 13 --rate-hz 10

Positives are windowed around ``event_time_s`` (``[te - window + tail, te +
tail]``) so the window always contains ``alert_time_s`` and the run-up to impact.
Negatives get a deterministic pseudo-random window seeded by clip id, so re-runs
are reproducible without storing the choice.

Every .npz holds ``X`` (T x 112 float32), ``t``, ``n_actors`` and a JSON ``meta``
carrying the label, event/alert times, the feature schema version and
``has_ego``. A 53 MB events.json reduces to a few hundred KB, which is what makes
it practical to extract on the GPU laptop and iterate on the CPU one.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run_pipeline hooks

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.common.splits import load_freeze, normalize_clip_id, verify_freeze  # noqa: E402
from a3ps.features.extract import (  # noqa: E402
    SCHEMA_VERSION,
    clip_features,
    frame_feature_dim,
    save_features,
)

DEFAULT_WINDOW_S = 13.0
DEFAULT_RATE_HZ = 10.0
# Seconds of clip kept AFTER the annotated event. A little tail is useful for
# sanity-checking that the model's score actually peaks near the event; the loss
# masks these frames out of the positive term regardless.
DEFAULT_TAIL_S = 1.0


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def load_rows(index_path, split):
    with open(index_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if split and (r.get("split") or "") != split:
            continue

        def f(key):
            try:
                return float(r.get(key))
            except (TypeError, ValueError):
                return None

        out.append({
            "clip_id": r.get("clip_id", ""),
            "path": r.get("path", ""),
            "label": int(float(r.get("label") or 0)),
            "event_time_s": f("event_time_s"),
            "alert_time_s": f("alert_time_s"),
            "duration_s": f("duration_s"),
            "fps": f("fps"),
        })
    return out


def window_for(row, window_s, tail_s):
    """(start_s, end_s) for one clip, or (None, None) to take the whole clip.

    Positives are anchored on the event so the actionable window is always
    inside. Negatives are anchored on a hash of the clip id -- deterministic
    across runs and machines, and not correlated with anything about the clip.
    """
    if window_s is None or window_s <= 0:
        return None, None
    dur = row.get("duration_s") or 0.0
    te = row.get("event_time_s")

    if row["label"] == 1 and te is not None:
        end = te + tail_s
        start = end - window_s
    else:
        if dur <= window_s:
            return 0.0, dur or None
        h = hashlib.sha1(str(row["clip_id"]).encode()).digest()
        frac = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        start = frac * (dur - window_s)
        end = start + window_s
    start = max(0.0, start)
    if dur:
        end = min(end, dur)
    return start, end


def out_path(out_dir, clip_id):
    return os.path.join(out_dir, f"{normalize_clip_id(clip_id)}.npz")


def base_meta(row, window, source, extra=None):
    start, end = window
    meta = {
        "clip_id": normalize_clip_id(row["clip_id"]),
        "label": int(row["label"]),
        "event_time_s": row.get("event_time_s"),
        "alert_time_s": row.get("alert_time_s"),
        "window_start_s": start,
        "window_end_s": end,
        "source": source,
    }
    meta.update(extra or {})
    return meta


# ---------------------------------------------------------------------------
# mode: from cached ClipResults (CPU)
# ---------------------------------------------------------------------------

def slice_clip_result(clip, start_s, end_s, rate_hz=None):
    """Restrict a ClipResult's frames to a time window and optional rate.

    Returns a NEW ClipResult sharing frame objects (no deep copy -- these are
    read-only here and the originals are 50 MB apiece).
    """
    frames = clip.frames
    if start_s is not None or end_s is not None:
        lo = -1e18 if start_s is None else float(start_s)
        hi = 1e18 if end_s is None else float(end_s)
        frames = [f for f in frames if lo <= f.t <= hi]
    if rate_hz:
        period = 1.0 / float(rate_hz)
        kept, next_t = [], None
        for f in frames:
            if next_t is None or f.t >= next_t - 1e-9:
                kept.append(f)
                next_t = f.t + period
        frames = kept
    out = ClipResult(meta=clip.meta, frames=frames, events=clip.events)
    return out


def run_from_cache(args, rows):
    done = skipped = failed = 0
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        cid = normalize_clip_id(row["clip_id"])
        dst = out_path(args.out, cid)
        if os.path.isfile(dst) and not args.force:
            skipped += 1
            continue
        src = os.path.join(args.from_cache, cid, "events.json")
        if not os.path.isfile(src):
            print(f"  [{i}/{len(rows)}] {cid}: no cached events.json -- skipped")
            failed += 1
            continue
        try:
            clip = ClipResult.load_json(src)
            window = window_for(row, args.window_s, args.tail_s)
            sliced = slice_clip_result(clip, window[0], window[1], args.rate_hz)
            if not sliced.frames:
                print(f"  [{i}/{len(rows)}] {cid}: window {window} contains no "
                      "frames -- skipped")
                failed += 1
                continue
            feats = clip_features(sliced, ego=None)     # no ego from cache
            save_features(dst, feats, base_meta(
                row, window, "cache",
                {"n_frames_src": len(clip.frames), "rate_hz": args.rate_hz,
                 "ego_reason": "cached pass predates ego-motion extraction"}))
            done += 1
            if done % 10 == 0 or i == len(rows):
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(rows)}] {done} written "
                      f"({el:.0f}s, {el / max(done, 1):.1f}s/clip)")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{i}/{len(rows)}] {cid}: FAILED {type(exc).__name__}: {exc}")
            failed += 1
    return done, skipped, failed


# ---------------------------------------------------------------------------
# mode: from video (GPU laptop)
# ---------------------------------------------------------------------------

def run_from_video(args, rows):
    """Perception + tracking + forecasting + ego flow over a window per clip.

    Imported lazily so ``--from-cache`` never needs cv2/ultralytics/torch-CUDA.
    """
    import cv2

    from a3ps.common.schema import FrameRecord
    from a3ps.features.ego import EgoFlowEstimator
    from a3ps.pipeline import Pipeline, load_config
    from run_pipeline import make_forecaster_hook, make_risk_hook

    config = load_config(args.config)
    pipeline = Pipeline(config,
                        forecaster=make_forecaster_hook(config),
                        risk_engine=make_risk_hook(config),
                        explainer=None)

    videos_root = args.videos_root or os.path.dirname(os.path.abspath(args.index))
    done = skipped = failed = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        cid = normalize_clip_id(row["clip_id"])
        dst = out_path(args.out, cid)
        if os.path.isfile(dst) and not args.force:
            skipped += 1
            continue
        video = os.path.normpath(os.path.join(videos_root, *row["path"].split("/")))
        if not os.path.isfile(video):
            print(f"  [{i}/{len(rows)}] {cid}: video not found: {video}")
            failed += 1
            continue

        try:
            start_s, end_s = window_for(row, args.window_s, args.tail_s)
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise IOError(f"cannot open {video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            pipeline.tracker.model.predictor = getattr(
                pipeline.tracker.model, "predictor", None)
            # Fresh tracker state per clip: track ids and the GMC reference frame
            # must not carry across clips.
            pipeline.tracker.buffer = type(pipeline.tracker.buffer)(config)
            ego_est = EgoFlowEstimator(downscale=args.ego_downscale)

            period = 1.0 / float(args.rate_hz) if args.rate_hz else 0.0
            frames, ego_rows = [], []
            idx = 0
            next_t = None
            while True:
                ok = cap.grab()
                if not ok:
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
                pipeline._fill_bev(tracks)
                ego_rows.append(ego_est.update(
                    frame, [tr.bbox for tr in tracks]))
                frames.append(FrameRecord(
                    frame_idx=idx, t=round(t, 3), tracks=tracks,
                    ego={"corridor_poly_img": pipeline._ego_corridor(width, height)},
                    context=None))
            cap.release()

            if not frames:
                print(f"  [{i}/{len(rows)}] {cid}: no frames in window "
                      f"[{start_s}, {end_s}] -- skipped")
                failed += 1
                continue

            clip = ClipResult(
                meta={"clip_id": cid, "fps": fps, "width": width, "height": height,
                      "n_frames": len(frames)},
                frames=frames, events=[])
            # Forecasts and risk probabilities are per-frame features here, so
            # replay the hooks over the decimated frame list.
            _apply_hooks(pipeline, clip, config)

            feats = clip_features(clip, ego=ego_rows)
            save_features(dst, feats, base_meta(
                row, (start_s, end_s), "video",
                {"rate_hz": args.rate_hz, "fps": fps,
                 "ego_stats": ego_est.stats()}))
            done += 1
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(rows)}] {cid}: {len(frames)} frames, "
                  f"ego fail {ego_est.stats()['fail_rate']:.0%} "
                  f"({el / max(done, 1):.1f}s/clip)")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{i}/{len(rows)}] {cid}: FAILED {type(exc).__name__}: {exc}")
            failed += 1
    return done, skipped, failed


def _apply_hooks(pipeline, clip, config):
    """Run the forecaster + risk hooks over an already-tracked ClipResult."""
    forecaster = pipeline.forecaster
    risk = pipeline.risk_engine
    if forecaster is None and risk is None:
        return
    ego = {"corridor_poly_img": clip.frames[0].ego["corridor_poly_img"]}
    for fr in clip.frames:
        ctx = {"frame_idx": fr.frame_idx, "t": fr.t,
               "fps": clip.meta.get("fps", 30.0), "config": config,
               "ego": ego, "context": {"flags": []},
               "buffer": pipeline.tracker.buffer, "ground": pipeline.ground,
               "forecast_space": pipeline.forecast_space}
        if forecaster is not None:
            for tr in fr.tracks:
                pred = forecaster(tr, ctx)
                if pred is not None:
                    tr.prediction = pred
        if risk is not None:
            risk(fr.tracks, ctx)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-cache", metavar="CLIPS_DIR",
                      help="Derive features from cached events.json (CPU only).")
    mode.add_argument("--from-video", action="store_true",
                      help="Run perception+tracking+ego flow over video (GPU).")

    p.add_argument("--index", default="eval/nexar_index_gpu.csv")
    p.add_argument("--split", default="eval")
    p.add_argument("--out", required=True, help="Output dir for the .npz files.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--videos-root", default=None,
                   help="Root for resolving index paths (default: index's dir).")
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S,
                   help="Seconds kept per clip (0 = whole clip). Default %(default)s.")
    p.add_argument("--tail-s", type=float, default=DEFAULT_TAIL_S,
                   help="Seconds kept after event_time_s on positives.")
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ,
                   help="Target frame rate (0 = native). Default %(default)s.")
    p.add_argument("--ego-downscale", type=int, default=2,
                   help="Optical-flow downscale factor (video mode).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N clips (smoke tests).")
    p.add_argument("--force", action="store_true",
                   help="Re-extract clips whose .npz already exists.")
    p.add_argument("--freeze", default="eval/split_freeze.json",
                   help="Verify the index against this freeze first ('' to skip).")
    args = p.parse_args()

    if not os.path.isfile(args.index):
        raise SystemExit(f"index not found: {args.index}")

    freeze = load_freeze(args.freeze) if args.freeze else None
    if freeze is not None:
        with open(args.index, newline="", encoding="utf-8-sig") as fh:
            recs = [{"clip_id": r["clip_id"], "label": int(float(r.get("label") or 0)),
                     "split": r.get("split", "") or ""} for r in csv.DictReader(fh)]
        ok, problems = verify_freeze(recs, freeze)
        if not ok:
            print(f"SPLIT DRIFT vs {args.freeze} ({len(problems)} problem(s)):")
            for msg in problems[:5]:
                print(f"  - {msg}")
            raise SystemExit(
                "Refusing to extract features against a drifted split -- a model "
                "trained on these would not be comparable. Reconcile with "
                "scripts/freeze_split.py first.")

    rows = load_rows(args.index, args.split)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no clips with split={args.split} in {args.index}")

    os.makedirs(args.out, exist_ok=True)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"{len(rows)} clip(s) in split={args.split} ({n_pos} positive)")
    print(f"feature width: {frame_feature_dim()} (schema v{SCHEMA_VERSION})")
    print(f"window: {args.window_s}s (+{args.tail_s}s tail)  rate: {args.rate_hz} Hz")
    print(f"out: {args.out}\n")

    if args.from_cache:
        print("mode: from-cache (CPU; ego features will be ZERO -- has_ego=False)")
        done, skipped, failed = run_from_cache(args, rows)
    else:
        print("mode: from-video (needs GPU for usable throughput)")
        done, skipped, failed = run_from_video(args, rows)

    print(f"\nwrote {done}, skipped {skipped} (already present), failed {failed}")
    manifest = os.path.join(args.out, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"index": args.index, "split": args.split,
                   "window_s": args.window_s, "tail_s": args.tail_s,
                   "rate_hz": args.rate_hz,
                   "source": "cache" if args.from_cache else "video",
                   "schema_version": SCHEMA_VERSION,
                   "feature_dim": frame_feature_dim(),
                   "n_written": done, "n_failed": failed}, fh, indent=2)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
