#!/usr/bin/env python
"""Mine training trajectories from Nexar normal-driving clips (Week 3).

Iterates clips with ``split == train_traj`` in data/nexar/index.csv, runs the
Tracker (no forecasting/risk) at stride 1, and exports sliding windows per
track: 2 s history (10 pts) + 4 s future (20 pts) at 5 Hz, stride 1 s, in BEV
metres (or image pixels if BEV is disabled via forecast_space: img).

Filtering:
  * drop tracks shorter than 6 s,
  * drop windows with an ID-switch artifact (a consecutive jump > 3x median),
  * drop near-static windows (net displacement < threshold) but KEEP 20% so the
    model still learns "stationary stays stationary".

Each window is normalized: translated so the last history point is the origin
and rotated so the mean history heading points +y; the inverse transform
(origin_x, origin_y, theta) is stored. Shards go to data/trajectories/*.npz
plus an aggregate stats.json.

    python scripts/mine_trajectories.py --index data/nexar/index.csv
    python scripts/mine_trajectories.py --limit-clips 1     # smoke test
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HZ = 5.0
HIST = 10                 # 2 s @ 5 Hz
FUT = 20                  # 4 s @ 5 Hz
WIN = HIST + FUT          # 30 points / 6 s
STRIDE_BUCKETS = int(round(1.0 * HZ))   # 1 s between windows
MIN_TRACK_S = 6.0
JUMP_FACTOR = 3.0
STATIC_KEEP_FRAC = 0.20
SEED = 20260708

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


# ---------------------------------------------------------------------------
# pure helpers (importable, no cv2/torch/dataset needed)
# ---------------------------------------------------------------------------

def resample_track(samples, hz=HZ):
    """Decimate (t, [x,y]) samples to one point per 1/hz-second bucket.

    Returns a dict {bucket_index: [x, y]} keeping the first sample per bucket.
    """
    out = {}
    for t, pt in samples:
        b = int(t * hz + 1e-9)          # floor into 1/hz-second buckets
        if b not in out:
            out[b] = [float(pt[0]), float(pt[1])]
    return out


def contiguous_windows(bucket_points, win=WIN, stride=STRIDE_BUCKETS):
    """Yield (start_bucket, [win points]) for every fully-contiguous window."""
    if not bucket_points:
        return
    b_min, b_max = min(bucket_points), max(bucket_points)
    start = b_min
    while start + win - 1 <= b_max:
        idxs = range(start, start + win)
        if all(b in bucket_points for b in idxs):
            yield start, [bucket_points[b] for b in idxs]
        start += stride


def is_id_switch(points, factor=JUMP_FACTOR):
    """True if any consecutive step exceeds ``factor`` x the median step."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    med = float(np.median(steps))
    if med <= 1e-9:
        # Degenerate (near-static): flag only a truly large absolute jump.
        return bool(np.max(steps) > 5.0)
    return bool(np.max(steps) > factor * med)


def net_displacement(points):
    """Straight-line distance from the first to the last point of the window."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    return float(np.linalg.norm(pts[-1] - pts[0]))


def normalize_window(points, hist_len=HIST):
    """Translate to last-history origin and rotate mean heading to +y.

    Returns (history[hist_len,2], future[N-hist_len,2], origin[2], theta).
    Absolute recovery: ``R(-theta) @ p_norm + origin``.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    origin = pts[hist_len - 1].copy()
    shifted = pts - origin

    hist = shifted[:hist_len]
    deltas = np.diff(hist, axis=0)
    mean_d = deltas.mean(axis=0)
    if float(np.hypot(mean_d[0], mean_d[1])) < 1e-6:
        theta = 0.0
    else:
        ang = float(np.arctan2(mean_d[1], mean_d[0]))
        theta = (np.pi / 2.0) - ang           # rotate heading onto +y

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rot = shifted @ R.T
    return rot[:hist_len], rot[hist_len:], origin, float(theta)


# ---------------------------------------------------------------------------
# per-clip tracking
# ---------------------------------------------------------------------------

def track_clip(video_path, config, use_bev):
    """Run the tracker over a whole clip; return {track_id: {'pts', 'cls'}}.

    Points are BEV metres when ``use_bev`` else image pixels.
    """
    import cv2

    from a3ps.common.geometry import GroundPlane
    from a3ps.tracking.tracker import Tracker

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ground = GroundPlane.from_config(config, width, height) if use_bev else None

    # Fresh tracker per clip so BoT-SORT ids never bleed across clips.
    tracker = Tracker(config)

    tracks = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        for tr in tracker.update(frame, idx, t):
            pt = (ground.img_to_bev([tr.centroid_img])[0] if ground is not None
                  else tr.centroid_img)
            rec = tracks.setdefault(tr.id, {"pts": [], "cls": {}})
            rec["pts"].append((t, [float(pt[0]), float(pt[1])]))
            rec["cls"][tr.cls] = rec["cls"].get(tr.cls, 0) + 1
        idx += 1
    cap.release()
    return tracks


def mine_clip(tracks, rng, stats):
    """Turn one clip's tracks into normalized windows. Returns list of dicts."""
    windows = []
    for tid, rec in tracks.items():
        samples = rec["pts"]
        if not samples:
            continue
        span = samples[-1][0] - samples[0][0]
        if span < MIN_TRACK_S:
            stats["short_tracks"] += 1
            continue
        cls = max(rec["cls"].items(), key=lambda kv: kv[1])[0]
        buckets = resample_track(samples)
        for _start, pts in contiguous_windows(buckets):
            if is_id_switch(pts):
                stats["id_switch_windows"] += 1
                continue
            if net_displacement(pts) < stats["static_thresh"]:
                if rng.random() >= STATIC_KEEP_FRAC:
                    stats["static_dropped"] += 1
                    continue
                stats["static_kept"] += 1
            hist, fut, origin, theta = normalize_window(pts)
            windows.append({
                "history": hist, "future": fut,
                "origin": origin, "theta": theta,
                "cls": cls,
            })
            stats["class_mix"][cls] = stats["class_mix"].get(cls, 0) + 1
    return windows


def plot_windows(out_dir, n=50, seed=SEED):
    """Scatter a random sample of mined windows (history vs future).

    Loads all shards in ``out_dir`` (normalized coords: last-history point at
    the origin, heading +y) and saves a PNG. Returns the output path or None.
    """
    import glob

    import numpy as np

    shards = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not shards:
        print(f"  (no shards in {out_dir} to plot)")
        return None

    hist_list, fut_list = [], []
    for sh in shards:
        d = np.load(sh)
        hist_list.append(d["history"])
        fut_list.append(d["future"])
    hist = np.concatenate(hist_list)      # (N, 10, 2)
    fut = np.concatenate(fut_list)        # (N, 20, 2)
    total = len(hist)
    if total == 0:
        print("  (no windows to plot)")
        return None

    rng = np.random.default_rng(seed)
    pick = rng.choice(total, size=min(n, total), replace=False)

    import matplotlib
    matplotlib.use("Agg")                 # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 8))
    for i in pick:
        h, f = hist[i], fut[i]
        ax.plot(h[:, 0], h[:, 1], "-", color="#1f77b4", alpha=0.5, lw=1)
        ax.plot(f[:, 0], f[:, 1], "-", color="#d62728", alpha=0.5, lw=1)
        ax.plot(0, 0, "k.", ms=3)
    ax.set_title(f"{len(pick)} random windows (blue=history, red=future)\n"
                 "normalized: origin=last history pt, heading +y")
    ax.set_xlabel("x (lateral)")
    ax.set_ylabel("y (forward, +)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path = os.path.join(out_dir, "windows_preview.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  plotted {len(pick)} of {total} windows -> {path}")
    return path


def save_shard(out_dir, clip_id, windows):
    import numpy as np

    if not windows:
        return
    hist = np.stack([w["history"] for w in windows]).astype(np.float32)
    fut = np.stack([w["future"] for w in windows]).astype(np.float32)
    transforms = np.array(
        [[w["origin"][0], w["origin"][1], w["theta"]] for w in windows],
        dtype=np.float32)
    class_ids = np.array(
        [CLASSES.index(w["cls"]) if w["cls"] in CLASSES else -1 for w in windows],
        dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"{clip_id}.npz"),
        history=hist, future=fut, transforms=transforms, class_ids=class_ids,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _recount_shards(out_dir):
    """Ground truth (clips, windows) by reading every shard actually on disk.

    Used to self-heal stats.json's cumulative counters: a prior interrupted
    or buggy run can leave them stale/wrong relative to what's really in
    --out, and a resumed run that only trusts the counter would never notice.
    """
    import numpy as np

    n_clips, n_windows = 0, 0
    for f in glob.glob(os.path.join(out_dir, "*.npz")):
        try:
            with np.load(f) as d:
                n_windows += len(d["history"])
            n_clips += 1
        except Exception:  # noqa: BLE001 - a corrupt shard shouldn't crash the count
            continue
    return n_clips, n_windows


def _resolve_clip_path(index_csv, rel_path):
    # index.csv `path` is relative to the directory containing index.csv
    # (prepare_nexar uses start=<root>), e.g. data/nexar/ + videos/00584.mp4.
    base = os.path.dirname(os.path.abspath(index_csv))
    return os.path.normpath(os.path.join(base, *rel_path.split("/")))


def main():
    p = argparse.ArgumentParser(description="Mine training trajectories.")
    p.add_argument("--index", default="data/nexar/index.csv")
    p.add_argument("--out", default="data/trajectories")
    p.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "default.yaml"))
    p.add_argument("--limit-clips", type=int, default=None,
                   help="Process at most N clips (smoke test).")
    p.add_argument("--split", default="train_traj",
                   help="Which index split to mine (default train_traj; use "
                        "e.g. 'dev' to verify on dev clips before real data).")
    p.add_argument("--plot", action="store_true",
                   help="After mining, save a scatter of random windows.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip mining; just plot existing shards in --out.")
    p.add_argument("--plot-n", type=int, default=50,
                   help="Number of random windows to plot (default 50).")
    p.add_argument("--force", action="store_true",
                   help="Re-mine clips even if a shard for them already exists "
                        "in --out (default: skip already-mined clips, so an "
                        "interrupted run can be resumed by just re-running).")
    args = p.parse_args()

    if args.plot_only:
        plot_windows(args.out, n=args.plot_n)
        return

    from a3ps.pipeline import load_config
    config = load_config(args.config)
    use_bev = config.get("forecast_space", "img") == "bev"
    space = "bev" if use_bev else "img"
    # Static threshold: 1 m in BEV, ~15 px in image space.
    static_thresh = 1.0 if use_bev else 15.0

    if not os.path.isfile(args.index):
        p.error(f"index not found: {args.index} (run scripts/prepare_nexar.py first)")

    with open(args.index, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("split") == args.split]
    if args.limit_clips is not None:
        rows = rows[:args.limit_clips]

    print(f"{args.split} clips: {len(rows)}  space={space}  "
          f"static_thresh={static_thresh}")
    if not rows:
        print(f"No '{args.split}' clips found. Populate data/nexar/ and run "
              "scripts/prepare_nexar.py (see TODO.md). Nothing to mine.")

    os.makedirs(args.out, exist_ok=True)
    stats_path = os.path.join(args.out, "stats.json")

    # Resume support: load any existing stats so counts accumulate across
    # interrupted/re-run sessions instead of resetting to zero each time.
    stats = {
        "space": space, "static_thresh": static_thresh,
        "clips_processed": 0, "total_windows": 0, "class_mix": {},
        "short_tracks": 0, "id_switch_windows": 0,
        "static_dropped": 0, "static_kept": 0, "clips_skipped_existing": 0,
    }
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, encoding="utf-8") as fh:
                prior = json.load(fh)
            stats.update({k: v for k, v in prior.items() if k in stats})
            print(f"resuming: loaded prior stats ({stats['clips_processed']} "
                  f"clips, {stats['total_windows']} windows already recorded)")
        except (json.JSONDecodeError, OSError):
            pass

    def checkpoint():
        # Self-heal clips_processed/total_windows from the actual shards on
        # disk rather than trusting the running counter, so staleness from a
        # prior interrupted/buggy run (or manual shard edits) can't persist.
        n_clips, n_windows = _recount_shards(args.out)
        stats["clips_processed"] = n_clips
        stats["total_windows"] = n_windows
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    rng = random.Random(SEED)
    for r in rows:
        clip_id = r["clip_id"]
        shard_path = os.path.join(args.out, f"{clip_id}.npz")
        if not args.force and os.path.isfile(shard_path):
            stats["clips_skipped_existing"] += 1
            continue
        video = _resolve_clip_path(args.index, r["path"])
        if not os.path.isfile(video):
            print(f"  ! missing video, skipping: {video}")
            continue
        print(f"  mining {clip_id} ...", flush=True)
        tracks = track_clip(video, config, use_bev)
        windows = mine_clip(tracks, rng, stats)
        save_shard(args.out, clip_id, windows)
        print(f"    {len(windows)} windows")
        checkpoint()          # persist after every clip so nothing is lost
                               # (also self-heals clips_processed/total_windows)

    checkpoint()  # unconditional final write: self-heals even if every clip
                  # this run was skipped (all already mined) and nothing else
                  # would have triggered a recount.

    if stats["clips_skipped_existing"]:
        print(f"skipped {stats['clips_skipped_existing']} already-mined clips "
              f"(use --force to re-mine them)")

    print(f"\nTotal windows mined: {stats['total_windows']} "
          f"from {stats['clips_processed']} clips -> {args.out}")
    print(f"class mix: {stats['class_mix']}")

    if args.plot:
        plot_windows(args.out, n=args.plot_n)


if __name__ == "__main__":
    main()
