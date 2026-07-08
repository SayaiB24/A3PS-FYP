#!/usr/bin/env python
"""Sanity-check BEV projection + forecasting from a clip's events.json.

    python scripts/verify_bev.py dashboard/clips/02134_pipe/events.json

Checks:
  1. velocity_bev magnitudes are physically sane (relative to the moving ego,
     so ~5-20 m/s is expected; parked cars approach at ~ego speed).
  2. centroid_bev depths/laterals are plausible (positive depth, within range).
  3. BEV predictions back-projected to mean_img land inside the frame
     ("hug the road" rather than flying off).

Prints a verdict and, if the default ground plane looks absurd, the
recommended next step (per-clip override, or fall back to forecast_space: img).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import ClipResult  # noqa: E402


def _pctl(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dashboard/clips/02134_pipe/events.json"
    if os.path.isdir(path):
        path = os.path.join(path, "events.json")
    clip = ClipResult.load_json(path)

    W = clip.meta.get("width", 1280)
    H = clip.meta.get("height", 720)
    print(f"clip: {clip.meta.get('clip_id')}  frames={len(clip.frames)}  size={W}x{H}")
    print(f"forecast config: {clip.meta.get('config', {})}\n")

    speeds, depths, laterals = [], [], []
    pred_pts_total = pred_pts_in_frame = 0
    pred_depths = []

    for f in clip.frames:
        for tr in f.tracks:
            if tr.velocity_bev is not None:
                vx, vy = tr.velocity_bev
                speeds.append((vx * vx + vy * vy) ** 0.5)
            if tr.centroid_bev is not None:
                laterals.append(tr.centroid_bev[0])
                depths.append(tr.centroid_bev[1])
            p = tr.prediction
            if p and p.mean_img:
                for (x, y) in p.mean_img:
                    pred_pts_total += 1
                    if 0 <= x < W and 0 <= y < H:
                        pred_pts_in_frame += 1
                if p.mean_bev:
                    pred_depths.extend(d[1] for d in p.mean_bev)

    # --- 1. velocity sanity ---
    print("== velocity_bev magnitude (m/s, RELATIVE to moving ego) ==")
    if speeds:
        in_band = sum(1 for s in speeds if 5.0 <= s <= 20.0) / len(speeds)
        print(f"  samples={len(speeds)}  median={_pctl(speeds,50):.1f}  "
              f"p10={_pctl(speeds,10):.1f}  p90={_pctl(speeds,90):.1f}  "
              f"max={max(speeds):.1f}")
        print(f"  fraction in 5-20 m/s: {in_band*100:.0f}%")
        sane_v = 0.30 <= in_band or _pctl(speeds, 50) <= 25.0
    else:
        print("  (no velocity_bev found)")
        sane_v = False

    # --- 2. position sanity ---
    print("\n== centroid_bev (metres) ==")
    if depths:
        neg = sum(1 for d in depths if d < 0) / len(depths)
        print(f"  depth  y: median={_pctl(depths,50):.1f}  "
              f"range=[{min(depths):.1f}, {max(depths):.1f}]  negative={neg*100:.0f}%")
        print(f"  lateral x: median={_pctl(laterals,50):.1f}  "
              f"range=[{min(laterals):.1f}, {max(laterals):.1f}]")
        sane_pos = neg < 0.20 and max(depths) < 120.0
    else:
        print("  (no centroid_bev found)")
        sane_pos = False

    # --- 3. predictions hug the road (stay in frame) ---
    print("\n== BEV predictions back-projected to image ==")
    if pred_pts_total:
        frac_in = pred_pts_in_frame / pred_pts_total
        print(f"  predicted points: {pred_pts_total}  in-frame: {frac_in*100:.0f}%")
        if pred_depths:
            print(f"  forecast BEV depth range: "
                  f"[{min(pred_depths):.1f}, {max(pred_depths):.1f}] m")
        sane_pred = frac_in >= 0.80
    else:
        print("  (no predictions found -- clip too short for buffer.ready?)")
        sane_pred = True  # not a calibration failure

    # --- verdict ---
    print("\n== verdict ==")
    ok = sane_v and sane_pos and sane_pred
    if ok:
        print("  OK: BEV projection looks plausible for this clip.")
        print("  Eyeball dotted paths in annotated.mp4 to confirm they hug the road.")
    else:
        print("  SUSPECT: the default ground-plane trapezoid may be wrong for this")
        print("  camera angle. Options:")
        print("   1. Create a per-clip override (ground_plane_override in config):")
        print("      pick 4 road points + their real distances, save as ground.yaml.")
        print("   2. If calibration fights you >2 days, set forecast_space: img in")
        print("      configs/default.yaml. The demo still works; only the")
        print("      'metres/seconds' labels become 'relative units'.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
