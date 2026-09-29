#!/usr/bin/env python
"""GPU re-timing of the fused perception+tracking call for Table VII.

Re-uses the EXACT production call the CPU numbers came from: a3ps.pipeline's
per-frame ``self.tracker.update(frame, idx, t)`` (a3ps/tracking/tracker.py),
which is a single ``model.track()`` call (YOLOv8s-seg detection+segmentation
fused with BoT-SORT association, since that's how a3ps.tracking.tracker.Tracker
is built -- see PER_STAGE_MS_NOTE in a3ps/pipeline.py). This script does not
modify Tracker or Pipeline; it drives ``Tracker`` directly (same config, same
model.track() call signature) so it can additionally read
``results[0].speed`` (Ultralytics' own CUDA-synchronized preprocess/inference/
postprocess timing) for the YOLO-vs-tracker/CMC breakdown, which
``Tracker.update()`` does not expose.

Timing protocol:
  - warm-up frames (first WARMUP_FRAMES frames of the first clip) run through
    the model but are NOT recorded, since the existing CPU protocol
    (a3ps/pipeline.py) has no warm-up step and the first CUDA call always
    includes one-time kernel/allocator setup.
  - every timed frame is wrapped in torch.cuda.synchronize() before and after
    the model.track() call, so wall-clock time reflects actual GPU completion
    rather than async kernel-launch return.
  - clips are drawn only from data/dev_clips/ (the frozen dev split); eval
    clips are never touched and this script has no "final report" flag.

Outputs:
  results/latency_gpu.csv         one row per timed frame
  results/latency_gpu_summary.md  aggregate stats + hardware info
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
import torch  # noqa: E402

from a3ps.pipeline import load_config  # noqa: E402
from a3ps.tracking.tracker import Tracker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CLIPS_DIR = REPO_ROOT / "data" / "dev_clips"
WARMUP_FRAMES = 20

# Existing CPU numbers (measured on the 8-core CPU dev laptop, same protocol,
# see eval/latency_table4.md / dashboard/clips/1118/meta.json). Not re-measured
# here -- only reused for the end-to-end combination.
CPU_FORECAST_MS = 19.76
CPU_RISK_GRU_MS = 0.252
CPU_RISK_LEGACY_MS = 1.94
CPU_PERCEPTION_TRACKING_MS = 81.65


def nvidia_smi_query() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        name, mem, driver = [p.strip() for p in out.split(",")]
        return {"gpu_name": name, "gpu_vram": mem, "driver_version": driver}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"gpu_name": "unknown", "gpu_vram": "unknown",
                "driver_version": f"unavailable ({exc})"}


def cpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_hw_info() -> dict:
    import ultralytics

    info = {
        "cpu": cpu_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics_version": ultralytics.__version__,
    }
    info.update(nvidia_smi_query())
    return info


def list_dev_clips() -> list[Path]:
    clips = sorted(DEV_CLIPS_DIR.glob("dev*.mp4"))
    if not clips:
        raise SystemExit(f"no dev clips found under {DEV_CLIPS_DIR}")
    return clips


def time_one_frame(tracker: Tracker, frame, idx: int, t: float) -> tuple[float, dict]:
    """Same model.track() call Tracker.update() makes, but also returns
    Ultralytics' own results[0].speed dict for the YOLO-only breakdown.
    Mirrors a3ps/tracking/tracker.py:Tracker.update() exactly (same args)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = tracker.model.track(
        frame,
        persist=True,
        tracker=tracker.tracker_cfg,
        conf=tracker.conf,
        imgsz=tracker.img_size,
        device=tracker.device,
        verbose=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    speed = dict(results[0].speed) if results else {}

    # Replicate Tracker.update()'s post-processing (box/mask extraction +
    # buffer push) so track state / history stay correct across frames if
    # anything downstream were to consume it -- kept for fidelity with the
    # production call even though this benchmark only needs the timing.
    r = results[0] if results else None
    boxes = getattr(r, "boxes", None) if r is not None else None
    if boxes is not None and boxes.id is not None and len(boxes) > 0:
        from a3ps.common.schema import TrackState
        masks = getattr(r, "masks", None)
        polys = masks.xy if masks is not None else [None] * len(boxes)
        out = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if tracker.class_ids and cls_id not in tracker.class_ids:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            foot = tracker._foot_point(bbox)
            out.append(TrackState(
                id=int(boxes.id[i].item()), cls=tracker.model.names.get(cls_id, str(cls_id)),
                bbox=[float(v) for v in bbox],
                centroid_img=[round(foot[0], 1), round(foot[1], 1)],
                centroid_bev=None, velocity_bev=None, mask_poly=None,
                history_img=[], prediction=None, risk_level="safe",
            ))
        tracker.buffer.push(idx, t, out)

    return total_ms, speed


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available on this machine -- refusing to "
                          "silently record CPU numbers as GPU numbers.")

    config = load_config(str(REPO_ROOT / "configs" / "default.yaml"))
    tracker = Tracker(config)
    assert tracker.use_half, "expected the pipeline to auto-select cuda+half"

    clips = list_dev_clips()
    rows = []
    warmed_up = False

    for clip_path in clips:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"[skip] cannot open {clip_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps

            if not warmed_up and idx < WARMUP_FRAMES:
                # Warm-up: run the identical call, discard the timing.
                tracker.model.track(
                    frame, persist=True, tracker=tracker.tracker_cfg,
                    conf=tracker.conf, imgsz=tracker.img_size,
                    device=tracker.device, verbose=False,
                )
                idx += 1
                if idx >= WARMUP_FRAMES:
                    warmed_up = True
                    # Reset track ids so BoT-SORT association starts clean for
                    # the timed run instead of persisting warm-up-only tracks.
                    tracker.model.predictor = None
                continue

            total_ms, speed = time_one_frame(tracker, frame, idx, t)
            pre = speed.get("preprocess", 0.0)
            inf = speed.get("inference", 0.0)
            post = speed.get("postprocess", 0.0)
            yolo_ms = pre + inf + post
            tracker_cmc_ms = max(0.0, total_ms - yolo_ms)
            rows.append({
                "clip": clip_path.stem,
                "frame_idx": idx,
                "total_fused_ms": round(total_ms, 4),
                "yolo_preprocess_ms": round(pre, 4),
                "yolo_inference_ms": round(inf, 4),
                "yolo_postprocess_ms": round(post, 4),
                "yolo_total_ms": round(yolo_ms, 4),
                "tracker_cmc_ms_approx": round(tracker_cmc_ms, 4),
            })
            idx += 1
        cap.release()
        print(f"[done] {clip_path.name}: {idx} frames")

    if not rows:
        raise SystemExit("no frames timed -- nothing to report")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "latency_gpu.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    totals = [r["total_fused_ms"] for r in rows]
    yolo_totals = [r["yolo_total_ms"] for r in rows]
    cmc_totals = [r["tracker_cmc_ms_approx"] for r in rows]
    n_frames = len(rows)
    n_clips = len({r["clip"] for r in rows})

    fused_mean = mean(totals)
    fused_median = median(totals)
    fused_std = pstdev(totals)

    end_to_end_gru = fused_mean + CPU_FORECAST_MS + CPU_RISK_GRU_MS
    end_to_end_legacy = fused_mean + CPU_FORECAST_MS + CPU_RISK_LEGACY_MS
    fps_gru = 1000.0 / end_to_end_gru
    fps_legacy = 1000.0 / end_to_end_legacy

    hw = collect_hw_info()

    summary_path = out_dir / "latency_gpu_summary.md"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("# GPU perception+tracking re-timing (Table VII)\n\n")
        fh.write("## Hardware / software\n\n")
        for k, v in hw.items():
            fh.write(f"- **{k}**: {v}\n")
        fh.write(f"- **precision**: half (fp16) -- Tracker auto-selects "
                  f"`.cuda().half()` whenever CUDA is available "
                  f"(a3ps/tracking/tracker.py); this machine has CUDA, so the "
                  f"pipeline ran fp16, not fp32.\n")
        fh.write("\n## Protocol\n\n")
        fh.write(f"- clips: {n_clips} dev-split clips from `data/dev_clips/` "
                  f"(`{', '.join(sorted({r['clip'] for r in rows}))}`)\n")
        fh.write(f"- frames timed: {n_frames} (after a {WARMUP_FRAMES}-frame "
                  f"warm-up on the first clip, discarded)\n")
        fh.write("- config: `configs/default.yaml` + `configs/botsort_a3ps.yaml` "
                  "(unchanged) -- yolov8s-seg, imgsz 1280, conf 0.35, six road-actor "
                  "classes, BoT-SORT motion-only (with_reid: False), sparse optical "
                  "flow CMC (gmc_method: sparseOptFlow), track_buffer 90\n")
        fh.write("- each frame: `torch.cuda.synchronize()` immediately before and "
                  "after the exact `model.track()` call `Tracker.update()` makes\n\n")
        fh.write("## Fused perception + tracking (GPU) -- matches the CPU row's "
                  "definition\n\n")
        fh.write(f"| mean ms/frame | median ms/frame | std ms/frame | n frames | n clips |\n")
        fh.write(f"|---|---|---|---|---|\n")
        fh.write(f"| {fused_mean:.4f} | {fused_median:.4f} | {fused_std:.4f} "
                  f"| {n_frames} | {n_clips} |\n\n")
        fh.write("## Breakdown (approximate)\n\n")
        fh.write("YOLO pre/inference/post is Ultralytics' own CUDA-synchronized "
                  "`results.speed`; tracker/CMC is the remainder of the wrapped "
                  "call (`total_fused_ms - yolo_total_ms`), i.e. BoT-SORT "
                  "association + sparse-optical-flow camera-motion compensation, "
                  "which runs on CPU and is not covered by Ultralytics' own "
                  "profiler.\n\n")
        fh.write(f"| yolo mean ms/frame | tracker/CMC mean ms/frame (approx) |\n")
        fh.write(f"|---|---|\n")
        fh.write(f"| {mean(yolo_totals):.4f} | {mean(cmc_totals):.4f} |\n\n")
        fh.write("## End-to-end (mixed device)\n\n")
        fh.write("GPU perception+tracking (measured here) + CPU forecasting "
                  "(Kalman-CV, 19.76 ms/frame, existing measurement) + CPU risk "
                  "stage (existing measurement, not re-measured):\n\n")
        fh.write("| risk stage | ms/frame | fps |\n")
        fh.write("|---|---|---|\n")
        fh.write(f"| learned GRU head ({CPU_RISK_GRU_MS} ms, CPU) | "
                  f"{end_to_end_gru:.4f} | {fps_gru:.4f} |\n")
        fh.write(f"| legacy threshold engine ({CPU_RISK_LEGACY_MS} ms, CPU) | "
                  f"{end_to_end_legacy:.4f} | {fps_legacy:.4f} |\n\n")
        fh.write(f"For reference, the CPU perception+tracking figure being "
                  f"replaced was {CPU_PERCEPTION_TRACKING_MS} ms/frame "
                  f"(speedup: {CPU_PERCEPTION_TRACKING_MS / fused_mean:.2f}x).\n")

    print(f"\nwrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"\nfused perception+tracking: mean={fused_mean:.2f} ms "
          f"median={fused_median:.2f} ms std={fused_std:.2f} ms "
          f"n={n_frames} frames / {n_clips} clips")
    print(f"end-to-end (GRU head): {end_to_end_gru:.2f} ms/frame -> {fps_gru:.2f} fps")
    print(f"end-to-end (legacy engine): {end_to_end_legacy:.2f} ms/frame -> {fps_legacy:.2f} fps")


if __name__ == "__main__":
    main()
