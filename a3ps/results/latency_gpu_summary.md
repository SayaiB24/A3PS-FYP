# GPU perception+tracking re-timing (Table VII)

## Hardware / software

- **cpu**: 12th Gen Intel(R) Core(TM) i7-12700H
- **torch_version**: 2.6.0+cu124
- **torch_cuda_version**: 12.4
- **cuda_available**: True
- **ultralytics_version**: 8.4.90
- **gpu_name**: NVIDIA GeForce RTX 3050 Laptop GPU
- **gpu_vram**: 4096 MiB
- **driver_version**: 566.07
- **precision**: half (fp16) -- Tracker auto-selects `.cuda().half()` whenever CUDA is available (a3ps/tracking/tracker.py); this machine has CUDA, so the pipeline ran fp16, not fp32.

## Protocol

- clips: 15 dev-split clips from `data/dev_clips/` (`dev01, dev02, dev03, dev04, dev05, dev06, dev07, dev08, dev09, dev10, dev11, dev12, dev13, dev14, dev15`)
- frames timed: 8170 (after a 20-frame warm-up on the first clip, discarded)
- config: `configs/default.yaml` + `configs/botsort_a3ps.yaml` (unchanged) -- yolov8s-seg, imgsz 1280, conf 0.35, six road-actor classes, BoT-SORT motion-only (with_reid: False), sparse optical flow CMC (gmc_method: sparseOptFlow), track_buffer 90
- each frame: `torch.cuda.synchronize()` immediately before and after the exact `model.track()` call `Tracker.update()` makes

## Fused perception + tracking (GPU) -- matches the CPU row's definition

| mean ms/frame | median ms/frame | std ms/frame | n frames | n clips |
|---|---|---|---|---|
| 61.9336 | 60.5819 | 7.4064 | 8170 | 15 |

## Breakdown (approximate)

YOLO pre/inference/post is Ultralytics' own CUDA-synchronized `results.speed`; tracker/CMC is the remainder of the wrapped call (`total_fused_ms - yolo_total_ms`), i.e. BoT-SORT association + sparse-optical-flow camera-motion compensation, which runs on CPU and is not covered by Ultralytics' own profiler.

| yolo mean ms/frame | tracker/CMC mean ms/frame (approx) |
|---|---|
| 34.7462 | 27.1874 |

## End-to-end (mixed device)

GPU perception+tracking (measured here) + CPU forecasting (Kalman-CV, 19.76 ms/frame, existing measurement) + CPU risk stage (existing measurement, not re-measured):

| risk stage | ms/frame | fps |
|---|---|---|
| learned GRU head (0.252 ms, CPU) | 81.9456 | 12.2032 |
| legacy threshold engine (1.94 ms, CPU) | 83.6336 | 11.9569 |

For reference, the CPU perception+tracking figure being replaced was 81.65 ms/frame (speedup: 1.32x).
