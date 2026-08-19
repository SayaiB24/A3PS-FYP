# Table IV — per-stage latency (ms per frame)

**Hardware matters more than the numbers here.** Everything below marked CPU was
measured on the development laptop (8 cores, no GPU). Perception and tracking are
20–40× slower there than on a GPU, so **CPU perception numbers must never be
presented as the deployed system's performance** — they are included only because
they are what has actually been measured.

## Measured

| stage | ms/frame | machine | source |
|---|---|---|---|
| Perception (YOLOv8-Seg) | *see note* | — | fused with tracking |
| Tracking (BoT-SORT) | **81.65** | CPU | `dashboard/clips/1118/meta.json` → `per_stage_ms` |
| Forecasting (Kalman-CV) | **19.76** | CPU | same |
| Risk & decision — old threshold engine | **1.94** | CPU | same |
| **Risk & decision — learned head (GRU)** | **0.252** | CPU, 1 thread | streaming inference, measured directly |
| End-to-end (old pipeline, CPU) | **~103.4** | CPU | sum of the above |

### Notes that must accompany the table

**Perception is not separately measurable in this architecture.** Detection,
segmentation and BoT-SORT association run as a *single* `model.track()` call
(`a3ps/tracking/tracker.py`) — splitting them would double inference cost. All of
that combined cost is attributed to "tracking", and perception is reported as 0.0
by convention. This is stated verbatim in every `meta.json` under
`per_stage_ms_note`; quote it rather than implying two independent measurements.

**The learned head is essentially free: 0.252 ms/frame** in streaming mode
(carrying hidden state frame to frame, which is how it would be deployed), for
34,465 parameters. Whole-clip batch inference is 6.14 ms for a 127-frame clip
(0.048 ms/frame), but streaming is the honest deployed figure.

That is **~7.7× cheaper than the threshold engine it replaced** (1.94 ms/frame) —
worth stating, because it means the Phase IV accuracy gain came with a latency
*reduction* in the risk stage, not a cost.

## Not yet measured — needed before this table is publishable

**GPU perception/tracking.** The dominant cost (81.65 ms/frame) is CPU-only.
Re-run 3+ clips on the GPU laptop and replace it:

```powershell
python scripts/run_pipeline.py --video data/nexar/videos/00621.mp4 --out /tmp/lat621
python -c "import json; print(json.load(open('/tmp/lat621/meta.json'))['per_stage_ms'])"
```

Expect roughly 20–60 ms/frame for tracking on GPU. **Confirm CUDA is live first**
(`python -c "import torch; print(torch.cuda.is_available())"` → `True`), or you
will silently record CPU numbers again.

**Real-time claim.** The paper asks whether the end-to-end mean beats the
real-time bar at the configured `process_fps`. With CPU tracking at 81.65 ms the
end-to-end mean is ~103 ms/frame ≈ 9.7 FPS, which is *below* 30 fps and roughly at
the 10 Hz feature rate the head actually consumes. State the frame rate the claim
is made against; do not claim 30 fps real-time from these numbers.

**Feature extraction is a separate cost** and is not in this table: 8.4 s/clip
over a 13 s window ≈ 65 ms/frame on GPU, which includes tracking plus the
optical-flow ego pass. If the paper describes the deployed pipeline end to end,
that is the number that governs throughput, not the sum above.
