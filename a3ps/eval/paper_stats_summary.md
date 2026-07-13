# A3PS — Paper 2 Stats Summary

Auto-collected from the repo. Paste the relevant parts of this file into the Paper 2 generation prompt in place of [FILL IN] / [RESULT: ...] placeholders. Sections marked NOT YET AVAILABLE need their listed command run first — do not hand-fill fabricated numbers.

## 1. Configuration / Hyperparameters (configs/default.yaml)

- **model_name**: yolov8s-seg.pt
- **tracker**: configs/botsort_a3ps.yaml
- **classes**:
  - person
  - bicycle
  - car
  - motorcycle
  - bus
  - truck
- **conf**: 0.35
- **img_size**: 1280
- **process_fps**: 30
- **predict_hz**: 5
- **horizon_s**: 4.0
- **history_s**: 2.0
- **base_threshold**: 0.75
- **caution_threshold**: 0.4
- **alert_margin**: 0.15
- **frames_to_confirm**: 3
- **event_cooldown_s**: 2.0
- **threshold_floor**: 0.45
- **risk_ema_alpha**: 0.4
- **mask_poly_max_points**: 40
- **actor_radius_m**:
  - **person**: 0.4
  - **bicycle**: 0.5
  - **motorcycle**: 0.6
  - **car**: 1.0
  - **bus**: 1.5
  - **truck**: 1.5
- **actor_radius_px**:
  - **person**: 8
  - **bicycle**: 10
  - **motorcycle**: 12
  - **car**: 20
  - **bus**: 28
  - **truck**: 28
- **ego_corridor**:
  - **bottom_y_frac**: 1.0
  - **top_y_frac**: 0.6
  - **bottom_left_frac**: 0.39
  - **bottom_right_frac**: 0.61
  - **top_left_frac**: 0.48
  - **top_right_frac**: 0.52
- **forecast_space**: img
- **ground_plane**:
  - **image_points_frac**:
    - [0.3, 1.0]
    - [0.7, 1.0]
    - [0.55, 0.62]
    - [0.45, 0.62]
  - **ground_points_m**:
    - [-1.8, 0.0]
    - [1.8, 0.0]
    - [1.8, 25.0]
    - [-1.8, 25.0]

## 2. Hardware & Environment

- **python_version**: 3.14.6
- **os**: Windows 11
- **cpu**: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- **gpu**: unknown — run `nvidia-smi --query-gpu=name,memory.total --format=csv` and paste here
- **torch_version**: 2.12.1+cpu
- **cuda_available**: False

## 3. Dataset Split Sizes (dev / train_traj / eval)

- **dev**:
  - **positives**: 5
  - **negatives**: 10
  - **total**: 15
- **eval**:
  - **positives**: 60
  - **negatives**: 60
  - **total**: 120
- **train_traj**:
  - **positives**: 0
  - **negatives**: 20
  - **total**: 20
- **_grand_total_clips_in_index**: 155

## 4. Per-Stage Latency — Table 3 material

- **_status**: found 5 meta.json file(s) but no recognizable per-stage timing field. Check the exact key your pipeline.py writes (per_stage_ms / timing / stage_timing_ms) and adjust this script's collect_latency() if it differs.

## 5. Anticipation Results — Table 1 material

- **_status**: NOT YET AVAILABLE — run: python scripts/eval_anticipation.py (requires the eval split from prepare_nexar.py)

## 6. Forecasting ADE/FDE — Table 2 ablation material

Raw report found on disk — copy directly into the paper:

```
# Forecast eval (val = 289 windows, units: px)

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 18.21 | 27.93 | 31.22 | 55.40 | 61.27 | 122.55 |
| Seq2Seq-LSTM | 18.85 | 30.67 | 32.68 | 56.33 | 56.33 | 99.74 |
```

## 7. Config Variants Actually Run

- **forecaster_variants_run**:
  - kalman_cv
- **forecast_space_variants_run**:
  - none recorded
- **base_thresholds_tested**:
  - 0.75
