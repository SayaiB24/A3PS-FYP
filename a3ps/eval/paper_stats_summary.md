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
- **base_threshold**: 0.65
- **caution_threshold**: 0.4
- **alert_margin**: 0.15
- **frames_to_confirm**: 3
- **event_cooldown_s**: 2.0
- **threshold_floor**: 0.45
- **risk_ema_alpha**: 0.4
- **mask_poly_max_points**: 40
- **forecaster**: kalman_cv
- **seq2seq_weights**: notebooks/models/seq2seq_v1.pt
- **dynamic_threshold**: True
- **ema_smoothing**: True
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

- **_status**: pandas not installed — cannot parse index.csv

## 4. Per-Stage Latency — Table 3 material

- **_clips_measured**: 33
- **perception**:
  - **mean_ms_per_frame**: 0.0
  - **std_ms_per_frame**: 0.0
  - **n_samples**: 30
- **tracking**:
  - **mean_ms_per_frame**: 70.0
  - **std_ms_per_frame**: 9.11
  - **n_samples**: 30
- **forecasting**:
  - **mean_ms_per_frame**: 3.04
  - **std_ms_per_frame**: 4.71
  - **n_samples**: 30
- **risk_decision**:
  - **mean_ms_per_frame**: 0.4
  - **std_ms_per_frame**: 0.55
  - **n_samples**: 30
- **_end_to_end_mean_ms_per_frame**: 73.44
- **_implied_fps**: 13.6

## 5. Anticipation Results — Table 1 material

Raw report found on disk — copy directly into the paper:

```
# Anticipation eval (split: eval)

- clips scored: **120** (60 positive, 60 negative)
- thresholds (A3PS): base **0.75**, alert **0.6** (base - alert_margin), floor **0.45**
- reactive baseline: BEV < 2.0 m (else img < 8% of frame height) to the ego corridor

| method | detection rate | false-alarm rate | mTTA (s) | useful warning rate | fired too early |
|---|---|---|---|---|---|
| **A3PS (proactive)** | 0.917 | 0.767 | 16.11 (n=55) | 0.050 (3/60) | 52/60 |
| Reactive-proximity (baseline) | 0.967 | 0.783 | 17.92 (n=58) | 0.033 (2/60) | 56/60 |

_Useful-warning window: **[alert_time_s - 0.00s, event_time_s] per clip (ground truth, not a chosen window)**. `time_of_alert` is Nexar's own annotation of the earliest actionable moment, so this window is the dataset's, not ours. The measured lead (`time_of_event - time_of_alert`) is 2.97-4.47 s over the 65 local positives (mean 3.49, sd 0.40)._

_The mTTA row above is averaged over each method's own true-positive set (different n, different clips) -- do NOT read the difference between these two mTTA values as an anticipation-gain claim. See the matched-subset comparison below for that._

_**Read the useful-warning-rate column, not mTTA, to judge whether warnings are actionable.** mTTA rewards warning early without bound, so a method that alarms on the first frame of every clip maximises it while telling the driver nothing -- which is exactly what the reactive baseline does here. The useful warning rate instead counts positives warned inside the dataset's own actionable window; misses, too-late alarms and alarms fired before `time_of_alert` all count against it. The 'fired too early' column isolates that last failure mode._

## Matched-subset mTTA (fair, same-clips comparison)

Over the **54** positive clip(s) BOTH methods correctly anticipate (removes the recall/false-alarm-driven bias of comparing mTTA across each method's own, differently-sized true-positive set):

- A3PS mTTA: **16.17 s**
- Reactive-baseline mTTA: **18.24 s**
- **Anticipation gain: A3PS is 2.07 s later** than the reactive baseline on this matched subset.

## Official-style Nexar AP at pre-event cutoffs

Each clip is ranked by the highest collision probability the model held using ONLY the frames it would have seen had the video been cut the stated interval before the annotated event. This is the dataset's own anticipation protocol, independent of our alert thresholds and decision state machine.

| cutoff before event | AP | clips scored |
|---|---|---|
| 500 ms | 0.547 | 120 (60 pos) |
| 1000 ms | 0.546 | 120 (60 pos) |
| 1500 ms | 0.546 | 120 (60 pos) |

**mean AP over the three cutoffs: 0.546**

_negatives scored over their full clip (official protocol; gives negatives more frames than positives, so this AP is a lower bound)._

_A3PS AP (whole-clip peak-prob ranking): 0.562 -- uses every frame including post-event ones, so it is NOT an anticipation number; compare the cutoff APs above instead._
```

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
