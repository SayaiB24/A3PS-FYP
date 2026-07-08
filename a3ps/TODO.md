# a3ps — TODO

## Data: feed in the actual Nexar dataset

The pipeline currently runs on placeholder/fake data and a single test clip.
Before real evaluation, populate `data/nexar/` with the actual dataset.

**Where to put it**

```
data/nexar/
├── positive/          # collision / near-collision clips   -> label 1
│   └── *.mp4
├── negative/          # normal-driving clips                -> label 0
│   └── *.mp4
└── <annotation file>  # OPTIONAL .csv/.json with per-clip event times (positives)
```

- Videos: any of `.mp4 .mov .avi .mkv .webm`; subfolders are fine (walked recursively).
- Annotation file (optional, positives only): a `.csv`/`.json` in `data/nexar/`.
  `scripts/prepare_nexar.py` auto-detects an id column (`id`, `clip_id`,
  `filename`, ...) and an event-time column (`time_of_event`, `event_time_s`,
  `time_of_alert`, ...). If absent, `event_time_s` is left blank.

**Counts needed for full splits**

- negatives: >= 70  (10 for `dev` + 60 for `eval`, rest -> `train_traj`)
- positives: >= 65  (5 for `dev` + 60 for `eval`)

With fewer, `prepare_nexar.py` clamps split sizes and prints a warning.

**Then run**

```
python scripts/prepare_nexar.py --root data/nexar
```

Rebuilds `data/nexar/index.csv` and refreshes `data/dev_clips/dev01..devNN.mp4`
+ `data/dev_clips/README.md`.

**Cleanup when the real data lands**

- Remove the test copy `data/nexar/negative/02134.mp4` (a duplicate of the
  dev clip, added only for a smoke test). Left in place for now on purpose.

## Week 4: per-clip BEV calibration for the 5 final demo clips
When Option 2 becomes worth it: later, in Week 4, when you're preparing your 5 demo clips. At that point you have exactly 5 clips to worry about, and spending 30 minutes each to give them proper metric units for the final demo is a good investment. Do it then, not now.

The default ground-plane trapezoid is generic and miscalibrated for real clips
(verified on 02134.mp4: BEV velocities too low, some predicted depths negative).
Pipeline default is `forecast_space: img` (relative units) until then.

For each of the 5 final demo clips:
- [ ] Pick 4 road points in the frame + estimate their real-world distances.
- [ ] Save `dashboard/clips/<clip>/ground.yaml` (image_points_frac + ground_points_m).
- [ ] Set `ground_plane_override` (or per-clip config) and `forecast_space: bev`.
- [ ] Re-run `scripts/verify_bev.py` — velocities should read ~5-20 m/s and
      predictions should stay in-frame (>=80%).

## Trajectory mining (`scripts/mine_trajectories.py`) — NOT done in this step

The script + its windowing/normalization logic are implemented and unit-tested
(`tests/test_mining.py`, 8 tests). What remains — all blocked on the real
dataset, since `train_traj` is currently empty (only clip 02134 exists, in `dev`):

- [ ] Run on the real `train_traj` split and hit the target of **10k+ windows**
      (`python scripts/mine_trajectories.py`; overnight on the laptop is fine).
- [ ] Verify the `--plot` scatter on real traffic: paths should be **smooth**
      (jump filter worked) and **forward-biased** (red futures extend +y).
- [ ] Confirm `stats.json` class mix / filter tallies look sane on real data
      (~20% static retained; short-track & ID-switch drops non-trivial).
- [ ] Currently mines in **image pixels** (`forecast_space: img`, static_thresh
      15 px). Re-mine in **BEV metres** after Week 4 per-clip calibration if the
      LSTM is to train in metric space.
- [ ] Verified so far only via a `--split dev` smoke run on 02134 (not the real
      train set). Re-run once real data lands.

## LSTM forecaster (STRETCH, Week 3) — not started

- [ ] Train `notebooks/train_forecaster.ipynb` on the mined windows.
- [ ] Implement `a3ps/forecasting/seq2seq.py` (currently raises NotImplementedError).

## Dashboard: shift from dummy to real data (later)

The dashboard ships a dummy clip `fake_demo` (`scripts/make_demo_clip.py`) so the
console / timeline / banner can be built before the risk engine exists. It reads
ONLY the ClipResult schema, so the switch to real data needs no dashboard code
changes:

- [ ] After Phase 4 (risk engine) fills real `collision_prob` + emits events,
      run the pipeline on the demo clips.
- [ ] Edit `dashboard/clips/manifest.json` to list the real clip id(s); drop
      `fake_demo`.
- [ ] Verify threats / threshold / event-log / banner / risk-timeline (incl. the
      ▼ A3PS vs ▽ reactive-ADAS gap) all populate from the real events.

## Downstream (needs the real dataset in place)

Implemented and runnable now (on any clip, CPU): YOLOv8-Seg segmenter, BoT-SORT
tracker + TrajectoryBuffer, Kalman-CV forecaster, and `pipeline.py`
(-> `annotated.mp4` + `events.json`). Still to do:

- [ ] `scripts/eval_forecast.py` — ADE/FDE, Kalman vs LSTM (still a stub).
- [ ] `scripts/eval_anticipation.py` — mTTA + AP + false-alarm rate on `eval`
      (still a stub; also needs the risk/decision phase wired into the pipeline).
