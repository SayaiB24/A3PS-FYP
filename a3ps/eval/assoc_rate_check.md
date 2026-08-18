# BoT-SORT association quality vs frame rate

- clips: **10** from split `eval`
- window: 13.0s (+1.0s tail after the event)
- device: `cuda`

| rate | tracks/clip | mean track (s) | **frac >=2s** | actors/frame | switches/s | inference (s/clip) |
|---|---|---|---|---|---|---|
| 30 Hz | 44.7 | 1.57 | **0.231** | 4.35 | 3.45 | 20.9 |
| 10 Hz | 25.2 | 2.34 | **0.349** | 4.00 | 1.95 | 8.4 |

## 10 Hz vs 30 Hz

- forecastable tracks (>=2 s): **+50.9%**
- actors per frame: **-7.9%**
- inference speed-up: **2.50x**

`frac >=2s` is the number that decides this: the trajectory buffer needs 2s of continuous history before it forecasts at all, so tracks shorter than that contribute no features rather than noisier ones.

**Verdict: borderline.** 10 Hz loses a few percent of forecastable tracks. Acceptable if the GPU hours matter; try an intermediate rate (15 Hz) first.
