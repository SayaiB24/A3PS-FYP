# BoT-SORT association quality vs frame rate

- clips: **10** from split `eval`
- window: 13.0s (+1.0s tail after the event)
- device: `cuda`

| rate | tracks/clip | mean track (s) | **frac >=2s** | actors/frame | switches/s | inference (s/clip) |
|---|---|---|---|---|---|---|
| 30 Hz | 44.7 | 1.57 | **0.231** | 4.35 | 3.45 | 19.0 |
| 15 Hz | 31.5 | 2.04 | **0.292** | 4.16 | 2.43 | 10.6 |

## 15 Hz vs 30 Hz

- forecastable tracks (>=2 s): **+26.5%**
- actors per frame: **-4.3%**
- inference speed-up: **1.79x**

`frac >=2s` is the number that decides this: the trajectory buffer needs 2s of continuous history before it forecasts at all, so tracks shorter than that contribute no features rather than noisier ones.

**Verdict: borderline.** 15 Hz loses a few percent of forecastable tracks. Acceptable if the GPU hours matter; try an intermediate rate (15 Hz) first.
