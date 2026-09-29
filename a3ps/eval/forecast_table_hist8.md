# Forecast eval (val = 289 windows, units: px, hist_len=8 (truncated))

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 13.02 | 22.38 | 26.68 | 53.53 | 61.32 | 131.90 |
| Seq2Seq-LSTM | 18.82 | 30.60 | 32.62 | 56.24 | 56.29 | 99.64 |
