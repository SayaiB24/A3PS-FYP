# Forecast eval (val = 289 windows, units: px, hist_len=9 (truncated))

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 13.00 | 22.29 | 26.56 | 53.21 | 60.95 | 131.03 |
| Seq2Seq-LSTM | 18.82 | 30.62 | 32.63 | 56.27 | 56.33 | 99.77 |
