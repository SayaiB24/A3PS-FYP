# Forecast eval (val = 289 windows, units: px, full 10-pt history)

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 12.94 | 22.16 | 26.41 | 52.89 | 60.62 | 130.36 |
| Seq2Seq-LSTM | 18.85 | 30.67 | 32.68 | 56.33 | 56.33 | 99.74 |
