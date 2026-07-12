# Forecast eval — PREVIOUS run (val = 172 windows, units: px)
#
# Saved as a fallback so these numbers are never lost. This is the earlier
# run (smaller mined set, ~93 clips / 1,725 windows) where Kalman and the
# LSTM were a near-tie at 4 s (51.24 vs 51.64). The CURRENT run's numbers
# (larger 289-window val set) live in forecast_table.md.

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 15.24 | 23.59 | 26.27 | 46.42 | 51.24 | 102.00 |
| Seq2Seq-LSTM | 16.06 | 26.90 | 29.22 | 52.04 | 51.64 | 90.46 |
