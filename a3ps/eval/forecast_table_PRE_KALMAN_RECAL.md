# Forecast eval — PRE-RECALIBRATION run (val = 289 windows, units: px)
#
# Archived so these numbers are never lost. This is the run that was checked
# in as forecast_table.md before a3ps/forecasting/kalman_cv.py's MEAS_VAR /
# PROCESS_VAR constants were recalibrated (see that file's header) for
# honestly-dispersed std (RMSE(k)/std(k) ~= 1 across the horizon). That
# recalibration also changed the filter's *mean* prediction, not just its
# reported std, so Kalman's ADE/FDE moved too, even though the val split and
# the Seq2Seq-LSTM weights are unchanged (LSTM numbers are identical below and
# in the regenerated forecast_table.md, confirming the split itself didn't
# move). The regenerated numbers are noticeably better for Kalman at 1s/2s.
# Rerun 2026-09-28 alongside the padded-history truncation check
# (forecast_table_hist8.md / _hist9.md) that motivated re-checking this file.

| model | ADE@1s | FDE@1s | ADE@2s | FDE@2s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|---|
| Kalman-CV | 18.21 | 27.93 | 31.22 | 55.40 | 61.27 | 122.55 |
| Seq2Seq-LSTM | 18.85 | 30.67 | 32.68 | 56.33 | 56.33 | 99.74 |
