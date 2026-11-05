# PredictIQ results (generated from result JSON files)

Target interval coverage: 90%. RUL cap: 125. Window: 30. Seeds: [0, 1, 2, 3, 4]. Device: cpu.

## Point accuracy on the official test engines (last window)

RMSE/MAE in cycles against the capped true RUL; `ens` is the seed ensemble, `single` is the mean ± std over individual seeds.

| subset | model | RMSE (ens) | RMSE (single) | MAE (ens) | NASA score (ens) |
|---|---|---|---|---|---|
| FD001 | ridge | 16.12 | n/a | 13.22 | 448 |
| FD001 | hgb | 14.28 | n/a | 10.50 | 399 |
| FD001 | lstm | 13.43 | 13.99 ± 0.42 | 9.86 | 288 |
| FD001 | gru | 13.82 | 14.14 ± 0.21 | 10.12 | 370 |
| FD001 | tcn | 14.68 | 15.42 ± 0.32 | 11.10 | 334 |
| FD001 | transformer | 13.48 | 14.30 ± 0.37 | 9.96 | 325 |
| FD002 | ridge | 16.40 | n/a | 13.05 | 1263 |
| FD002 | hgb | 13.57 | n/a | 9.90 | 774 |
| FD002 | lstm | 14.13 | 14.51 ± 0.15 | 10.24 | 1052 |
| FD002 | gru | 13.52 | 13.85 ± 0.21 | 9.81 | 885 |
| FD002 | tcn | 14.12 | 14.61 ± 0.01 | 10.43 | 1029 |
| FD002 | transformer | 13.08 | 13.75 ± 0.09 | 9.42 | 799 |
| FD003 | ridge | 16.87 | n/a | 13.57 | 549 |
| FD003 | hgb | 14.14 | n/a | 10.08 | 382 |
| FD003 | lstm | 14.91 | 15.44 ± 0.76 | 10.44 | 926 |
| FD003 | gru | 13.66 | 13.99 ± 0.39 | 9.41 | 429 |
| FD003 | tcn | 15.92 | 16.51 ± 1.18 | 11.33 | 982 |
| FD003 | transformer | 13.96 | 14.45 ± 0.23 | 9.39 | 502 |
| FD004 | ridge | 19.62 | n/a | 15.21 | 2025 |
| FD004 | hgb | 15.80 | n/a | 10.81 | 1286 |
| FD004 | lstm | 14.67 | 15.15 ± 0.16 | 10.68 | 1090 |
| FD004 | gru | 14.77 | 15.06 ± 0.35 | 10.75 | 1086 |
| FD004 | tcn | 16.86 | 17.46 ± 0.85 | 11.96 | 1697 |
| FD004 | transformer | 14.71 | 15.34 ± 0.15 | 9.59 | 1501 |

## Interval validity on the official test engines

Coverage with a 95% Wilson interval and mean width. `cal=trunc` calibrates on random cut-offs of held-out engines; `cal=all` on every cycle. Conformal variant: normalized (ensemble σ) when available, else split.

| subset | model | protocol | variant | coverage [95% CI] | width | coverage RUL≤30 (n) |
|---|---|---|---|---|---|---|
| FD001 | ridge | cal=trunc | split | 0.940 [0.88, 0.97] | 49.6 | 1.00 (25) |
| FD001 | ridge | cal=all | split | 0.910 [0.84, 0.95] | 46.7 | 0.92 (25) |
| FD001 | hgb | cal=trunc | split | 0.910 [0.84, 0.95] | 43.1 | 1.00 (25) |
| FD001 | hgb | cal=all | split | 0.910 [0.84, 0.95] | 43.0 | 1.00 (25) |
| FD001 | lstm | cal=trunc | normalized | 0.870 [0.79, 0.92] | 38.1 | 1.00 (25) |
| FD001 | lstm | cal=all | normalized | 0.910 [0.84, 0.95] | 43.8 | 1.00 (25) |
| FD001 | gru | cal=trunc | normalized | 0.880 [0.80, 0.93] | 43.2 | 1.00 (25) |
| FD001 | gru | cal=all | normalized | 0.910 [0.84, 0.95] | 46.5 | 1.00 (25) |
| FD001 | tcn | cal=trunc | normalized | 0.890 [0.81, 0.94] | 50.2 | 1.00 (25) |
| FD001 | tcn | cal=all | normalized | 0.890 [0.81, 0.94] | 50.1 | 1.00 (25) |
| FD001 | transformer | cal=trunc | normalized | 0.920 [0.85, 0.96] | 45.4 | 1.00 (25) |
| FD001 | transformer | cal=all | normalized | 0.920 [0.85, 0.96] | 44.8 | 1.00 (25) |
| FD002 | ridge | cal=trunc | split | 0.946 [0.91, 0.97] | 53.2 | 1.00 (61) |
| FD002 | ridge | cal=all | split | 0.931 [0.89, 0.96] | 50.4 | 0.98 (61) |
| FD002 | hgb | cal=trunc | split | 0.961 [0.93, 0.98] | 49.6 | 1.00 (61) |
| FD002 | hgb | cal=all | split | 0.942 [0.91, 0.96] | 46.1 | 1.00 (61) |
| FD002 | lstm | cal=trunc | normalized | 0.884 [0.84, 0.92] | 47.1 | 1.00 (61) |
| FD002 | lstm | cal=all | normalized | 0.884 [0.84, 0.92] | 47.1 | 1.00 (61) |
| FD002 | gru | cal=trunc | normalized | 0.861 [0.81, 0.90] | 39.6 | 0.97 (61) |
| FD002 | gru | cal=all | normalized | 0.892 [0.85, 0.92] | 43.5 | 0.97 (61) |
| FD002 | tcn | cal=trunc | normalized | 0.903 [0.86, 0.93] | 55.5 | 1.00 (61) |
| FD002 | tcn | cal=all | normalized | 0.896 [0.85, 0.93] | 52.6 | 1.00 (61) |
| FD002 | transformer | cal=trunc | normalized | 0.911 [0.87, 0.94] | 45.8 | 1.00 (61) |
| FD002 | transformer | cal=all | normalized | 0.923 [0.88, 0.95] | 49.8 | 1.00 (61) |
| FD003 | ridge | cal=trunc | split | 0.920 [0.85, 0.96] | 50.3 | 1.00 (20) |
| FD003 | ridge | cal=all | split | 0.870 [0.79, 0.92] | 45.6 | 1.00 (20) |
| FD003 | hgb | cal=trunc | split | 0.890 [0.81, 0.94] | 43.0 | 1.00 (20) |
| FD003 | hgb | cal=all | split | 0.880 [0.80, 0.93] | 37.6 | 1.00 (20) |
| FD003 | lstm | cal=trunc | normalized | 0.870 [0.79, 0.92] | 37.0 | 0.95 (20) |
| FD003 | lstm | cal=all | normalized | 0.880 [0.80, 0.93] | 37.5 | 0.95 (20) |
| FD003 | gru | cal=trunc | normalized | 0.860 [0.78, 0.91] | 42.4 | 0.95 (20) |
| FD003 | gru | cal=all | normalized | 0.820 [0.73, 0.88] | 35.5 | 0.95 (20) |
| FD003 | tcn | cal=trunc | normalized | 0.950 [0.89, 0.98] | 51.2 | 1.00 (20) |
| FD003 | tcn | cal=all | normalized | 0.890 [0.81, 0.94] | 43.6 | 1.00 (20) |
| FD003 | transformer | cal=trunc | normalized | 0.860 [0.78, 0.91] | 39.9 | 0.95 (20) |
| FD003 | transformer | cal=all | normalized | 0.820 [0.73, 0.88] | 34.6 | 0.90 (20) |
| FD004 | ridge | cal=trunc | split | 0.883 [0.84, 0.92] | 51.0 | 0.92 (53) |
| FD004 | ridge | cal=all | split | 0.883 [0.84, 0.92] | 51.3 | 0.92 (53) |
| FD004 | hgb | cal=trunc | split | 0.895 [0.85, 0.93] | 44.3 | 0.98 (53) |
| FD004 | hgb | cal=all | split | 0.891 [0.85, 0.92] | 42.8 | 0.98 (53) |
| FD004 | lstm | cal=trunc | normalized | 0.887 [0.84, 0.92] | 43.9 | 0.98 (53) |
| FD004 | lstm | cal=all | normalized | 0.907 [0.86, 0.94] | 47.3 | 0.98 (53) |
| FD004 | gru | cal=trunc | normalized | 0.863 [0.81, 0.90] | 41.9 | 1.00 (53) |
| FD004 | gru | cal=all | normalized | 0.887 [0.84, 0.92] | 43.9 | 1.00 (53) |
| FD004 | tcn | cal=trunc | normalized | 0.831 [0.78, 0.87] | 44.1 | 0.96 (53) |
| FD004 | tcn | cal=all | normalized | 0.839 [0.79, 0.88] | 45.7 | 0.96 (53) |
| FD004 | transformer | cal=trunc | normalized | 0.847 [0.80, 0.89] | 37.2 | 0.94 (53) |
| FD004 | transformer | cal=all | normalized | 0.863 [0.81, 0.90] | 40.2 | 0.96 (53) |

## Distribution shift

Models trained on the source subset, evaluated on the target subset. Rows compare in-domain calibration, source calibration under shift and recalibration on labelled target engines.

| source → target | model | RMSE src | RMSE tgt | setting | coverage [95% CI] | width |
|---|---|---|---|---|---|---|
| FD001 → FD002 | lstm | 13.43 | 19.61 | in_domain (src cal -> src test) | 0.870 [0.79, 0.92] | 38.1 |
| FD001 → FD002 | lstm | 13.43 | 19.61 | shifted, source calibration (src cal -> tgt test) | 0.772 [0.72, 0.82] | 47.7 |
| FD001 → FD002 | lstm | 13.43 | 19.61 | shifted, target recalibration (tgt cal -> tgt test) | 0.857 [0.81, 0.89] | 55.9 |
| FD001 → FD002 | gru | 13.82 | 20.58 | in_domain (src cal -> src test) | 0.880 [0.80, 0.93] | 43.2 |
| FD001 → FD002 | gru | 13.82 | 20.58 | shifted, source calibration (src cal -> tgt test) | 0.849 [0.80, 0.89] | 52.6 |
| FD001 → FD002 | gru | 13.82 | 20.58 | shifted, target recalibration (tgt cal -> tgt test) | 0.900 [0.86, 0.93] | 60.2 |
| FD001 → FD002 | tcn | 14.68 | 22.53 | in_domain (src cal -> src test) | 0.890 [0.81, 0.94] | 50.2 |
| FD001 → FD002 | tcn | 14.68 | 22.53 | shifted, source calibration (src cal -> tgt test) | 0.965 [0.94, 0.98] | 73.9 |
| FD001 → FD002 | tcn | 14.68 | 22.53 | shifted, target recalibration (tgt cal -> tgt test) | 0.900 [0.86, 0.93] | 62.8 |
| FD001 → FD002 | transformer | 13.48 | 18.87 | in_domain (src cal -> src test) | 0.920 [0.85, 0.96] | 45.4 |
| FD001 → FD002 | transformer | 13.48 | 18.87 | shifted, source calibration (src cal -> tgt test) | 0.846 [0.80, 0.88] | 49.6 |
| FD001 → FD002 | transformer | 13.48 | 18.87 | shifted, target recalibration (tgt cal -> tgt test) | 0.876 [0.83, 0.91] | 54.6 |
| FD001 → FD003 | lstm | 13.43 | 21.17 | in_domain (src cal -> src test) | 0.870 [0.79, 0.92] | 38.1 |
| FD001 → FD003 | lstm | 13.43 | 21.17 | shifted, source calibration (src cal -> tgt test) | 0.870 [0.79, 0.92] | 53.4 |
| FD001 → FD003 | lstm | 13.43 | 21.17 | shifted, target recalibration (tgt cal -> tgt test) | 0.910 [0.84, 0.95] | 65.6 |
| FD001 → FD003 | gru | 13.82 | 20.04 | in_domain (src cal -> src test) | 0.880 [0.80, 0.93] | 43.2 |
| FD001 → FD003 | gru | 13.82 | 20.04 | shifted, source calibration (src cal -> tgt test) | 0.840 [0.76, 0.90] | 49.6 |
| FD001 → FD003 | gru | 13.82 | 20.04 | shifted, target recalibration (tgt cal -> tgt test) | 0.840 [0.76, 0.90] | 51.8 |
| FD001 → FD003 | tcn | 14.68 | 29.30 | in_domain (src cal -> src test) | 0.890 [0.81, 0.94] | 50.2 |
| FD001 → FD003 | tcn | 14.68 | 29.30 | shifted, source calibration (src cal -> tgt test) | 0.790 [0.70, 0.86] | 64.2 |
| FD001 → FD003 | tcn | 14.68 | 29.30 | shifted, target recalibration (tgt cal -> tgt test) | 0.800 [0.71, 0.87] | 65.4 |
| FD001 → FD003 | transformer | 13.48 | 22.05 | in_domain (src cal -> src test) | 0.920 [0.85, 0.96] | 45.4 |
| FD001 → FD003 | transformer | 13.48 | 22.05 | shifted, source calibration (src cal -> tgt test) | 0.810 [0.72, 0.87] | 51.6 |
| FD001 → FD003 | transformer | 13.48 | 22.05 | shifted, target recalibration (tgt cal -> tgt test) | 0.860 [0.78, 0.91] | 59.9 |
| FD001 → FD004 | lstm | 13.43 | 22.88 | in_domain (src cal -> src test) | 0.870 [0.79, 0.92] | 38.1 |
| FD001 → FD004 | lstm | 13.43 | 22.88 | shifted, source calibration (src cal -> tgt test) | 0.766 [0.71, 0.81] | 47.6 |
| FD001 → FD004 | lstm | 13.43 | 22.88 | shifted, target recalibration (tgt cal -> tgt test) | 0.915 [0.87, 0.94] | 64.9 |
| FD001 → FD004 | gru | 13.82 | 23.44 | in_domain (src cal -> src test) | 0.880 [0.80, 0.93] | 43.2 |
| FD001 → FD004 | gru | 13.82 | 23.44 | shifted, source calibration (src cal -> tgt test) | 0.823 [0.77, 0.87] | 49.1 |
| FD001 → FD004 | gru | 13.82 | 23.44 | shifted, target recalibration (tgt cal -> tgt test) | 0.871 [0.82, 0.91] | 62.9 |
| FD001 → FD004 | tcn | 14.68 | 30.00 | in_domain (src cal -> src test) | 0.890 [0.81, 0.94] | 50.2 |
| FD001 → FD004 | tcn | 14.68 | 30.00 | shifted, source calibration (src cal -> tgt test) | 0.871 [0.82, 0.91] | 74.5 |
| FD001 → FD004 | tcn | 14.68 | 30.00 | shifted, target recalibration (tgt cal -> tgt test) | 0.855 [0.81, 0.89] | 70.3 |
| FD001 → FD004 | transformer | 13.48 | 22.72 | in_domain (src cal -> src test) | 0.920 [0.85, 0.96] | 45.4 |
| FD001 → FD004 | transformer | 13.48 | 22.72 | shifted, source calibration (src cal -> tgt test) | 0.823 [0.77, 0.87] | 51.4 |
| FD001 → FD004 | transformer | 13.48 | 22.72 | shifted, target recalibration (tgt cal -> tgt test) | 0.919 [0.88, 0.95] | 69.3 |
| FD002 → FD004 | lstm | 14.13 | 23.81 | in_domain (src cal -> src test) | 0.884 [0.84, 0.92] | 47.1 |
| FD002 → FD004 | lstm | 14.13 | 23.81 | shifted, source calibration (src cal -> tgt test) | 0.859 [0.81, 0.90] | 61.2 |
| FD002 → FD004 | lstm | 14.13 | 23.81 | shifted, target recalibration (tgt cal -> tgt test) | 0.899 [0.86, 0.93] | 67.4 |
| FD002 → FD004 | gru | 13.52 | 23.14 | in_domain (src cal -> src test) | 0.861 [0.81, 0.90] | 39.6 |
| FD002 → FD004 | gru | 13.52 | 23.14 | shifted, source calibration (src cal -> tgt test) | 0.819 [0.77, 0.86] | 49.4 |
| FD002 → FD004 | gru | 13.52 | 23.14 | shifted, target recalibration (tgt cal -> tgt test) | 0.899 [0.86, 0.93] | 62.1 |
| FD002 → FD004 | tcn | 14.12 | 25.67 | in_domain (src cal -> src test) | 0.903 [0.86, 0.93] | 55.5 |
| FD002 → FD004 | tcn | 14.12 | 25.67 | shifted, source calibration (src cal -> tgt test) | 0.875 [0.83, 0.91] | 68.9 |
| FD002 → FD004 | tcn | 14.12 | 25.67 | shifted, target recalibration (tgt cal -> tgt test) | 0.964 [0.93, 0.98] | 85.4 |
| FD002 → FD004 | transformer | 13.08 | 24.69 | in_domain (src cal -> src test) | 0.911 [0.87, 0.94] | 45.8 |
| FD002 → FD004 | transformer | 13.08 | 24.69 | shifted, source calibration (src cal -> tgt test) | 0.774 [0.72, 0.82] | 49.5 |
| FD002 → FD004 | transformer | 13.08 | 24.69 | shifted, target recalibration (tgt cal -> tgt test) | 0.895 [0.85, 0.93] | 63.8 |
| FD003 → FD004 | lstm | 14.91 | 22.29 | in_domain (src cal -> src test) | 0.870 [0.79, 0.92] | 37.0 |
| FD003 → FD004 | lstm | 14.91 | 22.29 | shifted, source calibration (src cal -> tgt test) | 0.875 [0.83, 0.91] | 49.2 |
| FD003 → FD004 | lstm | 14.91 | 22.29 | shifted, target recalibration (tgt cal -> tgt test) | 0.948 [0.91, 0.97] | 68.3 |
| FD003 → FD004 | gru | 13.66 | 22.97 | in_domain (src cal -> src test) | 0.860 [0.78, 0.91] | 42.4 |
| FD003 → FD004 | gru | 13.66 | 22.97 | shifted, source calibration (src cal -> tgt test) | 0.883 [0.84, 0.92] | 57.0 |
| FD003 → FD004 | gru | 13.66 | 22.97 | shifted, target recalibration (tgt cal -> tgt test) | 0.952 [0.92, 0.97] | 74.2 |
| FD003 → FD004 | tcn | 15.92 | 23.32 | in_domain (src cal -> src test) | 0.950 [0.89, 0.98] | 51.2 |
| FD003 → FD004 | tcn | 15.92 | 23.32 | shifted, source calibration (src cal -> tgt test) | 0.875 [0.83, 0.91] | 60.0 |
| FD003 → FD004 | tcn | 15.92 | 23.32 | shifted, target recalibration (tgt cal -> tgt test) | 0.911 [0.87, 0.94] | 67.1 |
| FD003 → FD004 | transformer | 13.96 | 20.21 | in_domain (src cal -> src test) | 0.860 [0.78, 0.91] | 39.9 |
| FD003 → FD004 | transformer | 13.96 | 20.21 | shifted, source calibration (src cal -> tgt test) | 0.899 [0.86, 0.93] | 56.2 |
| FD003 → FD004 | transformer | 13.96 | 20.21 | shifted, target recalibration (tgt cal -> tgt test) | 0.931 [0.89, 0.96] | 68.0 |

## Summary: in-distribution coverage (neural models, nominal 0.90)

Mean coverage over LSTM/GRU/TCN/Transformer per subset and calibration protocol (each model has n = number of test engines).

| subset | cal=trunc | cal=all | min over models (trunc) | max over models (trunc) |
|---|---|---|---|---|
| FD001 | 0.890 | 0.907 | 0.870 | 0.920 |
| FD002 | 0.890 | 0.899 | 0.861 | 0.911 |
| FD003 | 0.885 | 0.853 | 0.860 | 0.950 |
| FD004 | 0.857 | 0.874 | 0.831 | 0.887 |

## Summary: shift (neural models, mean over architectures)

| source → target | RMSE source | RMSE target | coverage in-domain | coverage shifted (source cal.) | coverage shifted (target recal.) | width in-domain | width shifted (source cal.) | width recal. |
|---|---|---|---|---|---|---|---|---|
| FD001 → FD002 | 13.85 | 20.39 | 0.890 | 0.858 | 0.883 | 44.2 | 56.0 | 58.4 |
| FD001 → FD003 | 13.85 | 23.14 | 0.890 | 0.828 | 0.853 | 44.2 | 54.7 | 60.7 |
| FD001 → FD004 | 13.85 | 24.76 | 0.890 | 0.821 | 0.890 | 44.2 | 55.6 | 66.8 |
| FD002 → FD004 | 13.71 | 24.33 | 0.890 | 0.832 | 0.914 | 47.0 | 57.3 | 69.6 |
| FD003 → FD004 | 14.61 | 22.20 | 0.885 | 0.883 | 0.935 | 42.6 | 55.6 | 69.4 |
