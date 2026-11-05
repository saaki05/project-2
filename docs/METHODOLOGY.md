# PredictIQ: methodology and paper outline

## Architecture

```mermaid
flowchart LR
  raw[C-MAPSS txt files] --> load[load_subset + validation]
  load --> prep[regime normalisation + windows]
  prep --> units[engine-wise train / val / calibration]
  units --> train[LSTM GRU TCN Transformer + baselines]
  train --> ens[seed ensemble: mean, sigma]
  ens --> cp[split / normalized / binned conformal]
  cp --> eval[coverage, width, RMSE, NASA score]
  ens --> export[predictiq-export bundle]
  export --> api[/predict /drift /health]
```

## Data

C-MAPSS FD001–FD004: simulated run-to-failure turbofan trajectories, 21 sensors and 3
operating settings. FD001/FD003 have one operating condition, FD002/FD004 six; FD003/FD004
add a second fault mode. Test engines are truncated before failure and come with the true
RUL at the last cycle.

## Protocol

- Target: piecewise-linear RUL, `min(RUL, 125)`; test truth is capped identically.
- 14 informative sensors; per-regime standardisation fitted on training rows.
- Window 30 cycles, front-padded; one window per cycle for training; the final window per
  engine for the official test metric; all windows reported separately.
- Training engines split 70/15/15 into train / validation (early stopping) / calibration
  (conformal). Test engines never influence any choice.
- Neural models: 5 seeds each; the seed ensemble provides the point prediction and σ.
- Conformal variants at nominal 90% coverage; calibration either on every calibration cycle
  (`all`) or on random cut-offs per engine (`trunc`), the latter matching how test engines
  were truncated.
- Shift experiments: models trained on a source subset are evaluated on a target subset
  with (a) the source calibration, (b) calibration on 25 labelled target training engines.

## Metrics

RMSE, MAE, NASA asymmetric score; empirical coverage with Wilson interval, mean width,
coverage restricted to true RUL ≤ 30 (the operationally critical region).

## Threats to validity

- Simulated data; real fleets differ.
- Coverage is estimated on 100–250 test engines (one window each): sampling error is a few
  points. Wilson intervals are reported.
- Early-stopping and calibration engines come from the training files, reducing training data.
- Capping RUL makes the task easier near the start of life; results do not transfer to
  un-capped RUL.

## Paper outline (fill from `results/REPORT.md` only)

1. Introduction: calibrated uncertainty in prognostics.
2. Background: C-MAPSS, conformal prediction, exchangeability under shift.
3. Method: models, ensembles, three conformal variants, calibration protocols.
4. Experiments: accuracy table; in-distribution coverage and the effect of calibration
   protocol; coverage under operating-condition and fault-mode shift; recalibration.
5. Discussion: when to trust intervals; monitoring with PSI/KS.
6. Limitations and conclusion.
