# PredictIQ

Predictive-maintenance research engineering on NASA's C-MAPSS turbofan degradation data:
**remaining-useful-life (RUL) prediction with prediction intervals whose validity is tested
under distribution shift.**

Three independently installable packages with one-way dependencies:

```
predictiq-data-pipeline  ──▶  predictiq-research  ──▶  predictiq-deployment
 load, normalise, window,      LSTM/GRU/TCN/Transformer,    export bundle, FastAPI
 engine-wise splits             conformal intervals,         service, drift checks
 (numpy, pandas, sklearn)       shift experiments (torch)
```

| Package | Responsibility |
|---|---|
| [`predictiq-data-pipeline`](predictiq-data-pipeline) | validated loading, RUL targets, regime-aware normalisation, windowing, leak-free engine splits |
| [`predictiq-research`](predictiq-research) | models, uncertainty methods (split / normalized / binned conformal written from scratch), experiment harness, report generator |
| [`predictiq-deployment`](predictiq-deployment) | model bundle export, `/predict` with interval and alert level, `/drift`, Dockerfile |

## Research question

Point accuracy on C-MAPSS is saturated, but a maintenance planner needs to know *how sure*
a prediction is. Split conformal prediction promises coverage under exchangeability, which
operating-condition and fault-mode changes break. We measure, per architecture:

1. How accurate are LSTM, GRU, TCN and Transformer regressors against feature-based
   baselines under one identical protocol (engine-wise splits, five seeds)?
2. Do conformal intervals reach their nominal coverage in-distribution, and does the
   calibration protocol (all cycles vs. random cut-offs that mimic test truncation) matter?
3. How much coverage is lost when models trained on FD001/FD003 (single operating condition)
   are applied to FD002/FD004 (six conditions, extra fault mode)? Does recalibrating on a few
   labelled target engines restore it?

The intended contribution is a careful empirical study with honest statistics, not a new
algorithm; a workshop paper on interval validity under shift in prognostics is the realistic
publication direction.

## Status

| Item | State |
|---|---|
| Data pipeline, models, conformal methods, harness, report generator, service | implemented; unit tests pass |
| Experiments on the real C-MAPSS data | see `predictiq-research/results/REPORT.md` for what has actually been run; anything missing there has not been run |
| Docker image | Dockerfile written, never built |
| Experiment-tracking server (MLflow) | not integrated |

## Quick start

```bash
git clone https://github.com/saaki05/project-2 && cd project-2
pip install -e predictiq-data-pipeline -e predictiq-research -e predictiq-deployment
# download and extract NASA C-MAPSS yourself, then:
cd predictiq-research && scripts/run_all.sh /path/to/CMAPSSData results
predictiq-research report --results results --out results/REPORT.md
```

Tests: `for d in predictiq-*; do (cd $d && pytest -q); done`. CI runs lint and tests for all
three packages (`.github/workflows/ci.yml`).

## Data and citation

C-MAPSS is distributed by the NASA Prognostics Center of Excellence; follow its terms of use.
A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM08. The data is not included here.

## License

MIT.
