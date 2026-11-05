# predictiq-research

Models, conformal prediction intervals and the experiment harness for the PredictIQ study:
**do uncertainty estimates for remaining-useful-life predictions stay valid under distribution
shift?**

## Implemented

| Area | Contents |
|---|---|
| Sequence models | LSTM, GRU, causal dilated TCN (residual blocks), pre-norm Transformer encoder with sinusoidal positions |
| Classical baselines | Ridge and histogram gradient boosting on hand-built window features (mean, std, min, max, last value, slope per sensor) |
| Uncertainty | seed-ensemble mean and standard deviation; **split conformal**, **normalized conformal** (ensemble σ as scale) and **binned (Mondrian-style) conformal**, written from scratch with the finite-sample quantile correction |
| Metrics | RMSE, MAE, NASA asymmetric score, coverage with Wilson interval, interval width, coverage on critical RUL ≤ 30 |
| Experiments | in-distribution runs per subset; shift runs (source models applied to a target subset) comparing source calibration with recalibration on labelled target engines |
| Reporting | `predictiq-research report` renders result JSON into Markdown tables |

## Protocol

- Training engines are split into disjoint train / validation / calibration sets.
- Early stopping uses only validation engines; conformal calibration uses only calibration
  engines; the official test engines are used for reporting only.
- Two calibration sets from the same engines: `all` (every cycle) and `trunc` (random cut-offs
  per engine, which mimics how the official test engines were truncated). They answer
  different questions and are both reported.
- Test targets are the capped RUL (cap 125), matching the training target.
- Intervals are clipped to the feasible range [0, cap]; clipping never removes the truth, so
  it cannot reduce coverage.

## Install and test

```bash
git clone https://github.com/saaki05/project-2 && cd project-2
pip install -e predictiq-data-pipeline
cd predictiq-research && pip install -e ".[dev]" && pytest
```

## Reproduce the study

```bash
scripts/run_all.sh /path/to/CMAPSSData results      # FD001-FD004 + shift experiments
predictiq-research report --results results --out results/REPORT.md
```

Single steps: `predictiq-research run --subset FD001 ...` and
`predictiq-research shift --source FD001 --target FD002 ...`. The CLI uses a CUDA GPU
automatically if one is present. Wall-clock on a 4-core CPU is hours for the full study.

## Results

See `results/REPORT.md`, generated from `results/*.json`; no number in this repository is
typed by hand. If a table is missing from the report, that experiment has not been run.

## Limitations

- Split conformal assumes exchangeability between calibration and test data. That is exactly
  what the shift experiments probe, and the results should be read as measurements of how it
  fails, not as guarantees.
- The test set has one final window per engine (100–250 engines), so coverage estimates carry
  sampling error of a few percentage points; the report prints Wilson intervals.
- Results are for the capped-RUL target. Intervals for un-capped RUL are not evaluated.
- Seed ensembles of five members give a noisy σ estimate; normalized conformal compensates
  only partly.

## License

MIT.
