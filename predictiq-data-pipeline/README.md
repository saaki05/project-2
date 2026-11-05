# predictiq-data-pipeline

Data layer for PredictIQ (remaining-useful-life prediction on NASA C-MAPSS turbofan data).
It owns everything between the raw text files and model-ready tensors.

## What it does

- **Loader** for `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt` with validation
  (26 columns, no missing values, consecutive cycles per engine, RUL file length matches the
  number of test engines).
- **RUL targets**: training RUL counted down to each engine's last cycle; test RUL
  reconstructed for every cycle from the provided final RUL; optional piecewise-linear cap
  (default 125 cycles, a common convention).
- **Regime-aware normalisation** (`Preprocessor`): operating-setting clusters (k-means on the
  standardised settings; 6 regimes for FD002/FD004, 1 otherwise) with per-regime, per-sensor
  standardisation, fitted on training rows only and serialisable to plain arrays.
- **Windowing**: one window per cycle, front-padded by repeating the first row, never
  mixing engines; `last_only` gives the official test protocol (final window per engine).
- **Engine-wise splits**: `split_units` produces disjoint train / validation / calibration
  engine sets; `audit_unit_split` raises if any engine appears twice.
- A checksummed on-disk format (`arrays.npz` + `manifest.json`).

## Install and test

```bash
git clone https://github.com/saaki05/project-2 && cd project-2/predictiq-data-pipeline
pip install -e ".[dev]" && pytest
# with the real data (see below), additionally:
PREDICTIQ_CMAPSS_DIR=/path/to/CMAPSSData pytest
```

## Data

The NASA Prognostics Center of Excellence repository distributes the C-MAPSS files
("Turbofan Engine Degradation Simulation Data Set"). Download and extract them yourself and
follow the repository's terms of use; this repository does not redistribute the data.
Citation: A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation*, PHM08.

Note: the dataset's readme lists FD004 as 248 train / 249 test engines; the files contain
249 train / 248 test (the loader checks this against the RUL file and the test suite encodes
the observed counts).

```bash
predictiq-data prepare --raw /path/to/CMAPSSData --subset FD001 --out data/fd001
```

## License

MIT.
