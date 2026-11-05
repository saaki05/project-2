# predictiq-deployment

Serving layer for PredictIQ: turns the seed ensembles trained by `predictiq-research` into an
HTTP service that returns a remaining-useful-life estimate, a conformal interval and an
alert level.

## Pipeline

```
predictiq-research  --(saved member weights)-->  predictiq-export  -->  bundle.pt
bundle.pt  -->  predictiq-serve  -->  /predict  /drift  /health
```

`predictiq-export` loads the ensemble members, calibrates a normalized-conformal quantile on
held-out calibration engines (random cut-offs, as in the research protocol) and writes one
bundle: weights, the fitted regime preprocessor, window length, RUL cap, target coverage and
the calibrated quantile.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | architecture, training subset, ensemble size, window, coverage target |
| `POST /predict` | time-ordered raw readings (`op1..op3, s1..s21`, oldest first) → `rul`, `sigma`, `lower`, `upper`, `alert` (`ok` / `warning` if the lower bound ≤ threshold / `critical` if the upper bound ≤ threshold) |
| `POST /drift` | PSI and KS per column between a reference and a current batch |

Invalid shapes and non-finite values return HTTP 422.

## Install, test, run

```bash
git clone https://github.com/saaki05/project-2 && cd project-2
pip install -e predictiq-data-pipeline -e predictiq-research
cd predictiq-deployment && pip install -e ".[dev]" && pytest

predictiq-export --raw /path/to/CMAPSSData --subset FD001 --arch gru \
    --models-dir ../predictiq-research/results/models --out bundle.pt
predictiq-serve --bundle bundle.pt --port 8000
```

Docker (from the repository root): `docker build -f predictiq-deployment/Dockerfile -t predictiq .`
(the image has not been built in the development environment).

## Notes and limitations

- The interval's coverage guarantee holds only if production data resembles the calibration
  engines. Use `/drift` and the shift results in `predictiq-research` before trusting it on a
  different fleet or operating profile.
- A new engine with fewer cycles than the window is handled by front-padding, as in training;
  such predictions are less reliable.
- Bundles are loaded with `torch.load(weights_only=False)`: load only bundles you created.
- No experiment tracking server (MLflow) is wired in; run metadata is stored in the result
  JSON files.

## License

MIT.
