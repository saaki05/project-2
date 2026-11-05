import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from predictiq_research.experiments import run_subset
from predictiq_research.train import TrainConfig

from predictiq_service.api import create_app
from predictiq_service.bundle import load_bundle
from predictiq_service.drift import drift_report, ks_statistic, psi
from predictiq_service.export import export_bundle


def write_fake(root, subset="FD001", n_train=30, n_test=10, seed=0):
    rng = np.random.default_rng(seed)

    def engine(unit, length):
        t = np.arange(1, length + 1)
        deg = (t / length)[:, None] * np.linspace(1, 2, 21)[None, :] * 3
        sens = 100 + deg + rng.normal(scale=0.3, size=(length, 21))
        return np.column_stack([np.full(length, unit), t, np.zeros((length, 3)), sens])

    np.savetxt(root / f"train_{subset}.txt",
               np.vstack([engine(u, int(rng.integers(60, 90))) for u in range(1, n_train + 1)]),
               fmt="%.4f")
    lengths = [int(rng.integers(30, 60)) for _ in range(n_test)]
    np.savetxt(root / f"test_{subset}.txt",
               np.vstack([engine(u, n) for u, n in enumerate(lengths, 1)]), fmt="%.4f")
    np.savetxt(root / f"RUL_{subset}.txt", rng.integers(10, 60, n_test), fmt="%d")


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("svc")
    raw = root / "raw"
    raw.mkdir()
    write_fake(raw)
    cfg = TrainConfig(hidden=16, epochs=3, patience=3)
    run_subset(raw, "FD001", ["gru"], [], [0, 1, 2], cfg, 0.1, root / "out", window=10,
               trunc_per_unit=5, device=torch.device("cpu"))
    path = root / "bundle.pt"
    info = export_bundle(raw, "FD001", "gru", [0, 1, 2], root / "out" / "models", path,
                         window=10, trunc_per_unit=5, cfg=cfg)
    assert info["members"] == 3 and info["q"] > 0
    return path, raw


def _rows(raw, n=25):
    data = np.loadtxt(raw / "test_FD001.txt")
    unit1 = data[data[:, 0] == 1][:n]
    return unit1[:, 2:].tolist()


def test_bundle_predicts_with_valid_interval(bundle_path):
    path, raw = bundle_path
    b = load_bundle(path)
    out = b.predict(np.asarray(_rows(raw)))
    assert 0.0 <= out["lower"] <= out["rul"] <= out["upper"] <= b.cap
    # a single row works (window is padded), as in the training pipeline
    assert b.predict(np.asarray(_rows(raw, 1)))["upper"] <= b.cap


def test_api_predict_alert_levels_and_validation(bundle_path):
    path, raw = bundle_path
    client = TestClient(create_app(path))
    assert client.get("/health").json()["members"] == 3
    r = client.post("/predict", json={"rows": _rows(raw), "alert_threshold": 1000})
    assert r.status_code == 200 and r.json()["alert"] == "critical"
    r = client.post("/predict", json={"rows": _rows(raw), "alert_threshold": 0})
    assert r.json()["alert"] in ("ok", "warning")
    assert client.post("/predict", json={"rows": [[1.0, 2.0]]}).status_code == 422
    with pytest.raises(ValueError, match="finite"):
        load_bundle(path).predict(np.full((3, 24), np.nan))


def test_drift_statistics_and_endpoint(bundle_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    assert psi(a, rng.normal(size=4000)) < 0.05
    assert psi(a, rng.normal(loc=1.5, size=4000)) > 0.25
    assert ks_statistic(a, a) == 0.0
    ref = rng.normal(size=(1000, 3))
    cur = ref + np.array([0, 3.0, 0])
    assert drift_report(ref, cur)["columns"][0]["name"] == "col1"
    client = TestClient(create_app(bundle_path[0]))
    r = client.post("/drift", json={"reference": ref.tolist(), "current": cur.tolist()})
    assert r.status_code == 200 and r.json()["columns"][0]["psi"] > 0.25
    assert client.post("/drift", json={"reference": [[1.0]], "current": [[1.0, 2.0]]}
                       ).status_code == 422
