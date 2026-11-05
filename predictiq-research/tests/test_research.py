import numpy as np
import pytest
import torch

from predictiq_research import metrics as M
from predictiq_research.conformal import (
    BinnedConformal,
    NormalizedConformal,
    SplitConformal,
    conformal_quantile,
)
from predictiq_research.experiments import run_shift, run_subset, truncation_mask
from predictiq_research.models import ARCHS, TCNRegressor, build_model
from predictiq_research.train import TrainConfig, train_nn, window_features


def test_conformal_quantile_hand_values():
    assert conformal_quantile(np.arange(1, 11), 0.2) == 9  # ceil(11 * 0.8) = 9
    assert conformal_quantile(np.arange(1, 4), 0.1) == float("inf")  # too few scores


def test_split_conformal_marginal_coverage_under_exchangeability():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        y = rng.uniform(0, 100, 600)
        p = y + rng.normal(scale=5, size=600)
        cal, te = slice(0, 200), slice(200, None)
        lo, hi = SplitConformal(0.1, -np.inf).fit(p[cal], y[cal]).interval(p[te])
        covs.append(M.coverage(y[te], lo, hi))
    assert 0.88 <= np.mean(covs) <= 0.93  # >= 1 - alpha, and not wildly conservative


def test_normalized_conformal_adapts_width_and_keeps_coverage():
    rng = np.random.default_rng(1)
    n = 6000
    sigma = rng.uniform(1, 10, n)
    y = rng.uniform(0, 100, n)
    p = y + rng.normal(size=n) * sigma
    cal, te = slice(0, 3000), slice(3000, None)
    nc = NormalizedConformal(0.1, -np.inf, floor=0.5).fit(p[cal], y[cal], sigma[cal])
    lo, hi = nc.interval(p[te], sigma[te])
    assert 0.87 <= M.coverage(y[te], lo, hi) <= 0.93
    assert np.corrcoef(hi - lo, sigma[te])[0, 1] > 0.99
    # coverage is balanced across noise levels, unlike a single global quantile
    hit = (y[te] >= lo) & (y[te] <= hi)
    assert abs(hit[sigma[te] < 3].mean() - hit[sigma[te] > 8].mean()) < 0.08


def test_binned_conformal_gives_locally_valid_coverage():
    rng = np.random.default_rng(2)
    n = 8000
    y = rng.uniform(0, 100, n)
    p = y + rng.normal(size=n) * np.where(y < 50, 2.0, 12.0)  # error grows with RUL
    cal, te = slice(0, 4000), slice(4000, None)
    lo, hi = BinnedConformal(0.1, 4, 20, -np.inf).fit(p[cal], y[cal]).interval(p[te])
    hit = (y[te] >= lo) & (y[te] <= hi)
    pt = p[te]  # the method conditions on the predicted value, so test on that
    assert abs(hit[pt < 35].mean() - 0.9) < 0.05
    assert abs(hit[pt > 65].mean() - 0.9) < 0.05
    g_lo, g_hi = SplitConformal(0.1, -np.inf).fit(p[cal], y[cal]).interval(p[te])
    g_hit = (y[te] >= g_lo) & (y[te] <= g_hi)
    assert g_hit[pt < 35].mean() > 0.97  # a global quantile over-covers the easy region
    assert g_hit[pt > 65].mean() < 0.87  # ... and under-covers the hard one


def test_clipping_never_hurts_coverage():
    y = np.array([0.0, 5.0, 125.0])
    lo, hi = SplitConformal(0.1, 0.0, 125.0).fit(y + 1, y).interval(y)
    assert lo.min() >= 0 and hi.max() <= 125
    assert M.coverage(y, lo, hi) == 1.0


def test_metrics_hand_values():
    y = np.array([50.0, 50.0])
    assert M.rmse(y, y + 3) == pytest.approx(3.0)
    assert M.nasa_score(np.array([50.0]), np.array([37.0])) == pytest.approx(np.e - 1)  # d=-13
    assert M.nasa_score(np.array([50.0]), np.array([60.0])) == pytest.approx(np.e - 1)  # d=+10
    lo, hi = M.wilson_interval(90, 100)
    assert lo < 0.9 < hi and 0.82 < lo and hi < 0.95
    s = M.interval_summary(np.array([10.0, 50.0]), np.array([0.0, 60.0]), np.array([20.0, 70.0]))
    assert s["coverage"] == 0.5 and s["coverage_critical"] == 1.0 and s["n_critical"] == 1


@pytest.mark.parametrize("arch", ARCHS)
def test_models_output_shape_and_gradient(arch):
    model = build_model(arch, in_dim=14, hidden=32)
    x = torch.randn(5, 30, 14)
    out = model(x)
    assert out.shape == (5,)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_tcn_is_causal():
    torch.manual_seed(0)
    tcn = TCNRegressor(4, hidden=8).eval()
    x = torch.randn(1, 20, 4)
    y = x.clone()
    y[:, 12:] += 5.0  # perturb the future of step 11
    a = tcn.net(x.transpose(1, 2))
    b = tcn.net(y.transpose(1, 2))
    assert torch.allclose(a[..., :12], b[..., :12], atol=1e-6)
    assert not torch.allclose(a[..., 12:], b[..., 12:])


def test_window_features_slope():
    X = np.tile(np.arange(10, dtype=np.float32)[None, :, None], (3, 1, 2)) * 2.0
    f = window_features(X)
    assert f.shape == (3, 12)
    assert np.allclose(f[:, 10:], 2.0, atol=1e-5)  # slope of 2 per step


def test_train_nn_learns_a_simple_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 10, 3)).astype(np.float32)
    y = np.clip(60 + 25 * X[:, -1, 0], 0, 125).astype(np.float32)
    cfg = TrainConfig(hidden=32, epochs=40, patience=40, lr=5e-3)
    m = train_nn("gru", X[:1200], y[:1200], X[1200:], y[1200:], cfg, seed=0,
                 device=torch.device("cpu"))
    with torch.no_grad():
        p = np.clip(m(torch.from_numpy(X[1200:])).numpy() * 125, 0, 125)
    assert np.sqrt(np.mean((p - y[1200:]) ** 2)) < 0.6 * y[1200:].std()


def test_truncation_mask_counts():
    unit = np.repeat([1, 2, 3], [40, 5, 25])
    m = truncation_mask(unit, 10, 0)
    assert m[unit == 1].sum() == 10 and m[unit == 2].sum() == 5


def _write_fake(root, subset, n_train, n_test, seed):
    rng = np.random.default_rng(seed)

    def engine(unit, length):
        t = np.arange(1, length + 1)
        deg = (t / length)[:, None] * np.linspace(1, 2, 21)[None, :] * 3
        sens = 100 + deg + rng.normal(scale=0.3, size=(length, 21))
        ops = np.zeros((length, 3))
        return np.column_stack([np.full(length, unit), t, ops, sens])

    def dump(name, rows):
        np.savetxt(root / name, rows, fmt="%.4f")

    dump(f"train_{subset}.txt",
         np.vstack([engine(u, int(rng.integers(60, 90))) for u in range(1, n_train + 1)]))
    lengths = [int(rng.integers(30, 60)) for _ in range(n_test)]
    dump(f"test_{subset}.txt", np.vstack([engine(u, L) for u, L in enumerate(lengths, 1)]))
    np.savetxt(root / f"RUL_{subset}.txt", rng.integers(10, 60, n_test), fmt="%d")


def test_run_subset_and_shift_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fake(raw, "FD001", 30, 12, 0)
    _write_fake(raw, "FD003", 30, 12, 1)
    cfg = TrainConfig(hidden=16, epochs=3, patience=3)
    res = run_subset(raw, "FD001", ["gru"], ["ridge"], [0, 1, 2], cfg, 0.1, tmp_path / "out",
                     window=10, trunc_per_unit=5, device=torch.device("cpu"))
    assert set(res["models"]) == {"ridge", "gru"}
    gru = res["models"]["gru"]
    assert gru["n_members"] == 3 and "single_member" in gru
    iv = gru["intervals"]["test_last|cal=trunc"]
    assert {"split", "binned", "normalized"} <= set(iv)
    assert 0.0 <= iv["split"]["coverage"] <= 1.0 and iv["split"]["n"] == 12
    assert "normalized" not in res["models"]["ridge"]["intervals"]["test_last|cal=trunc"]

    sh = run_shift(raw, "FD001", "FD003", ["gru"], [0, 1, 2], cfg, tmp_path / "out" / "models",
                   0.1, window=10, trunc_per_unit=5, recal_engines=10,
                   out_path=tmp_path / "out" / "shift_FD001_to_FD003.json")
    keys = set(sh["models"]["gru"]["intervals"])
    assert any("source calibration" in k for k in keys) and any("recalibration" in k for k in keys)
    from predictiq_research.report import build_report

    text = build_report(tmp_path / "out")
    assert "## Point accuracy" in text and "| FD001 | gru |" in text
    assert "Summary: in-distribution coverage" in text and "| FD001 |" in text
    assert "Summary: shift" in text and "FD001 → FD003" in text
