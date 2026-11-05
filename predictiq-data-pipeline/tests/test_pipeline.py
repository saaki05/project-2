import os
from pathlib import Path

import numpy as np
import pytest

from predictiq_data import (
    Preprocessor,
    audit_unit_split,
    load_subset,
    make_windows,
    prepare,
    split_units,
)
from predictiq_data.cli import main
from predictiq_data.cmapss import COLUMNS, add_test_rul, add_train_rul


def write_fake_subset(root: Path, subset="FD001", n_train=6, n_test=4, regimes=1, seed=0):
    """Write files in the C-MAPSS text layout (trailing space, 26 columns)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(regimes, 3)) * 10

    def engine_rows(unit, length):
        t = np.arange(1, length + 1)
        picked = centers[rng.integers(0, regimes, length)]
        ops = picked + rng.normal(scale=1e-3, size=(length, 3))
        drift = (t / length)[:, None] * rng.normal(size=(1, 21))
        noise = rng.normal(scale=0.1, size=(length, 21))
        sens = 100 + ops.sum(1, keepdims=True) + drift * 5 + noise
        return np.column_stack([np.full(length, unit), t, ops, sens])

    def write(name, rows):
        with open(root / name, "w") as fh:
            for r in rows:
                fh.write(" ".join(f"{v:.4f}" if i > 1 else f"{int(v)}" for i, v in enumerate(r)))
                fh.write(" \n")

    tr = np.vstack([engine_rows(u, int(rng.integers(40, 60))) for u in range(1, n_train + 1)])
    te_lengths = [int(rng.integers(20, 35)) for _ in range(n_test)]
    te = np.vstack([engine_rows(u, L) for u, L in enumerate(te_lengths, start=1)])
    write(f"train_{subset}.txt", tr)
    write(f"test_{subset}.txt", te)
    np.savetxt(root / f"RUL_{subset}.txt", rng.integers(5, 140, n_test), fmt="%d")
    return te_lengths


@pytest.fixture
def raw(tmp_path):
    write_fake_subset(tmp_path)
    return tmp_path


def test_loader_validates_and_shapes(raw):
    sub = load_subset(raw, "FD001")
    assert list(sub.train.columns) == COLUMNS
    assert sub.test["unit"].nunique() == len(sub.test_rul) == 4


def test_loader_rejects_gaps(raw):
    lines = (raw / "train_FD001.txt").read_text().splitlines()
    (raw / "train_FD001.txt").write_text("\n".join(lines[:5] + lines[6:]) + "\n")
    with pytest.raises(ValueError, match="consecutive"):
        load_subset(raw, "FD001")


def test_rul_definitions(raw):
    sub = load_subset(raw, "FD001")
    rul = add_train_rul(sub.train)
    last = sub.train["cycle"] == sub.train.groupby("unit")["cycle"].transform("max")
    assert (rul[last.to_numpy()] == 0).all()
    assert add_train_rul(sub.train, cap=10).max() == 10
    te = add_test_rul(sub.test, sub.test_rul)
    lastt = sub.test["cycle"] == sub.test.groupby("unit")["cycle"].transform("max")
    assert np.array_equal(te[lastt.to_numpy()], sub.test_rul)
    # RUL decreases by exactly one per cycle within an engine
    g = sub.test.assign(r=te)
    assert (g.groupby("unit")["r"].diff().dropna() == -1).all()


def test_windows_alignment_and_padding():
    feats = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    unit = np.zeros(10, dtype=int)
    cycle = np.arange(1, 11)
    w = make_windows(feats, unit, cycle, np.arange(10, dtype=float), window=4)
    assert w.X.shape == (10, 4, 2)
    assert np.array_equal(w.X[5, :, 0], [2, 3, 4, 5])  # window ends at row 5
    assert np.array_equal(w.X[0, :, 0], [0, 0, 0, 0])  # padded with first row
    last = make_windows(feats, unit, cycle, np.arange(10, dtype=float), window=4, last_only=True)
    assert len(last) == 1 and last.X[0, -1, 0] == 9


def test_windows_never_mix_engines():
    feats = np.concatenate([np.zeros((5, 1)), np.ones((5, 1))]).astype(np.float32)
    unit = np.repeat([1, 2], 5)
    cycle = np.tile(np.arange(1, 6), 2)
    w = make_windows(feats, unit, cycle, np.zeros(10), window=3)
    assert (w.X[w.unit == 1] == 0).all() and (w.X[w.unit == 2] == 1).all()


def test_unit_split_is_disjoint_and_audited():
    units = np.repeat(np.arange(1, 21), 3)
    a, b, c = split_units(units, seed=1)
    assert len(set(a) | set(b) | set(c)) == 20
    with pytest.raises(ValueError):
        audit_unit_split(a, np.concatenate([b, a[:1]]))
    assert np.array_equal(split_units(units, seed=1)[0], a)


def test_preprocessor_regimes_and_serialisation(tmp_path):
    write_fake_subset(tmp_path, "FD002", n_train=8, n_test=3, regimes=6)
    sub = load_subset(tmp_path, "FD002")
    pre = Preprocessor(n_regimes=6).fit(sub.train)
    x = pre.transform(sub.train)
    assert abs(x.mean()) < 0.2 and 0.7 < x.std() < 1.3
    clone = Preprocessor.from_dict(pre.to_dict())
    assert np.allclose(clone.transform(sub.test), pre.transform(sub.test), atol=1e-5)


def test_prepare_and_cli(raw, tmp_path):
    p = prepare(raw, "FD001", window=8, cap=50)
    assert p.train.X.shape[1:] == (8, 14)
    assert len(p.test_last) == 4
    assert p.train.y.max() <= 50 and p.test_last.y.max() <= 50
    out = tmp_path / "out"
    assert main(["prepare", "--raw", str(raw), "--subset", "FD001", "--out", str(out),
                 "--window", "8"]) == 0
    assert (out / "manifest.json").exists()


@pytest.mark.skipif("PREDICTIQ_CMAPSS_DIR" not in os.environ, reason="real C-MAPSS data not set")
@pytest.mark.parametrize(
    "subset,units_train,units_test",
    # Engine counts as found in the files (the dataset readme lists FD004 the other way round).
    [("FD001", 100, 100), ("FD002", 260, 259), ("FD003", 100, 100), ("FD004", 249, 248)],
)
def test_real_data_shapes(subset, units_train, units_test):
    p = prepare(os.environ["PREDICTIQ_CMAPSS_DIR"], subset)
    assert len(np.unique(p.train.unit)) == units_train
    assert len(p.test_last) == units_test
    assert np.isfinite(p.train.X).all()
