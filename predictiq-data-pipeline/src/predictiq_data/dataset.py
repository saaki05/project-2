"""End-to-end preparation of a C-MAPSS subset into model-ready windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from predictiq_data.cmapss import N_REGIMES, add_test_rul, add_train_rul, load_subset
from predictiq_data.preprocess import Preprocessor
from predictiq_data.windows import Windows, make_windows


@dataclass
class Prepared:
    subset: str
    window: int
    cap: int
    preprocessor: Preprocessor
    train: Windows  # every cycle of every training engine; y is the capped RUL
    test_last: Windows  # final window of each test engine; y is the capped true RUL
    test_all: Windows  # every window of every test engine; y is the capped RUL
    test_final_rul_raw: np.ndarray  # un-capped RUL provided with the dataset

    @property
    def num_features(self) -> int:
        return int(self.train.X.shape[2])


def prepare(
    raw_dir: str | Path, subset: str, window: int = 30, cap: int = 125
) -> Prepared:
    """Load a subset, normalise per operating regime and build windows.

    Targets are the piecewise-linear RUL ``min(RUL, cap)``; test truth is capped the same way
    so that prediction intervals are evaluated against a quantity the models can express.
    """
    sub = load_subset(raw_dir, subset)
    pre = Preprocessor(n_regimes=N_REGIMES[subset]).fit(sub.train)

    tr_x = pre.transform(sub.train)
    tr_y = add_train_rul(sub.train, cap)
    train = make_windows(tr_x, sub.train["unit"].to_numpy(), sub.train["cycle"].to_numpy(),
                         tr_y, window)

    te_x = pre.transform(sub.test)
    te_y = add_test_rul(sub.test, sub.test_rul, cap)
    args = (te_x, sub.test["unit"].to_numpy(), sub.test["cycle"].to_numpy(), te_y, window)
    return Prepared(
        subset=subset, window=window, cap=cap, preprocessor=pre, train=train,
        test_last=make_windows(*args, last_only=True), test_all=make_windows(*args),
        test_final_rul_raw=sub.test_rul,
    )


def save_prepared(p: Prepared, out_dir: str | Path) -> dict:
    """Write arrays to ``arrays.npz`` and a ``manifest.json`` with a checksum."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ("train", "test_last", "test_all"):
        w: Windows = getattr(p, name)
        arrays.update({f"{name}_X": w.X, f"{name}_y": w.y, f"{name}_unit": w.unit,
                       f"{name}_cycle": w.cycle})
    np.savez_compressed(out / "arrays.npz", **arrays)
    digest = hashlib.sha256((out / "arrays.npz").read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "subset": p.subset, "window": p.window, "cap": p.cap,
        "num_features": p.num_features, "n_train": len(p.train),
        "n_test_engines": len(p.test_last), "arrays_sha256": digest,
        "preprocessor": p.preprocessor.to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
