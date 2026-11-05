"""Operating-regime-aware sensor normalisation, fitted on training data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from predictiq_data.cmapss import DEFAULT_SENSORS, OP_COLS


class Preprocessor:
    """Standardises each sensor separately within each operating regime.

    Regimes are found by k-means on the standardised operating settings. With a single regime
    this reduces to plain per-sensor standardisation. Everything is fitted on training rows
    only, and the fitted state is plain arrays so it can be serialised with ``to_dict``.
    """

    def __init__(self, sensors: list[str] | None = None, n_regimes: int = 1, seed: int = 0):
        self.sensors = list(sensors or DEFAULT_SENSORS)
        self.n_regimes = n_regimes
        self.seed = seed
        self.op_mean: np.ndarray | None = None
        self.op_std: np.ndarray | None = None
        self.centers: np.ndarray | None = None
        self.mean: np.ndarray | None = None  # (R, F)
        self.std: np.ndarray | None = None  # (R, F)

    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def _assign(self, ops: np.ndarray) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(ops.shape[0], dtype=int)
        z = (ops - self.op_mean) / self.op_std
        d = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        ops = df[OP_COLS].to_numpy(float)
        if self.n_regimes > 1:
            self.op_mean = ops.mean(0)
            self.op_std = np.where(ops.std(0) < 1e-8, 1.0, ops.std(0))
            z = (ops - self.op_mean) / self.op_std
            km = KMeans(self.n_regimes, n_init=10, random_state=self.seed).fit(z)
            self.centers = km.cluster_centers_
        regime = self._assign(ops)
        x = df[self.sensors].to_numpy(float)
        mean = np.zeros((self.n_regimes, x.shape[1]))
        std = np.ones((self.n_regimes, x.shape[1]))
        for r in range(self.n_regimes):
            rows = x[regime == r]
            if len(rows) == 0:
                raise ValueError(f"regime {r} has no training rows")
            mean[r] = rows.mean(0)
            s = rows.std(0)
            std[r] = np.where(s < 1e-8, 1.0, s)
        self.mean, self.std = mean, std
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit first")
        regime = self._assign(df[OP_COLS].to_numpy(float))
        x = df[self.sensors].to_numpy(float)
        return ((x - self.mean[regime]) / self.std[regime]).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "sensors": self.sensors,
            "n_regimes": self.n_regimes,
            "op_mean": None if self.op_mean is None else self.op_mean.tolist(),
            "op_std": None if self.op_std is None else self.op_std.tolist(),
            "centers": None if self.centers is None else self.centers.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Preprocessor:
        p = cls(d["sensors"], d["n_regimes"])
        arr = lambda v: None if v is None else np.asarray(v, dtype=float)  # noqa: E731
        p.op_mean, p.op_std, p.centers = arr(d["op_mean"]), arr(d["op_std"]), arr(d["centers"])
        p.mean, p.std = arr(d["mean"]), arr(d["std"])
        return p
