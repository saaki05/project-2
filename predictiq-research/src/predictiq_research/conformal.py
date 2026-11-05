"""Split-conformal prediction intervals for regression, implemented from scratch.

Given calibration data that is exchangeable with the test data and was not used for
training, split conformal prediction yields intervals with marginal coverage of at least
``1 - alpha``. Three score variants are provided:

* ``SplitConformal``: absolute residuals, one global quantile.
* ``NormalizedConformal``: residuals divided by a per-example scale estimate (for example
  the standard deviation of an ensemble), giving locally adaptive widths.
* ``BinnedConformal``: a separate quantile per bin of the predicted value (a Mondrian
  variant), targeting coverage that varies less across the RUL range.

The coverage guarantee holds only under exchangeability. Under covariate or label shift it
can fail, and measuring that failure is the purpose of the shift experiments.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score (``inf`` if there are too few)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


class SplitConformal:
    def __init__(self, alpha: float = 0.1, lower: float = 0.0, upper: float | None = None):
        self.alpha, self.lower, self.upper = alpha, lower, upper
        self.q: float | None = None

    def fit(self, pred: np.ndarray, y: np.ndarray, scale: np.ndarray | None = None):
        self.q = conformal_quantile(np.abs(y - pred), self.alpha)
        return self

    def interval(self, pred: np.ndarray, scale: np.ndarray | None = None):
        lo, hi = pred - self.q, pred + self.q
        return _clip(lo, hi, self.lower, self.upper)


class NormalizedConformal:
    def __init__(self, alpha: float = 0.1, lower: float = 0.0, upper: float | None = None,
                 floor: float = 1.0):
        self.alpha, self.lower, self.upper, self.floor = alpha, lower, upper, floor
        self.q: float | None = None

    def fit(self, pred: np.ndarray, y: np.ndarray, scale: np.ndarray):
        s = np.maximum(scale, self.floor)
        self.q = conformal_quantile(np.abs(y - pred) / s, self.alpha)
        return self

    def interval(self, pred: np.ndarray, scale: np.ndarray):
        half = self.q * np.maximum(scale, self.floor)
        return _clip(pred - half, pred + half, self.lower, self.upper)


class BinnedConformal:
    """Per-bin quantiles over predicted-value bins; sparse bins fall back to the global one."""

    def __init__(self, alpha: float = 0.1, n_bins: int = 4, min_per_bin: int = 20,
                 lower: float = 0.0, upper: float | None = None):
        self.alpha, self.n_bins, self.min_per_bin = alpha, n_bins, min_per_bin
        self.lower, self.upper = lower, upper
        self.edges: np.ndarray | None = None
        self.q: np.ndarray | None = None
        self.q_global: float | None = None

    def _bin(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, pred, side="right") - 1, 0, self.n_bins - 1)

    def fit(self, pred: np.ndarray, y: np.ndarray, scale: np.ndarray | None = None):
        res = np.abs(y - pred)
        self.q_global = conformal_quantile(res, self.alpha)
        self.edges = np.quantile(pred, np.linspace(0, 1, self.n_bins + 1))
        b = self._bin(pred)
        self.q = np.array([
            conformal_quantile(res[b == k], self.alpha) if (b == k).sum() >= self.min_per_bin
            else self.q_global
            for k in range(self.n_bins)
        ])
        return self

    def interval(self, pred: np.ndarray, scale: np.ndarray | None = None):
        q = self.q[self._bin(pred)]
        return _clip(pred - q, pred + q, self.lower, self.upper)


def _clip(lo, hi, lower, upper):
    """Clipping to the feasible range never removes a true value, so coverage is unaffected."""
    lo = np.maximum(lo, lower)
    hi = hi if upper is None else np.minimum(hi, upper)
    return lo, hi
