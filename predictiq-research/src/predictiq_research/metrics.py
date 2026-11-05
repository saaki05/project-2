"""Point-prediction and interval metrics."""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(p - y)))


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    """Asymmetric PHM08 score: late predictions (p > y) are penalised more than early ones."""
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (coverage uncertainty)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def interval_summary(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, critical: float = 30.0) -> dict:
    """Coverage with a Wilson interval, mean width, and coverage where true RUL <= critical."""
    hit = (y >= lo) & (y <= hi)
    out = {
        "coverage": float(hit.mean()),
        "coverage_ci95": list(wilson_interval(int(hit.sum()), len(y))),
        "width": mean_width(lo, hi),
        "n": int(len(y)),
    }
    low = y <= critical
    out["coverage_critical"] = float(hit[low].mean()) if low.any() else None
    out["n_critical"] = int(low.sum())
    return out
