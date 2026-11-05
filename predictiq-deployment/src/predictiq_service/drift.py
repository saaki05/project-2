"""Population stability index and KS statistic for monitoring sensor drift."""

from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), eps, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), eps, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def drift_report(reference: np.ndarray, current: np.ndarray, names: list[str] | None = None):
    """Per-column PSI and KS, sorted by PSI (conventional alert levels: 0.1 and 0.25)."""
    if reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current must have the same number of columns")
    names = names or [f"col{j}" for j in range(reference.shape[1])]
    rows = [{"name": names[j], "psi": psi(reference[:, j], current[:, j]),
             "ks": ks_statistic(reference[:, j], current[:, j])}
            for j in range(reference.shape[1])]
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return {"mean_psi": float(np.mean([r["psi"] for r in rows])), "columns": rows}
