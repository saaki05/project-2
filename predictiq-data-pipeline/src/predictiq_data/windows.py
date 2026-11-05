"""Sliding-window construction and engine-wise splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Windows:
    X: np.ndarray  # (n, window, features) float32
    y: np.ndarray  # (n,) target RUL
    unit: np.ndarray  # (n,) engine id
    cycle: np.ndarray  # (n,) cycle at the window's last row

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def select(self, mask: np.ndarray) -> Windows:
        return Windows(self.X[mask], self.y[mask], self.unit[mask], self.cycle[mask])


def make_windows(
    feats: np.ndarray, unit: np.ndarray, cycle: np.ndarray, y: np.ndarray, window: int,
    last_only: bool = False,
) -> Windows:
    """One window per cycle ending at that cycle; the start is padded by repeating row 0.

    With ``last_only`` only each engine's final window is returned (the usual test protocol).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    xs, ys, us, cs = [], [], [], []
    for u in np.unique(unit):
        idx = np.flatnonzero(unit == u)
        idx = idx[np.argsort(cycle[idx], kind="stable")]
        f = feats[idx]
        padded = np.concatenate([np.repeat(f[:1], window - 1, axis=0), f], axis=0)
        w = sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)
        sel = slice(-1, None) if last_only else slice(None)
        xs.append(np.ascontiguousarray(w[sel]))
        ys.append(y[idx][sel])
        us.append(unit[idx][sel])
        cs.append(cycle[idx][sel])
    return Windows(
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(us),
        np.concatenate(cs),
    )


def split_units(
    units: np.ndarray, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint engine-level split (train / validation / calibration)."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    ids = np.array(sorted(np.unique(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr = int(round(fractions[0] * len(ids)))
    n_va = int(round(fractions[1] * len(ids)))
    parts = (ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:])
    audit_unit_split(*parts)
    return tuple(np.sort(p) for p in parts)


def audit_unit_split(*parts: np.ndarray) -> None:
    """Raise if any engine appears in more than one part, or a part is empty."""
    seen: set[int] = set()
    for p in parts:
        if len(p) == 0:
            raise ValueError("empty split")
        s = set(int(v) for v in p)
        if seen & s:
            raise ValueError(f"engines shared between splits: {sorted(seen & s)[:5]}")
        seen |= s
