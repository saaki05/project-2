"""A deployable RUL model: ensemble weights, preprocessing and a calibrated interval width."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from predictiq_data import Preprocessor, make_windows
from predictiq_data.cmapss import COLUMNS
from predictiq_research.conformal import NormalizedConformal
from predictiq_research.models import build_model

FORMAT_VERSION = 1
INPUT_COLUMNS = COLUMNS[2:]  # op1..op3, s1..s21 (unit and cycle are not model inputs)


@dataclass
class RulBundle:
    members: list[torch.nn.Module]
    preprocessor: Preprocessor
    window: int
    cap: float
    alpha: float
    q: float  # normalized-conformal quantile
    sigma_floor: float
    arch: str
    subset: str

    def predict(self, rows: np.ndarray) -> dict:
        """Predict RUL at the last row of ``rows`` (T x 24, oldest first)."""
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != len(INPUT_COLUMNS):
            raise ValueError(f"rows must have shape (T, {len(INPUT_COLUMNS)})")
        if rows.shape[0] < 1 or not np.isfinite(rows).all():
            raise ValueError("rows must be non-empty and finite")
        df = pd.DataFrame(rows, columns=INPUT_COLUMNS)
        feats = self.preprocessor.transform(df)
        n = len(feats)
        w = make_windows(feats, np.zeros(n, dtype=int), np.arange(1, n + 1), np.zeros(n),
                         self.window, last_only=True)
        with torch.no_grad():
            preds = np.stack([m.eval()(torch.from_numpy(w.X)).numpy() for m in self.members])
        preds = np.clip(preds * self.cap, 0.0, self.cap)[:, 0]
        mean, sigma = float(preds.mean()), float(preds.std())
        half = self.q * max(sigma, self.sigma_floor)
        return {"rul": mean, "sigma": sigma, "lower": max(mean - half, 0.0),
                "upper": min(mean + half, self.cap), "coverage_target": 1 - self.alpha}


def save_bundle(path: str | Path, members, preprocessor: Preprocessor, window: int, cap: float,
                conformal: NormalizedConformal, arch: str, subset: str, hidden: int,
                dropout: float) -> None:
    torch.save({
        "format_version": FORMAT_VERSION, "arch": arch, "subset": subset, "hidden": hidden,
        "dropout": dropout, "window": window, "cap": cap, "alpha": conformal.alpha,
        "q": conformal.q, "sigma_floor": conformal.floor,
        "preprocessor": preprocessor.to_dict(), "in_dim": len(preprocessor.sensors),
        "states": [m.state_dict() for m in members],
    }, path)


def load_bundle(path: str | Path) -> RulBundle:
    art = torch.load(path, map_location="cpu", weights_only=False)
    if art.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {art.get('format_version')!r}")
    members = []
    for state in art["states"]:
        m = build_model(art["arch"], art["in_dim"], art["hidden"], art["dropout"])
        m.load_state_dict(state)
        members.append(m.eval())
    return RulBundle(members, Preprocessor.from_dict(art["preprocessor"]), art["window"],
                     art["cap"], art["alpha"], art["q"], art["sigma_floor"], art["arch"],
                     art["subset"])
