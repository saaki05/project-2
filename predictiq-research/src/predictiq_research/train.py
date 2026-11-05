"""Training of neural RUL regressors and classical feature baselines."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn

from predictiq_research.models import build_model


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 6
    batch_size: int = 256
    cap: float = 125.0


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, cap: float, batch: int = 2048,
            device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


def train_nn(arch: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
             cfg: TrainConfig, seed: int, device: torch.device | None = None) -> nn.Module:
    """Fit with MSE on RUL/cap, early-stopping on validation RMSE; returns the best model."""
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(arch, Xtr.shape[2], cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    x_t = torch.from_numpy(Xtr).to(device)
    y_t = torch.from_numpy(ytr / cfg.cap).float().to(device)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(x_t))).to(device)
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_t[idx]), y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = float(np.sqrt(np.mean((predict(model, Xva, cfg.cap, device=device) - yva) ** 2)))
        if val < best - 1e-4:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu()


def window_features(X: np.ndarray) -> np.ndarray:
    """Per-sensor mean, std, min, max, last value and least-squares slope over the window."""
    t = np.arange(X.shape[1], dtype=np.float32)
    tc = t - t.mean()
    denom = max(float((tc**2).sum()), 1.0)
    slope = np.einsum("t,btf->bf", tc, X - X.mean(1, keepdims=True)) / denom
    return np.concatenate(
        [X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1], slope], axis=1
    ).astype(np.float32)


def fit_classical(name: str, Xtr: np.ndarray, ytr: np.ndarray, seed: int):
    f = window_features(Xtr)
    if name == "ridge":
        return Ridge(alpha=1.0).fit(f, ytr)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             random_state=seed).fit(f, ytr)
    raise ValueError(f"unknown classical model {name!r}")


def predict_classical(model, X: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(model.predict(window_features(X)), 0.0, cap)
