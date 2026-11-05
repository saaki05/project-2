"""Sequence regressors mapping a window (B, W, F) to a scalar RUL (B,)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

ARCHS = ("lstm", "gru", "tcn", "transformer")


class RNNRegressor(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_dim, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation, padding=self.pad)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        h = self.drop(self.act(self.conv1(x)[..., :n]))  # chop right padding => causal
        h = self.drop(self.act(self.conv2(h)[..., :n]))
        return self.act(h + self.skip(x))


class TCNRegressor(nn.Module):
    """Temporal convolutional network with exponentially dilated causal convolutions."""

    def __init__(self, in_dim: int, hidden: int = 64, levels: int = 3, kernel: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        blocks, c = [], in_dim
        for i in range(levels):
            blocks.append(_CausalBlock(c, hidden, kernel, 2**i, dropout))
            c = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.net(x.transpose(1, 2))
        return self.head(h[..., -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Pre-norm Transformer encoder with sinusoidal positions; reads the last position."""

    def __init__(self, in_dim: int, d_model: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x) + self.pe[: x.shape[1]]
        return self.head(self.encoder(h)[:, -1]).squeeze(-1)


def build_model(arch: str, in_dim: int, hidden: int = 64, dropout: float = 0.1) -> nn.Module:
    if arch in ("lstm", "gru"):
        return RNNRegressor(arch, in_dim, hidden, 2, dropout)
    if arch == "tcn":
        return TCNRegressor(in_dim, hidden, 3, 3, dropout)
    if arch == "transformer":
        return TransformerRegressor(in_dim, hidden, 4, 2, dropout)
    raise ValueError(f"unknown architecture {arch!r}")
