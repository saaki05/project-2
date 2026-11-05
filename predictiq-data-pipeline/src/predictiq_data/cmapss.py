"""Reader for the NASA C-MAPSS turbofan degradation files.

Each ``train_FDxxx.txt`` / ``test_FDxxx.txt`` has 26 whitespace-separated columns: unit,
cycle, three operating settings and 21 sensors. ``RUL_FDxxx.txt`` holds the true remaining
useful life at the last observed cycle of every test engine.

Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", PHM08.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_COLS, *SENSOR_COLS]
# Sensors that carry degradation signal in FD001-FD004 according to the usual practice in the
# literature (the others are constant or near-constant).
DEFAULT_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
# Number of distinct operating conditions per subset (from the dataset documentation).
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class Subset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray  # true RUL at the last cycle of each test engine, ordered by unit id


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if df.isna().any().any():
        raise ValueError(f"{path.name}: missing values")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    for unit, g in df.groupby("unit"):
        if not np.array_equal(g["cycle"].to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"{path.name}: unit {unit} cycles are not consecutive from 1")
    return df


def load_subset(raw_dir: str | Path, subset: str) -> Subset:
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    raw = Path(raw_dir)
    train = _read(raw / f"train_{subset}.txt")
    test = _read(raw / f"test_{subset}.txt")
    rul = np.loadtxt(raw / f"RUL_{subset}.txt").astype(float).ravel()
    if rul.shape[0] != test["unit"].nunique():
        raise ValueError("RUL file length does not match the number of test engines")
    return Subset(subset, train, test, rul)


def add_train_rul(df: pd.DataFrame, cap: int | None = None) -> np.ndarray:
    """RUL of every training row: cycles remaining until the engine's last recorded cycle."""
    rul = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).to_numpy(float)
    return rul if cap is None else np.minimum(rul, cap)


def add_test_rul(df: pd.DataFrame, final_rul: np.ndarray, cap: int | None = None) -> np.ndarray:
    """RUL of every test row, from the provided RUL at each engine's last observed cycle."""
    units = np.sort(df["unit"].unique())
    final = pd.Series(final_rul, index=units)
    last = df.groupby("unit")["cycle"].transform("max")
    rul = final.reindex(df["unit"]).to_numpy() + (last - df["cycle"]).to_numpy()
    return rul if cap is None else np.minimum(rul, cap)
