"""Small hyper-parameter study on FD001: model width and window length.

The configuration is chosen by *validation* RMSE (engines held out from training and
calibration); the test engines are only read afterwards, for every configuration, so the
table shows whether the validation choice carried over.

Usage: python scripts/tune_fd001.py /path/to/CMAPSSData results/tuning_fd001.json
"""
import json
import sys
from dataclasses import replace

import torch

from predictiq_research.experiments import run_subset
from predictiq_research.train import TrainConfig

raw, out = sys.argv[1], sys.argv[2]
base = TrainConfig(epochs=60, patience=8)
grid = [(64, 30), (128, 30), (64, 50), (128, 50)]  # (hidden, window)
results = {}
for hidden, window in grid:
    r = run_subset(raw, "FD001", ["gru"], [], [0, 1, 2], replace(base, hidden=hidden), 0.1,
                   f"{out}.work/h{hidden}_w{window}", window=window, device=torch.device("cpu"))
    m = r["models"]["gru"]
    iv = m["intervals"]["test_last|cal=trunc"]["normalized"]
    results[f"hidden={hidden}, window={window}"] = {
        "val_rmse": m["val_rmse"], "test_rmse": m["ensemble"]["rmse"],
        "test_nasa_score": m["ensemble"]["nasa_score"], "coverage": iv["coverage"],
        "width": iv["width"], "single_member_test_rmse": m["single_member"]["rmse"]}
    print(hidden, window, results[f"hidden={hidden}, window={window}"], flush=True)
best = min(results, key=lambda k: results[k]["val_rmse"])
json.dump({"selected_by_validation": best, "results": results}, open(out, "w"), indent=2)
print("selected by validation:", best)
