"""Turn models saved by ``predictiq-research`` into a servable bundle.

The normalized-conformal quantile is calibrated on engines that were held out from both
training and early stopping (the same engine split the research experiments use), using
random truncation cut-offs that mimic how a deployed model is queried mid-life.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from predictiq_data import prepare, split_units
from predictiq_research.conformal import NormalizedConformal
from predictiq_research.experiments import load_members, truncation_mask
from predictiq_research.train import TrainConfig, predict

from predictiq_service.bundle import save_bundle


def export_bundle(raw_dir, subset, arch, seeds, models_dir, out_path, alpha=0.1, window=30,
                  trunc_per_unit=10, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    prep = prepare(raw_dir, subset, window, int(cfg.cap))
    _, _, cal_units = split_units(prep.train.unit, seed=0)
    cal = prep.train.select(np.isin(prep.train.unit, cal_units))
    keep = truncation_mask(cal.unit, trunc_per_unit, 0)
    members = load_members(Path(models_dir), subset, arch, seeds, prep.num_features, cfg)
    import torch

    cpu = torch.device("cpu")
    preds = np.stack([predict(m, cal.X[keep], cfg.cap, device=cpu) for m in members])
    conformal = NormalizedConformal(alpha, 0.0, cfg.cap).fit(preds.mean(0), cal.y[keep],
                                                              preds.std(0))
    save_bundle(out_path, members, prep.preprocessor, window, cfg.cap, conformal, arch, subset,
                cfg.hidden, cfg.dropout)
    return {"q": conformal.q, "n_calibration": int(keep.sum()), "members": len(members)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="predictiq-export")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    info = export_bundle(a.raw, a.subset, a.arch, a.seeds, a.models_dir, a.out, a.alpha)
    print(f"wrote {a.out}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
