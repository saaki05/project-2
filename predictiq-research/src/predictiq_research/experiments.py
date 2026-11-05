"""In-distribution and distribution-shift experiments.

Protocol
--------
* Training engines are split into disjoint train / validation / calibration engines.
  Validation drives early stopping only; calibration engines feed the conformal procedures
  only; the official C-MAPSS test engines are used for reporting only.
* Neural models are trained once per seed; an "ensemble" averages the seeds and its standard
  deviation is the scale used by normalized conformal prediction.
* Test targets are the capped RUL, matching the training target (the cap is a parameter).
* Two calibration sets are built from the calibration engines, because the choice matters:
  ``all`` uses every cycle (matches the distribution of ``test_all``), ``trunc`` draws
  random truncation cycles per engine (matches how the official test engines are cut off,
  which is the distribution of ``test_last``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from predictiq_data import Prepared, prepare, split_units

from predictiq_research import metrics as M
from predictiq_research.conformal import BinnedConformal, NormalizedConformal, SplitConformal
from predictiq_research.models import build_model
from predictiq_research.train import (
    TrainConfig,
    fit_classical,
    pick_device,
    predict,
    predict_classical,
    train_nn,
)


def truncation_mask(unit: np.ndarray, per_unit: int, seed: int) -> np.ndarray:
    """Select ``per_unit`` random rows from every engine (calibration that mimics test cutoffs)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(unit), dtype=bool)
    for u in np.unique(unit):
        idx = np.flatnonzero(unit == u)
        mask[rng.choice(idx, size=min(per_unit, len(idx)), replace=False)] = True
    return mask


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"rmse": M.rmse(y, p), "mae": M.mae(y, p), "nasa_score": M.nasa_score(y, p)}


def interval_methods(pred_cal, y_cal, sigma_cal, pred_te, sigma_te, alpha, cap) -> dict:
    """Fit every conformal variant on calibration data and return the test intervals."""
    out = {"split": SplitConformal(alpha, 0.0, cap).fit(pred_cal, y_cal).interval(pred_te),
           "binned": BinnedConformal(alpha, 4, 20, 0.0, cap).fit(pred_cal, y_cal)
           .interval(pred_te)}
    if sigma_cal is not None and sigma_te is not None:
        nc = NormalizedConformal(alpha, 0.0, cap).fit(pred_cal, y_cal, sigma_cal)
        out["normalized"] = nc.interval(pred_te, sigma_te)
    return out


def evaluate_intervals(cal: dict, test: dict, alpha: float, cap: float) -> dict:
    """``cal`` / ``test`` hold pred, sigma, y for the calibration and evaluation windows."""
    ivs = interval_methods(cal["pred"], cal["y"], cal.get("sigma"), test["pred"],
                           test.get("sigma"), alpha, cap)
    return {name: M.interval_summary(test["y"], lo, hi) for name, (lo, hi) in ivs.items()}


class _Collector:
    """Gathers predictions of one model (or ensemble) for every evaluation set."""

    def __init__(self) -> None:
        self.runs: list[dict[str, np.ndarray]] = []

    def add(self, **preds: np.ndarray) -> None:
        self.runs.append(preds)

    def ensemble(self) -> tuple[dict, dict | None]:
        keys = self.runs[0].keys()
        mean = {k: np.mean([r[k] for r in self.runs], axis=0) for k in keys}
        std = ({k: np.std([r[k] for r in self.runs], axis=0) for k in keys}
               if len(self.runs) > 1 else None)
        return mean, std


def _summarise(name, collector, sets, alpha, cap, trunc_per_unit, seed) -> dict:
    mean, std = collector.ensemble()
    res: dict = {"n_members": len(collector.runs)}
    if len(collector.runs) > 1:
        per = [point_metrics(sets["test_last"].y, r["test_last"]) for r in collector.runs]
        res["single_member"] = {k: {"mean": float(np.mean([p[k] for p in per])),
                                    "std": float(np.std([p[k] for p in per]))} for k in per[0]}
    res["ensemble"] = point_metrics(sets["test_last"].y, mean["test_last"])
    res["ensemble_test_all"] = point_metrics(sets["test_all"].y, mean["test_all"])

    cal_unit = sets["cal"].unit
    trunc = truncation_mask(cal_unit, trunc_per_unit, seed)

    def pack(split, key, mask=None):
        m = slice(None) if mask is None else mask
        d = {"pred": mean[key][m], "y": sets[split].y[m]}
        if std is not None:
            d["sigma"] = std[key][m]
        return d

    res["intervals"] = {
        # official test cut-offs, calibrated with comparable random cut-offs
        "test_last|cal=trunc": evaluate_intervals(pack("cal", "cal", trunc),
                                                  pack("test_last", "test_last"), alpha, cap),
        "test_last|cal=all": evaluate_intervals(pack("cal", "cal"),
                                                pack("test_last", "test_last"), alpha, cap),
        "test_all|cal=all": evaluate_intervals(pack("cal", "cal"),
                                               pack("test_all", "test_all"), alpha, cap),
    }
    return res


def _sets(prep: Prepared, tr_u, va_u, ca_u) -> dict:
    d = {
        "train": prep.train.select(np.isin(prep.train.unit, tr_u)),
        "val": prep.train.select(np.isin(prep.train.unit, va_u)),
        "cal": prep.train.select(np.isin(prep.train.unit, ca_u)),
        "test_last": prep.test_last,
        "test_all": prep.test_all,
    }
    return d


def run_subset(raw_dir, subset, archs, classical, seeds, cfg: TrainConfig, alpha=0.1,
               out_dir="results", window=30, trunc_per_unit=10, device=None) -> dict:
    """Train every model on one subset and evaluate points and intervals."""
    device = device or pick_device()
    prep = prepare(raw_dir, subset, window, int(cfg.cap))
    tr_u, va_u, ca_u = split_units(prep.train.unit, seed=0)
    sets = _sets(prep, tr_u, va_u, ca_u)
    out = Path(out_dir)
    (out / "models" / subset).mkdir(parents=True, exist_ok=True)
    results: dict = {"subset": subset, "alpha": alpha, "cap": cfg.cap, "window": window,
                     "config": asdict(cfg), "seeds": list(seeds), "device": str(device),
                     "n_train_engines": int(len(tr_u)), "n_val_engines": int(len(va_u)),
                     "n_cal_engines": int(len(ca_u)),
                     "n_test_engines": int(len(prep.test_last)), "models": {}}
    ev = {"cal": sets["cal"].X, "test_last": sets["test_last"].X, "test_all": sets["test_all"].X}

    for name in classical:
        col = _Collector()
        model = fit_classical(name, sets["train"].X, sets["train"].y, seed=0)
        col.add(**{k: predict_classical(model, x, cfg.cap) for k, x in ev.items()})
        results["models"][name] = _summarise(
            name, col, sets, alpha, cfg.cap, trunc_per_unit, 0)

    for arch in archs:
        col = _Collector()
        for seed in seeds:
            model = train_nn(arch, sets["train"].X, sets["train"].y, sets["val"].X,
                             sets["val"].y, cfg, seed, device)
            torch.save(model.state_dict(), out / "models" / subset / f"{arch}_s{seed}.pt")
            col.add(**{k: predict(model, x, cfg.cap, device=torch.device("cpu"))
                       for k, x in ev.items()})
        results["models"][arch] = _summarise(arch, col, sets, alpha, cfg.cap, trunc_per_unit, 0)

    (out / f"{subset}.json").write_text(json.dumps(results, indent=2))
    return results


def load_members(models_dir: Path, subset: str, arch: str, seeds, in_dim, cfg: TrainConfig):
    members = []
    for s in seeds:
        m = build_model(arch, in_dim, cfg.hidden, cfg.dropout)
        m.load_state_dict(torch.load(models_dir / subset / f"{arch}_s{s}.pt", map_location="cpu"))
        members.append(m.eval())
    return members


def _ensemble(members, X, cap):
    p = np.stack([predict(m, X, cap, device=torch.device("cpu")) for m in members])
    return p.mean(0), (p.std(0) if len(members) > 1 else None)


def run_shift(raw_dir, source, target, archs, seeds, cfg: TrainConfig, models_dir, alpha=0.1,
              window=30, trunc_per_unit=10, recal_engines=25, out_path=None) -> dict:
    """Apply models trained on ``source`` to ``target`` and measure interval validity.

    Compares (a) the source calibration, (b) calibration on ``recal_engines`` labelled target
    engines (taken from the target *training* files, never seen by the source models), and
    (c) in-domain calibration as the reference (source models evaluated on source test).
    """
    models_dir = Path(models_dir)
    src = prepare(raw_dir, source, window, int(cfg.cap))
    tgt = prepare(raw_dir, target, window, int(cfg.cap))
    _, _, src_cal_u = split_units(src.train.unit, seed=0)
    src_cal = src.train.select(np.isin(src.train.unit, src_cal_u))
    rng = np.random.default_rng(1)
    tgt_units = np.array(sorted(np.unique(tgt.train.unit)))
    tgt_cal_u = rng.choice(tgt_units, size=min(recal_engines, len(tgt_units)), replace=False)
    tgt_cal = tgt.train.select(np.isin(tgt.train.unit, tgt_cal_u))

    results = {"source": source, "target": target, "alpha": alpha, "recal_engines": recal_engines,
               "models": {}}
    for arch in archs:
        members = load_members(models_dir, source, arch, seeds, src.num_features, cfg)

        parts = {k: _ensemble(members, w.X, cfg.cap) for k, w in {
            "src_cal": src_cal, "tgt_cal": tgt_cal,
            "src_test": src.test_last, "tgt_test": tgt.test_last}.items()}

        def pack(key, w, mask=None, parts=parts):
            m = slice(None) if mask is None else mask
            p, sd = parts[key]
            d = {"pred": p[m], "y": w.y[m]}
            if sd is not None:
                d["sigma"] = sd[m]
            return d

        t_src = truncation_mask(src_cal.unit, trunc_per_unit, 0)
        t_tgt = truncation_mask(tgt_cal.unit, trunc_per_unit, 0)
        results["models"][arch] = {
            "rmse_source_test": M.rmse(src.test_last.y, parts["src_test"][0]),
            "rmse_target_test": M.rmse(tgt.test_last.y, parts["tgt_test"][0]),
            "intervals": {
                "in_domain (src cal -> src test)": evaluate_intervals(
                    pack("src_cal", src_cal, t_src), pack("src_test", src.test_last), alpha,
                    cfg.cap),
                "shifted, source calibration (src cal -> tgt test)": evaluate_intervals(
                    pack("src_cal", src_cal, t_src), pack("tgt_test", tgt.test_last), alpha,
                    cfg.cap),
                "shifted, target recalibration (tgt cal -> tgt test)": evaluate_intervals(
                    pack("tgt_cal", tgt_cal, t_tgt), pack("tgt_test", tgt.test_last), alpha,
                    cfg.cap),
            },
        }
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results
