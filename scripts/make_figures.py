"""Generate every PredictIQ figure from the result JSON files.

Usage: python scripts/make_figures.py [--raw CMAPSS_DIR --bundle fd001_bundle.pt]
The optional arguments add the per-engine trajectory figure (needs the raw data and a bundle
created with ``predictiq-export``).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "predictiq-research" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fs.setup()
NEURAL = ("lstm", "gru", "tcn", "transformer")
NAME = {"ridge": "Ridge", "hgb": "Gradient boosting", "lstm": "LSTM", "gru": "GRU",
        "tcn": "TCN", "transformer": "Transformer"}


def runs():
    return {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(RES / "FD00?.json")))}


def best(iv):
    return iv["normalized"] if "normalized" in iv else iv["split"]


def accuracy():
    data = runs()
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharex=True)
    mean_rmse = {m: np.mean([r["models"][m]["ensemble"]["rmse"] for r in data.values()])
                 for m in NAME}
    models = sorted(NAME, key=lambda m: mean_rmse[m], reverse=True)  # one shared order
    for ax, (subset, r) in zip(axes, data.items(), strict=True):
        for i, m in enumerate(models):
            rm = r["models"][m]
            col = fs.BLUE if m in NEURAL else fs.GRAY
            ax.scatter([rm["ensemble"]["rmse"]], [i], s=55, color=col, zorder=3,
                       edgecolor=fs.SURFACE)
            if "single_member" in rm:
                ax.scatter([rm["single_member"]["rmse"]["mean"]], [i], s=40, facecolor="none",
                           edgecolor=col, lw=1.6, zorder=2)
            right = max(rm["ensemble"]["rmse"], rm.get("single_member", {}).get(
                "rmse", {}).get("mean", 0))
            ax.text(right + 0.35, i, f"{rm['ensemble']['rmse']:.1f}", va="center", fontsize=8,
                    color=fs.INK2)
        ax.set_yticks(range(len(models)), [NAME[m] for m in models] if ax is axes[0] else [])
        ax.set_title(subset)
        ax.set_xlim(12, 21)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RMSE (cycles)")
    axes[0].scatter([], [], s=55, color=fs.BLUE, label="neural: seed ensemble")
    axes[0].scatter([], [], s=40, facecolor="none", edgecolor=fs.BLUE, lw=1.6,
                    label="neural: single seed (mean)")
    axes[0].scatter([], [], s=55, color=fs.GRAY, label="classical (single model)")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Point accuracy on the official test engines (capped RUL)", x=0.07, ha="left",
                 y=1.02, fontweight="bold", fontsize=11)
    fs.save(fig, OUT / "accuracy.png")


def coverage():
    data = runs()
    rows = []
    for subset, r in data.items():
        for m in NEURAL:
            s = best(r["models"][m]["intervals"]["test_last|cal=trunc"])
            rows.append((subset, m, s["coverage"], *s["coverage_ci95"]))
    fig, ax = plt.subplots(figsize=(8, 6.4))
    y = 0
    ticks, labels = [], []
    for k, subset in enumerate(data):
        for r in [x for x in rows if x[0] == subset]:
            ok = r[3] <= 0.9 <= r[4]
            col = fs.BLUE if ok else fs.ORANGE
            ax.plot([r[3], r[4]], [y, y], color=col, lw=2.4)
            ax.scatter([r[2]], [y], color=col, s=42, zorder=3, edgecolor=fs.SURFACE)
            ticks.append(y)
            labels.append(f"{subset}  {NAME[r[1]]}")
            y += 1
        y += 0.8
    ax.axvline(0.9, color=fs.INK2, ls="--", lw=1.2, label="nominal 0.90")
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.72, 1.0)
    ax.set_xlabel("empirical coverage of 90% intervals (dot) with 95% Wilson interval")
    ax.set_title("In distribution, 13 of 16 models' intervals contain the nominal 0.90")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=fs.BLUE, label="interval contains 0.90")
    ax.scatter([], [], color=fs.ORANGE, label="below nominal")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.09))
    fs.save(fig, OUT / "coverage_in_distribution.png")


def shift():
    shifts = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "shift_*.json")))]
    cols = [fs.BLUE, fs.ORANGE, fs.AQUA, fs.VIOLET, fs.MAGENTA]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [0, 1, 2]
    for sh, col in zip(shifts, cols, strict=False):
        def mean_cov(key):
            vals = []
            for m in sh["models"].values():
                k = next(k for k in m["intervals"] if k.startswith(key))
                vals.append(best(m["intervals"][k])["coverage"])
            return float(np.mean(vals))
        ys = [mean_cov("in_domain"), mean_cov("shifted, source"), mean_cov("shifted, target")]
        ax.plot(xs, ys, color=col, marker="o", lw=2, ms=6, markeredgecolor=fs.SURFACE)
        ax.text(2.05, ys[2], f"{sh['source']} → {sh['target']}", va="center", fontsize=8.8,
                color=fs.INK2)
    ax.axhline(0.9, color=fs.INK2, ls="--", lw=1.2)
    ax.text(-0.03, 0.9015, "nominal 0.90", fontsize=8.5, color=fs.INK2, ha="left", va="bottom")
    ax.set_xticks(xs, ["in-domain\ncalibration", "shifted target,\nsource calibration",
                       "shifted target,\nrecalibrated on 25 target engines"])
    ax.set_xlim(-0.1, 2.55)
    ax.set_ylabel("coverage (mean over 4 neural models)")
    ax.set_title("Intervals lose coverage under shift; recalibration restores it")
    ax.grid(axis="x", visible=False)
    fs.save(fig, OUT / "shift_coverage.png")


def widths():
    shifts = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "shift_*.json")))]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    names = [f"{s['source']} → {s['target']}" for s in shifts]
    def w(sh, key):
        vals = []
        for m in sh["models"].values():
            k = next(k for k in m["intervals"] if k.startswith(key))
            vals.append(best(m["intervals"][k])["width"])
        return float(np.mean(vals))
    x = np.arange(len(shifts))
    for j, (key, lab, col) in enumerate([("in_domain", "in-domain", fs.GRAY),
                                         ("shifted, source", "shifted, source calibration", fs.BLUE),
                                         ("shifted, target", "shifted, recalibrated", fs.ORANGE)]):
        ax.bar(x + (j - 1) * 0.27, [w(s, key) for s in shifts], 0.25, color=col, label=lab)
    ax.set_xticks(x, names)
    ax.set_ylabel("mean interval width (cycles)")
    ax.set_title("Normalized intervals widen on their own under shift (ensemble disagreement)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.12))
    fs.save(fig, OUT / "interval_width.png")


def trajectories(raw, bundle_path):
    from predictiq_service.bundle import load_bundle

    bundle = load_bundle(bundle_path)
    test = np.loadtxt(Path(raw) / "test_FD001.txt")
    truth = np.loadtxt(Path(raw) / "RUL_FD001.txt")
    units = np.unique(test[:, 0]).astype(int)
    order = np.argsort(truth)
    picks = [units[order[3]], units[order[len(order) // 2]], units[order[-8]]]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    for ax, u in zip(axes, picks, strict=True):
        rows = test[test[:, 0] == u][:, 2:]
        T = len(rows)
        final = truth[u - 1]
        cycles = np.arange(1, T + 1)
        true = np.minimum(final + (T - cycles), 125)
        preds = [bundle.predict(rows[:t]) for t in cycles]
        mid = np.array([p["rul"] for p in preds])
        lo = np.array([p["lower"] for p in preds])
        hi = np.array([p["upper"] for p in preds])
        ax.fill_between(cycles, lo, hi, color=fs.BLUE, alpha=0.2, lw=0, label="90% interval")
        ax.plot(cycles, mid, color=fs.BLUE, label="prediction")
        ax.plot(cycles, true, color=fs.INK, lw=1.6, ls="--", label="true RUL (capped)")
        ax.set_title(f"test engine {u}")
        ax.set_xlabel("cycle")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("remaining useful life (cycles)")
    axes[0].legend(loc="lower left")
    fig.suptitle("FD001 LSTM ensemble: prediction and conformal interval at every cycle",
                 x=0.07, ha="left", y=1.04, fontweight="bold", fontsize=11)
    fs.save(fig, OUT / "trajectories.png")


def tuning():
    f = RES / "tuning_fd001.json"
    if not f.exists():
        return
    d = json.load(open(f))
    rows = list(d["results"].items())
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, (name, r) in enumerate(rows):
        sel = name == d["selected_by_validation"]
        ax.scatter([r["val_rmse"]], [i], color=fs.GRAY, s=70, zorder=3,
                   label="validation RMSE (used to choose)" if i == 0 else None)
        ax.scatter([r["test_rmse"]], [i], color=fs.BLUE, s=70, zorder=3,
                   edgecolor=fs.INK if sel else fs.SURFACE, linewidth=2 if sel else 1,
                   label="test RMSE" if i == 0 else None)
        ax.text(r["test_rmse"] + 0.2, i, f"{r['test_rmse']:.2f}" + ("  ← chosen on validation" if sel else ""),
                va="center", fontsize=8.5, color=fs.INK2)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(8.5, 17)
    ax.set_xlabel("RMSE (cycles), GRU seed ensemble on FD001")
    ax.set_title("Wider and longer-window GRUs: validation improves, test changes little")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    fs.save(fig, OUT / "tuning_fd001.png")


def architecture():
    fs.pipeline(OUT / "architecture.png", [
        ("data-pipeline", ["validated C-MAPSS loader", "regime-aware normalisation",
                           "windows, engine-wise splits", "leak audit"]),
        ("research", ["LSTM · GRU · TCN · Transformer", "seed ensembles (mean, σ)",
                      "split / normalized / binned", "conformal intervals", "shift experiments"]),
        ("deployment", ["bundle export + calibration", "/predict → RUL, interval, alert",
                        "/drift (PSI, KS)", "FastAPI + Docker"]),
    ], "PredictIQ: three packages, one-way dependencies", accent=fs.BLUE, width=10.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--bundle")
    a = ap.parse_args()
    accuracy()
    coverage()
    shift()
    widths()
    architecture()
    tuning()
    if a.raw and a.bundle:
        trajectories(a.raw, a.bundle)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
