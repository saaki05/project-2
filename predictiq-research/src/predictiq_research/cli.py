"""``predictiq-research run`` and ``predictiq-research shift``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictiq_research.experiments import run_shift, run_subset
from predictiq_research.train import TrainConfig


def _fmt(results: dict) -> str:
    lines = [f"{results['subset']}  (alpha={results['alpha']}, seeds={results['seeds']}, "
             f"device={results['device']})",
             f"{'model':<13}{'RMSE':>8}{'MAE':>8}{'score':>10}   coverage(test_last|trunc)  width"]
    for name, r in results["models"].items():
        e = r["ensemble"]
        iv = r["intervals"]["test_last|cal=trunc"]
        best = iv.get("normalized", iv["split"])
        lines.append(f"{name:<13}{e['rmse']:>8.2f}{e['mae']:>8.2f}{e['nasa_score']:>10.0f}"
                     f"   {best['coverage']:.3f}  width {best['width']:.1f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="predictiq-research")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="train and evaluate on one subset")
    r.add_argument("--raw", required=True)
    r.add_argument("--subset", required=True)
    r.add_argument("--archs", nargs="*", default=["lstm", "gru", "tcn", "transformer"])
    r.add_argument("--classical", nargs="*", default=["ridge", "hgb"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--alpha", type=float, default=0.1)
    r.add_argument("--out", default="results")

    s = sub.add_parser("shift", help="apply source-trained models to a target subset")
    s.add_argument("--raw", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--archs", nargs="*", default=["lstm", "gru", "tcn", "transformer"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    s.add_argument("--alpha", type=float, default=0.1)
    s.add_argument("--out", default="results")

    rp = sub.add_parser("report", help="render result JSON files as Markdown tables")
    rp.add_argument("--results", default="results")
    rp.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    if args.command == "report":
        from predictiq_research.report import build_report

        text = build_report(args.results)
        if args.out:
            Path(args.out).write_text(text)
        print(text)
        return 0
    cfg = TrainConfig(epochs=getattr(args, "epochs", 40))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        res = run_subset(args.raw, args.subset, args.archs, args.classical, args.seeds, cfg,
                         args.alpha, out)
        print(_fmt(res))
    else:
        res = run_shift(args.raw, args.source, args.target, args.archs, args.seeds, cfg,
                        out / "models", args.alpha,
                        out_path=out / f"shift_{args.source}_to_{args.target}.json")
        print(json.dumps({a: {"rmse_src": m["rmse_source_test"], "rmse_tgt": m["rmse_target_test"]}
                          for a, m in res["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
