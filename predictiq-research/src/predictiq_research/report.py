"""Render result JSON files into Markdown tables. Numbers are never typed by hand."""

from __future__ import annotations

import json
from pathlib import Path


def _load(results_dir: Path):
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("FD00?.json"))}
    shifts = [json.loads(p.read_text()) for p in sorted(results_dir.glob("shift_*.json"))]
    return runs, shifts


def _best_interval(iv: dict) -> tuple[str, dict]:
    name = "normalized" if "normalized" in iv else "split"
    return name, iv[name]


def build_report(results_dir: str | Path) -> str:
    runs, shifts = _load(Path(results_dir))
    out = ["# PredictIQ results (generated from result JSON files)", ""]
    if runs:
        any_run = next(iter(runs.values()))
        out += [
            f"Target interval coverage: {1 - any_run['alpha']:.0%}. RUL cap: {any_run['cap']:.0f}. "
            f"Window: {any_run['window']}. Seeds: {any_run['seeds']}. "
            f"Device: {any_run['device']}.", "",
            "## Point accuracy on the official test engines (last window)", "",
            "RMSE/MAE in cycles against the capped true RUL; `ens` is the seed ensemble, "
            "`single` is the mean ± std over individual seeds.", "",
            "| subset | model | RMSE (ens) | RMSE (single) | MAE (ens) | NASA score (ens) |",
            "|---|---|---|---|---|---|",
        ]
        for subset, r in runs.items():
            for name, m in r["models"].items():
                e = m["ensemble"]
                single = m.get("single_member")
                s = (f"{single['rmse']['mean']:.2f} ± {single['rmse']['std']:.2f}"
                     if single else "n/a")
                out.append(f"| {subset} | {name} | {e['rmse']:.2f} | {s} | {e['mae']:.2f} | "
                           f"{e['nasa_score']:.0f} |")
        out += ["", "## Interval validity on the official test engines", "",
                "Coverage with a 95% Wilson interval and mean width. `cal=trunc` calibrates on "
                "random cut-offs of held-out engines; `cal=all` on every cycle. Conformal "
                "variant: normalized (ensemble σ) when available, else split.", "",
                "| subset | model | protocol | variant | coverage [95% CI] | width | "
                "coverage RUL≤30 (n) |", "|---|---|---|---|---|---|---|"]
        for subset, r in runs.items():
            for name, m in r["models"].items():
                for proto in ("test_last|cal=trunc", "test_last|cal=all"):
                    variant, s = _best_interval(m["intervals"][proto])
                    ci = s["coverage_ci95"]
                    crit = ("n/a" if s["coverage_critical"] is None
                            else f"{s['coverage_critical']:.2f} ({s['n_critical']})")
                    out.append(f"| {subset} | {name} | {proto.split('|')[1]} | {variant} | "
                               f"{s['coverage']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                               f"{s['width']:.1f} | {crit} |")
    if shifts:
        out += ["", "## Distribution shift", "",
                "Models trained on the source subset, evaluated on the target subset. "
                "Rows compare in-domain calibration, source calibration under shift and "
                "recalibration on labelled target engines.", "",
                "| source → target | model | RMSE src | RMSE tgt | setting | coverage [95% CI] | "
                "width |", "|---|---|---|---|---|---|---|"]
        for sh in shifts:
            for arch, m in sh["models"].items():
                for setting, iv in m["intervals"].items():
                    variant, s = _best_interval(iv)
                    ci = s["coverage_ci95"]
                    out.append(f"| {sh['source']} → {sh['target']} | {arch} | "
                               f"{m['rmse_source_test']:.2f} | {m['rmse_target_test']:.2f} | "
                               f"{setting} | {s['coverage']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                               f"{s['width']:.1f} |")
    return "\n".join(out) + "\n"
