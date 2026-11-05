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
    out += _summary(runs, shifts)
    return "\n".join(out) + "\n"


def _summary(runs: dict, shifts: list) -> list[str]:
    """Aggregates over the neural architectures (mean of per-architecture values)."""
    out: list[str] = []
    if runs:
        out += ["", "## Summary: in-distribution coverage (neural models, nominal 0.90)", "",
                "Mean coverage over LSTM/GRU/TCN/Transformer per subset and calibration "
                "protocol (each model has n = number of test engines).", "",
                "| subset | cal=trunc | cal=all | min over models (trunc) | "
                "max over models (trunc) |",
                "|---|---|---|---|---|"]
        for subset, r in runs.items():
            neural = {k: v for k, v in r["models"].items() if v["n_members"] > 1}
            tr = [_best_interval(v["intervals"]["test_last|cal=trunc"])[1]["coverage"]
                  for v in neural.values()]
            al = [_best_interval(v["intervals"]["test_last|cal=all"])[1]["coverage"]
                  for v in neural.values()]
            out.append(f"| {subset} | {sum(tr) / len(tr):.3f} | {sum(al) / len(al):.3f} | "
                       f"{min(tr):.3f} | {max(tr):.3f} |")
    if shifts:
        out += ["", "## Summary: shift (neural models, mean over architectures)", "",
                "| source → target | RMSE source | RMSE target | coverage in-domain | "
                "coverage shifted (source cal.) | coverage shifted (target recal.) | "
                "width in-domain | width shifted (source cal.) | width recal. |",
                "|---|---|---|---|---|---|---|---|---|"]
        for sh in shifts:
            ms = list(sh["models"].values())

            def mean(f, ms=ms):
                return sum(f(m) for m in ms) / len(ms)

            def cov(m, key):
                k = next(k for k in m["intervals"] if k.startswith(key))
                return _best_interval(m["intervals"][k])[1]

            out.append(
                f"| {sh['source']} → {sh['target']} | "
                f"{mean(lambda m: m['rmse_source_test']):.2f} | "
                f"{mean(lambda m: m['rmse_target_test']):.2f} | "
                f"{mean(lambda m: cov(m, 'in_domain')['coverage']):.3f} | "
                f"{mean(lambda m: cov(m, 'shifted, source')['coverage']):.3f} | "
                f"{mean(lambda m: cov(m, 'shifted, target')['coverage']):.3f} | "
                f"{mean(lambda m: cov(m, 'in_domain')['width']):.1f} | "
                f"{mean(lambda m: cov(m, 'shifted, source')['width']):.1f} | "
                f"{mean(lambda m: cov(m, 'shifted, target')['width']):.1f} |")
    return out
