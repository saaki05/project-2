"""``predictiq-data prepare --raw DIR --subset FD001 --out DIR``."""

from __future__ import annotations

import argparse

from predictiq_data.cmapss import SUBSETS
from predictiq_data.dataset import prepare, save_prepared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="predictiq-data")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="normalise, window and store one C-MAPSS subset")
    p.add_argument("--raw", required=True, help="directory containing the extracted C-MAPSS files")
    p.add_argument("--subset", required=True, choices=SUBSETS)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--cap", type=int, default=125)
    args = ap.parse_args(argv)

    prepared = prepare(args.raw, args.subset, args.window, args.cap)
    m = save_prepared(prepared, args.out)
    print(f"{m['subset']}: {m['n_train']} train windows, {m['n_test_engines']} test engines, "
          f"{m['num_features']} features -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
