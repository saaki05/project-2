"""``predictiq-serve --bundle model.pt``."""

from __future__ import annotations

import argparse

import uvicorn

from predictiq_service.api import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="predictiq-serve")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(a.bundle), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
