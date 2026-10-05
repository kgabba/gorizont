"""CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_anisotropy3d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Estimate 3D continuity anisotropy (variogram ellipsoid)"
    )
    p.add_argument("--data", type=Path, required=True, help="CSV with X,Y,Z,Value")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: package config.yaml)",
    )
    args = p.parse_args(argv)
    run_anisotropy3d(args.data, args.out_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
