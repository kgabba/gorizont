"""CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_tuner


def main() -> None:
    p = argparse.ArgumentParser(
        description="3D ANISO OK tuner: fixed variogram + Optuna search neighbourhood"
    )
    p.add_argument("--data", required=True, help="CSV with X,Y,Z,Value [+HoleID]")
    p.add_argument(
        "--anisotropy",
        required=True,
        help="anisotropy3d result.json (or aniso_candidate JSON)",
    )
    p.add_argument("--config", default=None, help="YAML config (default: package config.yaml)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--trials", type=int, default=None, help="Override optuna.n_trials")
    args = p.parse_args()

    run_tuner(
        data_path=Path(args.data),
        anisotropy_path=Path(args.anisotropy),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config) if args.config else None,
        n_trials=args.trials,
    )


if __name__ == "__main__":
    main()
