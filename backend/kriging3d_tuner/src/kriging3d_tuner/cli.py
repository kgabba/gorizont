"""CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_tuner


def main() -> None:
    p = argparse.ArgumentParser(
        description="3D ANISO OK / IDW tuner: fixed orientation + Optuna neighbourhood"
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
    p.add_argument(
        "--mode",
        choices=["single", "multipass", "sequential_multipass_optuna_J"],
        default=None,
        help=(
            "single (default), multipass (30/60/100 gates), or "
            "sequential_multipass_optuna_J (minimize J, max 3 passes)"
        ),
    )
    p.add_argument(
        "--method",
        choices=["ordinary_kriging", "anisotropic_idw"],
        default=None,
        help="Interpolation method (default: ordinary_kriging)",
    )
    args = p.parse_args()

    run_tuner(
        data_path=Path(args.data),
        anisotropy_path=Path(args.anisotropy),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config) if args.config else None,
        n_trials=args.trials,
        mode=args.mode,
        method=args.method,
    )


if __name__ == "__main__":
    main()
