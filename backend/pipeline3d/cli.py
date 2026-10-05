"""CLI for gorizont3d end-to-end pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import PipelineError, run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Orchestrate anisotropy3d → aniso_candidate gate → kriging3d_tuner"
        )
    )
    p.add_argument("--data", type=Path, required=True, help="CSV with X,Y,Z,Value")
    p.add_argument(
        "--aniso-config",
        type=Path,
        required=True,
        help="anisotropy3d config.yaml",
    )
    p.add_argument(
        "--tuner-config",
        type=Path,
        required=True,
        help="kriging3d_tuner config.yaml",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="e.g. runs/test01")
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override kriging3d_tuner Optuna n_trials",
    )
    args = p.parse_args(argv)

    try:
        run_pipeline(
            data_path=args.data,
            out_dir=args.out_dir,
            aniso_config=args.aniso_config,
            tuner_config=args.tuner_config,
            n_trials=args.trials,
        )
    except PipelineError as e:
        print(f"[pipeline3d] ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
