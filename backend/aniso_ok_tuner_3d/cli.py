"""CLI entry for aniso_ok_tuner_3d."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .pipeline import load_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tune 3D anisotropic Ordinary Kriging per Domain "
            "(MOI az/dip + Optuna; CV=spatial_block|buffered_delete_d)."
        )
    )
    p.add_argument("--data", type=Path, required=True, help="CSV X,Y,Z,Grade[,Domain]")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--trials", type=int, default=None, help="Override optuna.n_trials")
    p.add_argument(
        "--cv",
        type=str,
        default=None,
        choices=["spatial_block", "buffered_delete_d"],
        help="Override cv.method",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = args.config
    if args.trials is not None or args.cv is not None:
        cfg = load_config(cfg_path)
        if args.trials is not None:
            cfg.setdefault("optuna", {})
            cfg["optuna"]["n_trials"] = int(args.trials)
        if args.cv is not None:
            cfg.setdefault("cv", {})
            cfg["cv"]["method"] = args.cv
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ephemeral = args.out_dir / "_run_config.yaml"
        ephemeral.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        cfg_path = ephemeral

    run_pipeline(args.data, args.out_dir, cfg_path)
    print(f"Done. See {args.out_dir / 'summary.json'} and report.md", flush=True)
    return 0
