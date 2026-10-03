"""Per-domain orchestration: 3D MOI → OK Optuna with trial logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .cv import (
    ModelFixed,
    buffered_delete_d_score,
    default_holdout_size,
    make_buffered_delete_d_folds,
    make_spatial_block_splits,
    spatial_block_score,
)
from .directional import fit_directional_variogram
from .domain import iter_domains, safe_dirname
from .geometry import az_dip_from_direction, ellipsoid_axes
from .io import load_points, points_from_records
from .moi import estimate_moi_and_ranges
from .pairs import compute_pairs
from .plotting import save_directional_variogram
from .report import build_report_markdown, mark_boundaries
from .search_space import build_search_space
from .tpe_search import run_tpe

ALLOWED_CV = frozenset({"spatial_block", "buffered_delete_d"})


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg(cfg: dict, *keys, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_domain(
    domain: str,
    df_domain: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Variography + Optuna OK tune for one Domain (isolated)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = df_domain[["X", "Y", "Z"]].to_numpy(dtype=float)
    z = df_domain["Grade"].to_numpy(dtype=float)
    n = len(df_domain)
    n_min = int(_cfg(cfg, "n_min_points", default=15))

    z_extent = {
        "z_min": float(xyz[:, 2].min()),
        "z_max": float(xyz[:, 2].max()),
        "z_span": float(xyz[:, 2].max() - xyz[:, 2].min()),
        "x_span": float(xyz[:, 0].max() - xyz[:, 0].min()),
        "y_span": float(xyz[:, 1].max() - xyz[:, 1].min()),
    }

    pairs = compute_pairs(xyz, z)
    moi = estimate_moi_and_ranges(
        pairs,
        azimuth_step_deg=float(_cfg(cfg, "moi", "azimuth_step_deg", default=15.0)),
        dip_step_deg=float(_cfg(cfg, "moi", "dip_step_deg", default=10.0)),
        dip_min=float(_cfg(cfg, "moi", "dip_min", default=-80.0)),
        dip_max=float(_cfg(cfg, "moi", "dip_max", default=80.0)),
        tolerance_deg=float(_cfg(cfg, "directional", "tolerance_deg", default=20.0)),
        n_lags=int(_cfg(cfg, "directional", "n_lags", default=12)),
        max_dist_percentile=float(
            _cfg(cfg, "directional", "max_dist_percentile", default=50.0)
        ),
        min_pairs=int(_cfg(cfg, "directional", "min_pairs", default=8)),
    )

    dir_kw = dict(
        tolerance_deg=float(_cfg(cfg, "directional", "tolerance_deg", default=20.0)),
        n_lags=int(_cfg(cfg, "directional", "n_lags", default=12)),
        max_dist_percentile=float(
            _cfg(cfg, "directional", "max_dist_percentile", default=50.0)
        ),
        min_pairs=int(_cfg(cfg, "directional", "min_pairs", default=8)),
    )
    inter_az = (moi.azimuth_deg + 90.0) % 180.0
    _, _, e3 = ellipsoid_axes(moi.azimuth_deg, moi.dip_deg)
    minor_az, minor_dip = az_dip_from_direction(e3)

    major_vg = fit_directional_variogram(
        pairs, moi.azimuth_deg, moi.dip_deg, **dir_kw
    )
    inter_vg = fit_directional_variogram(pairs, inter_az, 0.0, **dir_kw)
    minor_vg = fit_directional_variogram(pairs, minor_az, minor_dip, **dir_kw)

    save_directional_variogram(major_vg, out_dir / "variogram_major.png", f"{domain} major")
    save_directional_variogram(inter_vg, out_dir / "variogram_inter.png", f"{domain} inter")
    save_directional_variogram(minor_vg, out_dir / "variogram_minor.png", f"{domain} minor")

    nugget_fit = float(max(moi.nugget_fit, 0.0))
    sill_partial = float(max(moi.sill_partial, 1e-12))
    sill_total = max(nugget_fit + sill_partial, float(np.var(z)), 1e-6)

    fixed = ModelFixed(
        azimuth_deg=float(moi.azimuth_deg),
        dip_deg=float(moi.dip_deg),
        a_major=float(moi.a_major),
        a_inter=float(moi.a_inter),
        a_minor=float(moi.a_minor),
        sill_partial_base=sill_partial,
        sill_total=sill_total,
        nugget_fit=nugget_fit,
        isotropic=bool(_cfg(cfg, "isotropic", default=False)),
    )

    payload_base: dict[str, Any] = {
        "domain": domain,
        "n_points": n,
        "extent": z_extent,
        "moi": {
            "azimuth_deg": moi.azimuth_deg,
            "dip_deg": moi.dip_deg,
            "a_major": moi.a_major,
            "a_inter": moi.a_inter,
            "a_minor": moi.a_minor,
        },
        "directional": {
            "major": {
                "azimuth_deg": major_vg.azimuth_deg,
                "dip_deg": major_vg.dip_deg,
                "range": major_vg.range_,
                "nugget": major_vg.nugget,
                "sill": major_vg.sill,
            },
            "inter": {
                "azimuth_deg": inter_vg.azimuth_deg,
                "dip_deg": inter_vg.dip_deg,
                "range": inter_vg.range_,
                "nugget": inter_vg.nugget,
                "sill": inter_vg.sill,
            },
            "minor": {
                "azimuth_deg": minor_vg.azimuth_deg,
                "dip_deg": minor_vg.dip_deg,
                "range": minor_vg.range_,
                "nugget": minor_vg.nugget,
                "sill": minor_vg.sill,
            },
        },
        "fixed_model": {
            "azimuth_deg": fixed.azimuth_deg,
            "dip_deg": fixed.dip_deg,
            "a_major": fixed.a_major,
            "a_inter": fixed.a_inter,
            "a_minor": fixed.a_minor,
            "nugget_fit": fixed.nugget_fit,
            "sill_total": fixed.sill_total,
            "isotropic": fixed.isotropic,
        },
    }

    if n < n_min:
        payload_base["optimization"] = {
            "status": "SKIPPED",
            "reason": f"n_points={n} < n_min={n_min}",
        }
        (out_dir / "best_params.json").write_text(
            json.dumps(payload_base, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload_base

    space = build_search_space(
        fixed,
        radius_margin=float(_cfg(cfg, "search_space", "radius_margin", default=0.20)),
        nugget_margin=float(_cfg(cfg, "search_space", "nugget_margin", default=0.20)),
        range_scale_min=float(_cfg(cfg, "search_space", "range_scale_min", default=0.8)),
        range_scale_max=float(_cfg(cfg, "search_space", "range_scale_max", default=1.2)),
        nmax_min=int(_cfg(cfg, "search_space", "nmax_min", default=5)),
        nmax_max=int(_cfg(cfg, "search_space", "nmax_max", default=40)),
        radius_prior_scale=float(
            _cfg(cfg, "search_space", "radius_prior_scale", default=1.0)
        ),
    )

    penalty = float(_cfg(cfg, "optuna", "invalid_penalty", default=1e6))
    cv_method = str(_cfg(cfg, "cv", "method", default="spatial_block")).lower()
    if cv_method not in ALLOWED_CV:
        raise ValueError(
            f"cv.method must be one of {sorted(ALLOWED_CV)}, got {cv_method!r}"
        )
    cv_seed = int(_cfg(cfg, "cv", "seed", default=42))

    if cv_method == "spatial_block":
        grid_nx = int(_cfg(cfg, "cv", "grid_nx", default=3))
        grid_ny = int(_cfg(cfg, "cv", "grid_ny", default=3))
        grid_nz = int(_cfg(cfg, "cv", "grid_nz", default=2))
        test_folds = make_spatial_block_splits(
            xyz, grid_nx=grid_nx, grid_ny=grid_ny, grid_nz=grid_nz
        )
        if not test_folds:
            raise RuntimeError(f"spatial_block produced no folds for domain={domain}")

        def objective_fn(theta):
            return spatial_block_score(
                xyz,
                z,
                theta,
                fixed,
                test_folds,
                invalid_penalty=penalty,
                grid_nx=grid_nx,
                grid_ny=grid_ny,
                grid_nz=grid_nz,
            )

        cv_label = (
            f"spatial_block {grid_nx}x{grid_ny}x{grid_nz} n={n} folds={len(test_folds)}"
        )
    else:
        holdout = _cfg(cfg, "cv", "holdout_size", default=None)
        holdout = default_holdout_size(n) if holdout is None else int(holdout)
        holdout = min(holdout, n - 2)
        n_repeats = int(_cfg(cfg, "cv", "n_repeats", default=25))
        buffer_scale = _cfg(cfg, "cv", "buffer_scale", default=0.5)
        use_aniso_metric = bool(
            _cfg(cfg, "cv", "buffer_anisotropic", default=not fixed.isotropic)
        ) and (not fixed.isotropic)
        if buffer_scale is not None:
            buffer_radius = float(buffer_scale) * float(fixed.a_major)
            buffer_metric = "aniso_ok3d" if use_aniso_metric else "euclidean"
        else:
            buffer_radius = float(_cfg(cfg, "cv", "buffer_radius", default=0.4))
            buffer_metric = "aniso_ok3d" if use_aniso_metric else "euclidean"
            buffer_scale = None

        folds = make_buffered_delete_d_folds(
            xyz,
            holdout_size=holdout,
            n_repeats=n_repeats,
            buffer_radius=buffer_radius,
            seed=cv_seed,
            anisotropic=use_aniso_metric,
            azimuth_deg=fixed.azimuth_deg if use_aniso_metric else None,
            dip_deg=fixed.dip_deg if use_aniso_metric else None,
            a_major=fixed.a_major if use_aniso_metric else None,
            a_inter=fixed.a_inter if use_aniso_metric else None,
            a_minor=fixed.a_minor if use_aniso_metric else None,
        )
        if not folds:
            raise RuntimeError(
                f"buffered_delete_d produced no folds for domain={domain} "
                f"(d={holdout}, r={buffer_radius}, metric={buffer_metric})"
            )

        def objective_fn(theta):
            return buffered_delete_d_score(
                xyz,
                z,
                theta,
                fixed,
                folds,
                buffer_radius=buffer_radius,
                holdout_size=holdout,
                invalid_penalty=penalty,
                buffer_metric=buffer_metric,
                buffer_scale=float(buffer_scale) if buffer_scale is not None else None,
            )

        cv_label = (
            f"buffered_delete_d n={n} d={holdout} r={buffer_radius:.4g} "
            f"metric={buffer_metric} scale={buffer_scale} "
            f"repeats={len(folds)}/{n_repeats}"
        )

    print(
        f"[{domain}] OK3D Optuna cv={cv_method} {cv_label} "
        f"iso={fixed.isotropic} "
        f"az={fixed.azimuth_deg:.2f}° dip={fixed.dip_deg:.2f}° "
        f"a={fixed.a_major:.4g}/{fixed.a_inter:.4g}/{fixed.a_minor:.4g}",
        flush=True,
    )
    opt = run_tpe(
        space=space,
        fixed=fixed,
        objective_fn=objective_fn,
        n_trials=int(_cfg(cfg, "optuna", "n_trials", default=120)),
        seed=int(_cfg(cfg, "optuna", "seed", default=42)),
        out_csv=out_dir / "trials.csv",
        best_json=out_dir / "best_params_optuna.json",
        n_startup_trials=_cfg(cfg, "optuna", "n_startup_trials", default=None),
        n_ei_candidates=int(_cfg(cfg, "optuna", "n_ei_candidates", default=64)),
    )
    if opt.get("best_theta") and opt.get("search_space"):
        opt["boundary_flags"] = mark_boundaries(opt["best_theta"], opt["search_space"])
    payload_base["optimization"] = opt
    (out_dir / "best_params.json").write_text(
        json.dumps(payload_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload_base


def run_pipeline(
    data_path: str | Path | None = None,
    out_dir: str | Path = "results",
    config_path: str | Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune OK for every Domain; write summary.json + report.md."""
    cfg = load_config(config_path)
    if cfg_override:
        cfg = deep_merge(cfg, cfg_override)

    if df is None:
        if data_path is None:
            raise ValueError("Provide data_path or df")
        df = load_points(data_path)
        data_label = str(data_path)
    else:
        data_label = "in-memory"

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for domain, sub in iter_domains(df):
        print(f"=== Domain {domain} (n={len(sub)}) ===", flush=True)
        summaries.append(run_domain(domain, sub, root / safe_dirname(domain), cfg))

    cv_method = str(_cfg(cfg, "cv", "method", default="spatial_block"))
    report_md = build_report_markdown(
        summaries,
        data_label=data_label,
        out_dir=str(root),
        cv_method=cv_method,
        cfg=cfg,
    )
    (root / "report.md").write_text(report_md, encoding="utf-8")

    file_summary: dict[str, Any] = {
        "csv": data_label,
        "cv_method": cv_method,
        "primary_score": f"CV RMSE ({cv_method})",
        "n_domains": len(summaries),
        "n_points_total": int(len(df)),
        "domains": summaries,
        "report_path": str(root / "report.md"),
        "out_dir": str(root),
    }
    (root / "summary.json").write_text(
        json.dumps(file_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**file_summary, "report_markdown": report_md}


def run_pipeline_from_records(
    records: list[dict],
    out_dir: str | Path,
    config_path: str | Path | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = points_from_records(records)
    return run_pipeline(
        data_path=None,
        out_dir=out_dir,
        config_path=config_path,
        df=df,
        cfg_override=cfg_override,
    )
