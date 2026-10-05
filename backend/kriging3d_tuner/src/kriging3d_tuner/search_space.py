"""Optuna search space for hard search neighbourhood."""

from __future__ import annotations

from typing import Any

import optuna

from .models import SearchParams, SearchSpaceConfig, VariogramFixed


def build_search_space(vg: VariogramFixed, cfg: dict[str, Any]) -> SearchSpaceConfig:
    scfg = cfg.get("search_space") or {}
    r0 = float(vg.range_major)
    r_lo = float(scfg.get("r_lo", 0.5))
    r_hi = float(scfg.get("r_hi", 3.0))
    return SearchSpaceConfig(
        r_major_min=max(r0 * r_lo, 1e-12),
        r_major_max=max(r0 * r_hi, r0 * r_lo + 1e-12),
        k_min=float(scfg.get("k_min", 0.05)),
        nmin_lo=int(scfg.get("nmin_lo", 4)),
        nmin_hi=int(scfg.get("nmin_hi", 12)),
        nmax_lo=int(scfg.get("nmax_lo", 8)),
        nmax_hi=int(scfg.get("nmax_hi", 64)),
    )


def suggest_search_params(
    trial: optuna.Trial,
    space: SearchSpaceConfig,
) -> SearchParams:
    """Ordered sampling: K_minor <= K_inter <= 1; Nmin <= Nmax."""
    r_major = float(
        trial.suggest_float("R_major", space.r_major_min, space.r_major_max)
    )
    k_inter = float(
        trial.suggest_float("K_inter", space.k_min, 1.0)
    )
    k_minor = float(
        trial.suggest_float("K_minor", space.k_min, k_inter)
    )
    n_max = int(
        trial.suggest_int("Nmax", space.nmax_lo, space.nmax_hi)
    )
    nmin_hi_eff = min(space.nmin_hi, n_max)
    n_min = int(
        trial.suggest_int("Nmin", space.nmin_lo, max(nmin_hi_eff, space.nmin_lo))
    )
    return SearchParams(
        r_major=r_major,
        k_inter=k_inter,
        k_minor=k_minor,
        n_min=n_min,
        n_max=n_max,
    )
