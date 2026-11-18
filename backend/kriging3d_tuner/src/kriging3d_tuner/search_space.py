"""Optuna search space for hard search neighbourhood (conditional sectors)."""

from __future__ import annotations

from typing import Any

import optuna

from .models import (
    SearchParams,
    SearchSpaceConfig,
    VariogramFixed,
    sector_neighbour_cap,
)

IDW_SECTOR_CHOICES = (4, 8, 16)


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
        nmax_lo=int(scfg.get("nmax_lo", 4)),
        nmax_hi=int(scfg.get("nmax_hi", 64)),
        minoct_lo=int(scfg.get("minoct_lo", 1)),
        minoct_hi=int(scfg.get("minoct_hi", 4)),
        minperoc_lo=int(scfg.get("minperoc_lo", 1)),
        minperoc_hi=int(scfg.get("minperoc_hi", 3)),
        maxperoc_lo=int(scfg.get("maxperoc_lo", 1)),
        maxperoc_hi=int(scfg.get("maxperoc_hi", 4)),
    )


def parse_force_octant_search(cfg: dict[str, Any] | None) -> bool | None:
    """Return forced octant/sector mode, or ``None`` if Optuna should choose.

    Reads ``octants.force_octant_search`` (bool / null). Absent → auto.
    """
    raw = (cfg or {}).get("octants") or {}
    if "force_octant_search" not in raw:
        return None
    val = raw.get("force_octant_search")
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "null", "none", "auto"):
            return None
        return s in ("1", "true", "yes", "on")
    return bool(val)


def sector_implied_nmin_nmax(
    *,
    n_sectors: int,
    minoct: int,
    minperoc: int,
    maxperoc: int,
) -> tuple[int, int]:
    """Derive global ``Nmin`` / ``Nmax`` from sector limits (not Optuna params).

    ``Nmin = MINOCT × MINPEROC``, ``Nmax = n_sectors × MAXPEROC``.
    """
    n_min = max(1, int(minoct) * int(minperoc))
    n_max = max(n_min, sector_neighbour_cap(n_sectors, maxperoc))
    return n_min, n_max


def suggest_search_params(
    trial: optuna.Trial,
    space: SearchSpaceConfig,
    *,
    force_octant_search: bool | None = None,
    allow_variable_sectors: bool = False,
) -> SearchParams:
    """Conditional Optuna sampling.

    * ``octant_search=False``: ``R_major``, ``K_*``, ``Nmin``, ``Nmax``
      with ``Nmin ≤ Nmax``.
    * ``octant_search=True``: ``R_major``, ``K_*``, ``MINOCT``, ``MINPEROC``,
      ``MAXPEROC`` (and ``n_sectors`` for IDW). Global ``Nmin`` / ``Nmax`` are
      **not** suggested — they are implied as
      ``MINOCT×MINPEROC`` / ``n_sectors×MAXPEROC``.

    When ``allow_variable_sectors`` is True (IDW), also suggests
    ``n_sectors ∈ {4, 8, 16}`` and sets Micromine ``discard_underfilled``.
    Otherwise (OK) ``n_sectors`` is fixed at 8 (Datamine octants).

    If ``force_octant_search`` is set, that branch is fixed for the study
    (categorical with a single choice).
    """
    r_major = float(
        trial.suggest_float("R_major", space.r_major_min, space.r_major_max)
    )
    k_inter = float(trial.suggest_float("K_inter", space.k_min, 1.0))
    k_minor = float(trial.suggest_float("K_minor", space.k_min, k_inter))

    if force_octant_search is None:
        octant_search = bool(
            trial.suggest_categorical("octant_search", [False, True])
        )
    else:
        octant_search = bool(
            trial.suggest_categorical("octant_search", [bool(force_octant_search)])
        )

    if not octant_search:
        n_max = int(trial.suggest_int("Nmax", space.nmax_lo, space.nmax_hi))
        nmin_hi_eff = min(space.nmin_hi, n_max)
        n_min = int(
            trial.suggest_int(
                "Nmin", space.nmin_lo, max(nmin_hi_eff, space.nmin_lo)
            )
        )
        return SearchParams(
            r_major=r_major,
            k_inter=k_inter,
            k_minor=k_minor,
            n_min=n_min,
            n_max=n_max,
            octant_search=False,
        )

    if allow_variable_sectors:
        n_sectors = int(
            trial.suggest_categorical("n_sectors", list(IDW_SECTOR_CHOICES))
        )
        discard_underfilled = True
    else:
        n_sectors = 8
        discard_underfilled = False

    # --- sector branch: MAXPEROC → MINPEROC → MINOCT; Nmin/Nmax implied ---
    maxperoc_lo = max(1, space.maxperoc_lo)
    maxperoc_hi = max(maxperoc_lo, space.maxperoc_hi)
    maxperoc = int(trial.suggest_int("MAXPEROC", maxperoc_lo, maxperoc_hi))

    minperoc_hi_eff = min(space.minperoc_hi, maxperoc)
    minperoc_lo = min(max(1, space.minperoc_lo), minperoc_hi_eff)
    minperoc = int(
        trial.suggest_int(
            "MINPEROC", minperoc_lo, max(minperoc_lo, minperoc_hi_eff)
        )
    )

    minoct_lo = max(1, space.minoct_lo)
    minoct_hi = min(n_sectors, max(minoct_lo, space.minoct_hi))
    minoct = int(trial.suggest_int("MINOCT", minoct_lo, max(minoct_lo, minoct_hi)))

    n_min, n_max = sector_implied_nmin_nmax(
        n_sectors=n_sectors,
        minoct=minoct,
        minperoc=minperoc,
        maxperoc=maxperoc,
    )

    return SearchParams(
        r_major=r_major,
        k_inter=k_inter,
        k_minor=k_minor,
        n_min=n_min,
        n_max=n_max,
        octant_search=True,
        minoct=minoct,
        minperoc=minperoc,
        maxperoc=maxperoc,
        n_sectors=n_sectors,
        discard_underfilled=discard_underfilled,
    )


def search_params_from_trial_params(params: dict[str, Any]) -> SearchParams:
    """Rebuild ``SearchParams`` from a frozen Optuna trial ``params`` dict."""
    r_major = float(params["R_major"])
    k_inter = float(params["K_inter"])
    k_minor = float(params["K_minor"])
    octant_search = bool(params.get("octant_search", False))
    if octant_search:
        n_sectors = int(params.get("n_sectors", 8))
        minoct = int(params["MINOCT"])
        minperoc = int(params["MINPEROC"])
        maxperoc = int(params["MAXPEROC"])
        # New trials omit Nmin/Nmax; old trials may still carry them.
        if "Nmin" in params and "Nmax" in params:
            n_min = int(params["Nmin"])
            n_max = int(params["Nmax"])
        else:
            n_min, n_max = sector_implied_nmin_nmax(
                n_sectors=n_sectors,
                minoct=minoct,
                minperoc=minperoc,
                maxperoc=maxperoc,
            )
        # Micromine discard when variable sectors were suggested (IDW).
        discard = "n_sectors" in params
        return SearchParams(
            r_major=r_major,
            k_inter=k_inter,
            k_minor=k_minor,
            n_min=n_min,
            n_max=n_max,
            octant_search=True,
            minoct=minoct,
            minperoc=minperoc,
            maxperoc=maxperoc,
            n_sectors=n_sectors,
            discard_underfilled=discard,
        )
    n_min = int(params["Nmin"])
    n_max = int(params["Nmax"])
    return SearchParams(
        r_major=r_major,
        k_inter=k_inter,
        k_minor=k_minor,
        n_min=n_min,
        n_max=n_max,
        octant_search=False,
    )
