"""Dataclasses for kriging3d_tuner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VariogramFixed:
    """Frozen variogram ellipsoid from anisotropy3d."""

    variogram_model: str
    nugget: float
    sill: float  # partial sill
    range_major: float
    range_intermediate: float
    range_minor: float
    orientation_matrix: np.ndarray  # columns maj, int, min

    def as_dict(self) -> dict[str, Any]:
        return {
            "variogram_model": self.variogram_model,
            "nugget": float(self.nugget),
            "sill": float(self.sill),
            "range_major": float(self.range_major),
            "range_intermediate": float(self.range_intermediate),
            "range_minor": float(self.range_minor),
            "orientation_matrix": np.asarray(self.orientation_matrix, dtype=float).tolist(),
        }


@dataclass(frozen=True)
class OctantSettings:
    """Octant controls for constructing ``SearchParams`` manually.

    Optuna normally suggests these via ``suggest_search_params``; this helper
    remains for fixed A/B experiments and tests.
    """

    enabled: bool = False
    minoct: int = 1
    minperoc: int = 1
    maxperoc: int = 10**9

    def as_dict(self) -> dict[str, Any]:
        return {
            "octant_search": bool(self.enabled),
            "MINOCT": int(self.minoct),
            "MINPEROC": int(self.minperoc),
            "MAXPEROC": int(self.maxperoc),
        }


def octant_settings_from_cfg(cfg: dict[str, Any] | None) -> OctantSettings:
    """Parse ``octants:`` section (defaults / legacy fixed overrides)."""
    raw = (cfg or {}).get("octants") or {}
    enabled = raw.get("octant_search", raw.get("enabled", False))
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
    return OctantSettings(
        enabled=bool(enabled),
        minoct=int(raw.get("MINOCT", raw.get("minoct", 1))),
        minperoc=int(raw.get("MINPEROC", raw.get("minperoc", 1))),
        maxperoc=int(raw.get("MAXPEROC", raw.get("maxperoc", 10**9))),
    )


def sector_neighbour_cap(n_sectors: int, maxperoc: int) -> int:
    """Natural upper bound when sector search is on: ``n_sectors × MAXPEROC``."""
    return int(n_sectors) * int(maxperoc)


def octant_neighbour_cap(maxperoc: int) -> int:
    """Natural upper bound on selected samples when octants are on: ``8 × MAXPEROC``."""
    return sector_neighbour_cap(8, maxperoc)


@dataclass(frozen=True)
class SearchParams:
    """Hard search ellipsoid + neighbour count bounds (+ optional sectors)."""

    r_major: float
    k_inter: float
    k_minor: float
    n_min: int
    n_max: int
    octant_search: bool = False
    minoct: int = 1
    minperoc: int = 1
    maxperoc: int = 10**9
    n_sectors: int = 8
    discard_underfilled: bool = False

    @property
    def r_inter(self) -> float:
        return float(self.r_major) * float(self.k_inter)

    @property
    def r_minor(self) -> float:
        return float(self.r_major) * float(self.k_minor)

    def with_octants(self, octants: OctantSettings) -> "SearchParams":
        return SearchParams(
            r_major=self.r_major,
            k_inter=self.k_inter,
            k_minor=self.k_minor,
            n_min=self.n_min,
            n_max=self.n_max,
            octant_search=bool(octants.enabled),
            minoct=int(octants.minoct),
            minperoc=int(octants.minperoc),
            maxperoc=int(octants.maxperoc),
            n_sectors=8,
            discard_underfilled=False,
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "R_major": float(self.r_major),
            "R_inter": float(self.r_inter),
            "R_minor": float(self.r_minor),
            "K_inter": float(self.k_inter),
            "K_minor": float(self.k_minor),
            "Nmin": int(self.n_min),
            "Nmax": int(self.n_max),
            "octant_search": bool(self.octant_search),
            "sector_search": bool(self.octant_search),
        }
        if self.octant_search:
            out["n_sectors"] = int(self.n_sectors)
            out["MINOCT"] = int(self.minoct)
            out["MINPEROC"] = int(self.minperoc)
            out["MAXPEROC"] = int(self.maxperoc)
            out["discard_underfilled"] = bool(self.discard_underfilled)
            out["neighbour_cap"] = sector_neighbour_cap(
                self.n_sectors, self.maxperoc
            )
        return out


@dataclass
class PointCloud:
    """Validated sample cloud."""

    xyz: np.ndarray  # (n, 3)
    values: np.ndarray  # (n,)
    hole_ids: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class CVFold:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class CVPlan:
    method: str
    folds: list[CVFold]
    n_splits_requested: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)


@dataclass
class TrialMetrics:
    cv_rmse: float
    cv_mae: float
    prediction_coverage: float
    n_tgt: int
    n_valid: int
    n_invalid: int
    objective: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "CV_RMSE": float(self.cv_rmse),
            "CV_MAE": float(self.cv_mae),
            "prediction_coverage": float(self.prediction_coverage),
            "n_tgt": int(self.n_tgt),
            "n_valid": int(self.n_valid),
            "n_invalid": int(self.n_invalid),
            "objective": float(self.objective),
        }


@dataclass
class SearchSpaceConfig:
    r_major_min: float
    r_major_max: float
    k_min: float
    nmin_lo: int
    nmin_hi: int
    nmax_lo: int
    nmax_hi: int
    minoct_lo: int = 1
    minoct_hi: int = 4
    minperoc_lo: int = 1
    minperoc_hi: int = 3
    maxperoc_lo: int = 1
    maxperoc_hi: int = 4

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_major": [self.r_major_min, self.r_major_max],
            "K_inter": [self.k_min, 1.0],
            "K_minor": [self.k_min, 1.0],
            "Nmin": [self.nmin_lo, self.nmin_hi],
            "Nmax": [self.nmax_lo, self.nmax_hi],
            "MINOCT": [self.minoct_lo, self.minoct_hi],
            "MINPEROC": [self.minperoc_lo, self.minperoc_hi],
            "MAXPEROC": [self.maxperoc_lo, self.maxperoc_hi],
            "octant_search": [False, True],
            "n_sectors": [4, 8, 16],
        }


@dataclass
class ObjectiveConfig:
    """Coverage-gated RMSE objective (see ``compute_objective``)."""

    coverage_min: float = 0.97
    coverage_penalty: float = 1.0e6
    empty_rmse_fallback: float = 1.0e6


def compute_objective(
    *,
    cv_rmse: float,
    prediction_coverage: float,
    n_valid: int,
    cfg: ObjectiveConfig,
) -> float:
    """Pooled OOF RMSE with an explicit coverage gate (always finite).

    Missing forecasts stay NaN and are **excluded** from RMSE (never filled
    with zeros). ``prediction_coverage = n_valid / n_tgt``.

    Penalty (minimize):

    * If ``n_valid == 0``:
      ``objective = empty_rmse_fallback + coverage_penalty``
    * Else:
      ``objective = CV_RMSE + coverage_penalty * max(0, coverage_min − coverage)``

    With the default ``coverage_penalty = 1e6`` and ``coverage_min = 0.97``,
    any configuration below the gate is ranked strictly worse than every
    configuration that meets it with ``CV_RMSE < coverage_penalty``
    (practically all real RMSE values). This blocks Optuna from “winning”
    by estimating only the easiest subset of points.
    """
    if n_valid == 0:
        return float(cfg.empty_rmse_fallback + cfg.coverage_penalty)
    shortfall = max(0.0, float(cfg.coverage_min) - float(prediction_coverage))
    return float(cv_rmse + cfg.coverage_penalty * shortfall)
