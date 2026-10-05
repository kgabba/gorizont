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
class SearchParams:
    """Hard search ellipsoid + neighbour count bounds."""

    r_major: float
    k_inter: float
    k_minor: float
    n_min: int
    n_max: int

    @property
    def r_inter(self) -> float:
        return float(self.r_major) * float(self.k_inter)

    @property
    def r_minor(self) -> float:
        return float(self.r_major) * float(self.k_minor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_major": float(self.r_major),
            "R_inter": float(self.r_inter),
            "R_minor": float(self.r_minor),
            "K_inter": float(self.k_inter),
            "K_minor": float(self.k_minor),
            "Nmin": int(self.n_min),
            "Nmax": int(self.n_max),
        }


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_major": [self.r_major_min, self.r_major_max],
            "K_inter": [self.k_min, 1.0],
            "K_minor": [self.k_min, 1.0],
            "Nmin": [self.nmin_lo, self.nmin_hi],
            "Nmax": [self.nmax_lo, self.nmax_hi],
        }


@dataclass
class ObjectiveConfig:
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
    """Pooled RMSE + coverage penalty (always finite)."""
    if n_valid == 0:
        return float(cfg.empty_rmse_fallback + cfg.coverage_penalty)
    shortfall = max(0.0, float(cfg.coverage_min) - float(prediction_coverage))
    return float(cv_rmse + cfg.coverage_penalty * shortfall)
