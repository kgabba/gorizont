"""CV scorers and trial parameter containers for 3D OK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import anisotropic_metric_distance, rotate_to_principal
from .kriging import ordinary_kriging_ellipsoidal, ordinary_kriging_isotropic


@dataclass(frozen=True)
class Theta:
    """Optuna-tunable OK neighbourhood / model knobs (az/dip fixed outside)."""

    r_major: float
    r_inter: float
    r_minor: float
    n_max: int
    nugget: float
    range_scale: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "R_major": float(self.r_major),
            "R_inter": float(self.r_inter),
            "R_minor": float(self.r_minor),
            "R": float(self.r_major),
            "N_max": int(self.n_max),
            "nugget": float(self.nugget),
            "range_scale": float(self.range_scale),
        }


@dataclass(frozen=True)
class ModelFixed:
    """Quantities fixed from MOI + directional fit for one domain."""

    azimuth_deg: float
    dip_deg: float
    a_major: float
    a_inter: float
    a_minor: float
    sill_partial_base: float
    sill_total: float
    nugget_fit: float
    isotropic: bool = False


@dataclass
class CVResult:
    rmse: float
    mae: float
    valid: bool
    n_nan_predictions: int
    details: dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _nugget_sill(theta: Theta, fixed: ModelFixed) -> tuple[float, float]:
    nug = float(np.clip(theta.nugget, 0.0, 0.95 * fixed.sill_total))
    sil = max(fixed.sill_total - nug, 1e-12)
    return nug, sil


def _ok_predict(
    xyz: np.ndarray,
    z: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    *,
    nug: float,
    sil: float,
) -> np.ndarray:
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        return np.full(test_idx.size, np.nan)
    if fixed.isotropic:
        return ordinary_kriging_isotropic(
            xyz[train_idx],
            z[train_idx],
            xyz[test_idx],
            range_=fixed.a_major,
            nugget=nug,
            sill=sil,
            range_scale=theta.range_scale,
            radius=theta.r_major,
            n_max=theta.n_max,
        )
    return ordinary_kriging_ellipsoidal(
        xyz[train_idx],
        z[train_idx],
        xyz[test_idx],
        azimuth_deg=fixed.azimuth_deg,
        dip_deg=fixed.dip_deg,
        a_major=fixed.a_major,
        a_inter=fixed.a_inter,
        a_minor=fixed.a_minor,
        nugget=nug,
        sill=sil,
        range_scale=theta.range_scale,
        r_major=theta.r_major,
        r_inter=theta.r_inter,
        r_minor=theta.r_minor,
        n_max=theta.n_max,
    )


def _score_holdout_folds(
    xyz: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    method: str,
    invalid_penalty: float,
    extra_details: dict | None = None,
) -> CVResult:
    xyz = np.asarray(xyz, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    nug, sil = _nugget_sill(theta, fixed)
    rmses: list[float] = []
    maes: list[float] = []
    total_nan = 0

    for train_idx, test_idx in folds:
        pred = _ok_predict(xyz, z, train_idx, test_idx, theta, fixed, nug=nug, sil=sil)
        n_nan = int(np.isnan(pred).sum())
        total_nan += n_nan
        if n_nan > 0:
            return CVResult(
                rmse=float(invalid_penalty),
                mae=float(invalid_penalty),
                valid=False,
                n_nan_predictions=total_nan,
                details={"method": method, "reason": "NaN predictions"},
            )
        rmses.append(rmse(z[test_idx], pred))
        maes.append(mae(z[test_idx], pred))

    details = {
        "method": method,
        "n_folds": len(folds),
        "rmse_std": float(np.std(rmses)) if rmses else float("nan"),
        "mae_std": float(np.std(maes)) if maes else float("nan"),
        "nugget_used": nug,
        "sill_partial_used": sil,
    }
    if extra_details:
        details.update(extra_details)
    return CVResult(
        rmse=float(np.mean(rmses)) if rmses else float(invalid_penalty),
        mae=float(np.mean(maes)) if maes else float(invalid_penalty),
        valid=bool(rmses),
        n_nan_predictions=0,
        details=details,
    )


def default_holdout_size(n_sample: int) -> int:
    return int(min(max(5, n_sample // 10), n_sample - 2))


def make_spatial_block_splits(
    xyz: np.ndarray,
    *,
    grid_nx: int = 3,
    grid_ny: int = 3,
    grid_nz: int = 2,
) -> list[np.ndarray]:
    """Leave-one-cell-out test indices on an axis-aligned 3D grid over bbox."""
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    grid_nx, grid_ny, grid_nz = int(grid_nx), int(grid_ny), int(grid_nz)
    if n == 0:
        return []
    xmin, ymin, zmin = xyz.min(axis=0)
    xmax, ymax, zmax = xyz.max(axis=0)
    eps = 1e-12
    dx = max((xmax - xmin) / grid_nx, eps)
    dy = max((ymax - ymin) / grid_ny, eps)
    dz = max((zmax - zmin) / grid_nz, eps)
    ix = np.clip(((xyz[:, 0] - xmin) / dx).astype(int), 0, grid_nx - 1)
    iy = np.clip(((xyz[:, 1] - ymin) / dy).astype(int), 0, grid_ny - 1)
    iz = np.clip(((xyz[:, 2] - zmin) / dz).astype(int), 0, grid_nz - 1)
    cell = ix + iy * grid_nx + iz * grid_nx * grid_ny
    folds: list[np.ndarray] = []
    for c in range(grid_nx * grid_ny * grid_nz):
        test = np.where(cell == c)[0]
        if test.size == 0 or test.size >= n:
            continue
        folds.append(test.astype(int))
    return folds


def spatial_block_score(
    xyz: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    test_folds: Sequence[np.ndarray],
    *,
    invalid_penalty: float = 1e6,
    grid_nx: int = 3,
    grid_ny: int = 3,
    grid_nz: int = 2,
) -> CVResult:
    n = len(np.asarray(z).reshape(-1))
    all_idx = np.arange(n)
    folds = []
    for test_idx in test_folds:
        test_idx = np.asarray(test_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]
        if train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return _score_holdout_folds(
        xyz,
        z,
        theta,
        fixed,
        folds,
        method="spatial_block",
        invalid_penalty=invalid_penalty,
        extra_details={
            "grid_nx": int(grid_nx),
            "grid_ny": int(grid_ny),
            "grid_nz": int(grid_nz),
        },
    )


def make_buffered_delete_d_folds(
    xyz: np.ndarray,
    *,
    holdout_size: int,
    n_repeats: int,
    buffer_radius: float,
    seed: int,
    azimuth_deg: float | None = None,
    dip_deg: float | None = None,
    a_major: float | None = None,
    a_inter: float | None = None,
    a_minor: float | None = None,
    anisotropic: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Delete-d holdouts; train excludes points within buffer of any holdout."""
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    if holdout_size < 1 or holdout_size >= n:
        raise ValueError("holdout_size must be in [1, n)")
    r = float(buffer_radius)
    rng = np.random.default_rng(seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    if anisotropic:
        if any(v is None for v in (azimuth_deg, dip_deg, a_major, a_inter, a_minor)):
            raise ValueError(
                "anisotropic buffered folds need az/dip and a_major/inter/minor"
            )
        xyz_m = rotate_to_principal(xyz, float(azimuth_deg), float(dip_deg))
        a_maj, a_int, a_min = float(a_major), float(a_inter), float(a_minor)
    else:
        xyz_m = xyz

    for _ in range(int(n_repeats)):
        hold_idx = rng.choice(n, size=int(holdout_size), replace=False)
        if anisotropic:
            du = xyz_m[:, None, :] - xyz_m[hold_idx][None, :, :]
            d_min = np.min(
                anisotropic_metric_distance(
                    du[:, :, 0], du[:, :, 1], du[:, :, 2], a_maj, a_int, a_min
                ),
                axis=1,
            )
            train_mask = d_min > r
        else:
            d2 = np.min(
                np.sum((xyz[:, None, :] - xyz[hold_idx][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            train_mask = d2 > (r * r)
        train_idx = np.where(train_mask)[0]
        if train_idx.size == 0:
            continue
        folds.append((train_idx, np.asarray(hold_idx, dtype=int)))
    return folds


def buffered_delete_d_score(
    xyz: np.ndarray,
    z: np.ndarray,
    theta: Theta,
    fixed: ModelFixed,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    buffer_radius: float,
    holdout_size: int,
    invalid_penalty: float = 1e6,
    buffer_metric: str = "euclidean",
    buffer_scale: float | None = None,
) -> CVResult:
    if not folds:
        return CVResult(
            rmse=float(invalid_penalty),
            mae=float(invalid_penalty),
            valid=False,
            n_nan_predictions=0,
            details={
                "method": "buffered_delete_d",
                "reason": "no valid folds",
                "buffer_radius": float(buffer_radius),
                "holdout_size": int(holdout_size),
                "buffer_metric": buffer_metric,
                "buffer_scale": buffer_scale,
            },
        )
    return _score_holdout_folds(
        xyz,
        z,
        theta,
        fixed,
        folds,
        method="buffered_delete_d",
        invalid_penalty=invalid_penalty,
        extra_details={
            "buffer_radius": float(buffer_radius),
            "holdout_size": int(holdout_size),
            "n_repeats_used": len(folds),
            "buffer_metric": buffer_metric,
            "buffer_scale": buffer_scale,
        },
    )
