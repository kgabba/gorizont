"""Data loading, fold construction, pooled OOF evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .idw import anisotropic_idw_at_points
from .kriging import ordinary_kriging_at_points, select_neighbour_indices
from .models import (
    CVFold,
    CVPlan,
    PointCloud,
    SearchParams,
    TrialMetrics,
    VariogramFixed,
    compute_objective,
    ObjectiveConfig,
)

METHOD_ORDINARY_KRIGING = "ordinary_kriging"
METHOD_ANISOTROPIC_IDW = "anisotropic_idw"


def _predict_at_points(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    vg: VariogramFixed,
    search: SearchParams,
    *,
    method: str = METHOD_ORDINARY_KRIGING,
    power: float | None = None,
) -> np.ndarray:
    method_eff = (method or METHOD_ORDINARY_KRIGING).strip().lower()
    if method_eff == METHOD_ANISOTROPIC_IDW:
        if power is None:
            raise ValueError("power is required for anisotropic_idw")
        return anisotropic_idw_at_points(
            known_xyz, known_z, pred_xyz, vg, search, float(power)
        )
    if method_eff != METHOD_ORDINARY_KRIGING:
        raise ValueError(
            f"unknown method {method!r}; expected "
            f"{METHOD_ORDINARY_KRIGING!r} or {METHOD_ANISOTROPIC_IDW!r}"
        )
    return ordinary_kriging_at_points(known_xyz, known_z, pred_xyz, vg, search)


class DataError(ValueError):
    pass


_REQUIRED_VG = (
    "variogram_model",
    "nugget",
    "sill",
    "range_major",
    "range_intermediate",
    "range_minor",
    "orientation_matrix",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_points_csv(path: str | Path, cfg: dict[str, Any]) -> PointCloud:
    df = pd.read_csv(path)
    for col in ("X", "Y", "Z", "Value"):
        if col not in df.columns:
            raise DataError(f"missing required column {col!r}")
    m = np.isfinite(df[["X", "Y", "Z", "Value"]].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[m].reset_index(drop=True)
    n_min = int(cfg.get("n_min_points", 30))
    if len(df) < n_min:
        raise DataError(f"too few points after filter: {len(df)} < {n_min}")
    hole_ids = None
    if "HoleID" in df.columns:
        hole_ids = df["HoleID"].to_numpy()
    return PointCloud(
        xyz=df[["X", "Y", "Z"]].to_numpy(dtype=float),
        values=df["Value"].to_numpy(dtype=float),
        hole_ids=hole_ids,
    )


def _extract_vg_dict(data: dict[str, Any]) -> dict[str, Any]:
    cand = data.get("aniso_candidate")
    if isinstance(cand, dict) and all(k in cand for k in _REQUIRED_VG):
        return cand
    if all(k in data for k in _REQUIRED_VG):
        return data
    raise DataError(
        "anisotropy JSON must contain aniso_candidate or top-level variogram fields"
    )


def _linked_sort_ranges_axes(
    ranges: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ranges)[::-1]
    return ranges[order], Q[:, order]


def load_variogram_json(path: str | Path) -> VariogramFixed:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    src = _extract_vg_dict(data)
    model = str(src["variogram_model"]).strip().lower()
    if model != "spherical":
        raise DataError(f"v0 supports variogram_model=spherical only, got {model!r}")
    Q = np.asarray(src["orientation_matrix"], dtype=float)
    if Q.shape != (3, 3):
        raise DataError(f"orientation_matrix must be 3x3, got {Q.shape}")
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1.0
    ranges = np.array(
        [
            float(src["range_major"]),
            float(src["range_intermediate"]),
            float(src["range_minor"]),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise DataError("variogram ranges must be finite and positive")
    ranges, Q = _linked_sort_ranges_axes(ranges, Q)
    nug = float(src["nugget"])
    sil = float(src["sill"])
    if nug < 0 or sil <= 0:
        raise DataError("nugget >= 0 and sill > 0 required")
    return VariogramFixed(
        variogram_model="spherical",
        nugget=nug,
        sill=sil,
        range_major=float(ranges[0]),
        range_intermediate=float(ranges[1]),
        range_minor=float(ranges[2]),
        orientation_matrix=Q,
    )


def build_folds(cloud: PointCloud, cfg: dict[str, Any]) -> CVPlan:
    ccfg = cfg.get("cv") or {}
    if cloud.hole_ids is not None:
        groups = cloud.hole_ids
        n_holes = len(np.unique(groups))
        if n_holes < 2:
            raise DataError("need at least 2 unique HoleID for group CV")
        n_req = int(ccfg.get("n_splits", 5))
        n_splits = min(n_req, n_holes)
        gkf = GroupKFold(n_splits=n_splits)
        folds = [
            CVFold(train_idx=tr, test_idx=te)
            for tr, te in gkf.split(cloud.xyz, groups=groups)
        ]
        return CVPlan(
            method="group_kfold",
            folds=folds,
            n_splits_requested=n_req,
        )

    nx = int(ccfg.get("grid_nx", 3))
    ny = int(ccfg.get("grid_ny", 3))
    nz = int(ccfg.get("grid_nz", 2))
    xyz = cloud.xyz
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    rel = (xyz - lo) / span
    ix = np.clip((rel[:, 0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((rel[:, 1] * ny).astype(int), 0, ny - 1)
    iz = np.clip((rel[:, 2] * nz).astype(int), 0, nz - 1)
    cell = ix + nx * (iy + ny * iz)
    n_cells = nx * ny * nz
    folds: list[CVFold] = []
    for c in range(n_cells):
        test_idx = np.where(cell == c)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(cell != c)[0]
        folds.append(CVFold(train_idx=train_idx, test_idx=test_idx))
    if len(folds) < 2:
        raise DataError("spatial block CV: fewer than 2 non-empty blocks")
    return CVPlan(
        method="spatial_block_3d",
        folds=folds,
        n_splits_requested=n_cells,
    )


def evaluate_oof(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    search: SearchParams,
    obj_cfg: ObjectiveConfig,
    *,
    method: str = METHOD_ORDINARY_KRIGING,
    power: float | None = None,
) -> TrialMetrics:
    """Pooled OOF RMSE/MAE over all fold test points."""
    preds: list[float] = []
    obs: list[float] = []
    n_tgt = 0
    n_valid = 0

    for fold in plan.folds:
        tr = fold.train_idx
        te = fold.test_idx
        if te.size == 0:
            continue
        n_tgt += int(te.size)
        pred = _predict_at_points(
            cloud.xyz[tr],
            cloud.values[tr],
            cloud.xyz[te],
            vg,
            search,
            method=method,
            power=power,
        )
        z = cloud.values[te]
        for p, y in zip(pred, z):
            if np.isfinite(p):
                preds.append(float(p))
                obs.append(float(y))
                n_valid += 1

    n_invalid = n_tgt - n_valid
    if n_valid == 0:
        cv_rmse = float(obj_cfg.empty_rmse_fallback)
        cv_mae = float(obj_cfg.empty_rmse_fallback)
    else:
        err = np.asarray(preds) - np.asarray(obs)
        cv_rmse = float(np.sqrt(np.mean(err**2)))
        cv_mae = float(np.mean(np.abs(err)))
    coverage = float(n_valid / n_tgt) if n_tgt > 0 else 0.0
    objective = compute_objective(
        cv_rmse=cv_rmse,
        prediction_coverage=coverage,
        n_valid=n_valid,
        cfg=obj_cfg,
    )
    return TrialMetrics(
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        prediction_coverage=coverage,
        n_tgt=n_tgt,
        n_valid=n_valid,
        n_invalid=n_invalid,
        objective=objective,
    )


def evaluate_oof_masked(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    search: SearchParams,
    candidate_global: np.ndarray,
    *,
    method: str = METHOD_ORDINARY_KRIGING,
    power: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict only on OOF test points that appear in ``candidate_global``.

    Returns ``(indices, preds, n_used)`` aligned arrays for every candidate
    that was a fold test point (preds may be NaN). Does not touch non-candidates.
    """
    cand = set(int(i) for i in np.asarray(candidate_global, dtype=int).ravel())
    Q = vg.orientation_matrix
    indices: list[int] = []
    preds: list[float] = []
    n_used: list[int] = []

    for fold in plan.folds:
        tr = fold.train_idx
        te = fold.test_idx
        if te.size == 0:
            continue
        # Which test points are still candidates?
        te_list = [int(g) for g in te.tolist() if int(g) in cand]
        if not te_list:
            continue
        te_arr = np.asarray(te_list, dtype=int)
        known_xyz = cloud.xyz[tr]
        known_z = cloud.values[tr]
        pred = _predict_at_points(
            known_xyz,
            known_z,
            cloud.xyz[te_arr],
            vg,
            search,
            method=method,
            power=power,
        )
        for j, gi in enumerate(te_list):
            delta = known_xyz - cloud.xyz[gi]
            nb = select_neighbour_indices(delta, Q, search)
            indices.append(gi)
            preds.append(float(pred[j]) if np.isfinite(pred[j]) else float("nan"))
            n_used.append(int(nb.size))

    return (
        np.asarray(indices, dtype=int),
        np.asarray(preds, dtype=float),
        np.asarray(n_used, dtype=int),
    )


def oof_target_indices(plan: CVPlan) -> np.ndarray:
    """All global indices that appear as fold test points (pooled OOF set)."""
    parts: list[np.ndarray] = []
    for fold in plan.folds:
        if fold.test_idx.size:
            parts.append(fold.test_idx.astype(int, copy=False))
    if not parts:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(parts))
