"""Load and validate X,Y,Z,Value tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidatedPoints:
    xyz: np.ndarray  # (n, 3)
    values: np.ndarray  # (n,)
    n_raw: int
    n_dropped: int
    diagnostics: dict[str, Any]


class DataValidationError(ValueError):
    """Raised when points cannot support 3D anisotropy estimation."""


def _cfg(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_points(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ("X", "Y", "Z"):
        if want in df.columns:
            continue
        if want.lower() in cols:
            rename[cols[want.lower()]] = want
    if "Value" not in df.columns:
        for alt in ("Grade", "value", "V", "z_value"):
            if alt in df.columns:
                rename[alt] = "Value"
                break
            if alt.lower() in cols:
                rename[cols[alt.lower()]] = "Value"
                break
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in ("X", "Y", "Z", "Value") if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}; have {list(df.columns)}")
    return df[["X", "Y", "Z", "Value"]].copy()


def validate_points(df: pd.DataFrame, cfg: dict[str, Any]) -> ValidatedPoints:
    n_raw = len(df)
    arr = df[["X", "Y", "Z", "Value"]].to_numpy(dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    n_dropped = int(n_raw - len(arr))

    n_min = int(_cfg(cfg, "n_min_points", default=30))
    if len(arr) < n_min:
        raise DataValidationError(f"n_points={len(arr)} < n_min_points={n_min}")

    xyz = arr[:, :3]
    values = arr[:, 3]
    var = float(np.var(values))
    eps = float(_cfg(cfg, "near_zero_var_eps", default=1e-12))
    if var < eps:
        raise DataValidationError(f"Value variance {var} ~ 0")

    nd = int(_cfg(cfg, "coord_dup_decimals", default=6))
    keys = np.round(xyz, nd)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    n_dup = int(len(xyz) - len(uniq_idx))
    if n_dup > 0:
        from collections import defaultdict

        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, row in enumerate(keys):
            buckets[tuple(row)].append(i)
        keep_xyz = []
        keep_z = []
        for idxs in buckets.values():
            keep_xyz.append(xyz[idxs[0]])
            keep_z.append(float(np.mean(values[idxs])))
        xyz = np.asarray(keep_xyz, dtype=float)
        values = np.asarray(keep_z, dtype=float)
        if len(xyz) < n_min:
            raise DataValidationError(
                f"after dedup n_points={len(xyz)} < n_min_points={n_min}"
            )

    # spatial dimensionality via PCA of coordinates
    c = xyz - xyz.mean(axis=0, keepdims=True)
    # covariance
    cov = (c.T @ c) / max(len(xyz) - 1, 1)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
    evals = np.maximum(evals, 0.0)
    total = float(evals.sum()) + 1e-18
    fracs = evals / total
    lin_thr = float(_cfg(cfg, "degeneracy_linear_frac", default=0.02))
    pln_thr = float(_cfg(cfg, "degeneracy_planar_frac", default=0.05))
    spatial_dim = 3
    degeneracy = None
    if fracs[0] > 1.0 - lin_thr:
        spatial_dim = 1
        degeneracy = "linear"
    elif fracs[0] + fracs[1] > 1.0 - pln_thr:
        spatial_dim = 2
        degeneracy = "planar"

    diag = {
        "n_raw": n_raw,
        "n_points": int(len(xyz)),
        "n_dropped_nonfinite": n_dropped,
        "n_duplicate_coords_merged": n_dup,
        "value_variance": var,
        "value_mean": float(np.mean(values)),
        "pca_eigenvalue_fracs": fracs.tolist(),
        "spatial_dim": spatial_dim,
        "spatial_degeneracy": degeneracy,
        "bbox": {
            "xmin": float(xyz[:, 0].min()),
            "xmax": float(xyz[:, 0].max()),
            "ymin": float(xyz[:, 1].min()),
            "ymax": float(xyz[:, 1].max()),
            "zmin": float(xyz[:, 2].min()),
            "zmax": float(xyz[:, 2].max()),
        },
    }
    return ValidatedPoints(
        xyz=xyz,
        values=values,
        n_raw=n_raw,
        n_dropped=n_dropped + n_dup,
        diagnostics=diag,
    )
