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


def _median_nn_dist(xyz: np.ndarray, *, max_n: int = 1500, seed: int = 42) -> float:
    """Median nearest-neighbour distance (subsample for large clouds)."""
    n = int(len(xyz))
    if n < 2:
        return 0.0
    if n > max_n:
        rng = np.random.default_rng(seed)
        pts = xyz[rng.choice(n, size=max_n, replace=False)]
    else:
        pts = xyz
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(pts).query(pts, k=2)
        return float(np.median(dist[:, 1]))
    except Exception:
        # Tiny fallback without scipy: mean of a few brute-force NNs
        m = min(len(pts), 64)
        dmin = []
        for i in range(m):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf
            dmin.append(float(np.min(d)))
        return float(np.median(dmin)) if dmin else 0.0


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
        for alt in ("Fe", "Grade", "value", "V", "z_value"):
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

    # Spatial dimensionality.
    # PCA aspect ratio alone is NOT enough: drillhole clouds are often
    # "pancake" (XY ≫ Z) yet fully 3D for variography/MOI. Hard degeneracy
    # only when the cloud is literally a line or thinner than sampling scale.
    c = xyz - xyz.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(xyz) - 1, 1)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
    evals = np.maximum(evals, 0.0)
    total = float(evals.sum()) + 1e-18
    fracs = evals / total
    spans = xyz.max(axis=0) - xyz.min(axis=0)
    min_span = float(np.min(spans))
    max_span = float(np.max(spans))
    median_nn = _median_nn_dist(xyz)

    lin_thr = float(_cfg(cfg, "degeneracy_linear_frac", default=0.02))
    # Absolute coplanar: thickness ≪ typical sample spacing (or ≪ bbox).
    nn_mult = float(_cfg(cfg, "degeneracy_coplanar_nn_mult", default=2.0))
    rel_span = float(_cfg(cfg, "degeneracy_coplanar_rel_span", default=1.0e-4))
    # Soft PCA pancake flag (diagnostic only; never blocks MOI).
    # Accept legacy key degeneracy_planar_frac as the aspect threshold.
    aspect_thr = float(
        _cfg(
            cfg,
            "degeneracy_aspect_pancake_frac",
            default=_cfg(cfg, "degeneracy_planar_frac", default=0.05),
        )
    )

    spatial_dim = 3
    degeneracy = None
    aspect_pancake = bool(fracs[0] + fracs[1] > 1.0 - aspect_thr)
    coplanar_thr = max(nn_mult * median_nn, rel_span * max_span, 1e-12)

    if fracs[0] > 1.0 - lin_thr:
        spatial_dim = 1
        degeneracy = "linear"
    elif min_span <= coplanar_thr:
        spatial_dim = 2
        degeneracy = "planar"
    # else: keep spatial_dim=3 even if aspect_pancake — MOI may proceed

    diag = {
        "n_raw": n_raw,
        "n_points": int(len(xyz)),
        "n_dropped_nonfinite": n_dropped,
        "n_duplicate_coords_merged": n_dup,
        "value_variance": var,
        "value_mean": float(np.mean(values)),
        "pca_eigenvalue_fracs": fracs.tolist(),
        "bbox_spans": spans.tolist(),
        "min_span": min_span,
        "max_span": max_span,
        "median_nn_dist": median_nn,
        "coplanar_thickness_thr": coplanar_thr,
        "aspect_pancake": aspect_pancake,
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
