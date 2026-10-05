"""SPD continuity ellipsoid fit from directional ranges."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import DirectionalEstimate


def _cfg(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def stabilize_axis(v: np.ndarray) -> np.ndarray:
    """Flip undirected axis for stable reporting (prefer +z, else first >0)."""
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-18)
    if v[2] < -1e-12:
        v = -v
    elif abs(v[2]) <= 1e-12:
        for k in range(3):
            if abs(v[k]) > 1e-12:
                if v[k] < 0:
                    v = -v
                break
    return v


def eigen_ranges_axes(
    A: np.ndarray, *, lambda_min: float = 1e-18
) -> tuple[np.ndarray, np.ndarray]:
    """Return ranges (3,) descending and orientation_matrix columns maj,int,min."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, float(lambda_min))
    ranges = 1.0 / np.sqrt(w)
    # eigh returns ascending eigenvalues → ascending ranges; reverse for maj>=...
    order = np.argsort(ranges)[::-1]
    ranges = ranges[order]
    axes = V[:, order]
    for j in range(3):
        axes[:, j] = stabilize_axis(axes[:, j])
    # right-handed: minor = major × intermediate
    cross = np.cross(axes[:, 0], axes[:, 1])
    if np.dot(cross, axes[:, 2]) < 0:
        axes[:, 2] = -axes[:, 2]
    axes[:, 2] = stabilize_axis(axes[:, 2])
    # re-orthonormalize minor via cross for exact RH
    axes[:, 2] = stabilize_axis(np.cross(axes[:, 0], axes[:, 1]))
    return ranges, axes


def fit_ellipsoid(
    estimates: list[DirectionalEstimate],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit SPD A via weighted linear LS on 1/r(u)^2 = uᵀ A u.

    Directions with ``range_boundary_hit`` are excluded (unreliable range).
    """
    valid = [
        e
        for e in estimates
        if e.valid
        and np.isfinite(e.range_)
        and e.range_ > 0
        and not e.range_boundary_hit
    ]
    n_boundary_excluded = sum(
        1 for e in estimates if e.valid and e.range_boundary_hit
    )
    if len(valid) < 6:
        return {
            "ok": False,
            "reason": f"too_few_valid_directions ({len(valid)}; "
            f"boundary_excluded={n_boundary_excluded})",
            "n_valid": len(valid),
            "n_boundary_excluded": n_boundary_excluded,
        }

    p = float(_cfg(cfg, "ellipsoid", "weight_n_pairs_pow", default=1.0))
    q = float(_cfg(cfg, "ellipsoid", "weight_n_lags_pow", default=0.5))
    eps = float(_cfg(cfg, "ellipsoid", "weight_sse_eps", default=1e-6))
    lam_min = float(_cfg(cfg, "ellipsoid", "lambda_min", default=1e-18))
    max_cond = float(_cfg(cfg, "ellipsoid", "max_condition", default=1e10))
    neg_tol = float(_cfg(cfg, "ellipsoid", "neg_eig_rel_tol", default=1e-4))

    U = np.stack([e.u for e in valid], axis=0)
    r = np.array([e.range_ for e in valid], dtype=float)
    target = 1.0 / (r**2)
    w = np.array(
        [
            (max(e.n_pairs, 1) ** p)
            * (max(e.n_valid_lags, 1) ** q)
            / (float(e.sse) + eps)
            for e in valid
        ],
        dtype=float,
    )
    w = w / (np.mean(w) + 1e-18)

    ux, uy, uz = U[:, 0], U[:, 1], U[:, 2]
    # 1/r² = a11 ux² + a22 uy² + a33 uz² + 2 a12 ux uy + 2 a13 ux uz + 2 a23 uy uz
    X = np.column_stack(
        [ux * ux, uy * uy, uz * uz, 2.0 * ux * uy, 2.0 * ux * uz, 2.0 * uy * uz]
    )
    sw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sw[:, None]
    yw = target * sw

    s = np.linalg.svd(Xw, compute_uv=False)
    s = np.asarray(s, dtype=float)
    smax = float(s[0]) if s.size else 0.0
    rank_tol = max(smax * 1e-10, 1e-14)
    rank = int(np.sum(s > rank_tol))
    cond = float(smax / (float(s[-1]) + 1e-18)) if s.size >= 6 else float("inf")
    if rank < 6 or (not np.isfinite(cond)) or cond > max_cond:
        return {
            "ok": False,
            "reason": (
                f"ellipsoid_unidentifiable (rank={rank}<6 or cond={cond:.3g}>{max_cond:g})"
            ),
            "n_valid": len(valid),
            "n_boundary_excluded": n_boundary_excluded,
            "design_rank": rank,
            "design_condition": cond,
        }

    theta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    a11, a22, a33, a12, a13, a23 = map(float, theta)
    A = np.array(
        [[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]],
        dtype=float,
    )
    A = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A)
    emax = float(np.max(np.abs(eigvals))) + 1e-18
    if float(np.min(eigvals)) < -neg_tol * emax:
        return {
            "ok": False,
            "reason": (
                f"ellipsoid_not_SPD (min_eig={float(np.min(eigvals)):.3g}, "
                f"emax={emax:.3g})"
            ),
            "n_valid": len(valid),
            "n_boundary_excluded": n_boundary_excluded,
            "design_rank": rank,
            "design_condition": cond,
            "eigenvalues": eigvals.tolist(),
        }

    # minimal SPD projection: clip tiny noise negatives (and zeros)
    eig_clipped = np.maximum(eigvals, lam_min)
    A = (eigvecs * eig_clipped) @ eigvecs.T

    pred = np.einsum("ij,jk,ik->i", U, A, U)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    rel = rmse / (float(np.sqrt(np.mean(target**2))) + 1e-18)
    ranges, axes = eigen_ranges_axes(A, lambda_min=lam_min)
    obj = float(np.sum(w * (pred - target) ** 2))

    return {
        "ok": True,
        "A": A,
        "ranges": ranges,
        "axes": axes,  # columns maj, int, min
        "orientation_matrix": axes.copy(),
        "rmse_inv_r2": rmse,
        "rmse_inv_r2_rel": rel,
        "n_valid": len(valid),
        "n_boundary_excluded": n_boundary_excluded,
        "obj": obj,
        "design_rank": rank,
        "design_condition": cond,
        "success": True,
    }


def ellipsoid_range(u: np.ndarray, A: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-18)
    q = float(u @ A @ u)
    return float(1.0 / np.sqrt(max(q, 1e-18)))
