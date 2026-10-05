"""Hemisphere direction set, cone filter, and geometric coverage check."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import DirectionalEstimate, PairCloud3D
from .variogram import experimental_directional, fit_spherical_weighted


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """Approximate equal-area unit vectors on the upper hemisphere (z >= 0).

    Returns shape (n, 3). Undirected axes: z folded to >= 0.
    """
    n = max(int(n), 1)
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = (i + 0.5) / n
    rad = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = phi * i
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    out = np.column_stack([x, y, z])
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-18
    return out


def directional_coverage_ok(
    estimates: list[DirectionalEstimate],
    cfg: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Check that usable directions span 3D (not a plane / narrow cone).

    Usable = valid fit, finite range > 0, and not ``range_boundary_hit``
    (same set as SPD ellipsoid fit).

    Uses SVD of the (n × 3) matrix of unit vectors: require rank 3 and
    σ_min/σ_max above ``confidence.coverage_min_sv_ratio``.
    """
    ccfg = cfg.get("confidence") or {}
    min_ratio = float(ccfg.get("coverage_min_sv_ratio", 0.05))
    usable_u = [
        e.u
        for e in estimates
        if e.valid
        and np.isfinite(e.range_)
        and e.range_ > 0
        and not e.range_boundary_hit
    ]
    diag: dict[str, Any] = {
        "n_valid_for_coverage": len(usable_u),
        "coverage_singular_values": None,
        "coverage_sv_ratio": None,
        "coverage_rank": 0,
    }
    if len(usable_u) < 3:
        return False, diag
    U = np.stack(usable_u, axis=0)
    s = np.linalg.svd(U, compute_uv=False)
    s = np.asarray(s, dtype=float)
    smax = float(s[0]) + 1e-18
    rank = int(np.sum(s > 1e-2 * smax))
    sv_ratio = float(s[-1] / smax) if s.size >= 3 else 0.0
    diag["coverage_singular_values"] = s.tolist()
    diag["coverage_sv_ratio"] = sv_ratio
    diag["coverage_rank"] = rank
    ok = rank >= 3 and sv_ratio >= min_ratio
    return bool(ok), diag


def scan_directions(
    pairs: PairCloud3D,
    *,
    omni_nugget: float,
    omni_sill: float,
    max_dist: float,
    cfg: dict[str, Any],
) -> list[DirectionalEstimate]:
    """Coarse directional range scan on Fibonacci hemisphere."""
    dcfg = cfg.get("directions") or {}
    vcfg = cfg.get("variogram") or {}
    n_dir = int(dcfg.get("n_directions", 96))
    tol = float(dcfg.get("angular_tolerance_deg", 22.5))
    bandwidth = dcfg.get("bandwidth", None)
    if bandwidth is None and dcfg.get("bandwidth_frac_max_dist") is not None:
        bandwidth = float(dcfg["bandwidth_frac_max_dist"]) * float(max_dist)
    min_pairs = int(dcfg.get("min_pairs", 40))
    min_lags = int(dcfg.get("min_valid_lags", 4))
    n_lags = int(vcfg.get("n_lags", 12))
    min_pairs_lag = int(vcfg.get("min_pairs_per_lag", 8))
    boundary_frac = float(dcfg.get("range_boundary_frac", 0.98))
    # Range fit upper bound tied to the same max_dist as the pair cloud
    range_upper = float(max_dist) * 2.5

    dirs = fibonacci_hemisphere(n_dir)
    out: list[DirectionalEstimate] = []
    for u in dirs:
        lag, gamma, counts, n_cone = experimental_directional(
            pairs,
            u,
            angular_tolerance_deg=tol,
            bandwidth=float(bandwidth) if bandwidth is not None else None,
            n_lags=n_lags,
            max_dist=max_dist,
            min_pairs=min_pairs_lag,
        )
        valid = n_cone >= min_pairs and int(lag.size) >= min_lags
        if not valid:
            out.append(
                DirectionalEstimate(
                    u=u.copy(),
                    range_=float("nan"),
                    sse=float("inf"),
                    n_pairs=n_cone,
                    n_valid_lags=int(lag.size),
                    valid=False,
                    fit=None,
                    range_boundary_hit=False,
                    range_upper_bound=range_upper,
                )
            )
            continue
        fit = fit_spherical_weighted(
            lag,
            gamma,
            counts,
            nugget=float(omni_nugget),
            sill=float(omni_sill),
            max_range=range_upper,
            boundary_frac=boundary_frac,
        )
        out.append(
            DirectionalEstimate(
                u=u.copy(),
                range_=float(fit.range_),
                sse=float(fit.sse),
                n_pairs=n_cone,
                n_valid_lags=int(fit.n_valid_lags),
                valid=True,
                fit=fit,
                range_boundary_hit=bool(fit.range_boundary_hit),
                range_upper_bound=fit.range_upper_bound,
            )
        )
    return out
