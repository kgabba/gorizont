"""Experimental variograms and spherical model fitting (n_pairs-weighted)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from .models import PairCloud3D, VariogramFit


def spherical_model(
    h: np.ndarray, nugget: float, sill: float, range_: float
) -> np.ndarray:
    """Spherical γ(h); sill is partial sill."""
    h = np.asarray(h, dtype=float)
    a = max(float(range_), 1e-12)
    out = np.full_like(h, float(nugget) + float(sill), dtype=float)
    m = (h > 0) & (h < a)
    hr = h[m] / a
    out[m] = float(nugget) + float(sill) * (1.5 * hr - 0.5 * hr**3)
    out[h == 0] = 0.0
    return out


def experimental_omni(
    pairs: PairCloud3D,
    *,
    n_lags: int = 12,
    max_dist: float | None = None,
    min_pairs: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairs.n == 0:
        e = np.array([], dtype=float)
        return e, e, e
    if max_dist is None:
        max_dist = float(np.max(pairs.h))
    max_dist = float(max_dist)
    h = pairs.h
    g = pairs.gamma
    m = h <= max_dist
    h, g = h[m], g[m]
    edges = np.linspace(0.0, max_dist, int(n_lags) + 1)
    lags, gammas, counts = [], [], []
    for i in range(int(n_lags)):
        sel = (h >= edges[i]) & (h < edges[i + 1])
        c = int(sel.sum())
        if c < int(min_pairs):
            continue
        lags.append(0.5 * (edges[i] + edges[i + 1]))
        gammas.append(float(np.mean(g[sel])))
        counts.append(c)
    return (
        np.asarray(lags, dtype=float),
        np.asarray(gammas, dtype=float),
        np.asarray(counts, dtype=float),
    )


def angle_to_axis_deg(u_pairs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Acute angle (degrees) between pair unit vectors and ±axis."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-18)
    c = np.abs(u_pairs @ axis)
    c = np.clip(c, 0.0, 1.0)
    return np.degrees(np.arccos(c))


def experimental_directional(
    pairs: PairCloud3D,
    axis: np.ndarray,
    *,
    angular_tolerance_deg: float = 22.5,
    bandwidth: float | None = None,
    n_lags: int = 12,
    max_dist: float | None = None,
    min_pairs: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Directional experimental VG along undirected axis.

    Returns lag, gamma, n_pairs, n_pairs_in_cone.
    """
    if pairs.n == 0:
        e = np.array([], dtype=float)
        return e, e, e, 0
    if max_dist is None:
        max_dist = float(np.max(pairs.h))
    max_dist = float(max_dist)
    u = pairs.u_stack()
    ang = angle_to_axis_deg(u, axis)
    near = ang <= float(angular_tolerance_deg)
    near &= pairs.h <= max_dist
    if bandwidth is not None and float(bandwidth) > 0:
        # perpendicular distance from chord ≈ h * sin(theta)
        theta = np.radians(ang)
        band = pairs.h * np.sin(theta)
        near &= band <= float(bandwidth)
    n_cone = int(near.sum())
    h = pairs.h[near]
    g = pairs.gamma[near]
    if h.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, 0
    edges = np.linspace(0.0, max_dist, int(n_lags) + 1)
    lags, gammas, counts = [], [], []
    for i in range(int(n_lags)):
        sel = (h >= edges[i]) & (h < edges[i + 1])
        c = int(sel.sum())
        if c < int(min_pairs):
            continue
        lags.append(0.5 * (edges[i] + edges[i + 1]))
        gammas.append(float(np.mean(g[sel])))
        counts.append(c)
    return (
        np.asarray(lags, dtype=float),
        np.asarray(gammas, dtype=float),
        np.asarray(counts, dtype=float),
        n_cone,
    )


def _weighted_sse(
    lag: np.ndarray,
    gamma: np.ndarray,
    n_pairs: np.ndarray,
    nugget: float,
    sill: float,
    range_: float,
) -> float:
    pred = spherical_model(lag, nugget, sill, range_)
    w = np.maximum(n_pairs.astype(float), 1.0)
    return float(np.sum(w * (gamma - pred) ** 2))


def fit_spherical_weighted(
    lag: np.ndarray,
    gamma: np.ndarray,
    n_pairs: np.ndarray,
    *,
    nugget: float | None = None,
    sill: float | None = None,
    range_: float | None = None,
    max_range: float | None = None,
    boundary_frac: float = 0.98,
) -> VariogramFit:
    """Fit spherical model with weights = n_pairs.

    Fixed parameters stay fixed; free ones optimized.
    Does **not** use max(sill, variance) correction.
    """
    lag = np.asarray(lag, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n_pairs = np.asarray(n_pairs, dtype=float)
    empty = VariogramFit(
        lag=lag,
        gamma=gamma,
        n_pairs=n_pairs,
        nugget=0.0,
        sill=0.0,
        range_=0.0,
        sse=float("inf"),
        n_valid_lags=int(lag.size),
    )
    if lag.size < 2:
        return empty

    g_max = float(np.max(gamma))
    g_min = float(np.min(gamma))
    h_max = float(np.max(lag))
    if max_range is None:
        max_range = max(h_max * 2.0, h_max + 1e-6)
    max_range = float(max_range)
    lo_range = max(h_max * 0.05, 1e-12)

    fix_nug = nugget is not None
    fix_sil = sill is not None
    fix_rng = range_ is not None

    nug0 = float(nugget) if fix_nug else max(0.0, g_min * 0.5)
    sil0 = float(sill) if fix_sil else max(g_max - nug0, 1e-12)
    rng0 = float(range_) if fix_rng else max(h_max * 0.6, 1e-6)

    def pack(nug, sil, rng):
        return _weighted_sse(lag, gamma, n_pairs, nug, sil, rng)

    def _finish(nug, sil, rng, sse):
        hit = (not fix_rng) and (float(rng) >= float(boundary_frac) * max_range)
        return VariogramFit(
            lag,
            gamma,
            n_pairs,
            float(nug),
            float(sil),
            float(rng),
            float(sse),
            n_valid_lags=int(lag.size),
            range_upper_bound=max_range,
            range_boundary_hit=bool(hit),
        )

    if fix_nug and fix_sil and not fix_rng:
        def obj(r):
            return pack(nug0, sil0, float(r))

        res = minimize_scalar(obj, bounds=(lo_range, max_range), method="bounded")
        return _finish(nug0, sil0, float(res.x), float(res.fun))

    x0 = []
    bounds = []
    if not fix_nug:
        x0.append(nug0)
        bounds.append((0.0, g_max))
    if not fix_sil:
        x0.append(sil0)
        bounds.append((1e-12, max(g_max * 3.0, 1e-6)))
    if not fix_rng:
        x0.append(rng0)
        bounds.append((lo_range, max_range))

    if not x0:
        sse = pack(nug0, sil0, rng0)
        return _finish(nug0, sil0, rng0, sse)

    def unpack(x):
        nug, sil, rng = nug0, sil0, rng0
        k = 0
        if not fix_nug:
            nug = float(x[k])
            k += 1
        if not fix_sil:
            sil = float(x[k])
            k += 1
        if not fix_rng:
            rng = float(x[k])
        return nug, sil, rng

    def obj(x):
        return pack(*unpack(x))

    res = minimize(obj, np.asarray(x0, dtype=float), method="L-BFGS-B", bounds=bounds)
    nug, sil, rng = unpack(res.x)
    return _finish(nug, sil, rng, float(res.fun))


def fit_omni(
    pairs: PairCloud3D,
    cfg: dict[str, Any],
) -> tuple[VariogramFit, float]:
    """Omnidirectional fit on the already distance-filtered pair cloud.

    Uses ``pairs.max_dist`` (single source of truth). Does **not** re-cut
    distances by percentile.
    """
    vg = cfg.get("variogram") or {}
    n_lags = int(vg.get("n_lags", 12))
    min_pairs = int(vg.get("min_pairs_per_lag", 8))
    if pairs.n == 0:
        empty = VariogramFit(
            np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, float("inf")
        )
        return empty, float(pairs.max_dist)
    max_dist = float(pairs.max_dist) if pairs.max_dist > 0 else float(np.max(pairs.h))
    lag, gamma, counts = experimental_omni(
        pairs, n_lags=n_lags, max_dist=max_dist, min_pairs=min_pairs
    )
    fit = fit_spherical_weighted(lag, gamma, counts, max_range=max_dist * 2.5)
    return fit, max_dist
