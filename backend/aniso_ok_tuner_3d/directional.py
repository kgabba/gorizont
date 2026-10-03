"""Directional experimental variograms and spherical fitting (3D cones)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .geometry import direction_from_az_dip
from .pairs import PairCloud3D


@dataclass(frozen=True)
class DirectionalVariogram3D:
    azimuth_deg: float
    dip_deg: float
    lag: np.ndarray
    gamma: np.ndarray
    n_pairs: np.ndarray
    nugget: float
    sill: float
    range_: float


def spherical_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    out = np.full_like(h, nugget + sill, dtype=float)
    m = (h > 0) & (h < range_)
    hr = h[m] / max(float(range_), 1e-12)
    out[m] = nugget + sill * (1.5 * hr - 0.5 * hr**3)
    out[h == 0] = 0.0
    return out


def _cone_mask(
    pairs: PairCloud3D,
    direction: np.ndarray,
    tolerance_deg: float,
    max_dist: float,
) -> np.ndarray:
    """Pairs whose lag vector lies within ``tolerance_deg`` of ±direction."""
    if pairs.h.size == 0:
        return np.array([], dtype=bool)
    d = direction / max(np.linalg.norm(direction), 1e-12)
    # lag unit vectors
    h = pairs.h
    valid = h > 1e-12
    ux = np.zeros_like(h)
    uy = np.zeros_like(h)
    uz = np.zeros_like(h)
    ux[valid] = pairs.hx[valid] / h[valid]
    uy[valid] = pairs.hy[valid] / h[valid]
    uz[valid] = pairs.hz[valid] / h[valid]
    cosang = np.abs(ux * d[0] + uy * d[1] + uz * d[2])
    cosang = np.clip(cosang, 0.0, 1.0)
    ang = np.rad2deg(np.arccos(cosang))
    near = valid & (ang <= float(tolerance_deg)) & (h <= float(max_dist))
    return near


def experimental_directional(
    pairs: PairCloud3D,
    azimuth_deg: float,
    dip_deg: float,
    *,
    tolerance_deg: float = 20.0,
    n_lags: int = 12,
    max_dist: float | None = None,
    max_dist_percentile: float = 50.0,
    min_pairs: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin pairs in a 3D angular cone into an experimental variogram."""
    if pairs.h.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    if max_dist is None:
        max_dist = float(np.percentile(pairs.h, max_dist_percentile))

    direction = direction_from_az_dip(azimuth_deg, dip_deg)
    near = _cone_mask(pairs, direction, tolerance_deg, float(max_dist))
    h = pairs.h[near]
    g = pairs.gamma[near]
    if h.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    edges = np.linspace(0.0, float(max_dist), int(n_lags) + 1)
    lags, gammas, counts = [], [], []
    for i in range(int(n_lags)):
        m = (h >= edges[i]) & (h < edges[i + 1])
        c = int(m.sum())
        if c < int(min_pairs):
            continue
        lags.append(float(h[m].mean()))
        gammas.append(float(g[m].mean()))
        counts.append(c)
    return (
        np.asarray(lags, dtype=float),
        np.asarray(gammas, dtype=float),
        np.asarray(counts, dtype=float),
    )


def fit_spherical(lag: np.ndarray, gamma: np.ndarray) -> tuple[float, float, float]:
    """Grid-search spherical (nugget, sill, range)."""
    lag = np.asarray(lag, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if lag.size < 2:
        g0 = float(np.mean(gamma)) if gamma.size else 1.0
        r0 = float(lag.max()) if lag.size else 1.0
        return 0.0, max(g0, 1e-6), max(r0, 1e-6)

    best_sse = np.inf
    best = (0.0, float(np.max(gamma)), float(np.max(lag)))
    g_max = float(np.max(gamma))
    for nugget, sill, range_ in product(
        np.linspace(0.0, max(np.percentile(gamma, 25), 1e-9), 8),
        np.linspace(max(np.percentile(gamma, 40), 1e-9), g_max * 1.3 + 1e-9, 12),
        np.linspace(max(float(lag.min()), 1e-6), float(lag.max()), 15),
    ):
        pred = spherical_model(lag, nugget, sill, range_)
        sse = float(np.sum((gamma - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (float(nugget), float(sill), float(range_))
    return best


def fit_directional_variogram(
    pairs: PairCloud3D,
    azimuth_deg: float,
    dip_deg: float,
    **kwargs,
) -> DirectionalVariogram3D:
    lag, gamma, n_pairs = experimental_directional(
        pairs, azimuth_deg, dip_deg, **kwargs
    )
    nugget, sill, range_ = fit_spherical(lag, gamma)
    return DirectionalVariogram3D(
        azimuth_deg=float(azimuth_deg),
        dip_deg=float(dip_deg),
        lag=lag,
        gamma=gamma,
        n_pairs=n_pairs,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )
