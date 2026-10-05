"""3D Ordinary Kriging with hard search ellipsoid + fixed variogram ellipsoid."""

from __future__ import annotations

import numpy as np

from .geometry import search_distance, to_principal, variogram_hprime
from .models import SearchParams, VariogramFixed


def spherical_covariance(h_prime: np.ndarray, sill: float) -> np.ndarray:
    """Covariance C(h') with partial sill and range=1 in h'-units."""
    h = np.asarray(h_prime, dtype=float)
    sil = max(float(sill), 1e-18)
    gamma = np.zeros_like(h, dtype=float)
    m = (h > 0) & (h < 1.0)
    hr = h[m]
    gamma[m] = sil * (1.5 * hr - 0.5 * hr**3)
    gamma[h >= 1.0] = sil
    return sil - gamma


def select_neighbour_indices(
    delta_xyz: np.ndarray,
    orientation: np.ndarray,
    search: SearchParams,
) -> np.ndarray:
    """Hard search ellipsoid mask; cap to Nmax by d_search; no outside fallback."""
    xp = to_principal(delta_xyz, orientation)
    d = search_distance(xp, search.r_major, search.r_inter, search.r_minor)
    inside = d <= 1.0 + 1e-12
    idx = np.where(inside)[0]
    if idx.size == 0:
        return idx
    if idx.size > search.n_max:
        d_in = d[idx]
        order = np.lexsort((idx, d_in))
        idx = idx[order[: search.n_max]]
    return idx


def ordinary_kriging_at_points(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    vg: VariogramFixed,
    search: SearchParams,
) -> np.ndarray:
    """Predict at each row of pred_xyz; NaN if neighbourhood < Nmin."""
    known_xyz = np.asarray(known_xyz, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xyz = np.asarray(pred_xyz, dtype=float)
    Q = vg.orientation_matrix
    n_known = known_xyz.shape[0]
    m = pred_xyz.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    preds = np.full(m, np.nan, dtype=float)
    nug = max(float(vg.nugget), 0.0)
    sil = max(float(vg.sill), 1e-18)
    a_maj = float(vg.range_major)
    a_int = float(vg.range_intermediate)
    a_min = float(vg.range_minor)

    for i in range(m):
        delta = known_xyz - pred_xyz[i]
        idx = select_neighbour_indices(delta, Q, search)
        if idx.size < search.n_min:
            continue

        nb = known_xyz[idx]
        zz = known_z[idx]
        n = idx.size

        delta_nn = nb[:, None, :] - nb[None, :, :]
        xp_nn = to_principal(delta_nn.reshape(-1, 3), Q).reshape(n, n, 3)
        h_nn = variogram_hprime(
            xp_nn.reshape(-1, 3), a_maj, a_int, a_min
        ).reshape(n, n)

        delta_n0 = nb - pred_xyz[i]
        xp_n0 = to_principal(delta_n0, Q)
        h_n0 = variogram_hprime(xp_n0, a_maj, a_int, a_min)

        K = spherical_covariance(h_nn, sil)
        np.fill_diagonal(K, sil + nug)
        k0 = spherical_covariance(h_n0, sil)

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = K
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = k0

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))

    return preds
