"""Ellipsoidal anisotropic Ordinary Kriging (3D)."""

from __future__ import annotations

import numpy as np

from .geometry import (
    anisotropic_metric_distance,
    ellipsoid_mask,
    rotate_to_principal,
)


def spherical_gamma(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical semivariogram γ(h) with nugget, partial sill, range."""
    h = np.asarray(h, dtype=float)
    out = np.full_like(h, nugget + sill, dtype=float)
    m = (h > 0) & (h < range_)
    hr = h[m] / max(float(range_), 1e-12)
    out[m] = nugget + sill * (1.5 * hr - 0.5 * hr**3)
    out[h == 0] = 0.0
    return out


def ordinary_kriging_ellipsoidal(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    *,
    azimuth_deg: float,
    dip_deg: float,
    a_major: float,
    a_inter: float,
    a_minor: float,
    nugget: float,
    sill: float,
    range_scale: float,
    r_major: float,
    r_inter: float,
    r_minor: float,
    n_max: int,
) -> np.ndarray:
    """Point OK with 3D anisotropic metric + ellipsoidal neighbourhood.

    Geometric ratios K2, K3 use *base* ranges (scale cancels); spherical
    isotropic range in metric space is ``a_major * range_scale``.
    """
    known_xyz = np.asarray(known_xyz, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xyz = np.asarray(pred_xyz, dtype=float)
    n_known = known_xyz.shape[0]
    m = pred_xyz.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    a_maj = max(float(a_major) * float(range_scale), 1e-8)
    k_maj = max(float(a_major), 1e-8)
    k_int = max(float(a_inter), 1e-8)
    k_min = max(float(a_minor), 1e-8)

    known_r = rotate_to_principal(known_xyz, azimuth_deg, dip_deg)
    pred_r = rotate_to_principal(pred_xyz, azimuth_deg, dip_deg)

    preds = np.empty(m, dtype=float)
    n_max_i = max(int(n_max), 1)
    nug = max(float(nugget), 0.0)
    sil = max(float(sill), 1e-12)

    for i in range(m):
        du = known_r - pred_r[i]
        u1, u2, u3 = du[:, 0], du[:, 1], du[:, 2]
        inside = ellipsoid_mask(u1, u2, u3, r_major, r_inter, r_minor)
        if np.any(inside):
            idx = np.where(inside)[0]
        else:
            d_all = anisotropic_metric_distance(u1, u2, u3, k_maj, k_int, k_min)
            k = min(n_max_i, n_known)
            idx = np.argpartition(d_all, k - 1)[:k]

        if idx.size > n_max_i:
            d_met = anisotropic_metric_distance(
                u1[idx], u2[idx], u3[idx], k_maj, k_int, k_min
            )
            keep = np.argpartition(d_met, n_max_i - 1)[:n_max_i]
            idx = idx[keep]

        c = known_r[idx]
        zz = known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue

        du_ij = c[:, None, :] - c[None, :, :]
        D = anisotropic_metric_distance(
            du_ij[:, :, 0], du_ij[:, :, 1], du_ij[:, :, 2], k_maj, k_int, k_min
        )
        du0 = c - pred_r[i]
        d0 = anisotropic_metric_distance(
            du0[:, 0], du0[:, 1], du0[:, 2], k_maj, k_int, k_min
        )

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = spherical_gamma(D, nug, sil, a_maj)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = spherical_gamma(d0, nug, sil, a_maj)

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))

    return preds


def ordinary_kriging_isotropic(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    *,
    range_: float,
    nugget: float,
    sill: float,
    range_scale: float,
    radius: float,
    n_max: int,
) -> np.ndarray:
    """Point OK with Euclidean distance and spherical neighbourhood."""
    known_xyz = np.asarray(known_xyz, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xyz = np.asarray(pred_xyz, dtype=float)
    n_known = known_xyz.shape[0]
    m = pred_xyz.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    a = max(float(range_) * float(range_scale), 1e-8)
    r = max(float(radius), 1e-12)
    n_max_i = max(int(n_max), 1)
    nug = max(float(nugget), 0.0)
    sil = max(float(sill), 1e-12)

    preds = np.empty(m, dtype=float)
    for i in range(m):
        d0_all = np.linalg.norm(known_xyz - pred_xyz[i], axis=1)
        inside = d0_all <= r
        if np.any(inside):
            idx = np.where(inside)[0]
        else:
            k = min(n_max_i, n_known)
            idx = np.argpartition(d0_all, k - 1)[:k]

        if idx.size > n_max_i:
            keep = np.argpartition(d0_all[idx], n_max_i - 1)[:n_max_i]
            idx = idx[keep]

        c = known_xyz[idx]
        zz = known_z[idx]
        n = len(idx)
        if n == 0:
            preds[i] = np.nan
            continue

        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        d0 = np.linalg.norm(c - pred_xyz[i], axis=1)

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = spherical_gamma(D, nug, sil, a)
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = spherical_gamma(d0, nug, sil, a)

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))

    return preds
