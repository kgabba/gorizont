"""Principal-axis geometry: search ellipsoid vs variogram metric."""

from __future__ import annotations

import numpy as np


def normalize_orientation(Q: np.ndarray) -> np.ndarray:
    """Ensure right-handed orthonormal orientation (columns maj, int, min)."""
    Q = np.asarray(Q, dtype=float).copy()
    if Q.shape != (3, 3):
        raise ValueError(f"orientation_matrix must be 3x3, got {Q.shape}")
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1.0
    return Q


def to_principal(delta_xyz: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Rotate lags into principal frame: x' = Q.T @ delta."""
    d = np.asarray(delta_xyz, dtype=float)
    Q = normalize_orientation(orientation)
    if d.ndim == 1:
        return Q.T @ d
    return d @ Q


def search_distance(
    xp: np.ndarray,
    r_major: float,
    r_inter: float,
    r_minor: float,
) -> np.ndarray:
    """Anisotropic search distance d_search in principal frame."""
    xp = np.asarray(xp, dtype=float)
    rm = max(float(r_major), 1e-18)
    ri = max(float(r_inter), 1e-18)
    rn = max(float(r_minor), 1e-18)
    if xp.ndim == 1:
        return float(
            np.sqrt((xp[0] / rm) ** 2 + (xp[1] / ri) ** 2 + (xp[2] / rn) ** 2)
        )
    return np.sqrt((xp[:, 0] / rm) ** 2 + (xp[:, 1] / ri) ** 2 + (xp[:, 2] / rn) ** 2)


def search_inside(d_search: np.ndarray) -> np.ndarray:
    return np.asarray(d_search, dtype=float) <= 1.0 + 1e-12


def variogram_hprime(
    xp: np.ndarray,
    a_major: float,
    a_intermediate: float,
    a_minor: float,
) -> np.ndarray:
    """Normalized variogram lag h' using variogram semi-axes."""
    xp = np.asarray(xp, dtype=float)
    am = max(float(a_major), 1e-18)
    ai = max(float(a_intermediate), 1e-18)
    an = max(float(a_minor), 1e-18)
    if xp.ndim == 1:
        return float(
            np.sqrt((xp[0] / am) ** 2 + (xp[1] / ai) ** 2 + (xp[2] / an) ** 2)
        )
    return np.sqrt((xp[:, 0] / am) ** 2 + (xp[:, 1] / ai) ** 2 + (xp[:, 2] / an) ** 2)
