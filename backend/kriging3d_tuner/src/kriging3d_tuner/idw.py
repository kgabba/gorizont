"""3D Anisotropic Inverse Distance Weighting (IDW).

Weights use anisotropic search distance (same ellipsoid as neighbour selection).
Variogram / covariance is not used.
"""

from __future__ import annotations

import numpy as np

from .geometry import search_distance, to_principal
from .kriging import select_neighbour_indices
from .models import SearchParams, VariogramFixed

_ZERO_DIST = 1e-18


def anisotropic_idw_at_points(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    vg: VariogramFixed,
    search: SearchParams,
    power: float,
) -> np.ndarray:
    """Predict at each row of ``pred_xyz``; NaN if neighbourhood < Nmin.

    Distance for both neighbour selection and weights is anisotropic
    ``search_distance`` in the search-ellipsoid principal frame.
    If any selected neighbour has d ≈ 0, return the mean of those coincident
    sample values (no division by zero).
    """
    known_xyz = np.asarray(known_xyz, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xyz = np.asarray(pred_xyz, dtype=float)
    p = float(power)
    if not np.isfinite(p) or p < 0.0:
        raise ValueError(f"IDW power must be finite and >= 0, got {power!r}")

    Q = vg.orientation_matrix
    n_known = known_xyz.shape[0]
    m = pred_xyz.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    preds = np.full(m, np.nan, dtype=float)
    r_maj = float(search.r_major)
    r_int = float(search.r_inter)
    r_min = float(search.r_minor)

    for i in range(m):
        delta = known_xyz - pred_xyz[i]
        idx = select_neighbour_indices(delta, Q, search)
        if idx.size < search.n_min:
            continue

        xp = to_principal(delta[idx], Q)
        d = np.asarray(
            search_distance(xp, r_maj, r_int, r_min), dtype=float
        ).ravel()
        zz = known_z[idx]

        zero = d <= _ZERO_DIST
        if np.any(zero):
            preds[i] = float(np.mean(zz[zero]))
            continue

        w = np.power(d, -p)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0.0:
            continue
        preds[i] = float(np.dot(w, zz) / sw)

    return preds
