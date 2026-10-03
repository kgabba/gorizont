"""Pairwise lag vectors and semivariances in 3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairCloud3D:
    """All unordered point pairs within one domain."""

    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    h: np.ndarray
    gamma: np.ndarray


def compute_pairs(xyz: np.ndarray, z: np.ndarray) -> PairCloud3D:
    """Build full pairwise cloud for 3D variography."""
    xyz = np.asarray(xyz, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = xyz.shape[0]
    if n < 2:
        empty = np.array([], dtype=float)
        return PairCloud3D(empty, empty, empty, empty, empty)

    i_idx, j_idx = np.triu_indices(n, k=1)
    d = xyz[j_idx] - xyz[i_idx]
    hx, hy, hz = d[:, 0], d[:, 1], d[:, 2]
    h = np.sqrt(hx * hx + hy * hy + hz * hz)
    gamma = 0.5 * (z[i_idx] - z[j_idx]) ** 2
    return PairCloud3D(hx=hx, hy=hy, hz=hz, h=h, gamma=gamma)
