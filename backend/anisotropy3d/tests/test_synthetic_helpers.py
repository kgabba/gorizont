"""Synthetic Gaussian random fields for anisotropy verification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _chol_embed_cov(
    xyz: np.ndarray, ranges: np.ndarray, Q: np.ndarray, sill: float, nugget: float
) -> np.ndarray:
    """Spherical covariance in rotated anisotropic metric (range-normalized).

    h' = || diag(1/ranges) Q^T (x_i - x_j) ||; C = sill * spherical(h').
    Nugget added on the diagonal by the caller if needed.
    """
    n = len(xyz)
    inv_r = 1.0 / np.maximum(ranges, 1e-12)
    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        C[i, i] = float(sill)
        for j in range(i + 1, n):
            dx = xyz[i] - xyz[j]
            hp = float(np.linalg.norm(inv_r * (Q.T @ dx)))
            if hp >= 1.0:
                cov = 0.0
            elif hp <= 0:
                cov = float(sill)
            else:
                cov = float(sill) * (1.0 - 1.5 * hp + 0.5 * hp**3)
            C[i, j] = C[j, i] = cov
    return C


def sample_anisotropic_field(
    *,
    n: int,
    ranges: tuple[float, float, float],
    Q: np.ndarray | None = None,
    sill: float = 1.0,
    nugget: float = 0.05,
    seed: int = 0,
    bbox: tuple[float, float, float] = (200.0, 200.0, 200.0),
) -> pd.DataFrame:
    """Draw points uniformly in bbox and values ~ N(0, C) spherical anisotropic."""
    rng = np.random.default_rng(seed)
    if Q is None:
        Q = np.eye(3)
    Q = np.asarray(Q, dtype=float)
    # RH orthonormalize
    q0 = Q[:, 0] / (np.linalg.norm(Q[:, 0]) + 1e-18)
    q1 = Q[:, 1] - q0 * (q0 @ Q[:, 1])
    q1 = q1 / (np.linalg.norm(q1) + 1e-18)
    q2 = np.cross(q0, q1)
    Q = np.column_stack([q0, q1, q2])
    ranges_a = np.asarray(ranges, dtype=float)
    xyz = rng.uniform(0.0, 1.0, size=(n, 3)) * np.asarray(bbox, dtype=float)
    C = _chol_embed_cov(xyz, ranges_a, Q, sill=sill, nugget=0.0)
    C = C + np.eye(n) * nugget
    # jitter for PD
    C = C + np.eye(n) * 1e-8
    L = np.linalg.cholesky(C)
    z = L @ rng.normal(size=n)
    return pd.DataFrame({"X": xyz[:, 0], "Y": xyz[:, 1], "Z": xyz[:, 2], "Value": z})


def undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a / (np.linalg.norm(a) + 1e-18)
    b = b / (np.linalg.norm(b) + 1e-18)
    c = float(np.clip(abs(a @ b), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))
