"""3D geometric anisotropy: azimuth + dip orientation, metric, search ellipsoid.

Convention (rake = 0)
---------------------
- ``azimuth_deg``: major direction in the XY plane, math CCW from +X
  (same convention as 2D ``alpha_deg``).
- ``dip_deg``: elevation of major from horizontal in [-90, +90];
  positive = toward +Z.
- Intermediate axis is horizontal, 90° CCW from azimuth in plan.
- Minor = major × intermediate (right-handed).

Local coordinates ``u = R @ (x - x0)`` put major along +u1.
"""

from __future__ import annotations

import numpy as np


def ellipsoid_axes(
    azimuth_deg: float,
    dip_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit vectors (e_major, e_inter, e_minor) in world XYZ."""
    az = np.deg2rad(float(azimuth_deg))
    dip = np.deg2rad(float(dip_deg))
    c_a, s_a = np.cos(az), np.sin(az)
    c_d, s_d = np.cos(dip), np.sin(dip)

    e1 = np.array([c_d * c_a, c_d * s_a, s_d], dtype=float)
    e2 = np.array([-s_a, c_a, 0.0], dtype=float)
    e3 = np.cross(e1, e2)
    # numerical safety
    n3 = np.linalg.norm(e3)
    if n3 < 1e-12:
        # major nearly vertical: pick horizontal intermediate from az
        e2 = np.array([-s_a, c_a, 0.0], dtype=float)
        e3 = np.cross(e1, e2)
        n3 = np.linalg.norm(e3)
    e3 = e3 / max(n3, 1e-12)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    return e1, e2, e3


def rotation_matrix_az_dip(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """3×3 matrix with rows = (e1, e2, e3); ``u = xyz @ R.T``."""
    e1, e2, e3 = ellipsoid_axes(azimuth_deg, dip_deg)
    return np.stack([e1, e2, e3], axis=0)


def rotate_to_principal(
    xyz: np.ndarray,
    azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:
    """Rotate world XYZ into ellipsoid principal frame (major → +u1)."""
    R = rotation_matrix_az_dip(azimuth_deg, dip_deg)
    return np.asarray(xyz, dtype=float) @ R.T


def direction_from_az_dip(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """Unit major direction in world coords."""
    e1, _, _ = ellipsoid_axes(azimuth_deg, dip_deg)
    return e1


def az_dip_from_direction(vec: np.ndarray) -> tuple[float, float]:
    """Recover (azimuth_deg, dip_deg) from a direction vector."""
    v = np.asarray(vec, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 0.0, 0.0
    v = v / n
    dip = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    az = float(np.rad2deg(np.arctan2(v[1], v[0]))) % 180.0
    return az, dip


def anisotropic_metric_distance(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    a_major: float,
    a_inter: float,
    a_minor: float,
) -> np.ndarray:
    """Geometric-anisotropy distance in principal frame.

    h' = sqrt(u1² + (u2/K2)² + (u3/K3)²) with K2=a_inter/a_major,
    K3=a_minor/a_major. Spherical model then uses range = a_major (* scale).
    """
    a1 = max(float(a_major), 1e-12)
    k2 = float(np.clip(float(a_inter) / a1, 1e-6, 1.0))
    k3 = float(np.clip(float(a_minor) / a1, 1e-6, 1.0))
    return np.sqrt(u1 * u1 + (u2 / k2) ** 2 + (u3 / k3) ** 2)


def ellipsoid_mask(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    r_major: float,
    r_inter: float,
    r_minor: float,
) -> np.ndarray:
    """Neighbours inside search ellipsoid (u1/R1)²+(u2/R2)²+(u3/R3)² ≤ 1."""
    r1 = max(float(r_major), 1e-12)
    r2 = max(float(r_inter), 1e-12)
    r3 = max(float(r_minor), 1e-12)
    return (u1 / r1) ** 2 + (u2 / r2) ** 2 + (u3 / r3) ** 2 <= 1.0


def order_ranges_descending(
    a1: float,
    a2: float,
    a3: float,
    azimuth_deg: float,
    dip_deg: float,
) -> tuple[float, float, float, float, float]:
    """Ensure a_major ≥ a_inter ≥ a_minor; reassign az/dip if major axis swaps.

    When the longest fitted range is not along the MOI major direction,
    we rotate az/dip so the new major aligns with that axis.
    """
    ranges = np.array([float(a1), float(a2), float(a3)], dtype=float)
    ranges = np.maximum(ranges, 1e-8)
    e1, e2, e3 = ellipsoid_axes(azimuth_deg, dip_deg)
    axes = [e1, e2, e3]
    order = np.argsort(-ranges)  # longest first
    new_ranges = ranges[order]
    new_e1 = axes[int(order[0])]
    # rebuild az/dip from new major; intermediate kept as horizontal 90° CCW
    az, dip = az_dip_from_direction(new_e1)
    # After reorient, a_inter/a_minor correspond to remaining two; keep sorted
    a_maj, a_int, a_min = float(new_ranges[0]), float(new_ranges[1]), float(new_ranges[2])
    if a_int < a_min:
        a_int, a_min = a_min, a_int
    return a_maj, a_int, a_min, float(az), float(dip)
