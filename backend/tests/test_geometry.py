"""Unit tests for 3D geometry and OK invariants."""

from __future__ import annotations

import numpy as np

from aniso_ok_tuner_3d.geometry import (
    anisotropic_metric_distance,
    az_dip_from_direction,
    direction_from_az_dip,
    ellipsoid_axes,
    ellipsoid_mask,
    order_ranges_descending,
    rotate_to_principal,
)
from aniso_ok_tuner_3d.kriging import ordinary_kriging_ellipsoidal, ordinary_kriging_isotropic


def test_axes_orthonormal():
    e1, e2, e3 = ellipsoid_axes(35.0, 20.0)
    R = np.stack([e1, e2, e3], axis=0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_major_aligns_with_u1():
    az, dip = 40.0, -15.0
    e1 = direction_from_az_dip(az, dip)
    u = rotate_to_principal(e1.reshape(1, 3), az, dip)[0]
    assert np.allclose(u, [1.0, 0.0, 0.0], atol=1e-10)


def test_az_dip_roundtrip():
    for az, dip in [(0.0, 0.0), (45.0, 30.0), (120.0, -40.0), (90.0, 80.0)]:
        e1 = direction_from_az_dip(az, dip)
        az2, dip2 = az_dip_from_direction(e1)
        e1b = direction_from_az_dip(az2, dip2)
        # direction may flip 180° in plan fold — compare absolute cos
        assert abs(np.dot(e1, e1b)) > 0.999


def test_isotropic_metric_equals_euclidean_scaled():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(50, 3))
    a = 10.0
    h = anisotropic_metric_distance(u[:, 0], u[:, 1], u[:, 2], a, a, a)
    assert np.allclose(h, np.linalg.norm(u, axis=1), atol=1e-10)


def test_metric_compresses_short_axes():
    # point along u2 at distance 1 → metric = 1/K2 = a_maj/a_inter
    h = anisotropic_metric_distance(
        np.array([0.0]), np.array([1.0]), np.array([0.0]), 10.0, 5.0, 2.0
    )
    assert np.isclose(h[0], 2.0, atol=1e-10)


def test_ellipsoid_mask_unit_sphere_when_equal_r():
    u1 = np.array([0.5, 1.1, 0.0])
    u2 = np.array([0.5, 0.0, 0.0])
    u3 = np.array([0.5, 0.0, 0.0])
    m = ellipsoid_mask(u1, u2, u3, 1.0, 1.0, 1.0)
    assert m[0]
    assert not m[1]


def test_order_ranges_descending():
    a1, a2, a3, az, dip = order_ranges_descending(3.0, 10.0, 5.0, 0.0, 0.0)
    assert a1 >= a2 >= a3
    assert np.isclose(a1, 10.0)


def test_ok_recovers_constant_field():
    rng = np.random.default_rng(1)
    known = rng.uniform(0, 10, size=(30, 3))
    z = np.full(30, 7.5)
    pred = rng.uniform(0, 10, size=(5, 3))
    out = ordinary_kriging_ellipsoidal(
        known,
        z,
        pred,
        azimuth_deg=20.0,
        dip_deg=10.0,
        a_major=8.0,
        a_inter=5.0,
        a_minor=3.0,
        nugget=0.01,
        sill=1.0,
        range_scale=1.0,
        r_major=15.0,
        r_inter=12.0,
        r_minor=10.0,
        n_max=20,
    )
    assert np.allclose(out, 7.5, atol=1e-6)


def test_ok_isotropic_matches_ellipsoidal_when_equal():
    rng = np.random.default_rng(2)
    known = rng.uniform(0, 5, size=(20, 3))
    z = known[:, 0] + 0.1 * known[:, 2]
    pred = rng.uniform(0, 5, size=(4, 3))
    a = 4.0
    r = 6.0
    kwargs = dict(nugget=0.05, sill=1.0, range_scale=1.0, n_max=12)
    iso = ordinary_kriging_isotropic(
        known, z, pred, range_=a, radius=r, **kwargs
    )
    aniso = ordinary_kriging_ellipsoidal(
        known,
        z,
        pred,
        azimuth_deg=0.0,
        dip_deg=0.0,
        a_major=a,
        a_inter=a,
        a_minor=a,
        r_major=r,
        r_inter=r,
        r_minor=r,
        **kwargs,
    )
    assert np.allclose(iso, aniso, atol=1e-8, equal_nan=True)
