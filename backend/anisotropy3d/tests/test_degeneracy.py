"""Hard degeneracy vs pancake drillhole clouds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anisotropy3d.io import validate_points
from anisotropy3d.pipeline import load_config


def _cfg():
    return load_config()


def test_pancake_drillholes_not_planar():
    """XY ≫ Z but real thickness ≫ sample spacing → no hard planar gate."""
    rng = np.random.default_rng(0)
    rows = []
    for h in range(40):
        x0, y0 = rng.uniform(0, 1000, size=2)
        for k in range(20):
            z = 300.0 - k * 8.0  # ~160 m vertical
            rows.append((x0 + rng.normal(0, 0.5), y0 + rng.normal(0, 0.5), z, 40.0 + rng.normal()))
    df = pd.DataFrame(rows, columns=["X", "Y", "Z", "Value"])
    pts = validate_points(df, _cfg())
    assert pts.diagnostics["aspect_pancake"] is True
    assert pts.diagnostics["spatial_degeneracy"] is None
    assert pts.diagnostics["spatial_dim"] == 3


def test_absolute_coplanar_is_planar():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(200, 2))
    z = np.full(200, 10.0)  # zero thickness
    df = pd.DataFrame(
        {"X": xy[:, 0], "Y": xy[:, 1], "Z": z, "Value": rng.normal(size=200)}
    )
    pts = validate_points(df, _cfg())
    assert pts.diagnostics["spatial_degeneracy"] == "planar"
    assert pts.diagnostics["spatial_dim"] == 2


def test_line_is_linear():
    t = np.linspace(0, 100, 80)
    df = pd.DataFrame(
        {
            "X": t,
            "Y": 0.01 * t,
            "Z": -0.02 * t,
            "Value": np.sin(t / 10) + 1.0,
        }
    )
    pts = validate_points(df, _cfg())
    assert pts.diagnostics["spatial_degeneracy"] == "linear"
    assert pts.diagnostics["spatial_dim"] == 1


def test_synthetic_fe_csv_not_planar():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "datasets"
        / "synthetic_Fe_3D_1000_samples.csv"
    )
    if not path.is_file():
        pytest.skip("synthetic Fe CSV not present")
    from anisotropy3d.io import load_points

    df = load_points(path)
    pts = validate_points(df, _cfg())
    assert pts.diagnostics["spatial_degeneracy"] is None
    assert pts.diagnostics["spatial_dim"] == 3
    assert pts.diagnostics.get("aspect_pancake") is True
