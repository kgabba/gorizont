"""Tests for 3D anisotropic IDW."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kriging3d_tuner.cv import (
    METHOD_ANISOTROPIC_IDW,
    METHOD_ORDINARY_KRIGING,
    evaluate_oof,
)
from kriging3d_tuner.geometry import search_distance, to_principal
from kriging3d_tuner.idw import anisotropic_idw_at_points
from kriging3d_tuner.models import (
    CVFold,
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    VariogramFixed,
)
from kriging3d_tuner.pipeline import run_tuner


def _vg(orientation: np.ndarray | None = None) -> VariogramFixed:
    return VariogramFixed(
        variogram_model="spherical",
        nugget=0.0,
        sill=1.0,
        range_major=10.0,
        range_intermediate=10.0,
        range_minor=10.0,
        orientation_matrix=np.eye(3) if orientation is None else orientation,
    )


def _search(**kw) -> SearchParams:
    base = dict(
        r_major=10.0,
        k_inter=1.0,
        k_minor=1.0,
        n_min=1,
        n_max=10,
        octant_search=False,
    )
    base.update(kw)
    return SearchParams(**base)


def test_idw_formula_two_points():
    """ŷ = Σ w y / Σ w with w = 1/d^p on anisotropic (here spherical) d."""
    known_xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    known_z = np.array([0.0, 10.0])
    pred_xyz = np.array([[1.0, 0.0, 0.0]])
    search = _search(r_major=5.0, n_min=1, n_max=2)
    power = 2.0
    # Both at anisotropic d = |Δ|/R = 1/5
    d = 1.0 / 5.0
    w = 1.0 / d**power
    expected = (w * 0.0 + w * 10.0) / (w + w)
    got = anisotropic_idw_at_points(
        known_xyz, known_z, pred_xyz, _vg(), search, power
    )
    assert got[0] == pytest.approx(expected)
    assert got[0] == pytest.approx(5.0)


def test_anisotropic_distance_changes_weights():
    """Compressing R_inter changes anisotropic d and therefore ŷ."""
    # Sample along intermediate axis (Y)
    known_xyz = np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
    known_z = np.array([0.0, 10.0])
    pred_xyz = np.array([[0.0, 0.0, 0.0]])
    power = 2.0
    sph = _search(r_major=10.0, k_inter=1.0, k_minor=1.0, n_min=1, n_max=2)
    aniso = _search(r_major=10.0, k_inter=0.2, k_minor=0.2, n_min=1, n_max=2)
    y_sph = anisotropic_idw_at_points(
        known_xyz, known_z, pred_xyz, _vg(), sph, power
    )[0]
    y_an = anisotropic_idw_at_points(
        known_xyz, known_z, pred_xyz, _vg(), aniso, power
    )[0]
    # Symmetric samples → both give 5; check distances differ
    xp = to_principal(known_xyz - pred_xyz[0], np.eye(3))
    d_sph = search_distance(xp, sph.r_major, sph.r_inter, sph.r_minor)
    d_an = search_distance(xp, aniso.r_major, aniso.r_inter, aniso.r_minor)
    assert float(np.max(d_an)) > float(np.max(d_sph))
    assert y_sph == pytest.approx(5.0)
    assert y_an == pytest.approx(5.0)

    # Asymmetric values + offset target: anisotropy must change weights ratio
    known_z2 = np.array([0.0, 10.0])
    known_xyz2 = np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    pred2 = np.array([[0.0, 0.0, 0.0]])
    y1 = anisotropic_idw_at_points(
        known_xyz2, known_z2, pred2, _vg(), sph, power
    )[0]
    y2 = anisotropic_idw_at_points(
        known_xyz2, known_z2, pred2, _vg(), aniso, power
    )[0]
    assert y1 != pytest.approx(y2)


def test_spherical_matches_euclidean_idw():
    known_xyz = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=float
    )
    known_z = np.array([1.0, 2.0, 3.0])
    pred = np.array([[0.0, 0.0, 0.0]])
    R = 10.0
    power = 1.5
    search = _search(r_major=R, k_inter=1.0, k_minor=1.0, n_min=1, n_max=3)
    got = anisotropic_idw_at_points(
        known_xyz, known_z, pred, _vg(), search, power
    )[0]
    # Euclidean d / R == anisotropic d when K=1
    d_euc = np.linalg.norm(known_xyz - pred[0], axis=1)
    d_an = d_euc / R
    w = d_an ** (-power)
    expected = float(np.dot(w, known_z) / np.sum(w))
    assert got == pytest.approx(expected)


def test_nmin_returns_nan():
    known_xyz = np.array([[1.0, 0.0, 0.0]])
    known_z = np.array([5.0])
    pred = np.array([[0.0, 0.0, 0.0]])
    search = _search(r_major=10.0, n_min=2, n_max=5)
    got = anisotropic_idw_at_points(
        known_xyz, known_z, pred, _vg(), search, 2.0
    )
    assert np.isnan(got[0])


def test_nmax_and_ellipsoid_boundary():
    # 5 samples on a line; Nmax=2 keeps nearest two
    known_xyz = np.array(
        [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]],
        dtype=float,
    )
    known_z = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    pred = np.array([[0.0, 0.0, 0.0]])
    search = _search(r_major=10.0, n_min=1, n_max=2)
    got = anisotropic_idw_at_points(
        known_xyz, known_z, pred, _vg(), search, 1.0
    )[0]
    # Only first two: d=0.1 and 0.2 → w=10 and 5
    expected = (10 * 10.0 + 5 * 20.0) / (10 + 5)
    assert got == pytest.approx(expected)

    # Outside ellipsoid → NaN
    search_small = _search(r_major=0.5, n_min=1, n_max=5)
    got2 = anisotropic_idw_at_points(
        known_xyz, known_z, pred, _vg(), search_small, 1.0
    )
    assert np.isnan(got2[0])


def test_zero_distance_returns_sample_value():
    known_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    known_z = np.array([7.5, 99.0])
    pred = np.array([[0.0, 0.0, 0.0]])
    search = _search(r_major=5.0, n_min=1, n_max=2)
    got = anisotropic_idw_at_points(
        known_xyz, known_z, pred, _vg(), search, 2.0
    )
    assert got[0] == pytest.approx(7.5)


def test_cv_no_test_leakage():
    n = 8
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.arange(n, dtype=float)
    values = np.arange(n, dtype=float)
    cloud = PointCloud(xyz=xyz, values=values)
    plan = CVPlan(
        method="manual",
        folds=[
            CVFold(
                train_idx=np.array([0, 1, 2, 3]),
                test_idx=np.array([4, 5, 6, 7]),
            )
        ],
        n_splits_requested=1,
    )
    search = _search(r_major=100.0, n_min=1, n_max=8)
    obj = ObjectiveConfig(coverage_min=0.0, coverage_penalty=0.0)

    # Monkeypatch-free check: evaluate fold manually — known must be train only
    from kriging3d_tuner.idw import anisotropic_idw_at_points as idw_fn

    pred = idw_fn(
        cloud.xyz[plan.folds[0].train_idx],
        cloud.values[plan.folds[0].train_idx],
        cloud.xyz[plan.folds[0].test_idx],
        _vg(),
        search,
        2.0,
    )
    # Predictions finite and not equal to test true values identity leak
    assert np.all(np.isfinite(pred))
    assert not np.allclose(pred, cloud.values[plan.folds[0].test_idx])

    m = evaluate_oof(
        cloud,
        plan,
        _vg(),
        search,
        obj,
        method=METHOD_ANISOTROPIC_IDW,
        power=2.0,
    )
    assert m.n_valid == 4


def test_seed_reproducibility(tmp_path: Path, monkeypatch):
    """Same seed → same best params for a tiny Optuna run."""
    rng = np.random.default_rng(0)
    n = 40
    xyz = rng.normal(size=(n, 3)) * 5.0
    values = xyz[:, 0] + 0.1 * rng.normal(size=n)
    data = tmp_path / "pts.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "X": xyz[:, 0],
            "Y": xyz[:, 1],
            "Z": xyz[:, 2],
            "Value": values,
            "HoleID": np.arange(n) % 4,
        }
    ).to_csv(data, index=False)

    aniso = {
        "variogram_model": "spherical",
        "nugget": 0.05,
        "sill": 1.0,
        "range_major": 8.0,
        "range_intermediate": 5.0,
        "range_minor": 3.0,
        "orientation_matrix": np.eye(3).tolist(),
    }
    aniso_path = tmp_path / "aniso.json"
    import json

    aniso_path.write_text(json.dumps(aniso), encoding="utf-8")

    cfg = {
        "n_min_points": 10,
        "seed": 123,
        "mode": "single",
        "method": "anisotropic_idw",
        "cv": {"n_splits": 2, "grid_nx": 2, "grid_ny": 2, "grid_nz": 1},
        "objective": {
            "coverage_min": 0.0,
            "coverage_penalty": 0.0,
            "empty_rmse_fallback": 1e6,
        },
        "search_space": {
            "r_lo": 0.5,
            "r_hi": 2.0,
            "k_min": 0.2,
            "nmin_lo": 2,
            "nmin_hi": 6,
            "nmax_lo": 2,
            "nmax_hi": 12,
        },
        "optuna": {
            "n_trials": 8,
            "n_startup_trials": 4,
            "n_ei_candidates": 16,
            "flush_every": 10,
        },
        "idw": {"power_min": 1.0, "power_max": 3.0},
        "octants": {"octant_search": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    a = run_tuner(data, aniso_path, out1, cfg_path, n_trials=8, method="anisotropic_idw")
    b = run_tuner(data, aniso_path, out2, cfg_path, n_trials=8, method="anisotropic_idw")
    assert a["power"] == pytest.approx(b["power"])
    assert a["R_major"] == pytest.approx(b["R_major"])
    assert a["Nmin"] == b["Nmin"]
    assert a["Nmax"] == b["Nmax"]
    assert a["CV_RMSE"] == pytest.approx(b["CV_RMSE"])


def test_ordinary_kriging_method_still_dispatched(monkeypatch, tmp_path: Path):
    called = {"ok": False, "idw": False}

    def fake_ok(*a, **k):
        called["ok"] = True
        return np.array([1.0])

    def fake_idw(*a, **k):
        called["idw"] = True
        return np.array([1.0])

    monkeypatch.setattr(
        "kriging3d_tuner.cv.ordinary_kriging_at_points", fake_ok
    )
    monkeypatch.setattr(
        "kriging3d_tuner.cv.anisotropic_idw_at_points", fake_idw
    )
    from kriging3d_tuner.cv import _predict_at_points

    cloud_xyz = np.zeros((2, 3))
    z = np.array([1.0, 2.0])
    pred = np.zeros((1, 3))
    _predict_at_points(
        cloud_xyz, z, pred, _vg(), _search(), method=METHOD_ORDINARY_KRIGING
    )
    assert called["ok"] and not called["idw"]
    called = {"ok": False, "idw": False}
    _predict_at_points(
        cloud_xyz,
        z,
        pred,
        _vg(),
        _search(),
        method=METHOD_ANISOTROPIC_IDW,
        power=2.0,
    )
    assert called["idw"] and not called["ok"]


def test_idw_rejects_non_single_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="mode=single"):
        run_tuner(
            tmp_path / "d.csv",
            tmp_path / "a.json",
            tmp_path / "out",
            method="anisotropic_idw",
            mode="multipass",
        )
