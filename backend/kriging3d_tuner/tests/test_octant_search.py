"""Unit tests for octant neighbour selection and conditional Optuna space."""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from kriging3d_tuner.kriging import select_neighbour_indices
from kriging3d_tuner.models import (
    ObjectiveConfig,
    SearchParams,
    SearchSpaceConfig,
    VariogramFixed,
    compute_objective,
    octant_neighbour_cap,
)
from kriging3d_tuner.search_space import (
    search_params_from_trial_params,
    suggest_search_params,
)


def _search(**kwargs) -> SearchParams:
    base = dict(
        r_major=10.0,
        k_inter=1.0,
        k_minor=1.0,
        n_min=1,
        n_max=64,
        octant_search=False,
        minoct=1,
        minperoc=1,
        maxperoc=10**9,
    )
    base.update(kwargs)
    return SearchParams(**base)


def _space() -> SearchSpaceConfig:
    return SearchSpaceConfig(
        r_major_min=10.0,
        r_major_max=100.0,
        k_min=0.05,
        nmin_lo=4,
        nmin_hi=12,
        nmax_lo=4,
        nmax_hi=64,
        minoct_lo=1,
        minoct_hi=4,
        minperoc_lo=1,
        minperoc_hi=3,
        maxperoc_lo=1,
        maxperoc_hi=4,
    )


def test_octant_off_matches_legacy_nmax_cap():
    """With octants off, keep nearest Nmax by d_search (legacy path)."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(40, 3)) * 3.0
    Q = np.eye(3)
    search = _search(n_max=5, octant_search=False)
    idx = select_neighbour_indices(xyz, Q, search)
    assert idx.size == 5
    d = np.linalg.norm(xyz, axis=1) / 10.0
    inside = np.where(d <= 1.0 + 1e-12)[0]
    order = np.lexsort((inside, d[inside]))
    expected = inside[order[:5]]
    np.testing.assert_array_equal(idx, expected)


def test_uneven_cluster_maxperoc_spreads_selection():
    """Dense NE cluster must not monopolise neighbours when MAXPEROC is small."""
    Q = np.eye(3)
    ne = np.array([[0.4, 0.4, 0.4 + 0.01 * i] for i in range(12)], dtype=float)
    others = np.array(
        [
            [-3.0, -3.0, -3.0],
            [-3.0, 3.0, -3.0],
            [3.0, -3.0, -3.0],
            [-3.0, -3.0, 3.0],
            [3.0, 3.0, -3.0],
            [-3.0, 3.0, 3.0],
            [3.0, -3.0, 3.0],
        ],
        dtype=float,
    )
    xyz = np.vstack([ne, others])

    off = select_neighbour_indices(
        xyz, Q, _search(n_max=8, octant_search=False)
    )
    assert set(off.tolist()).issubset(set(range(12)))

    on = select_neighbour_indices(
        xyz,
        Q,
        _search(
            n_max=16,
            octant_search=True,
            minoct=4,
            minperoc=1,
            maxperoc=2,
        ),
    )
    n_ne = sum(1 for i in on.tolist() if i < 12)
    assert n_ne <= 2
    assert on.size >= 4
    assert on.size <= min(16, octant_neighbour_cap(2))


def test_nmax_minperoc_conflict_returns_empty():
    """Cannot satisfy Nmax while keeping MINPEROC in enough octants."""
    Q = np.eye(3)
    samples = []
    signs = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ]
    for sx, sy, sz in signs:
        samples.append([sx * 2.0, sy * 2.0, sz * 2.0])
        samples.append([sx * 2.5, sy * 2.5, sz * 2.5])
    xyz = np.asarray(samples, dtype=float)

    # MINOCT=4, MINPEROC=2 → need all 8; Nmax=4 cannot keep 2 per 4 octants
    idx = select_neighbour_indices(
        xyz,
        Q,
        _search(
            octant_search=True,
            minoct=4,
            minperoc=2,
            maxperoc=2,
            n_max=4,
            n_min=1,
        ),
    )
    assert idx.size == 0


def test_octant_mode_applies_global_nmax():
    """After MAXPEROC pooling, global Nmax trims furthest with MINPEROC protection."""
    Q = np.eye(3)
    samples = []
    signs = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ]
    for sx, sy, sz in signs:
        samples.append([sx * 2.0, sy * 2.0, sz * 2.0])
        samples.append([sx * 2.5, sy * 2.5, sz * 2.5])
    xyz = np.asarray(samples, dtype=float)

    idx = select_neighbour_indices(
        xyz,
        Q,
        _search(
            octant_search=True,
            minoct=4,
            minperoc=1,
            maxperoc=2,
            n_max=6,
            n_min=1,
        ),
    )
    assert idx.size == 6


def test_rotated_ellipsoid_octants_use_principal_axes():
    from kriging3d_tuner.geometry import to_principal
    from kriging3d_tuner.kriging import _octant_ids

    xyz = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0]], dtype=float)
    c0 = np.array([1.0, 1.0, 0.0])
    c0 /= np.linalg.norm(c0)
    c1 = np.array([-1.0, 1.0, 0.0])
    c1 /= np.linalg.norm(c1)
    c2 = np.array([0.0, 0.0, 1.0])
    Q = np.column_stack([c0, c1, c2])
    assert np.linalg.det(Q) > 0

    id_oct = _octant_ids(to_principal(xyz, np.eye(3)))
    rot_oct = _octant_ids(to_principal(xyz, Q))
    assert id_oct[0] == id_oct[1]
    assert rot_oct[0] != rot_oct[1]

    id_sel = select_neighbour_indices(
        xyz,
        np.eye(3),
        _search(octant_search=True, minoct=1, minperoc=1, maxperoc=1, n_max=2),
    )
    rot_sel = select_neighbour_indices(
        xyz,
        Q,
        _search(octant_search=True, minoct=1, minperoc=1, maxperoc=1, n_max=2),
    )
    assert id_sel.size == 1
    assert rot_sel.size == 2


def test_minoct_not_met_returns_empty():
    Q = np.eye(3)
    xyz = np.array([[1.0, 1.0, 1.0], [1.5, 1.2, 1.1], [2.0, 1.0, 1.3]], dtype=float)
    idx = select_neighbour_indices(
        xyz,
        Q,
        _search(
            octant_search=True,
            minoct=3,
            minperoc=1,
            maxperoc=5,
            n_max=10,
        ),
    )
    assert idx.size == 0


def test_on_plane_samples_assigned_to_positive_halfspace():
    Q = np.eye(3)
    xyz = np.array(
        [
            [2.0, 0.0, 2.0],
            [-2.0, 0.0, -2.0],
        ],
        dtype=float,
    )
    idx = select_neighbour_indices(
        xyz,
        Q,
        _search(
            octant_search=True,
            minoct=2,
            minperoc=1,
            maxperoc=1,
            n_max=2,
        ),
    )
    assert idx.size == 2


def test_ok_nan_when_octants_fail():
    from kriging3d_tuner.kriging import ordinary_kriging_at_points

    Q = np.eye(3)
    known = np.array([[1.0, 1.0, 1.0], [1.2, 1.1, 1.0]], dtype=float)
    z = np.array([1.0, 1.1])
    pred = np.array([[0.0, 0.0, 0.0]])
    vg = VariogramFixed(
        variogram_model="spherical",
        nugget=0.1,
        sill=1.0,
        range_major=5.0,
        range_intermediate=5.0,
        range_minor=5.0,
        orientation_matrix=Q,
    )
    search = _search(
        r_major=10.0,
        octant_search=True,
        minoct=4,
        minperoc=1,
        maxperoc=2,
        n_min=1,
        n_max=8,
    )
    out = ordinary_kriging_at_points(known, z, pred, vg, search)
    assert np.isnan(out[0])


# --- Optuna conditional space -------------------------------------------------


def test_optuna_legacy_branch_has_nmax_no_octant_params():
    space = _space()

    def obj(trial: optuna.Trial) -> float:
        params = _suggest_forced(trial, space, octant=False)
        assert params.octant_search is False
        assert "Nmax" in trial.params
        assert "MINOCT" not in trial.params
        assert "MINPEROC" not in trial.params
        assert "MAXPEROC" not in trial.params
        assert params.n_min <= params.n_max
        rebuilt = search_params_from_trial_params(dict(trial.params))
        assert rebuilt.octant_search is False
        assert rebuilt.as_dict()["Nmax"] == rebuilt.n_max
        return 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=5)


def _suggest_forced(
    trial: optuna.Trial, space: SearchSpaceConfig, *, octant: bool
) -> SearchParams:
    """Suggest params while forcing the octant categorical via distributions."""
    # Use the real suggest_search_params but first register octant_search
    # by calling suggest_categorical with a single allowed value through
    # a FixedTrial-like approach: run suggest_search_params on a study that
    # only ever sees one branch via relative sampling.
    # Simpler: duplicate the branch logic with forced flag for the test helper,
    # then assert the public function via FixedTrial.

    class _Forced:
        def __init__(self, inner: optuna.Trial, forced: bool):
            self._inner = inner
            self._forced = forced

        def suggest_float(self, name, low, high, **kw):
            return self._inner.suggest_float(name, low, high, **kw)

        def suggest_int(self, name, low, high, **kw):
            return self._inner.suggest_int(name, low, high, **kw)

        def suggest_categorical(self, name, choices):
            if name == "octant_search":
                # Register the forced value as the only choice
                return self._inner.suggest_categorical(name, [self._forced])
            return self._inner.suggest_categorical(name, choices)

        @property
        def params(self):
            return self._inner.params

    return suggest_search_params(_Forced(trial, octant), space)  # type: ignore[arg-type]


def test_optuna_octant_branch_implies_nmin_nmax():
    space = _space()

    def obj(trial: optuna.Trial) -> float:
        params = _suggest_forced(trial, space, octant=True)
        assert params.octant_search is True
        assert "Nmax" not in trial.params
        assert "Nmin" not in trial.params
        assert "MAXPEROC" in trial.params
        assert "MINPEROC" in trial.params
        assert "MINOCT" in trial.params
        assert params.minperoc <= params.maxperoc
        assert params.n_min == params.minoct * params.minperoc
        assert params.n_max == octant_neighbour_cap(params.maxperoc)
        assert params.n_min <= params.n_max
        rebuilt = search_params_from_trial_params(dict(trial.params))
        assert rebuilt.octant_search is True
        assert rebuilt.n_min == params.n_min
        assert rebuilt.n_max == params.n_max
        assert rebuilt.as_dict()["neighbour_cap"] == 8 * rebuilt.maxperoc
        return 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=8)


def test_optuna_both_branches_examples():
    """Smoke: one legacy and one octant sample look as documented."""
    space = _space()
    legacy_params: dict = {}
    octant_params: dict = {}

    def make_obj(forced: bool):
        def obj(trial: optuna.Trial) -> float:
            p = _suggest_forced(trial, space, octant=forced)
            if forced:
                octant_params.update(dict(trial.params))
                assert p.octant_search
            else:
                legacy_params.update(dict(trial.params))
                assert not p.octant_search
            return float(p.r_major)

        return obj

    # Separate studies: Optuna forbids changing categorical choice sets mid-study
    optuna.create_study(direction="minimize").optimize(make_obj(False), n_trials=1)
    optuna.create_study(direction="minimize").optimize(make_obj(True), n_trials=1)

    assert set(legacy_params) >= {
        "octant_search",
        "R_major",
        "K_inter",
        "K_minor",
        "Nmin",
        "Nmax",
    }
    assert "MINOCT" not in legacy_params
    assert set(octant_params) >= {
        "octant_search",
        "R_major",
        "K_inter",
        "K_minor",
        "MINOCT",
        "MINPEROC",
        "MAXPEROC",
    }
    assert "Nmin" not in octant_params
    assert "Nmax" not in octant_params


def test_coverage_penalty_ranks_low_coverage_worse():
    cfg = ObjectiveConfig(
        coverage_min=0.97, coverage_penalty=1.0e6, empty_rmse_fallback=1.0e6
    )
    ok = compute_objective(
        cv_rmse=0.9, prediction_coverage=0.98, n_valid=100, cfg=cfg
    )
    bad = compute_objective(
        cv_rmse=0.1, prediction_coverage=0.50, n_valid=50, cfg=cfg
    )
    empty = compute_objective(
        cv_rmse=0.0, prediction_coverage=0.0, n_valid=0, cfg=cfg
    )
    assert ok < bad
    assert ok < empty
    assert bad < empty
    shortfall = 0.97 - 0.50
    assert bad == pytest.approx(0.1 + 1.0e6 * shortfall)


def test_as_dict_includes_nmax_in_octant_mode():
    p = _search(
        octant_search=True, minoct=2, minperoc=2, maxperoc=4, n_min=6, n_max=12
    )
    d = p.as_dict()
    assert d["Nmax"] == 12
    assert d["neighbour_cap"] == 32
    p2 = _search(octant_search=False, n_max=16)
    assert p2.as_dict()["Nmax"] == 16
    assert "neighbour_cap" not in p2.as_dict()
