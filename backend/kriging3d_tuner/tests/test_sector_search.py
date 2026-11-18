"""Micromine-style sector ids (4/8/16) and IDW Optuna conditional space."""

from __future__ import annotations

import numpy as np
import optuna

from kriging3d_tuner.kriging import _sector_ids, select_neighbour_indices
from kriging3d_tuner.models import SearchParams, SearchSpaceConfig, sector_neighbour_cap
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
        octant_search=True,
        minoct=1,
        minperoc=1,
        maxperoc=10**9,
        n_sectors=8,
        discard_underfilled=False,
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


def test_sector_ids_4_plan_quadrants_ignore_z():
    xp = np.array(
        [
            [1.0, 1.0, 5.0],
            [1.0, -1.0, -9.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    ids = _sector_ids(xp, 4)
    # ox + 2*oy: (++)=3, (+-)=1, (-+)=2, (--)=0
    np.testing.assert_array_equal(ids, [3, 1, 2, 0])
    # Same plan signs → same quadrant regardless of Z
    assert _sector_ids(np.array([[1.0, 1.0, 1.0]]), 4)[0] == _sector_ids(
        np.array([[1.0, 1.0, -1.0]]), 4
    )[0]


def test_sector_ids_8_matches_octants():
    xp = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    ids = _sector_ids(xp, 8)
    np.testing.assert_array_equal(ids, [1 + 2 + 4, 1 + 2, 0])


def test_sector_ids_16_wedges_and_hemisphere():
    # +X axis → wedge 0; upper vs lower by z
    upper = _sector_ids(np.array([[1.0, 0.0, 1.0]]), 16)[0]
    lower = _sector_ids(np.array([[1.0, 0.0, -1.0]]), 16)[0]
    assert upper == 0
    assert lower == 8
    # +Y (CCW 90°) is clockwise 270° from +X → wedge 6
    assert _sector_ids(np.array([[0.0, 1.0, 1.0]]), 16)[0] == 6
    # -Y (CW 90°) → wedge 2
    assert _sector_ids(np.array([[0.0, -1.0, 1.0]]), 16)[0] == 2


def test_micromine_discard_underfilled_sector():
    """Sector with 1 sample and MINPEROC=2 contributes nothing when discarding."""
    Q = np.eye(3)
    # Two filled quadrants (≥2) + one singleton underfilled
    xyz = np.array(
        [
            [2.0, 2.0, 0.0],
            [2.5, 2.5, 0.0],
            [-2.0, -2.0, 0.0],
            [-2.5, -2.5, 0.0],
            [2.0, -2.0, 0.0],  # singleton quadrant
        ],
        dtype=float,
    )
    keep = select_neighbour_indices(
        xyz,
        Q,
        _search(
            n_sectors=4,
            discard_underfilled=True,
            minoct=2,
            minperoc=2,
            maxperoc=2,
            n_max=8,
        ),
    )
    assert 4 not in set(keep.tolist())
    assert keep.size == 4

    legacy = select_neighbour_indices(
        xyz,
        Q,
        _search(
            n_sectors=4,
            discard_underfilled=False,
            minoct=2,
            minperoc=2,
            maxperoc=2,
            n_max=8,
        ),
    )
    assert 4 in set(legacy.tolist())


def test_min_sectors_not_met_returns_empty():
    Q = np.eye(3)
    xyz = np.array([[1.0, 1.0, 0.0], [1.5, 1.2, 0.0]], dtype=float)
    idx = select_neighbour_indices(
        xyz,
        Q,
        _search(
            n_sectors=4,
            discard_underfilled=True,
            minoct=3,
            minperoc=1,
            maxperoc=5,
            n_max=10,
        ),
    )
    assert idx.size == 0


def test_neighbour_cap_uses_n_sectors():
    assert sector_neighbour_cap(4, 3) == 12
    assert sector_neighbour_cap(16, 2) == 32
    p = _search(n_sectors=16, maxperoc=3, minoct=1, minperoc=1, n_max=20)
    assert p.as_dict()["neighbour_cap"] == 48
    assert p.as_dict()["n_sectors"] == 16
    assert p.as_dict()["sector_search"] is True


def test_optuna_idw_sector_branch_suggests_n_sectors():
    space = _space()

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
                return self._inner.suggest_categorical(name, [self._forced])
            return self._inner.suggest_categorical(name, choices)

        @property
        def params(self):
            return self._inner.params

    def obj(trial: optuna.Trial) -> float:
        params = suggest_search_params(
            _Forced(trial, True),  # type: ignore[arg-type]
            space,
            allow_variable_sectors=True,
        )
        assert params.octant_search is True
        assert params.n_sectors in (4, 8, 16)
        assert params.discard_underfilled is True
        assert "n_sectors" in trial.params
        assert "Nmin" not in trial.params
        assert "Nmax" not in trial.params
        assert params.n_min == params.minoct * params.minperoc
        assert params.n_max == sector_neighbour_cap(
            params.n_sectors, params.maxperoc
        )
        assert params.minoct <= params.n_sectors
        rebuilt = search_params_from_trial_params(dict(trial.params))
        assert rebuilt.n_sectors == params.n_sectors
        assert rebuilt.discard_underfilled is True
        assert rebuilt.n_min == params.n_min
        assert rebuilt.n_max == params.n_max
        return 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=12)


def test_optuna_ok_sector_branch_keeps_n_sectors_8():
    space = _space()

    def obj(trial: optuna.Trial) -> float:
        params = suggest_search_params(
            trial, space, force_octant_search=True, allow_variable_sectors=False
        )
        assert params.n_sectors == 8
        assert params.discard_underfilled is False
        assert "n_sectors" not in trial.params
        return 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=3)
