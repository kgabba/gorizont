"""CV fold construction smoke tests."""

from __future__ import annotations

import numpy as np

from aniso_ok_tuner_3d.cv import (
    default_holdout_size,
    make_buffered_delete_d_folds,
    make_spatial_block_splits,
)


def test_spatial_block_3d_produces_folds():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, size=(40, 3))
    folds = make_spatial_block_splits(xyz, grid_nx=2, grid_ny=2, grid_nz=2)
    assert len(folds) >= 2
    n = xyz.shape[0]
    for test in folds:
        assert 0 < test.size < n


def test_buffered_delete_d_aniso_folds():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 10, size=(30, 3))
    folds = make_buffered_delete_d_folds(
        xyz,
        holdout_size=default_holdout_size(30),
        n_repeats=5,
        buffer_radius=2.0,
        seed=1,
        anisotropic=True,
        azimuth_deg=30.0,
        dip_deg=10.0,
        a_major=5.0,
        a_inter=3.0,
        a_minor=2.0,
    )
    assert len(folds) >= 1
    for train, hold in folds:
        assert train.size > 0
        assert hold.size > 0
        assert len(np.intersect1d(train, hold)) == 0
