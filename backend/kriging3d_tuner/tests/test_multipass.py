"""Tests for three-pass OOF store, objectives, and masked evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from kriging3d_tuner.cv import evaluate_oof, evaluate_oof_masked, oof_target_indices
from kriging3d_tuner.models import (
    CVFold,
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    VariogramFixed,
    compute_objective,
)
from kriging3d_tuner.oof_store import (
    OOFStore,
    lexico_coverage_rmse_objective,
    risky_extrapolation,
)


def _vg() -> VariogramFixed:
    return VariogramFixed(
        variogram_model="spherical",
        nugget=0.1,
        sill=1.0,
        range_major=5.0,
        range_intermediate=5.0,
        range_minor=5.0,
        orientation_matrix=np.eye(3),
    )


def _cloud(n: int = 20, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)) * 2.0
    # Two hole groups for GroupKFold-like splits we build manually
    holes = np.array([0] * (n // 2) + [1] * (n - n // 2))
    values = xyz[:, 0] + 0.1 * rng.normal(size=n)
    return PointCloud(xyz=xyz, values=values, hole_ids=holes)


def _plan_two_fold(n: int) -> CVPlan:
    mid = n // 2
    return CVPlan(
        method="manual",
        folds=[
            CVFold(
                train_idx=np.arange(mid, n),
                test_idx=np.arange(0, mid),
            ),
            CVFold(
                train_idx=np.arange(0, mid),
                test_idx=np.arange(mid, n),
            ),
        ],
        n_splits_requested=2,
    )


def _search(**kw) -> SearchParams:
    base = dict(
        r_major=10.0,
        k_inter=1.0,
        k_minor=1.0,
        n_min=2,
        n_max=8,
        octant_search=False,
    )
    base.update(kw)
    return SearchParams(**base)


def test_merge_never_overwrites():
    y = np.arange(5, dtype=float)
    store = OOFStore(n_tgt=5, y_true=y)
    store.merge(
        np.array([0, 1]),
        np.array([10.0, 20.0]),
        np.array([3, 4]),
        pass_no=1,
    )
    n2 = store.merge(
        np.array([1, 2]),
        np.array([99.0, 30.0]),
        np.array([9, 5]),
        pass_no=2,
    )
    assert n2 == 1  # only index 2
    assert store.pred[1] == 20.0
    assert store.pass_id[1] == 1
    assert store.pass_id[2] == 2
    assert store.coverage == pytest.approx(3 / 5)


def test_coverage_relative_to_n_tgt_not_residual():
    store = OOFStore(n_tgt=100, y_true=np.zeros(100))
    store.merge(np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0]), np.array([1, 1, 1]), 1)
    assert store.coverage == pytest.approx(0.03)


def test_lexico_higher_coverage_beats_lower_rmse():
    # coverage 0.9 with RMSE 10 vs coverage 0.5 with RMSE 0.01
    a = lexico_coverage_rmse_objective(0.9, 10.0)
    b = lexico_coverage_rmse_objective(0.5, 0.01)
    assert a < b
    # equal coverage → lower RMSE wins
    c = lexico_coverage_rmse_objective(0.9, 1.0)
    d = lexico_coverage_rmse_objective(0.9, 2.0)
    assert c < d


def test_gate_objective_low_coverage_worse():
    cfg = ObjectiveConfig(coverage_min=0.30, coverage_penalty=1e6, empty_rmse_fallback=1e6)
    ok = compute_objective(
        cv_rmse=0.8, prediction_coverage=0.35, n_valid=35, cfg=cfg
    )
    bad = compute_objective(
        cv_rmse=0.1, prediction_coverage=0.10, n_valid=10, cfg=cfg
    )
    assert ok < bad


def test_pooled_rmse_not_mean_of_pass_rmses():
    store = OOFStore(n_tgt=4, y_true=np.array([0.0, 0.0, 0.0, 0.0]))
    # pass1: two points error 0
    store.merge(np.array([0, 1]), np.array([0.0, 0.0]), np.array([2, 2]), 1)
    # pass2: two points error 3 and 4 → pass-local RMSE = 3.535
    store.merge(np.array([2, 3]), np.array([3.0, 4.0]), np.array([2, 2]), 2)
    rmse_all, _ = store.rmse_mae_all()
    local1 = store.pass_local_metrics(1)["CV_RMSE"]
    local2 = store.pass_local_metrics(2)["CV_RMSE"]
    mean_locals = 0.5 * (local1 + local2)
    assert rmse_all == pytest.approx(np.sqrt((0 + 0 + 9 + 16) / 4))
    assert abs(rmse_all - mean_locals) > 0.1


def test_evaluate_oof_masked_only_candidates():
    cloud = _cloud(16)
    plan = _plan_two_fold(16)
    vg = _vg()
    search = _search()
    # Only first half of OOF targets as candidates
    oof = oof_target_indices(plan)
    assert oof.size == 16
    cand = oof[:4]
    idx, preds, n_used = evaluate_oof_masked(cloud, plan, vg, search, cand)
    assert set(idx.tolist()) <= set(cand.tolist())
    assert idx.size == cand.size
    assert preds.shape == idx.shape
    assert n_used.shape == idx.shape


def test_masked_then_merge_matches_full_when_all_candidates():
    cloud = _cloud(12)
    plan = _plan_two_fold(12)
    vg = _vg()
    search = _search(n_min=2, n_max=6)
    obj = ObjectiveConfig(coverage_min=0.0, coverage_penalty=0.0)
    full = evaluate_oof(cloud, plan, vg, search, obj)
    oof = oof_target_indices(plan)
    store = OOFStore(n_tgt=int(oof.size), y_true=cloud.values.copy())
    idx, preds, n_used = evaluate_oof_masked(cloud, plan, vg, search, oof)
    store.merge(idx, preds, n_used, pass_no=1)
    rmse, mae = store.rmse_mae_all()
    assert store.n_valid == full.n_valid
    assert store.coverage == pytest.approx(full.prediction_coverage)
    assert rmse == pytest.approx(full.cv_rmse, rel=1e-9)
    assert mae == pytest.approx(full.cv_mae, rel=1e-9)


def test_risky_extrapolation_flag():
    assert risky_extrapolation(11.0, 5.0, factor=2.0) is True
    assert risky_extrapolation(9.0, 5.0, factor=2.0) is False


def test_pass2_cannot_see_already_estimated(monkeypatch):
    """evaluate_oof_masked must not request OK for non-candidates."""
    cloud = _cloud(10)
    plan = _plan_two_fold(10)
    vg = _vg()
    search = _search()
    called_targets: list[int] = []

    import kriging3d_tuner.cv as cv_mod

    real_ok = cv_mod.ordinary_kriging_at_points

    def wrap(known_xyz, known_z, pred_xyz, vg_, search_):
        # Map pred rows back is hard; just count calls and sizes
        out = real_ok(known_xyz, known_z, pred_xyz, vg_, search_)
        called_targets.append(int(pred_xyz.shape[0]))
        return out

    monkeypatch.setattr(cv_mod, "ordinary_kriging_at_points", wrap)
    # Only 2 candidates in fold0 test set
    cand = plan.folds[0].test_idx[:2]
    evaluate_oof_masked(cloud, plan, vg, search, cand)
    # Each fold call should only predict candidates in that fold
    assert sum(called_targets) == 2
