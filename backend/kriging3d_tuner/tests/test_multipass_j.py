"""Tests for sequential multipass Optuna minimizing loss J."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kriging3d_tuner.models import CVFold, CVPlan, PointCloud
from kriging3d_tuner.multipass_j import (
    build_point_fold_map,
    compute_J,
    compute_fold_lambdas,
)
from kriging3d_tuner.oof_store import OOFStore
from kriging3d_tuner.pipeline import run_tuner


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


def test_compute_fold_lambdas_train_only():
    # Fold0 train = [2,3] → values 10, 12; Fold1 train = [0,1] → 0, 2
    values = np.array([0.0, 2.0, 10.0, 12.0])
    cloud = PointCloud(xyz=np.zeros((4, 3)), values=values)
    plan = _plan_two_fold(4)
    lambdas = compute_fold_lambdas(cloud, plan)
    assert lambdas[0] == pytest.approx(float(np.var([10.0, 12.0], ddof=1)))
    assert lambdas[1] == pytest.approx(float(np.var([0.0, 2.0], ddof=1)))
    # Test points must not affect λ
    assert lambdas[0] != pytest.approx(float(np.var(values, ddof=1)))


def test_compute_fold_lambdas_degenerate():
    values = np.array([5.0, 5.0, 5.0])
    cloud = PointCloud(xyz=np.zeros((3, 3)), values=values)
    plan = CVPlan(
        method="manual",
        folds=[
            CVFold(train_idx=np.array([0]), test_idx=np.array([1, 2])),
        ],
        n_splits_requested=1,
    )
    lambdas = compute_fold_lambdas(cloud, plan)
    assert lambdas[0] == pytest.approx(max(1e-12, 25.0))


def test_compute_J_estimated_plus_lambda_unevaluated():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    store = OOFStore(n_tgt=4, y_true=y)
    # Estimate first two: errors 1 and 2 → SE = 1+4=5
    store.merge(
        np.array([0, 1]),
        np.array([1.0, 2.0]),
        np.array([2, 2]),
        pass_no=1,
    )
    oof = np.arange(4)
    point_fold = {0: 0, 1: 0, 2: 1, 3: 1}
    lambdas = {0: 3.0, 1: 7.0}
    # J = (1 + 4 + 7 + 7) / 4 = 19/4
    j = compute_J(store, oof, point_fold, lambdas)
    assert j == pytest.approx(19.0 / 4.0)


def test_merge_never_overwrite_between_passes():
    y = np.arange(5, dtype=float)
    store = OOFStore(n_tgt=5, y_true=y)
    store.merge(np.array([0, 1]), np.array([10.0, 20.0]), np.array([3, 4]), 1)
    n2 = store.merge(np.array([1, 2]), np.array([99.0, 30.0]), np.array([9, 5]), 2)
    assert n2 == 1
    assert store.pred[1] == 20.0
    assert store.pass_id[1] == 1


def test_reject_pass_restores_store_and_stops():
    """Pass2 with worse J → store rolled back, stop_reason=no_j_improvement."""
    n = 12
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n, 3))
    values = xyz[:, 0] + 0.05 * rng.normal(size=n)
    cloud = PointCloud(xyz=xyz, values=values, hole_ids=np.arange(n) % 3)
    plan = _plan_two_fold(n)
    oof = np.arange(n)
    point_fold = build_point_fold_map(plan)
    lambdas = compute_fold_lambdas(cloud, plan)
    store = OOFStore(n_tgt=n, y_true=values.copy())

    half = np.arange(n // 2)
    store.merge(half, values[half].copy(), np.full(n // 2, 4), pass_no=1)
    j1 = compute_J(store, oof, point_fold, lambdas)
    snap = (store.pred.copy(), store.pass_id.copy(), store.n_used.copy())

    rem = np.arange(n // 2, n)
    store.merge(rem, values[rem] + 100.0, np.full(n - n // 2, 4), pass_no=2)
    j2 = compute_J(store, oof, point_fold, lambdas)
    assert j2 > j1

    if not (j2 < j1):
        store.pred[:] = snap[0]
        store.pass_id[:] = snap[1]
        store.n_used[:] = snap[2]
        stop = "no_j_improvement"
    else:
        stop = "accepted"

    assert stop == "no_j_improvement"
    assert store.n_valid == n // 2
    assert np.all(store.pass_id[rem] == 0)
    assert compute_J(store, oof, point_fold, lambdas) == pytest.approx(j1)


def test_stop_at_full_coverage_skips_extra_passes():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    store = OOFStore(n_tgt=4, y_true=y)
    store.merge(np.arange(4), y.copy(), np.ones(4, dtype=int), pass_no=1)
    assert store.coverage >= 1.0 - 1e-15
    max_passes = 3
    accepted = 1
    stop_reason = "coverage_100"
    next_pass = 2 if store.coverage < 1.0 - 1e-15 else None
    assert next_pass is None
    assert accepted < max_passes
    assert stop_reason == "coverage_100"


def test_pipeline_dispatches_j_mode(monkeypatch, tmp_path: Path):
    sentinel = {"ok": False}

    def fake_run(*a, **k):
        sentinel["ok"] = True
        return {"mode": "sequential_multipass_optuna_J", "J": 1.0}

    monkeypatch.setattr(
        "kriging3d_tuner.multipass_j.run_sequential_multipass_optuna_j",
        fake_run,
    )
    monkeypatch.setattr(
        "kriging3d_tuner.pipeline.load_config",
        lambda _p=None: {"mode": "single", "multipass": {}},
    )
    out = run_tuner("d.csv", "a.json", tmp_path, mode="sequential_multipass_optuna_J")
    assert sentinel["ok"]
    assert out["mode"] == "sequential_multipass_optuna_J"


def test_single_mode_does_not_call_multipass_j(monkeypatch, tmp_path: Path):
    """mode=single must not enter sequential_multipass_optuna_J."""
    j_called = {"v": False}

    def boom(*a, **k):
        j_called["v"] = True
        raise AssertionError("multipass_j must not run in single mode")

    monkeypatch.setattr(
        "kriging3d_tuner.multipass_j.run_sequential_multipass_optuna_j",
        boom,
    )
    monkeypatch.setattr(
        "kriging3d_tuner.pipeline.load_config",
        lambda _p=None: {
            "mode": "single",
            "multipass": {},
            "seed": 1,
            "objective": {},
            "optuna": {"n_trials": 1, "n_ei_candidates": 8, "flush_every": 1},
            "search_space": {},
            "cv": {"n_splits": 2},
        },
    )

    def fake_load_points(*a, **k):
        raise RuntimeError("stop_after_dispatch_ok")

    monkeypatch.setattr(
        "kriging3d_tuner.pipeline.load_points_csv",
        fake_load_points,
    )
    with pytest.raises(RuntimeError, match="stop_after_dispatch_ok"):
        run_tuner("d.csv", "a.json", tmp_path, mode="single")
    assert j_called["v"] is False
