"""Optuna objective factory."""

from __future__ import annotations

from typing import Callable

from .cv import evaluate_oof
from .models import (
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    SearchSpaceConfig,
    TrialMetrics,
    VariogramFixed,
)
from .search_space import suggest_search_params


def make_objective(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    obj_cfg: ObjectiveConfig,
) -> Callable[[SearchParams], TrialMetrics]:
    def _eval(search: SearchParams) -> TrialMetrics:
        return evaluate_oof(cloud, plan, vg, search, obj_cfg)

    return _eval


def make_optuna_objective(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    obj_cfg: ObjectiveConfig,
) -> Callable:
    """Return Optuna objective; stores last metrics on trial.user_attrs."""
    eval_fn = make_objective(cloud, plan, vg, space, obj_cfg)

    def _objective(trial) -> float:
        params = suggest_search_params(trial, space)
        metrics = eval_fn(params)
        trial.set_user_attr("CV_RMSE", metrics.cv_rmse)
        trial.set_user_attr("CV_MAE", metrics.cv_mae)
        trial.set_user_attr("prediction_coverage", metrics.prediction_coverage)
        trial.set_user_attr("n_valid", metrics.n_valid)
        trial.set_user_attr("n_invalid", metrics.n_invalid)
        trial.set_user_attr("n_tgt", metrics.n_tgt)
        trial.set_user_attr("R_inter", params.r_inter)
        trial.set_user_attr("R_minor", params.r_minor)
        return float(metrics.objective)

    return _objective
