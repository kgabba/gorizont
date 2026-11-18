"""Optuna objective factory."""

from __future__ import annotations

from typing import Callable

from .cv import (
    METHOD_ANISOTROPIC_IDW,
    METHOD_ORDINARY_KRIGING,
    evaluate_oof,
    evaluate_oof_masked,
    oof_target_indices,
)
from .models import (
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    SearchSpaceConfig,
    TrialMetrics,
    VariogramFixed,
)
from .oof_store import OOFStore
from .search_space import suggest_search_params


def make_objective(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    obj_cfg: ObjectiveConfig,
    *,
    method: str = METHOD_ORDINARY_KRIGING,
    power: float | None = None,
) -> Callable[[SearchParams], TrialMetrics]:
    def _eval(search: SearchParams) -> TrialMetrics:
        return evaluate_oof(
            cloud, plan, vg, search, obj_cfg, method=method, power=power
        )

    return _eval


def make_optuna_objective(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    obj_cfg: ObjectiveConfig,
    *,
    force_octant_search: bool | None = None,
    method: str = METHOD_ORDINARY_KRIGING,
    power_min: float = 1.0,
    power_max: float = 4.0,
    loss: str = "rmse",
) -> Callable:
    """Return Optuna objective; stores last metrics on trial.user_attrs.

    Suggests a conditional neighbourhood (legacy ``Nmax`` vs sector/octant
    controls). For ``anisotropic_idw`` also suggests ``power`` and allows
    Micromine-style ``n_sectors ∈ {4, 8, 16}`` when sector search is on.

    ``loss``:
      - ``rmse`` (default): coverage-gated RMSE via ``compute_objective``
      - ``j``: pooled J = (Σ SE estimated + Σ λ_f unevaluated) / N
    """
    method_eff = (method or METHOD_ORDINARY_KRIGING).strip().lower()
    allow_variable_sectors = method_eff == METHOD_ANISOTROPIC_IDW

    loss_eff = (loss or "rmse").strip().lower()
    if loss_eff not in ("rmse", "j"):
        raise ValueError(f"unknown loss {loss!r}; expected 'rmse' or 'j'")

    oof_indices = oof_target_indices(plan)
    lambdas: dict = {}
    point_fold: dict = {}
    if loss_eff == "j":
        # Lazy import avoids circular import with multipass_j → objective
        from .multipass_j import (
            build_point_fold_map,
            compute_J,
            compute_fold_lambdas,
        )

        lambdas = compute_fold_lambdas(cloud, plan)
        point_fold = build_point_fold_map(plan)

    def _objective(trial) -> float:
        params = suggest_search_params(
            trial,
            space,
            force_octant_search=force_octant_search,
            allow_variable_sectors=allow_variable_sectors,
        )
        power: float | None = None
        if method_eff == METHOD_ANISOTROPIC_IDW:
            p_lo = float(min(power_min, power_max))
            p_hi = float(max(power_min, power_max))
            power = float(trial.suggest_float("power", p_lo, p_hi))

        if loss_eff == "j":
            store = OOFStore(n_tgt=int(oof_indices.size), y_true=cloud.values.copy())
            idx, preds, n_used = evaluate_oof_masked(
                cloud,
                plan,
                vg,
                params,
                oof_indices,
                method=method_eff,
                power=power,
            )
            store.merge(idx, preds, n_used, pass_no=1)
            j = compute_J(store, oof_indices, point_fold, lambdas)
            rmse, mae = store.rmse_mae_all(
                empty_fallback=float(obj_cfg.empty_rmse_fallback)
            )
            trial.set_user_attr("J", float(j))
            trial.set_user_attr("CV_RMSE", float(rmse))
            trial.set_user_attr("CV_MAE", float(mae))
            trial.set_user_attr("prediction_coverage", float(store.coverage))
            trial.set_user_attr("n_valid", int(store.n_valid))
            trial.set_user_attr("n_invalid", int(store.n_invalid))
            trial.set_user_attr("n_tgt", int(store.n_tgt))
            objective_value = float(j)
        else:
            metrics = evaluate_oof(
                cloud,
                plan,
                vg,
                params,
                obj_cfg,
                method=method_eff,
                power=power,
            )
            trial.set_user_attr("CV_RMSE", metrics.cv_rmse)
            trial.set_user_attr("CV_MAE", metrics.cv_mae)
            trial.set_user_attr("prediction_coverage", metrics.prediction_coverage)
            trial.set_user_attr("n_valid", metrics.n_valid)
            trial.set_user_attr("n_invalid", metrics.n_invalid)
            trial.set_user_attr("n_tgt", metrics.n_tgt)
            objective_value = float(metrics.objective)

        trial.set_user_attr("R_inter", params.r_inter)
        trial.set_user_attr("R_minor", params.r_minor)
        trial.set_user_attr("octant_search", params.octant_search)
        trial.set_user_attr("sector_search", params.octant_search)
        trial.set_user_attr("Nmin", params.n_min)
        trial.set_user_attr("Nmax", params.n_max)
        trial.set_user_attr("method", method_eff)
        trial.set_user_attr("loss", loss_eff)
        if power is not None:
            trial.set_user_attr("power", float(power))
        if params.octant_search:
            trial.set_user_attr("n_sectors", int(params.n_sectors))
            trial.set_user_attr("MINOCT", params.minoct)
            trial.set_user_attr("MINPEROC", params.minperoc)
            trial.set_user_attr("MAXPEROC", params.maxperoc)
            trial.set_user_attr(
                "neighbour_cap",
                int(params.n_sectors) * int(params.maxperoc),
            )
        return objective_value

    return _objective
