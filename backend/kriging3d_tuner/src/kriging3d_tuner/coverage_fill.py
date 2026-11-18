"""Optional second Optuna pass to fill remaining OOF coverage.

Pass 1 stays as configured (OK RMSE-gated / IDW J). If pooled OOF coverage
is below ``coverage_fill.trigger_below`` (default 1.0), pass 2 re-optimizes a
neighbourhood on still-unevaluated OOF targets with loss=RMSE and a hard
coverage gate at ``pass2_coverage_min`` (default 1.0). Works for both
``ordinary_kriging`` and ``anisotropic_idw``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .cv import (
    METHOD_ANISOTROPIC_IDW,
    METHOD_ORDINARY_KRIGING,
    evaluate_oof_masked,
    oof_target_indices,
)
from .diagnostics import write_oof_diagnostic_plots, write_swath_plots
from .models import (
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    SearchSpaceConfig,
    VariogramFixed,
    compute_objective,
)
from .oof_store import OOFStore, diagnose_residuals
from .search_space import search_params_from_trial_params, suggest_search_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _expand_space(space: SearchSpaceConfig, factor: float) -> SearchSpaceConfig:
    f = max(float(factor), 1.0)
    return SearchSpaceConfig(
        r_major_min=float(space.r_major_min),
        r_major_max=max(float(space.r_major_max) * f, float(space.r_major_min) + 1e-12),
        k_min=float(space.k_min),
        nmin_lo=int(space.nmin_lo),
        nmin_hi=int(space.nmin_hi),
        nmax_lo=int(space.nmax_lo),
        nmax_hi=int(space.nmax_hi),
        minoct_lo=int(space.minoct_lo),
        minoct_hi=int(space.minoct_hi),
        minperoc_lo=int(space.minperoc_lo),
        minperoc_hi=int(space.minperoc_hi),
        maxperoc_lo=int(space.maxperoc_lo),
        maxperoc_hi=int(space.maxperoc_hi),
    )


def _seed_store_from_pass1(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    search: SearchParams,
    *,
    method: str,
    power: float | None,
) -> OOFStore:
    oof = oof_target_indices(plan)
    store = OOFStore(n_tgt=int(oof.size), y_true=cloud.values.copy())
    idx, preds, n_used = evaluate_oof_masked(
        cloud,
        plan,
        vg,
        search,
        oof,
        method=method,
        power=power,
    )
    store.merge(idx, preds, n_used, pass_no=1)
    return store


def _apply_pass2_trial(
    store: OOFStore,
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    search: SearchParams,
    *,
    method: str,
    power: float | None,
    frozen_pred: np.ndarray,
    frozen_pass: np.ndarray,
    frozen_n_used: np.ndarray,
) -> dict[str, Any]:
    store.pred[:] = frozen_pred
    store.pass_id[:] = frozen_pass
    store.n_used[:] = frozen_n_used
    unset = [
        int(i)
        for i in oof_target_indices(plan).tolist()
        if store.pass_id[int(i)] == 0
    ]
    idx, preds, n_used = evaluate_oof_masked(
        cloud,
        plan,
        vg,
        search,
        np.asarray(unset, dtype=int),
        method=method,
        power=power,
    )
    n_new = store.merge(idx, preds, n_used, pass_no=2)
    rmse_all, mae_all = store.rmse_mae_all()
    local = store.pass_local_metrics(2)
    return {
        "n_new": n_new,
        "coverage": store.coverage,
        "n_valid": store.n_valid,
        "n_invalid": store.n_invalid,
        "n_tgt": store.n_tgt,
        "CV_RMSE_all": rmse_all,
        "CV_MAE_all": mae_all,
        **{f"local_{k}": v for k, v in local.items()},
    }


def _write_oof_artifacts(
    out: Path,
    cloud: PointCloud,
    store: OOFStore,
    *,
    title_prefix: str,
) -> None:
    rows: list[dict[str, Any]] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    xyz: list[list[float]] = []
    pass_ids: list[int] = []
    for gi in np.where(store.estimated_mask)[0]:
        gi_i = int(gi)
        yt = float(cloud.values[gi_i])
        yp = float(store.pred[gi_i])
        rows.append(
            {
                "global_idx": gi_i,
                "X": float(cloud.xyz[gi_i, 0]),
                "Y": float(cloud.xyz[gi_i, 1]),
                "Z": float(cloud.xyz[gi_i, 2]),
                "y_true": yt,
                "y_pred": yp,
                "n_used": int(store.n_used[gi_i]),
                "pass_id": int(store.pass_id[gi_i]),
            }
        )
        y_true.append(yt)
        y_pred.append(yp)
        xyz.append(
            [
                float(cloud.xyz[gi_i, 0]),
                float(cloud.xyz[gi_i, 1]),
                float(cloud.xyz[gi_i, 2]),
            ]
        )
        pass_ids.append(int(store.pass_id[gi_i]))
    pd.DataFrame(rows).to_csv(out / "oof_predictions.csv", index=False)
    if y_true:
        diag = out / "oof_diagnostics"
        yt_arr = np.asarray(y_true, dtype=float)
        yp_arr = np.asarray(y_pred, dtype=float)
        xyz_arr = np.asarray(xyz, dtype=float)
        write_oof_diagnostic_plots(
            yt_arr,
            yp_arr,
            diag,
            pass_id=np.asarray(pass_ids, dtype=int),
            title_prefix=title_prefix,
        )
        write_swath_plots(xyz_arr, yt_arr, yp_arr, diag, title_prefix=title_prefix)


def maybe_run_coverage_fill(
    *,
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    cfg: dict[str, Any],
    method: str,
    pass1_result: dict[str, Any],
    pass1_search: SearchParams,
    pass1_power: float | None,
    out_dir: str | Path,
    progress_path: str | Path | None,
    n_trials_pass1: int,
    power_min: float,
    power_max: float,
    force_octant: bool | None,
    seed: int,
    write_progress_fn: Any | None = None,
) -> dict[str, Any]:
    """Possibly append pass-2 fill; returns updated result dict."""
    cfill = cfg.get("coverage_fill") or {}
    if not bool(cfill.get("enabled", False)):
        pass1_result = dict(pass1_result)
        pass1_result["coverage_fill"] = {"enabled": False, "triggered": False}
        return pass1_result

    trigger = float(cfill.get("trigger_below", 1.0))
    cov_min = float(cfill.get("pass2_coverage_min", 1.0))
    expand = float(cfill.get("r_expand_factor", 2.0))
    ocfg = cfg.get("optuna") or {}
    n_trials_p2 = int(
        cfill.get("n_trials")
        if cfill.get("n_trials") is not None
        else n_trials_pass1
    )
    n_startup = ocfg.get("n_startup_trials")
    if n_startup is None:
        n_startup = max(10, int(0.3 * n_trials_p2))
    else:
        n_startup = int(n_startup)
    n_ei_default = 256 if method == METHOD_ANISOTROPIC_IDW else 64
    n_ei = int(ocfg.get("n_ei_candidates", n_ei_default))
    flush_every = int(ocfg.get("flush_every", 25))
    raw_obj = cfg.get("objective") or {}
    penalty = float(raw_obj.get("coverage_penalty", 1.0e6))
    empty_fb = float(raw_obj.get("empty_rmse_fallback", 1.0e6))

    out = Path(out_dir)
    method_eff = (method or METHOD_ORDINARY_KRIGING).strip().lower()
    force_oct = force_octant
    allow_variable_sectors = method_eff == METHOD_ANISOTROPIC_IDW

    store = _seed_store_from_pass1(
        cloud,
        plan,
        vg,
        pass1_search,
        method=method_eff,
        power=pass1_power,
    )
    pass1_cov = float(store.coverage)
    pass1_rmse, pass1_mae = store.rmse_mae_all(empty_fallback=empty_fb)
    pass1_local = store.pass_local_metrics(1, empty_fallback=empty_fb)

    result = dict(pass1_result)
    result["coverage_fill"] = {
        "enabled": True,
        "triggered": False,
        "trigger_below": trigger,
        "pass1_coverage": pass1_cov,
        "pass1_CV_RMSE": pass1_rmse,
        "pass1_CV_MAE": pass1_mae,
        "pass1_n_valid": store.n_valid,
        "pass1_n_invalid": store.n_invalid,
    }
    result["passes"] = [
        {
            "pass": 1,
            **pass1_search.as_dict(),
            "power": pass1_power,
            "loss": pass1_result.get("loss"),
            "CV_RMSE": pass1_rmse,
            "CV_MAE": pass1_mae,
            "prediction_coverage": pass1_cov,
            "n_new": int(pass1_local["n_new"]),
            "local_CV_RMSE": float(pass1_local["CV_RMSE"]),
            "local_CV_MAE": float(pass1_local["CV_MAE"]),
        }
    ]

    if pass1_cov + 1e-12 >= trigger:
        print(
            f"[coverage_fill] skip pass2: coverage={pass1_cov:.4f} "
            f">= trigger={trigger:.4f}",
            flush=True,
        )
        return result

    print(
        f"[coverage_fill] pass1 coverage={pass1_cov:.4f} < {trigger:.4f} → "
        f"pass2 RMSE with coverage_min={cov_min:.4f} "
        f"trials={n_trials_p2} r_expand={expand:g}",
        flush=True,
    )

    space2 = _expand_space(space, expand)
    out_p2 = out / "pass_2_coverage_fill"
    out_p2.mkdir(parents=True, exist_ok=True)
    frozen_pred, frozen_pass, frozen_nu = (
        store.pred.copy(),
        store.pass_id.copy(),
        store.n_used.copy(),
    )
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total_trials_ui = int(n_trials_pass1) + int(n_trials_p2)

    def _progress(done_p2: int, best_obj: float | None = None) -> None:
        if write_progress_fn is None or progress_path is None:
            return
        write_progress_fn(
            Path(progress_path),
            trial=int(n_trials_pass1) + int(done_p2),
            n_trials=total_trials_ui,
            best_objective=best_obj,
            stage_label="Дозаполнение покрытия",
        )

    _progress(0)

    gate_cfg = ObjectiveConfig(
        coverage_min=cov_min,
        coverage_penalty=penalty,
        empty_rmse_fallback=empty_fb,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_search_params(
            trial,
            space2,
            force_octant_search=force_oct,
            allow_variable_sectors=allow_variable_sectors,
        )
        power: float | None = None
        if method_eff == METHOD_ANISOTROPIC_IDW:
            p_lo = float(min(power_min, power_max))
            p_hi = float(max(power_min, power_max))
            power = float(trial.suggest_float("power", p_lo, p_hi))
        metrics = _apply_pass2_trial(
            store,
            cloud,
            plan,
            vg,
            params,
            method=method_eff,
            power=power,
            frozen_pred=frozen_pred,
            frozen_pass=frozen_pass,
            frozen_n_used=frozen_nu,
        )
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("R_inter", params.r_inter)
        trial.set_user_attr("R_minor", params.r_minor)
        trial.set_user_attr("octant_search", params.octant_search)
        trial.set_user_attr("sector_search", params.octant_search)
        trial.set_user_attr("Nmin", params.n_min)
        trial.set_user_attr("Nmax", params.n_max)
        if params.octant_search:
            trial.set_user_attr("n_sectors", int(params.n_sectors))
            trial.set_user_attr("MINOCT", params.minoct)
            trial.set_user_attr("MINPEROC", params.minperoc)
            trial.set_user_attr("MAXPEROC", params.maxperoc)
        if power is not None:
            trial.set_user_attr("power", power)
        return compute_objective(
            cv_rmse=float(metrics["CV_RMSE_all"]),
            prediction_coverage=float(metrics["coverage"]),
            n_valid=int(metrics["n_valid"]),
            cfg=gate_cfg,
        )

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        p = trial.params
        ua = trial.user_attrs
        rows.append(
            {
                "trial": trial.number,
                "method": method_eff,
                "power": p.get("power"),
                "octant_search": p.get("octant_search"),
                "sector_search": p.get("octant_search"),
                "n_sectors": p.get("n_sectors", ua.get("n_sectors")),
                "R_major": p.get("R_major"),
                "K_inter": p.get("K_inter"),
                "K_minor": p.get("K_minor"),
                "R_inter": ua.get("R_inter"),
                "R_minor": ua.get("R_minor"),
                "Nmin": p.get("Nmin", ua.get("Nmin")),
                "Nmax": p.get("Nmax", ua.get("Nmax")),
                "MINOCT": p.get("MINOCT"),
                "MINPEROC": p.get("MINPEROC"),
                "MAXPEROC": p.get("MAXPEROC"),
                "objective": trial.value,
                "CV_RMSE_all": ua.get("CV_RMSE_all"),
                "CV_MAE_all": ua.get("CV_MAE_all"),
                "prediction_coverage": ua.get("coverage"),
                "n_new": ua.get("n_new"),
                "local_CV_RMSE": ua.get("local_CV_RMSE"),
                "local_CV_MAE": ua.get("local_CV_MAE"),
                "n_valid": ua.get("n_valid"),
                "n_invalid": ua.get("n_invalid"),
                "n_tgt": ua.get("n_tgt"),
            }
        )
        if len(rows) % flush_every == 0:
            pd.DataFrame(rows).to_csv(out_p2 / "trials.csv", index=False)
        best_obj = (
            float(study.best_value)
            if study.best_trial is not None and study.best_value is not None
            else None
        )
        _progress(int(trial.number) + 1, best_obj)
        if study.best_trial is not None and trial.number == study.best_trial.number:
            print(
                f"[coverage_fill] pass2 new best trial={trial.number} "
                f"obj={trial.value:.4g} RMSE_all={ua.get('CV_RMSE_all'):.4g} "
                f"cov={ua.get('coverage'):.3f} n_new={ua.get('n_new')}",
                flush=True,
            )

    sampler = TPESampler(
        seed=int(seed) + 17,
        n_startup_trials=int(n_startup),
        n_ei_candidates=n_ei,
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials_p2, callbacks=[callback])
    pd.DataFrame(rows).to_csv(out_p2 / "trials.csv", index=False)

    best = study.best_trial
    search2 = search_params_from_trial_params(best.params)
    power2: float | None = None
    if method_eff == METHOD_ANISOTROPIC_IDW:
        power2 = float(best.params["power"])
    final = _apply_pass2_trial(
        store,
        cloud,
        plan,
        vg,
        search2,
        method=method_eff,
        power=power2,
        frozen_pred=frozen_pred,
        frozen_pass=frozen_pass,
        frozen_n_used=frozen_nu,
    )
    elapsed_p2 = time.perf_counter() - t0

    pass2_summary = {
        "pass": 2,
        **search2.as_dict(),
        "power": power2,
        "loss": "rmse",
        "coverage_min": cov_min,
        "n_trials": n_trials_p2,
        "best_trial_number": int(best.number),
        "objective": float(best.value),
        "CV_RMSE_all": float(final["CV_RMSE_all"]),
        "CV_MAE_all": float(final["CV_MAE_all"]),
        "prediction_coverage": float(final["coverage"]),
        "n_new": int(final["n_new"]),
        "local_CV_RMSE": float(final["local_CV_RMSE"]),
        "local_CV_MAE": float(final["local_CV_MAE"]),
        "n_valid": int(final["n_valid"]),
        "n_invalid": int(final["n_invalid"]),
        "n_tgt": int(final["n_tgt"]),
        "r_expand_factor": expand,
        "runtime_seconds": elapsed_p2,
    }
    (out_p2 / "best_params.json").write_text(
        json.dumps(pass2_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    residual: dict[str, int] = {}
    if store.n_invalid > 0:
        residual = diagnose_residuals(cloud, plan, vg, store, search2)

    _write_oof_artifacts(
        out, cloud, store, title_prefix="OOF coverage_fill"
    )

    pooled_rmse, pooled_mae = store.rmse_mae_all(empty_fallback=empty_fb)
    result["passes"].append(pass2_summary)
    result["coverage_fill"].update(
        {
            "triggered": True,
            "pass2_coverage_min": cov_min,
            "r_expand_factor": expand,
            "n_trials": n_trials_p2,
            "pass2_coverage": float(store.coverage),
            "pass2_CV_RMSE_all": pooled_rmse,
            "pass2_n_new": int(final["n_new"]),
            "n_by_pass": {
                "1": int(np.sum(store.pass_id == 1)),
                "2": int(np.sum(store.pass_id == 2)),
            },
            "residual_failure_reasons": residual,
        }
    )
    # Pooled OOF metrics become the headline result; keep pass-1 search at top
    # level for backward compatibility, expose both under passes[].
    result["CV_RMSE"] = pooled_rmse
    result["CV_MAE"] = pooled_mae
    result["prediction_coverage"] = float(store.coverage)
    result["n_valid"] = int(store.n_valid)
    result["n_invalid"] = int(store.n_invalid)
    result["n_tgt"] = int(store.n_tgt)
    result["search_params_by_pass"] = [
        result["passes"][0],
        {k: pass2_summary[k] for k in pass2_summary if k != "pass"},
    ]
    result["runtime_seconds"] = float(result.get("runtime_seconds") or 0.0) + elapsed_p2
    result["n_trials"] = int(n_trials_pass1) + int(n_trials_p2)

    print(
        f"[coverage_fill] done pooled RMSE={pooled_rmse:.4g} "
        f"cov={store.coverage:.3f} n_by_pass="
        f"{{1:{int(np.sum(store.pass_id == 1))}, 2:{int(np.sum(store.pass_id == 2))}}}",
        flush=True,
    )
    return result
