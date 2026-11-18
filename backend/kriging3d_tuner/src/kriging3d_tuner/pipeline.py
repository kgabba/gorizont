"""Orchestration: load → folds → Optuna → artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler

from .cv import (
    METHOD_ANISOTROPIC_IDW,
    METHOD_ORDINARY_KRIGING,
    build_folds,
    evaluate_oof_masked,
    load_config,
    load_points_csv,
    load_variogram_json,
    oof_target_indices,
)
from .diagnostics import write_oof_diagnostic_plots, write_swath_plots
from .models import ObjectiveConfig
from .objective import make_optuna_objective
from .search_space import (
    build_search_space,
    parse_force_octant_search,
    search_params_from_trial_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _write_search_progress(
    progress_path: Path | None,
    *,
    trial: int,
    n_trials: int,
    best_objective: float | None = None,
    stage_label: str = "Подбор области поиска",
) -> None:
    if progress_path is None:
        return
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "search",
        "stage_label": str(stage_label),
        "trial": int(trial),
        "n_trials": int(n_trials),
        "updated_at": time.time(),
    }
    if best_objective is not None:
        payload["best_objective"] = float(best_objective)
    progress_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_idw_report(path: Path, result: dict[str, Any]) -> None:
    sector_on = bool(
        result.get("octant_search", result.get("sector_search", False))
    )
    lines = [
        "# anisotropic_idw",
        "",
        f"- Method: **{result.get('method')}**",
        f"- power (p): **{result.get('power'):.6g}**",
        f"- R_major / R_inter / R_minor: **{result.get('R_major'):.6g}** / "
        f"**{result.get('R_inter'):.6g}** / **{result.get('R_minor'):.6g}**",
        f"- K_inter / K_minor: **{result.get('K_inter'):.6g}** / "
        f"**{result.get('K_minor'):.6g}**",
        f"- Nmin / Nmax: **{result.get('Nmin')}** / **{result.get('Nmax')}**"
        + (" *(implied by sector limits)*" if sector_on else ""),
        f"- CV_RMSE / CV_MAE: **{result.get('CV_RMSE'):.6g}** / "
        f"**{result.get('CV_MAE'):.6g}**",
        f"- Coverage: **{100.0 * float(result.get('prediction_coverage', 0)):.4g}%** "
        f"({result.get('n_valid')}/{result.get('n_tgt')})",
        f"- mean_n_used: **{result.get('mean_n_used')}**",
        f"- n_trials / seed / runtime_s: **{result.get('n_trials')}** / "
        f"**{result.get('seed')}** / **{float(result.get('runtime_seconds', 0)):.4g}**",
        "",
        "## Sector search (Micromine-style)",
        "",
        f"- **sector_search: {'ON' if sector_on else 'OFF'}**",
    ]
    if sector_on:
        lines += [
            f"- n_sectors: **{result.get('n_sectors')}** "
            f"(4=quadrants, 8=octants, 16=sixteen)",
            f"- MINOCT / MINPEROC / MAXPEROC: **{result.get('MINOCT')}** / "
            f"**{result.get('MINPEROC')}** / **{result.get('MAXPEROC')}**",
            f"- neighbour_cap (= n_sectors × MAXPEROC): "
            f"**{result.get('neighbour_cap')}**",
            f"- discard_underfilled: **{result.get('discard_underfilled')}**",
            "",
        ]
    else:
        lines += [
            "- Optuna chose legacy ellipsoid + Nmin/Nmax (no sector limits).",
            "",
        ]
    idw = result.get("idw_config") or {}
    ss = result.get("search_space") or {}
    lines += [
        "## Search ranges",
        "",
        f"- power: [{idw.get('power_min')}, {idw.get('power_max')}]",
        f"- search_space: `{ss}`",
        "",
        "OOF predictions: `oof_predictions.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tuner(
    data_path: str | Path,
    anisotropy_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    *,
    n_trials: int | None = None,
    progress_path: str | Path | None = None,
    mode: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    mcfg = cfg.get("multipass") or {}
    mode_eff = (mode or cfg.get("mode") or "").strip().lower()
    if not mode_eff:
        mode_eff = "multipass" if mcfg.get("enabled") else "single"

    method_eff = (
        method or cfg.get("method") or METHOD_ORDINARY_KRIGING
    ).strip().lower()
    if method_eff not in (METHOD_ORDINARY_KRIGING, METHOD_ANISOTROPIC_IDW):
        raise ValueError(
            f"unknown method {method_eff!r}; expected "
            f"{METHOD_ORDINARY_KRIGING!r} or {METHOD_ANISOTROPIC_IDW!r}"
        )
    if method_eff == METHOD_ANISOTROPIC_IDW and mode_eff != "single":
        raise ValueError(
            "anisotropic_idw requires mode=single "
            "(multipass / sequential_multipass_optuna_J are not supported)"
        )

    if mode_eff == "multipass":
        from .multipass import run_multipass_tuner

        return run_multipass_tuner(
            data_path,
            anisotropy_path,
            out_dir,
            config_path,
            n_trials=n_trials,
        )
    if mode_eff in ("sequential_multipass_optuna_j", "multipass_j"):
        from .multipass_j import run_sequential_multipass_optuna_j

        return run_sequential_multipass_optuna_j(
            data_path,
            anisotropy_path,
            out_dir,
            config_path,
            n_trials=n_trials,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_file = Path(progress_path) if progress_path else None

    cloud = load_points_csv(data_path, cfg)
    vg = load_variogram_json(anisotropy_path)
    plan = build_folds(cloud, cfg)
    space = build_search_space(vg, cfg)

    ocfg_raw = cfg.get("objective") or {}
    idw_cfg = cfg.get("idw") or {}

    # Validated anisotropic IDW setup (idw_630_1500_j_p0.5_6): minimize J,
    # no coverage gate, power ∈ [0.5, 6]; Optuna may enable Micromine sectors.
    if method_eff == METHOD_ANISOTROPIC_IDW:
        loss_eff = "j"
        obj_cfg = ObjectiveConfig(
            coverage_min=0.0,
            coverage_penalty=0.0,
            empty_rmse_fallback=float(
                ocfg_raw.get("empty_rmse_fallback", 1.0e6)
            ),
        )
        power_min = float(idw_cfg.get("power_min", 0.5))
        power_max = float(idw_cfg.get("power_max", 6.0))
    else:
        obj_cfg = ObjectiveConfig(
            coverage_min=float(ocfg_raw.get("coverage_min", 0.97)),
            coverage_penalty=float(ocfg_raw.get("coverage_penalty", 1.0e6)),
            empty_rmse_fallback=float(
                ocfg_raw.get("empty_rmse_fallback", 1.0e6)
            ),
        )
        loss_eff = str(ocfg_raw.get("loss", "rmse")).strip().lower()
        if loss_eff not in ("rmse", "j"):
            raise ValueError(
                f"objective.loss must be 'rmse' or 'j', got {loss_eff!r}"
            )
        power_min = float(idw_cfg.get("power_min", 1.0))
        power_max = float(idw_cfg.get("power_max", 4.0))

    seed = int(cfg.get("seed", 42))
    ocfg = cfg.get("optuna") or {}
    n_trials = int(n_trials if n_trials is not None else ocfg.get("n_trials", 200))
    n_startup = ocfg.get("n_startup_trials")
    if n_startup is None:
        n_startup = max(10, int(0.3 * n_trials))
    n_ei_default = 256 if method_eff == METHOD_ANISOTROPIC_IDW else 64
    n_ei = int(ocfg.get("n_ei_candidates", n_ei_default))
    flush_every = int(ocfg.get("flush_every", 25))

    force_octant = parse_force_octant_search(cfg)
    force_label = (
        "auto"
        if force_octant is None
        else ("true" if force_octant else "false")
    )
    print(
        f"[kriging3d_tuner] method={method_eff} loss={loss_eff} n={cloud.n} "
        f"cv={plan.method} folds={plan.n_folds} "
        f"vg ranges={vg.range_major:.4g}/{vg.range_intermediate:.4g}/{vg.range_minor:.4g} "
        f"force_octant_search={force_label}",
        flush=True,
    )

    cfg_dump = dict(cfg)
    cfg_dump["method"] = method_eff
    cfg_dump["mode"] = mode_eff
    if method_eff == METHOD_ANISOTROPIC_IDW:
        cfg_dump["objective"] = {
            **(cfg.get("objective") or {}),
            "loss": loss_eff,
            "coverage_min": obj_cfg.coverage_min,
            "coverage_penalty": obj_cfg.coverage_penalty,
            "empty_rmse_fallback": obj_cfg.empty_rmse_fallback,
        }
        cfg_dump["idw"] = {
            **(cfg.get("idw") or {}),
            "power_min": power_min,
            "power_max": power_max,
        }
        cfg_dump["optuna"] = {
            **(cfg.get("optuna") or {}),
            "n_trials": n_trials,
            "n_startup_trials": n_startup,
            "n_ei_candidates": n_ei,
            "flush_every": flush_every,
        }
    (out / "_run_config.yaml").write_text(
        yaml.safe_dump(cfg_dump, sort_keys=False), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    _write_search_progress(progress_file, trial=0, n_trials=n_trials)

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
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
                "J": ua.get("J"),
                "CV_RMSE": ua.get("CV_RMSE"),
                "CV_MAE": ua.get("CV_MAE"),
                "prediction_coverage": ua.get("prediction_coverage"),
                "n_valid": ua.get("n_valid"),
                "n_invalid": ua.get("n_invalid"),
                "n_tgt": ua.get("n_tgt"),
            }
        )
        if len(rows) % flush_every == 0:
            pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)
        best_obj = (
            float(study.best_value)
            if study.best_trial is not None and study.best_value is not None
            else None
        )
        # trial.number is 0-based; report completed count for the UI bar
        _write_search_progress(
            progress_file,
            trial=int(trial.number) + 1,
            n_trials=n_trials,
            best_objective=best_obj,
        )
        if study.best_trial is not None and trial.number == study.best_trial.number:
            j_part = (
                f" J={ua.get('J'):.4g}" if ua.get("J") is not None else ""
            )
            print(
                f"[kriging3d_tuner] new best trial={trial.number} "
                f"objective={trial.value:.4g}{j_part} RMSE={ua.get('CV_RMSE'):.4g} "
                f"coverage={ua.get('prediction_coverage'):.3f}",
                flush=True,
            )

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=int(n_startup),
        n_ei_candidates=n_ei,
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_optuna_objective(
        cloud,
        plan,
        vg,
        space,
        obj_cfg,
        force_octant_search=force_octant,
        method=method_eff,
        power_min=power_min,
        power_max=power_max,
        loss=loss_eff,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)

    best = study.best_trial
    bp = best.params
    ua = best.user_attrs
    search = search_params_from_trial_params(bp)
    power_best: float | None = None
    if method_eff == METHOD_ANISOTROPIC_IDW:
        power_best = float(bp["power"])

    mean_n_used = float("nan")
    oof = oof_target_indices(plan)
    idx, preds, n_used = evaluate_oof_masked(
        cloud,
        plan,
        vg,
        search,
        oof,
        method=method_eff,
        power=power_best,
    )
    rows_oof: list[dict[str, Any]] = []
    used_for_mean: list[int] = []
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    xyz_list: list[list[float]] = []
    for gi, pr, nu in zip(idx.tolist(), preds.tolist(), n_used.tolist()):
        if not np.isfinite(pr):
            continue
        gi_i = int(gi)
        yt = float(cloud.values[gi_i])
        rows_oof.append(
            {
                "global_idx": gi_i,
                "X": float(cloud.xyz[gi_i, 0]),
                "Y": float(cloud.xyz[gi_i, 1]),
                "Z": float(cloud.xyz[gi_i, 2]),
                "y_true": yt,
                "y_pred": float(pr),
                "n_used": int(nu),
            }
        )
        used_for_mean.append(int(nu))
        y_true_list.append(yt)
        y_pred_list.append(float(pr))
        xyz_list.append(
            [
                float(cloud.xyz[gi_i, 0]),
                float(cloud.xyz[gi_i, 1]),
                float(cloud.xyz[gi_i, 2]),
            ]
        )
    pd.DataFrame(rows_oof).to_csv(out / "oof_predictions.csv", index=False)
    if used_for_mean:
        mean_n_used = float(np.mean(used_for_mean))

    diag_dir = out / "oof_diagnostics"
    if y_true_list:
        yt_arr = np.asarray(y_true_list, dtype=float)
        yp_arr = np.asarray(y_pred_list, dtype=float)
        xyz_arr = np.asarray(xyz_list, dtype=float)
        write_oof_diagnostic_plots(
            yt_arr, yp_arr, diag_dir, title_prefix="OOF"
        )
        write_swath_plots(
            xyz_arr, yt_arr, yp_arr, diag_dir, title_prefix="OOF"
        )

    result: dict[str, Any] = {
        "method": method_eff,
        "loss": loss_eff,
        **search.as_dict(),
        "CV_RMSE": float(ua["CV_RMSE"]),
        "CV_MAE": float(ua["CV_MAE"]),
        "prediction_coverage": float(ua["prediction_coverage"]),
        "n_valid": int(ua["n_valid"]),
        "n_invalid": int(ua["n_invalid"]),
        "n_tgt": int(ua["n_tgt"]),
        "objective": float(best.value),
        "n_trials": n_trials,
        "best_trial_number": int(best.number),
        "cv": {
            "method": plan.method,
            "n_folds": plan.n_folds,
            "n_splits_requested": plan.n_splits_requested,
        },
        "variogram_fixed": vg.as_dict(),
        "search_space": space.as_dict(),
        "objective_config": {
            "loss": loss_eff,
            "coverage_min": obj_cfg.coverage_min,
            "coverage_penalty": obj_cfg.coverage_penalty,
            "empty_rmse_fallback": obj_cfg.empty_rmse_fallback,
        },
        "seed": seed,
        "runtime_seconds": time.perf_counter() - t0,
    }
    if ua.get("J") is not None:
        result["J"] = float(ua["J"])
    if power_best is not None:
        result["power"] = power_best
        result["p"] = power_best
        result["mean_n_used"] = mean_n_used
        result["idw_config"] = {
            "power_min": power_min,
            "power_max": power_max,
        }

    # Optional pass-2: if OOF coverage < 100%, fill with RMSE + coverage gate.
    from .coverage_fill import maybe_run_coverage_fill

    result = maybe_run_coverage_fill(
        cloud=cloud,
        plan=plan,
        vg=vg,
        space=space,
        cfg=cfg,
        method=method_eff,
        pass1_result=result,
        pass1_search=search,
        pass1_power=power_best,
        out_dir=out,
        progress_path=progress_file,
        n_trials_pass1=n_trials,
        power_min=power_min,
        power_max=power_max,
        force_octant=force_octant,
        seed=seed,
        write_progress_fn=_write_search_progress,
    )

    if power_best is not None or result.get("method") == METHOD_ANISOTROPIC_IDW:
        _write_idw_report(out / "REPORT_idw.md", result)

    (out / "best_params.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fill = result.get("coverage_fill") or {}
    fill_note = ""
    if fill.get("triggered"):
        fill_note = (
            f" +coverage_fill cov={result['prediction_coverage']:.3f}"
        )
    print(
        f"[kriging3d_tuner] done method={method_eff} best trial={best.number} "
        f"RMSE={result['CV_RMSE']:.4g} coverage={result['prediction_coverage']:.3f}"
        f"{fill_note} → {out}",
        flush=True,
    )
    return result
