"""Orchestration: load → folds → Optuna → artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler

from .cv import build_folds, load_config, load_points_csv, load_variogram_json
from .models import ObjectiveConfig, SearchParams
from .objective import make_optuna_objective
from .search_space import build_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_tuner(
    data_path: str | Path,
    anisotropy_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    *,
    n_trials: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cloud = load_points_csv(data_path, cfg)
    vg = load_variogram_json(anisotropy_path)
    plan = build_folds(cloud, cfg)
    space = build_search_space(vg, cfg)

    ocfg_raw = cfg.get("objective") or {}
    obj_cfg = ObjectiveConfig(
        coverage_min=float(ocfg_raw.get("coverage_min", 0.97)),
        coverage_penalty=float(ocfg_raw.get("coverage_penalty", 1.0e6)),
        empty_rmse_fallback=float(ocfg_raw.get("empty_rmse_fallback", 1.0e6)),
    )

    seed = int(cfg.get("seed", 42))
    ocfg = cfg.get("optuna") or {}
    n_trials = int(n_trials if n_trials is not None else ocfg.get("n_trials", 200))
    n_startup = ocfg.get("n_startup_trials")
    if n_startup is None:
        n_startup = max(10, int(0.3 * n_trials))
    n_ei = int(ocfg.get("n_ei_candidates", 64))
    flush_every = int(ocfg.get("flush_every", 25))

    print(
        f"[kriging3d_tuner] n={cloud.n} cv={plan.method} folds={plan.n_folds} "
        f"vg ranges={vg.range_major:.4g}/{vg.range_intermediate:.4g}/{vg.range_minor:.4g}",
        flush=True,
    )

    (out / "_run_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        p = trial.params
        ua = trial.user_attrs
        rows.append(
            {
                "trial": trial.number,
                "R_major": p.get("R_major"),
                "K_inter": p.get("K_inter"),
                "K_minor": p.get("K_minor"),
                "R_inter": ua.get("R_inter"),
                "R_minor": ua.get("R_minor"),
                "Nmin": p.get("Nmin"),
                "Nmax": p.get("Nmax"),
                "objective": trial.value,
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
        if study.best_trial is not None and trial.number == study.best_trial.number:
            print(
                f"[kriging3d_tuner] new best trial={trial.number} "
                f"objective={trial.value:.4g} RMSE={ua.get('CV_RMSE'):.4g} "
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
    objective = make_optuna_objective(cloud, plan, vg, space, obj_cfg)
    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)

    best = study.best_trial
    bp = best.params
    ua = best.user_attrs
    search = SearchParams(
        r_major=float(bp["R_major"]),
        k_inter=float(bp["K_inter"]),
        k_minor=float(bp["K_minor"]),
        n_min=int(bp["Nmin"]),
        n_max=int(bp["Nmax"]),
    )
    result = {
        **search.as_dict(),
        "CV_RMSE": float(ua["CV_RMSE"]),
        "CV_MAE": float(ua["CV_MAE"]),
        "prediction_coverage": float(ua["prediction_coverage"]),
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
        "seed": seed,
        "runtime_seconds": time.perf_counter() - t0,
    }
    (out / "best_params.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[kriging3d_tuner] done best trial={best.number} "
        f"RMSE={result['CV_RMSE']:.4g} coverage={result['prediction_coverage']:.3f} "
        f"→ {out}",
        flush=True,
    )
    return result
