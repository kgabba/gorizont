"""Three-pass Optuna neighbourhood tuning with cumulative coverage gates."""

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
    build_folds,
    evaluate_oof_masked,
    load_config,
    load_points_csv,
    load_variogram_json,
    oof_target_indices,
)
from .models import (
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    SearchSpaceConfig,
    VariogramFixed,
    compute_objective,
)
from .oof_store import (
    OOFStore,
    diagnose_residuals,
    lexico_coverage_rmse_objective,
    risky_extrapolation,
)
from .search_space import (
    build_search_space,
    parse_force_octant_search,
    search_params_from_trial_params,
    suggest_search_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _obj_cfg_from(cfg: dict[str, Any], coverage_min: float) -> ObjectiveConfig:
    raw = cfg.get("objective") or {}
    return ObjectiveConfig(
        coverage_min=float(coverage_min),
        coverage_penalty=float(raw.get("coverage_penalty", 1.0e6)),
        empty_rmse_fallback=float(raw.get("empty_rmse_fallback", 1.0e6)),
    )


def _snapshot_store(store: OOFStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return store.pred.copy(), store.pass_id.copy(), store.n_used.copy()


def _restore_store(
    store: OOFStore,
    pred: np.ndarray,
    pass_id: np.ndarray,
    n_used: np.ndarray,
) -> None:
    store.pred[:] = pred
    store.pass_id[:] = pass_id
    store.n_used[:] = n_used


def _apply_trial_to_store(
    store: OOFStore,
    cloud: PointCloud,
    plan,
    vg: VariogramFixed,
    search: SearchParams,
    pass_no: int,
    frozen_pred: np.ndarray,
    frozen_pass: np.ndarray,
    frozen_n_used: np.ndarray,
) -> dict[str, Any]:
    """Restore freeze, evaluate candidates, merge; return metrics dict."""
    _restore_store(store, frozen_pred, frozen_pass, frozen_n_used)
    unset = [
        int(i)
        for i in oof_target_indices(plan).tolist()
        if store.pass_id[i] == 0
    ]
    idx, preds, n_used = evaluate_oof_masked(
        cloud, plan, vg, search, np.asarray(unset, dtype=int)
    )
    n_new = store.merge(idx, preds, n_used, pass_no)
    rmse_all, mae_all = store.rmse_mae_all()
    local = store.pass_local_metrics(pass_no)
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


def _run_pass_study(
    *,
    pass_no: int,
    cloud: PointCloud,
    plan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    store: OOFStore,
    frozen_pred: np.ndarray,
    frozen_pass: np.ndarray,
    frozen_n_used: np.ndarray,
    coverage_gate: float | None,
    lexico: bool,
    n_trials: int,
    seed: int,
    n_startup: int,
    n_ei: int,
    flush_every: int,
    force_octant: bool | None,
    out_pass: Path,
    obj_penalty_cfg: ObjectiveConfig,
) -> tuple[SearchParams, dict[str, Any], optuna.Study]:
    out_pass.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_search_params(
            trial, space, force_octant_search=force_octant
        )
        metrics = _apply_trial_to_store(
            store,
            cloud,
            plan,
            vg,
            params,
            pass_no,
            frozen_pred,
            frozen_pass,
            frozen_n_used,
        )
        trial.set_user_attr("CV_RMSE_all", metrics["CV_RMSE_all"])
        trial.set_user_attr("CV_MAE_all", metrics["CV_MAE_all"])
        trial.set_user_attr("prediction_coverage", metrics["coverage"])
        trial.set_user_attr("n_valid", metrics["n_valid"])
        trial.set_user_attr("n_invalid", metrics["n_invalid"])
        trial.set_user_attr("n_tgt", metrics["n_tgt"])
        trial.set_user_attr("n_new", metrics["n_new"])
        trial.set_user_attr("local_CV_RMSE", metrics["local_CV_RMSE"])
        trial.set_user_attr("local_CV_MAE", metrics["local_CV_MAE"])
        trial.set_user_attr("local_mean_n_used", metrics["local_mean_n_used"])
        trial.set_user_attr("R_inter", params.r_inter)
        trial.set_user_attr("R_minor", params.r_minor)
        trial.set_user_attr("octant_search", params.octant_search)
        trial.set_user_attr("Nmax", params.n_max)
        if params.octant_search:
            trial.set_user_attr("MINOCT", params.minoct)
            trial.set_user_attr("MINPEROC", params.minperoc)
            trial.set_user_attr("MAXPEROC", params.maxperoc)

        if lexico:
            return lexico_coverage_rmse_objective(
                metrics["coverage"], metrics["CV_RMSE_all"]
            )
        assert coverage_gate is not None
        gate_cfg = ObjectiveConfig(
            coverage_min=float(coverage_gate),
            coverage_penalty=obj_penalty_cfg.coverage_penalty,
            empty_rmse_fallback=obj_penalty_cfg.empty_rmse_fallback,
        )
        return compute_objective(
            cv_rmse=metrics["CV_RMSE_all"],
            prediction_coverage=metrics["coverage"],
            n_valid=metrics["n_valid"],
            cfg=gate_cfg,
        )

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        p = trial.params
        ua = trial.user_attrs
        rows.append(
            {
                "trial": trial.number,
                "octant_search": p.get("octant_search"),
                "R_major": p.get("R_major"),
                "K_inter": p.get("K_inter"),
                "K_minor": p.get("K_minor"),
                "R_inter": ua.get("R_inter"),
                "R_minor": ua.get("R_minor"),
                "Nmin": p.get("Nmin"),
                "Nmax": p.get("Nmax"),
                "MINOCT": p.get("MINOCT"),
                "MINPEROC": p.get("MINPEROC"),
                "MAXPEROC": p.get("MAXPEROC"),
                "objective": trial.value,
                "CV_RMSE_all": ua.get("CV_RMSE_all"),
                "CV_MAE_all": ua.get("CV_MAE_all"),
                "prediction_coverage": ua.get("prediction_coverage"),
                "n_new": ua.get("n_new"),
                "local_CV_RMSE": ua.get("local_CV_RMSE"),
                "local_CV_MAE": ua.get("local_CV_MAE"),
                "local_mean_n_used": ua.get("local_mean_n_used"),
                "n_valid": ua.get("n_valid"),
                "n_invalid": ua.get("n_invalid"),
                "n_tgt": ua.get("n_tgt"),
            }
        )
        if len(rows) % flush_every == 0:
            pd.DataFrame(rows).to_csv(out_pass / "trials.csv", index=False)
        if study.best_trial is not None and trial.number == study.best_trial.number:
            print(
                f"[multipass] pass={pass_no} new best trial={trial.number} "
                f"obj={trial.value:.4g} RMSE_all={ua.get('CV_RMSE_all'):.4g} "
                f"cov={ua.get('prediction_coverage'):.3f} n_new={ua.get('n_new')}",
                flush=True,
            )

    sampler = TPESampler(
        seed=seed + pass_no,
        n_startup_trials=int(n_startup),
        n_ei_candidates=n_ei,
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    pd.DataFrame(rows).to_csv(out_pass / "trials.csv", index=False)

    best = study.best_trial
    search = search_params_from_trial_params(best.params)
    # Commit best into store
    final_metrics = _apply_trial_to_store(
        store,
        cloud,
        plan,
        vg,
        search,
        pass_no,
        frozen_pred,
        frozen_pass,
        frozen_n_used,
    )
    ua = best.user_attrs
    pass_summary = {
        **search.as_dict(),
        "pass": pass_no,
        "best_trial_number": int(best.number),
        "objective": float(best.value),
        "n_trials": n_trials,
        "CV_RMSE_all": float(final_metrics["CV_RMSE_all"]),
        "CV_MAE_all": float(final_metrics["CV_MAE_all"]),
        "prediction_coverage": float(final_metrics["coverage"]),
        "n_valid": int(final_metrics["n_valid"]),
        "n_invalid": int(final_metrics["n_invalid"]),
        "n_tgt": int(final_metrics["n_tgt"]),
        "n_new": int(final_metrics["n_new"]),
        "pct_new": float(final_metrics["local_pct_new"]),
        "local_CV_RMSE": float(final_metrics["local_CV_RMSE"]),
        "local_CV_MAE": float(final_metrics["local_CV_MAE"]),
        "mean_n_used": final_metrics["local_mean_n_used"],
        "n_remaining": int(final_metrics["n_invalid"]),
        "risky_extrapolation": risky_extrapolation(
            search.r_major, vg.range_major
        ),
    }
    (out_pass / "best_params.json").write_text(
        json.dumps(pass_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return search, pass_summary, study


def _write_report(
    path: Path,
    passes: list[dict[str, Any]],
    total: dict[str, Any],
) -> None:
    lines = [
        "# Трёхпроходная оптимизация соседства OK",
        "",
        "## Сравнение проходов",
        "",
        "| Проход | octant | R_major | Nmin | Nmax | MINOCT | MINPEROC | MAXPEROC | RMSE (новые) | MAE (новые) | n_new | % новых | coverage накопл. % | remaining | mean_n_used | risky |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in passes:
        nu = p["mean_n_used"]
        nu_s = "—" if nu != nu else f"{nu:.4g}"
        lines.append(
            "| {pno} | {oct} | {R:.4g} | {Nmin} | {Nmax} | {MINOCT} | {MINPEROC} | {MAXPEROC} | "
            "{rmse:.4g} | {mae:.4g} | {n_new} | {pct:.2f} | {cov:.2f} | {rem} | {nu} | {risky} |".format(
                pno=p["pass"],
                oct=p.get("octant_search"),
                R=p["R_major"],
                Nmin=p["Nmin"],
                Nmax=p["Nmax"],
                MINOCT=p.get("MINOCT", "—"),
                MINPEROC=p.get("MINPEROC", "—"),
                MAXPEROC=p.get("MAXPEROC", "—"),
                rmse=p["local_CV_RMSE"],
                mae=p["local_CV_MAE"],
                n_new=p["n_new"],
                pct=p["pct_new"],
                cov=100.0 * p["prediction_coverage"],
                rem=p["n_remaining"],
                nu=nu_s,
                risky=p["risky_extrapolation"],
            )
        )
    lines += [
        "",
        "## Итоговые результаты",
        "",
        f"- CV RMSE (объединённый OOF): **{total['CV_RMSE']:.6g}**",
        f"- CV MAE (объединённый OOF): **{total['CV_MAE']:.6g}**",
        f"- Итоговое coverage: **{100.0 * total['prediction_coverage']:.4g}%** "
        f"({total['n_valid']}/{total['n_tgt']})",
        f"- Неоценённых точек: **{total['n_invalid']}**",
        f"- Оценено по проходам: "
        + ", ".join(f"pass{k}={total['n_by_pass'].get(k, 0)}" for k in (1, 2, 3)),
        f"- Время оптимизации, с: **{total['runtime_seconds']:.4g}**",
        "",
    ]
    q = total.get("quality") or {}
    if q:
        lines += [
            "### Доп. качество (scatter / QQ)",
            "",
            f"- R²: **{q.get('R2', float('nan')):.4g}**",
            f"- Pearson r: **{q.get('pearson_r', float('nan')):.4g}**",
            f"- Spearman r: **{q.get('spearman_r', float('nan')):.4g}**",
            f"- Bias (mean pred−obs): **{q.get('bias_mean_error', float('nan')):.4g}**",
            "- Графики: `diagnostics/scatter.png`, `diagnostics/qq.png`, "
            "`diagnostics/qq_obs_pred.png`",
            "",
        ]
    if total["n_invalid"] > 0:
        lines.append("### Остаток без оценки")
        lines.append(
            f"Максимальное достигнутое покрытие: "
            f"{100.0 * total['prediction_coverage']:.4g}% "
            f"(цель 100% не достигнута)."
        )
        reasons = total.get("residual_failure_reasons") or {}
        if reasons:
            lines.append("Причины отказов на остатке (под best pass-3 search):")
            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{k}`: {v}")
        lines.append("")
        lines.append(
            "Прогнозы не подменялись нулями и не экстраполировались "
            "вне допустимого поискового эллипсоида."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multipass_tuner(
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

    mcfg = cfg.get("multipass") or {}
    targets = mcfg.get("coverage_targets") or [0.30, 0.60, 1.0]
    if len(targets) != 3:
        raise ValueError("multipass.coverage_targets must have 3 values")
    c1, c2, _c3 = float(targets[0]), float(targets[1]), float(targets[2])
    risky_factor = float(mcfg.get("risky_r_major_factor", 2.0))

    cloud = load_points_csv(data_path, cfg)
    vg = load_variogram_json(anisotropy_path)
    plan = build_folds(cloud, cfg)
    space = build_search_space(vg, cfg)
    force_octant = parse_force_octant_search(cfg)

    oof_idx = oof_target_indices(plan)
    n_tgt = int(oof_idx.size)
    store = OOFStore(n_tgt=n_tgt, y_true=cloud.values.copy())

    seed = int(cfg.get("seed", 42))
    ocfg = cfg.get("optuna") or {}
    raw_trials = mcfg.get("n_trials_per_pass", ocfg.get("n_trials", 200))
    if n_trials is not None:
        trials_per_pass = [int(n_trials)] * 3
    elif isinstance(raw_trials, (list, tuple)):
        if len(raw_trials) != 3:
            raise ValueError("multipass.n_trials_per_pass list must have 3 ints")
        trials_per_pass = [int(x) for x in raw_trials]
    else:
        trials_per_pass = [int(raw_trials)] * 3
    n_startup = ocfg.get("n_startup_trials")
    n_ei = int(ocfg.get("n_ei_candidates", 64))
    flush_every = int(ocfg.get("flush_every", 25))
    penalty_cfg = _obj_cfg_from(cfg, coverage_min=c1)

    (out / "_run_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    print(
        f"[multipass] n={cloud.n} oof_tgt={n_tgt} folds={plan.n_folds} "
        f"targets={c1}/{c2}/1.0 trials/pass={trials_per_pass}",
        flush=True,
    )

    t0 = time.perf_counter()
    pass_summaries: list[dict[str, Any]] = []
    searches: list[SearchParams] = []

    def _startup_for(n_tr: int) -> int:
        if n_startup is None:
            return max(10, int(0.3 * n_tr))
        return int(n_startup)

    # --- Pass 1 ---
    fp, fpass, fnu = _snapshot_store(store)
    s1, sum1, _ = _run_pass_study(
        pass_no=1,
        cloud=cloud,
        plan=plan,
        vg=vg,
        space=space,
        store=store,
        frozen_pred=fp,
        frozen_pass=fpass,
        frozen_n_used=fnu,
        coverage_gate=c1,
        lexico=False,
        n_trials=trials_per_pass[0],
        seed=seed,
        n_startup=_startup_for(trials_per_pass[0]),
        n_ei=n_ei,
        flush_every=flush_every,
        force_octant=force_octant,
        out_pass=out / "pass_1",
        obj_penalty_cfg=penalty_cfg,
    )
    sum1["risky_extrapolation"] = risky_extrapolation(
        s1.r_major, vg.range_major, risky_factor
    )
    (out / "pass_1" / "best_params.json").write_text(
        json.dumps(sum1, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pass_summaries.append(sum1)
    searches.append(s1)

    # --- Pass 2 ---
    fp, fpass, fnu = _snapshot_store(store)
    s2, sum2, _ = _run_pass_study(
        pass_no=2,
        cloud=cloud,
        plan=plan,
        vg=vg,
        space=space,
        store=store,
        frozen_pred=fp,
        frozen_pass=fpass,
        frozen_n_used=fnu,
        coverage_gate=c2,
        lexico=False,
        n_trials=trials_per_pass[1],
        seed=seed,
        n_startup=_startup_for(trials_per_pass[1]),
        n_ei=n_ei,
        flush_every=flush_every,
        force_octant=force_octant,
        out_pass=out / "pass_2",
        obj_penalty_cfg=penalty_cfg,
    )
    sum2["risky_extrapolation"] = risky_extrapolation(
        s2.r_major, vg.range_major, risky_factor
    )
    (out / "pass_2" / "best_params.json").write_text(
        json.dumps(sum2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pass_summaries.append(sum2)
    searches.append(s2)

    # --- Pass 3 (lexicographic coverage, then RMSE) ---
    fp, fpass, fnu = _snapshot_store(store)
    s3, sum3, _ = _run_pass_study(
        pass_no=3,
        cloud=cloud,
        plan=plan,
        vg=vg,
        space=space,
        store=store,
        frozen_pred=fp,
        frozen_pass=fpass,
        frozen_n_used=fnu,
        coverage_gate=None,
        lexico=True,
        n_trials=trials_per_pass[2],
        seed=seed,
        n_startup=_startup_for(trials_per_pass[2]),
        n_ei=n_ei,
        flush_every=flush_every,
        force_octant=force_octant,
        out_pass=out / "pass_3",
        obj_penalty_cfg=penalty_cfg,
    )
    sum3["risky_extrapolation"] = risky_extrapolation(
        s3.r_major, vg.range_major, risky_factor
    )
    (out / "pass_3" / "best_params.json").write_text(
        json.dumps(sum3, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pass_summaries.append(sum3)
    searches.append(s3)

    rmse, mae = store.rmse_mae_all()
    residual_reasons: dict[str, int] = {}
    if store.n_invalid > 0:
        residual_reasons = diagnose_residuals(cloud, plan, vg, store, s3)

    n_by_pass = {
        1: int(np.sum(store.pass_id == 1)),
        2: int(np.sum(store.pass_id == 2)),
        3: int(np.sum(store.pass_id == 3)),
    }
    runtime = time.perf_counter() - t0
    total = {
        "CV_RMSE": rmse,
        "CV_MAE": mae,
        "prediction_coverage": store.coverage,
        "n_valid": store.n_valid,
        "n_invalid": store.n_invalid,
        "n_tgt": store.n_tgt,
        "n_by_pass": n_by_pass,
        "runtime_seconds": runtime,
        "residual_failure_reasons": residual_reasons,
        "coverage_targets": [c1, c2, 1.0],
        "variogram_fixed": vg.as_dict(),
        "cv": {
            "method": plan.method,
            "n_folds": plan.n_folds,
            "n_splits_requested": plan.n_splits_requested,
        },
        "seed": seed,
        "passes": pass_summaries,
        "search_params": [s.as_dict() for s in searches],
    }

    pred_rows = store.to_predictions_frame()
    pd.DataFrame(pred_rows).to_csv(out / "oof_predictions.csv", index=False)

    from .diagnostics import write_oof_diagnostic_plots

    if pred_rows:
        diag = write_oof_diagnostic_plots(
            np.array([r["y_true"] for r in pred_rows]),
            np.array([r["y_pred"] for r in pred_rows]),
            out / "diagnostics",
            pass_id=np.array([r["pass_id"] for r in pred_rows]),
            title_prefix="Multipass OOF",
        )
        total["quality"] = diag
    else:
        total["quality"] = {}

    (out / "multipass_result.json").write_text(
        json.dumps(total, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(out / "REPORT_multipass.md", pass_summaries, total)

    print(
        f"[multipass] done RMSE={rmse:.4g} coverage={store.coverage:.3f} "
        f"n_by_pass={n_by_pass} → {out}",
        flush=True,
    )
    return total
