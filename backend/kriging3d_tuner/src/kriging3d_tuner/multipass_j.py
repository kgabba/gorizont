"""Experimental sequential multipass Optuna minimizing loss J.

J = (sum SE over estimated OOF + sum λ_f over unevaluated OOF) / N
λ_f = Var(y_train_f) per fold (no test leakage).
"""

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
    evaluate_oof,
    evaluate_oof_masked,
    load_config,
    load_points_csv,
    load_variogram_json,
    oof_target_indices,
)
from .models import (
    CVPlan,
    ObjectiveConfig,
    PointCloud,
    SearchParams,
    SearchSpaceConfig,
    VariogramFixed,
)
from .oof_store import OOFStore
from .objective import make_optuna_objective
from .search_space import (
    build_search_space,
    search_params_from_trial_params,
    suggest_search_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_EPS = 1e-12


def compute_fold_lambdas(cloud: PointCloud, plan: CVPlan) -> dict[int, float]:
    """λ_f = Var(y_train) per fold index; train-only."""
    out: dict[int, float] = {}
    for fi, fold in enumerate(plan.folds):
        y = np.asarray(cloud.values[fold.train_idx], dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 2:
            lam = float(max(_EPS, float(np.mean(y * y)) if y.size else _EPS))
        else:
            lam = float(np.var(y, ddof=1))
            if not np.isfinite(lam) or lam <= _EPS:
                lam = float(max(_EPS, float(np.mean(y * y))))
        out[fi] = lam
    return out


def build_point_fold_map(plan: CVPlan) -> dict[int, int]:
    """Map global OOF test index → fold index (first fold if duplicated)."""
    m: dict[int, int] = {}
    for fi, fold in enumerate(plan.folds):
        for gi in fold.test_idx.tolist():
            gi = int(gi)
            if gi not in m:
                m[gi] = fi
    return m


def compute_J(
    store: OOFStore,
    oof_indices: np.ndarray,
    point_fold: dict[int, int],
    lambdas: dict[int, float],
) -> float:
    """Pooled J over all OOF targets in ``oof_indices``."""
    oof = np.asarray(oof_indices, dtype=int).ravel()
    n = int(oof.size)
    if n == 0:
        return float(_EPS)
    total = 0.0
    for gi in oof:
        gi = int(gi)
        fi = point_fold[gi]
        lam = float(lambdas[fi])
        if store.pass_id[gi] > 0 and np.isfinite(store.pred[gi]):
            e = float(store.pred[gi] - store.y_true[gi])
            total += e * e
        else:
            total += lam
    return float(total / n)


def _snapshot(store: OOFStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return store.pred.copy(), store.pass_id.copy(), store.n_used.copy()


def _restore(
    store: OOFStore, pred: np.ndarray, pass_id: np.ndarray, n_used: np.ndarray
) -> None:
    store.pred[:] = pred
    store.pass_id[:] = pass_id
    store.n_used[:] = n_used


def _apply_search(
    store: OOFStore,
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    search: SearchParams,
    pass_no: int,
    frozen: tuple[np.ndarray, np.ndarray, np.ndarray],
    oof_indices: np.ndarray,
    point_fold: dict[int, int],
    lambdas: dict[int, float],
) -> dict[str, Any]:
    _restore(store, *frozen)
    unset = [int(i) for i in oof_indices.tolist() if store.pass_id[int(i)] == 0]
    idx, preds, n_used = evaluate_oof_masked(
        cloud, plan, vg, search, np.asarray(unset, dtype=int)
    )
    n_new = store.merge(idx, preds, n_used, pass_no)
    j = compute_J(store, oof_indices, point_fold, lambdas)
    rmse, mae = store.rmse_mae_all()
    local = store.pass_local_metrics(pass_no)
    return {
        "J": j,
        "n_new": n_new,
        "coverage": store.coverage,
        "n_valid": store.n_valid,
        "n_invalid": store.n_invalid,
        "n_tgt": store.n_tgt,
        "CV_RMSE": rmse,
        "CV_MAE": mae,
        "local_CV_RMSE": local["CV_RMSE"],
        "local_CV_MAE": local["CV_MAE"],
        "local_mean_n_used": local["mean_n_used"],
        "pct_new": local["pct_new"],
    }


def _run_pass_optuna(
    *,
    pass_no: int,
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    store: OOFStore,
    frozen: tuple[np.ndarray, np.ndarray, np.ndarray],
    oof_indices: np.ndarray,
    point_fold: dict[int, int],
    lambdas: dict[int, float],
    n_trials: int,
    seed: int,
    n_startup: int,
    n_ei: int,
    flush_every: int,
    out_pass: Path,
) -> tuple[SearchParams, dict[str, Any], float]:
    out_pass.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_search_params(
            trial, space, force_octant_search=False
        )
        m = _apply_search(
            store,
            cloud,
            plan,
            vg,
            params,
            pass_no,
            frozen,
            oof_indices,
            point_fold,
            lambdas,
        )
        trial.set_user_attr("J", m["J"])
        trial.set_user_attr("CV_RMSE", m["CV_RMSE"])
        trial.set_user_attr("CV_MAE", m["CV_MAE"])
        trial.set_user_attr("prediction_coverage", m["coverage"])
        trial.set_user_attr("n_new", m["n_new"])
        trial.set_user_attr("n_valid", m["n_valid"])
        trial.set_user_attr("n_invalid", m["n_invalid"])
        trial.set_user_attr("R_inter", params.r_inter)
        trial.set_user_attr("R_minor", params.r_minor)
        return float(m["J"])

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        p = trial.params
        ua = trial.user_attrs
        rows.append(
            {
                "trial": trial.number,
                "R_major": p.get("R_major"),
                "K_inter": p.get("K_inter"),
                "K_minor": p.get("K_minor"),
                "Nmin": p.get("Nmin"),
                "Nmax": p.get("Nmax"),
                "objective_J": trial.value,
                "CV_RMSE": ua.get("CV_RMSE"),
                "CV_MAE": ua.get("CV_MAE"),
                "prediction_coverage": ua.get("prediction_coverage"),
                "n_new": ua.get("n_new"),
            }
        )
        if len(rows) % flush_every == 0:
            pd.DataFrame(rows).to_csv(out_pass / "trials.csv", index=False)
        if study.best_trial is not None and trial.number == study.best_trial.number:
            print(
                f"[multipass_j] pass={pass_no} best trial={trial.number} "
                f"J={trial.value:.6g} cov={ua.get('prediction_coverage'):.3f} "
                f"n_new={ua.get('n_new')}",
                flush=True,
            )

    sampler = TPESampler(
        seed=seed + pass_no,
        n_startup_trials=int(n_startup),
        n_ei_candidates=n_ei,
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    elapsed = time.perf_counter() - t0
    pd.DataFrame(rows).to_csv(out_pass / "trials.csv", index=False)

    search = search_params_from_trial_params(study.best_trial.params)
    final = _apply_search(
        store,
        cloud,
        plan,
        vg,
        search,
        pass_no,
        frozen,
        oof_indices,
        point_fold,
        lambdas,
    )
    summary = {
        "pass": pass_no,
        "accepted": True,
        "search": search.as_dict(),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": n_trials,
        "runtime_seconds": elapsed,
        "J": final["J"],
        "CV_RMSE": final["CV_RMSE"],
        "CV_MAE": final["CV_MAE"],
        "prediction_coverage": final["coverage"],
        "n_new": final["n_new"],
        "pct_new": final["pct_new"],
        "coverage_delta": final["pct_new"],
        "n_valid": final["n_valid"],
        "n_invalid": final["n_invalid"],
        "n_tgt": final["n_tgt"],
        "local_CV_RMSE": final["local_CV_RMSE"],
        "local_CV_MAE": final["local_CV_MAE"],
        "mean_n_used": final["local_mean_n_used"],
    }
    return search, summary, float(final["J"])


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# sequential_multipass_optuna_J",
        "",
        f"- Accepted passes: **{result['n_accepted_passes']}**",
        f"- Stop reason: **{result['stop_reason']}**",
        f"- Final J: **{result['J']:.6g}**",
        f"- Final RMSE / MAE: **{result['CV_RMSE']:.6g}** / **{result['CV_MAE']:.6g}**",
        f"- Coverage: **{100.0 * result['prediction_coverage']:.4g}%**",
        "",
        "## Passes",
        "",
        "| pass | accepted | J | RMSE | MAE | n_new | cov% | trials | t_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in result["passes"]:
        lines.append(
            "| {pno} | {acc} | {J:.4g} | {rmse:.4g} | {mae:.4g} | {n_new} | "
            "{cov:.2f} | {tr} | {t:.3g} |".format(
                pno=p["pass"],
                acc=p.get("accepted"),
                J=p["J"],
                rmse=p["CV_RMSE"],
                mae=p["CV_MAE"],
                n_new=p.get("n_new", 0),
                cov=100.0 * p.get("prediction_coverage", 0.0),
                tr=p.get("n_trials", "—"),
                t=p.get("runtime_seconds", float("nan")),
            )
        )
    base = result.get("baseline_single") or {}
    if base:
        lines += [
            "",
            "## Baseline single-pass OK",
            "",
            f"- J: **{base.get('J', float('nan')):.6g}**",
            f"- RMSE / MAE: **{base.get('CV_RMSE', float('nan')):.6g}** / "
            f"**{base.get('CV_MAE', float('nan')):.6g}**",
            f"- Coverage: **{100.0 * float(base.get('prediction_coverage', 0)):.4g}%**",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_baseline_single(
    cloud: PointCloud,
    plan: CVPlan,
    vg: VariogramFixed,
    space: SearchSpaceConfig,
    oof_indices: np.ndarray,
    point_fold: dict[int, int],
    lambdas: dict[int, float],
    obj_cfg: ObjectiveConfig,
    *,
    n_trials: int,
    seed: int,
    n_startup: int,
    n_ei: int,
) -> dict[str, Any]:
    """Single-pass Optuna with the same coverage-gated objective as ``run_tuner``."""
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=int(n_startup),
        n_ei_candidates=n_ei,
        multivariate=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_optuna_objective(
        cloud, plan, vg, space, obj_cfg, force_octant_search=False
    )
    study.optimize(objective, n_trials=n_trials)
    search = search_params_from_trial_params(study.best_trial.params)
    metrics = evaluate_oof(cloud, plan, vg, search, obj_cfg)
    store = OOFStore(n_tgt=int(oof_indices.size), y_true=cloud.values.copy())
    idx, preds, n_used = evaluate_oof_masked(
        cloud, plan, vg, search, oof_indices
    )
    store.merge(idx, preds, n_used, pass_no=1)
    j = compute_J(store, oof_indices, point_fold, lambdas)
    return {
        "search": search.as_dict(),
        "J": j,
        "CV_RMSE": metrics.cv_rmse,
        "CV_MAE": metrics.cv_mae,
        "prediction_coverage": metrics.prediction_coverage,
        "n_valid": metrics.n_valid,
        "n_invalid": metrics.n_invalid,
        "n_tgt": metrics.n_tgt,
        "n_trials": n_trials,
        "best_trial_number": int(study.best_trial.number),
        "objective": float(study.best_value),
        "objective_used": "coverage_gated_rmse",
    }


def run_sequential_multipass_optuna_j(
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

    jcfg = cfg.get("multipass_j") or {}
    max_passes = int(jcfg.get("max_passes", 3))
    max_passes = max(1, min(3, max_passes))
    compare_single = bool(jcfg.get("compare_single", True))
    raw_trials = jcfg.get("n_trials_per_pass", (cfg.get("optuna") or {}).get("n_trials", 200))
    if n_trials is not None:
        trials_per_pass = [int(n_trials)] * max_passes
    elif isinstance(raw_trials, (list, tuple)):
        trials_per_pass = [int(x) for x in raw_trials][:max_passes]
        while len(trials_per_pass) < max_passes:
            trials_per_pass.append(trials_per_pass[-1])
    else:
        trials_per_pass = [int(raw_trials)] * max_passes

    cloud = load_points_csv(data_path, cfg)
    vg = load_variogram_json(anisotropy_path)
    plan = build_folds(cloud, cfg)
    space = build_search_space(vg, cfg)

    oof_indices = oof_target_indices(plan)
    n_tgt = int(oof_indices.size)
    store = OOFStore(n_tgt=n_tgt, y_true=cloud.values.copy())
    lambdas = compute_fold_lambdas(cloud, plan)
    point_fold = build_point_fold_map(plan)

    seed = int(cfg.get("seed", 42))
    ocfg = cfg.get("optuna") or {}
    n_ei = int(ocfg.get("n_ei_candidates", 64))
    flush_every = int(ocfg.get("flush_every", 25))

    def _startup(n_tr: int) -> int:
        ns = ocfg.get("n_startup_trials")
        if ns is None:
            return max(10, int(0.3 * n_tr))
        return int(ns)

    (out / "_run_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    print(
        f"[multipass_j] n={cloud.n} oof={n_tgt} folds={plan.n_folds} "
        f"max_passes={max_passes} trials/pass={trials_per_pass}",
        flush=True,
    )

    t0 = time.perf_counter()
    pass_records: list[dict[str, Any]] = []
    accepted_searches: list[dict[str, Any]] = []
    j_prev: float | None = None
    stop_reason = "max_passes"

    for pass_no in range(1, max_passes + 1):
        frozen = _snapshot(store)
        cov_before = store.coverage
        search, summary, j_best = _run_pass_optuna(
            pass_no=pass_no,
            cloud=cloud,
            plan=plan,
            vg=vg,
            space=space,
            store=store,
            frozen=frozen,
            oof_indices=oof_indices,
            point_fold=point_fold,
            lambdas=lambdas,
            n_trials=trials_per_pass[pass_no - 1],
            seed=seed,
            n_startup=_startup(trials_per_pass[pass_no - 1]),
            n_ei=n_ei,
            flush_every=flush_every,
            out_pass=out / f"pass_{pass_no}",
        )
        summary["coverage_before"] = cov_before
        summary["coverage_delta"] = summary["prediction_coverage"] - cov_before

        if pass_no == 1:
            summary["accepted"] = True
            (out / f"pass_{pass_no}" / "best_params.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            pass_records.append(summary)
            accepted_searches.append(search.as_dict())
            j_prev = j_best
        else:
            assert j_prev is not None
            if j_best < j_prev:
                summary["accepted"] = True
                (out / f"pass_{pass_no}" / "best_params.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                pass_records.append(summary)
                accepted_searches.append(search.as_dict())
                j_prev = j_best
            else:
                _restore(store, *frozen)
                summary["accepted"] = False
                summary["reject_reason"] = "no_j_improvement"
                (out / f"pass_{pass_no}" / "rejected.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                pass_records.append(summary)
                stop_reason = "no_j_improvement"
                print(
                    f"[multipass_j] pass={pass_no} rejected "
                    f"(J={j_best:.6g} >= prev {j_prev:.6g})",
                    flush=True,
                )
                break

        if store.coverage >= 1.0 - 1e-15:
            stop_reason = "coverage_100"
            print(f"[multipass_j] coverage 100% after pass {pass_no}", flush=True)
            break
    else:
        stop_reason = "max_passes"

    rmse, mae = store.rmse_mae_all()
    j_final = compute_J(store, oof_indices, point_fold, lambdas)

    ocfg_raw = cfg.get("objective") or {}
    obj_cfg = ObjectiveConfig(
        coverage_min=float(ocfg_raw.get("coverage_min", 0.97)),
        coverage_penalty=float(ocfg_raw.get("coverage_penalty", 1.0e6)),
        empty_rmse_fallback=float(ocfg_raw.get("empty_rmse_fallback", 1.0e6)),
    )

    baseline: dict[str, Any] | None = None
    if compare_single:
        print("[multipass_j] baseline single-pass Optuna…", flush=True)
        baseline = _run_baseline_single(
            cloud,
            plan,
            vg,
            space,
            oof_indices,
            point_fold,
            lambdas,
            obj_cfg,
            n_trials=trials_per_pass[0],
            seed=seed,
            n_startup=_startup(trials_per_pass[0]),
            n_ei=n_ei,
        )

    result: dict[str, Any] = {
        "mode": "sequential_multipass_optuna_J",
        "n_accepted_passes": sum(1 for p in pass_records if p.get("accepted")),
        "stop_reason": stop_reason,
        "J": j_final,
        "CV_RMSE": rmse,
        "CV_MAE": mae,
        "prediction_coverage": store.coverage,
        "n_valid": store.n_valid,
        "n_invalid": store.n_invalid,
        "n_tgt": store.n_tgt,
        "fold_lambdas": {str(k): v for k, v in lambdas.items()},
        "passes": pass_records,
        "accepted_search_params": accepted_searches,
        "baseline_single": baseline,
        "runtime_seconds": time.perf_counter() - t0,
        "seed": seed,
        "cv": {
            "method": plan.method,
            "n_folds": plan.n_folds,
            "n_splits_requested": plan.n_splits_requested,
        },
        "variogram_fixed": vg.as_dict(),
        "trials_per_pass": trials_per_pass,
    }

    pd.DataFrame(store.to_predictions_frame()).to_csv(
        out / "oof_predictions.csv", index=False
    )
    (out / "multipass_j_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(out / "REPORT_multipass_j.md", result)
    print(
        f"[multipass_j] done accepted={result['n_accepted_passes']} "
        f"stop={stop_reason} J={j_final:.6g} RMSE={rmse:.6g} "
        f"cov={store.coverage:.3f} → {out}",
        flush=True,
    )
    return result
