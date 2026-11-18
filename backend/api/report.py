"""Assemble analyze report JSON from existing run artifacts — no math."""

from __future__ import annotations

import csv
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROXIMITY_FRAC = 0.05
COVERAGE_WARN = 0.97

CV_LABELS = {
    "group_kfold": "По скважинам (GroupKFold, HoleID)",
    "spatial_block": "Блочная пространственная кросс-валидация",
    "kfold": "K-fold кросс-валидация",
}

ANISO_PNG_LABELS = {
    "omnidirectional_variogram.png": "Всенаправленная вариограмма",
    "principal_axes.png": "Главные оси анизотропии",
    "major_variogram.png": "Направленная вариограмма, ось 1 (главная)",
    "intermediate_variogram.png": "Направленная вариограмма, ось 2 (промежуточная)",
    "minor_variogram.png": "Направленная вариограмма, ось 3 (малая)",
}

# Directional VGs and principal_axes hidden from report UI for now
# (still generated as run artifacts).
ANISO_PNG_ORDER = (
    "omnidirectional_variogram.png",
)

ANISO_PNG_HIDDEN = frozenset(
    {
        "principal_axes.png",
        "major_variogram.png",
        "intermediate_variogram.png",
        "minor_variogram.png",
    }
)

# OOF validation plots (QQ / scatter / swath) — shown after KPI metrics.
VALIDATION_PNG_LABELS = {
    "scatter.png": "Наблюдённое vs прогноз (OOF)",
    "qq_obs_pred.png": "QQ: квантили факта vs прогноза",
    "qq.png": "QQ: остатки vs нормальное распределение",
    "swath_x.png": "Swath вдоль X (средние OOF)",
    "swath_y.png": "Swath вдоль Y (средние OOF)",
    "swath_z.png": "Swath вдоль Z (средние OOF)",
}
VALIDATION_PNG_ORDER = (
    "scatter.png",
    "qq_obs_pred.png",
    "qq.png",
    "swath_x.png",
    "swath_y.png",
    "swath_z.png",
)


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _vector_to_azimuth_dip(v: Any) -> dict[str, float] | None:
    """Same reporting convention as anisotropy3d.report.vector_to_azimuth_dip.

    az 0=+X CCW to +Y; dip 0=horizontal, 90=+Z. Undirected: prefer +Z.
    """
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return None
    try:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
    except (TypeError, ValueError):
        return None
    import math

    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-18:
        return None
    x, y, z = x / norm, y / norm, z / norm
    if z < 0:
        x, y, z = -x, -y, -z
    horiz = math.hypot(x, y)
    dip = math.degrees(math.atan2(z, horiz))
    az = math.degrees(math.atan2(y, x)) % 360.0
    return {"azimuth_deg": az, "dip_deg": dip}


def _safe_int(v: Any) -> int | None:
    f = _safe_float(v)
    if f is None:
        return None
    return int(f)


def _safe_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _sector_fields(src: dict[str, Any]) -> dict[str, Any]:
    """Normalize Micromine / Datamine sector fields for the report UI."""
    octant = _safe_bool(src.get("octant_search"))
    if octant is None:
        octant = _safe_bool(src.get("sector_search"))
    out: dict[str, Any] = {
        "octant_search": octant,
        "sector_search": octant,
    }
    if octant:
        out["n_sectors"] = _safe_int(src.get("n_sectors"))
        out["MINOCT"] = _safe_int(src.get("MINOCT"))
        out["MINPEROC"] = _safe_int(src.get("MINPEROC"))
        out["MAXPEROC"] = _safe_int(src.get("MAXPEROC"))
        out["discard_underfilled"] = _safe_bool(src.get("discard_underfilled"))
        out["neighbour_cap"] = _safe_int(src.get("neighbour_cap"))
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _inspect_input_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"n_points": None, "has_hole_id": None}
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {"n_points": 0, "has_hole_id": False}
            cols = {c.strip() for c in reader.fieldnames if c}
            rows = list(reader)
        return {
            "n_points": len(rows),
            "has_hole_id": "HoleID" in cols,
        }
    except OSError:
        return {"n_points": None, "has_hole_id": None}


def _load_trials(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    trials: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trials.append(
                    {
                        "trial": _safe_int(row.get("trial")),
                        "method": (row.get("method") or None),
                        "power": _safe_float(row.get("power")),
                        "J": _safe_float(row.get("J")),
                        "R_major": _safe_float(row.get("R_major")),
                        "K_inter": _safe_float(row.get("K_inter")),
                        "K_minor": _safe_float(row.get("K_minor")),
                        "R_inter": _safe_float(row.get("R_inter")),
                        "R_minor": _safe_float(row.get("R_minor")),
                        "Nmin": _safe_int(row.get("Nmin")),
                        "Nmax": _safe_int(row.get("Nmax")),
                        "octant_search": _safe_bool(row.get("octant_search")),
                        "sector_search": _safe_bool(
                            row.get("sector_search", row.get("octant_search"))
                        ),
                        "n_sectors": _safe_int(row.get("n_sectors")),
                        "MINOCT": _safe_int(row.get("MINOCT")),
                        "MINPEROC": _safe_int(row.get("MINPEROC")),
                        "MAXPEROC": _safe_int(row.get("MAXPEROC")),
                        "objective": _safe_float(row.get("objective")),
                        "CV_RMSE": _safe_float(row.get("CV_RMSE")),
                        "CV_MAE": _safe_float(row.get("CV_MAE")),
                        "prediction_coverage": _safe_float(
                            row.get("prediction_coverage")
                        ),
                        "n_valid": _safe_int(row.get("n_valid")),
                        "n_invalid": _safe_int(row.get("n_invalid")),
                        "n_tgt": _safe_int(row.get("n_tgt")),
                    }
                )
    except OSError:
        return []
    return trials


def _proximity(
    value: float | int | None,
    lo: float | None,
    hi: float | None,
) -> dict[str, Any]:
    if value is None or lo is None or hi is None:
        return {
            "status": "unknown",
            "label": "Нет данных",
            "near_bound": None,
        }
    span = float(hi) - float(lo)
    if span <= 0:
        return {
            "status": "inside",
            "label": "В пределах диапазона",
            "near_bound": None,
        }
    margin = PROXIMITY_FRAC * span
    v = float(value)
    near_lo = v - float(lo) <= margin
    near_hi = float(hi) - v <= margin
    if near_lo or near_hi:
        return {
            "status": "near_bound",
            "label": "У края диапазона",
            "near_bound": "min" if near_lo and not near_hi else (
                "max" if near_hi and not near_lo else "both"
            ),
        }
    return {
        "status": "inside",
        "label": "В пределах диапазона",
        "near_bound": None,
    }


def _cv_label(cv: dict[str, Any] | None) -> str | None:
    if not cv or not isinstance(cv, dict):
        return None
    method = str(cv.get("method") or "").strip().lower()
    label = CV_LABELS.get(method, method or None)
    if not label:
        return None
    n_folds = cv.get("n_folds") or cv.get("n_splits_requested")
    if n_folds is not None:
        return f"{label}, {n_folds} фолдов"
    return label


def _list_aniso_images(aniso_dir: Path) -> list[dict[str, str]]:
    if not aniso_dir.is_dir():
        return []
    found: dict[str, Path] = {}
    for p in aniso_dir.glob("*.png"):
        found[p.name] = p
    images: list[dict[str, str]] = []
    for name in ANISO_PNG_ORDER:
        if name in found:
            images.append(
                {
                    "name": name,
                    "path": f"anisotropy/{name}",
                    "label": ANISO_PNG_LABELS.get(
                        name, name.replace(".png", "").replace("_", " ")
                    ),
                }
            )
            del found[name]
    for name in sorted(found):
        if name in ANISO_PNG_HIDDEN:
            continue
        images.append(
            {
                "name": name,
                "path": f"anisotropy/{name}",
                "label": ANISO_PNG_LABELS.get(
                    name, name.replace(".png", "").replace("_", " ")
                ),
            }
        )
    return images


def _list_validation_images(tune_dir: Path) -> list[dict[str, str]]:
    diag = tune_dir / "oof_diagnostics"
    if not diag.is_dir():
        return []
    found = {p.name: p for p in diag.glob("*.png")}
    images: list[dict[str, str]] = []
    for name in VALIDATION_PNG_ORDER:
        if name not in found:
            continue
        images.append(
            {
                "name": name,
                "path": f"kriging_tuning/oof_diagnostics/{name}",
                "label": VALIDATION_PNG_LABELS.get(
                    name, name.replace(".png", "").replace("_", " ")
                ),
            }
        )
    return images


def write_run_meta(
    run_dir: Path,
    *,
    original_filename: str | None,
    n_points: int,
    has_hole_id: bool,
    method: str | None = None,
) -> None:
    meta: dict[str, Any] = {
        "original_filename": original_filename or None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_points": n_points,
        "has_hole_id": has_hole_id,
    }
    if method:
        meta["method"] = method
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_report(run_id: str, run_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    meta = _load_json(run_dir / "run_meta.json") or {}
    input_meta = _inspect_input_csv(run_dir / "input.csv")

    n_points = meta.get("n_points")
    if n_points is None:
        n_points = input_meta.get("n_points")
    has_hole_id = meta.get("has_hole_id")
    if has_hole_id is None:
        has_hole_id = input_meta.get("has_hole_id")

    created_at = meta.get("created_at")
    if not created_at:
        try:
            created_at = datetime.fromtimestamp(
                run_dir.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            created_at = None

    result = _load_json(run_dir / "anisotropy" / "result.json")
    if result is None:
        missing.append("anisotropy/result.json")
        result = {}
    best = _load_json(run_dir / "kriging_tuning" / "best_params.json")
    if best is None:
        missing.append("kriging_tuning/best_params.json")
        best = {}

    cand = result.get("aniso_candidate") or {}
    if not isinstance(cand, dict):
        cand = {}

    trials = _load_trials(run_dir / "kriging_tuning" / "trials.csv")
    if not trials and not (run_dir / "kriging_tuning" / "trials.csv").is_file():
        missing.append("kriging_tuning/trials.csv")

    best_trial_number = best.get("best_trial_number")
    best_trial_row: dict[str, Any] | None = None
    if best_trial_number is not None:
        for t in trials:
            if t.get("trial") == int(best_trial_number):
                best_trial_row = t
                break

    search_space = best.get("search_space") or {}
    if not isinstance(search_space, dict):
        search_space = {}

    method = best.get("method") or meta.get("method") or "ordinary_kriging"
    idw_cfg = best.get("idw_config") if isinstance(best.get("idw_config"), dict) else {}
    # Expose power bounds in search_space for proximity / report
    if method == "anisotropic_idw" and "power" not in search_space:
        p_lo = _safe_float(idw_cfg.get("power_min"))
        p_hi = _safe_float(idw_cfg.get("power_max"))
        if p_lo is not None and p_hi is not None:
            search_space = {**search_space, "power": [p_lo, p_hi]}

    param_keys = [
        ("R_major", "R_major"),
        ("K_inter", "K_inter"),
        ("K_minor", "K_minor"),
        ("Nmin", "Nmin"),
        ("Nmax", "Nmax"),
    ]
    if method == "anisotropic_idw":
        param_keys.insert(0, ("power", "power"))
    proximity: dict[str, Any] = {}
    any_near_bound = False
    for key, space_key in param_keys:
        bounds = search_space.get(space_key)
        lo = hi = None
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            lo = _safe_float(bounds[0])
            hi = _safe_float(bounds[1])
        val = best.get(key)
        if key in ("Nmin", "Nmax"):
            val = _safe_int(val) if val is not None else None
        else:
            val = _safe_float(val)
        prox = _proximity(val, lo, hi)
        proximity[key] = {
            **prox,
            "value": val,
            "lo": lo,
            "hi": hi,
        }
        if prox["status"] == "near_bound":
            any_near_bound = True

    oof_path = run_dir / "kriging_tuning" / "oof_predictions.csv"
    report_idw_path = run_dir / "kriging_tuning" / "REPORT_idw.md"
    oof_available = oof_path.is_file()
    extra_artifacts: list[dict[str, str]] = []
    if report_idw_path.is_file():
        extra_artifacts.append(
            {
                "name": "REPORT_idw.md",
                "path": "kriging_tuning/REPORT_idw.md",
                "label": "Отчёт IDW (markdown)",
            }
        )
    if oof_available:
        extra_artifacts.append(
            {
                "name": "oof_predictions.csv",
                "path": "kriging_tuning/oof_predictions.csv",
                "label": "OOF-прогнозы",
            }
        )

    ranked = sorted(
        [t for t in trials if t.get("objective") is not None],
        key=lambda t: float(t["objective"]),
    )
    top10 = ranked[:10]

    cv = best.get("cv") if isinstance(best.get("cv"), dict) else None
    coverage = _safe_float(best.get("prediction_coverage"))
    coverage_below = coverage is not None and coverage < COVERAGE_WARN

    images = _list_aniso_images(run_dir / "anisotropy")
    validation_images = _list_validation_images(run_dir / "kriging_tuning")

    major_xyz = cand.get("major_axis_xyz", result.get("major_axis_xyz"))
    inter_xyz = cand.get(
        "intermediate_axis_xyz", result.get("intermediate_axis_xyz")
    )
    minor_xyz = cand.get("minor_axis_xyz", result.get("minor_axis_xyz"))
    major_ang = _vector_to_azimuth_dip(major_xyz)
    inter_ang = _vector_to_azimuth_dip(inter_xyz)
    minor_ang = _vector_to_azimuth_dip(minor_xyz)

    spatial = {
        "anisotropy_type": result.get("type"),
        "variogram_model": cand.get("variogram_model")
        or result.get("variogram_model"),
        "nugget": cand.get("nugget", result.get("nugget")),
        "sill": cand.get("sill", result.get("sill")),
        "range_major": cand.get("range_major", result.get("range_major")),
        "range_intermediate": cand.get(
            "range_intermediate", result.get("range_intermediate")
        ),
        "range_minor": cand.get("range_minor", result.get("range_minor")),
        "orientation_matrix": cand.get(
            "orientation_matrix", result.get("orientation_matrix")
        ),
        "major_axis_xyz": major_xyz,
        "intermediate_axis_xyz": inter_xyz,
        "minor_axis_xyz": minor_xyz,
        "major_azimuth_deg": (major_ang or {}).get("azimuth_deg"),
        "major_dip_deg": (major_ang or {}).get("dip_deg"),
        "intermediate_azimuth_deg": (inter_ang or {}).get("azimuth_deg"),
        "intermediate_dip_deg": (inter_ang or {}).get("dip_deg"),
        "minor_azimuth_deg": (minor_ang or {}).get("azimuth_deg"),
        "minor_dip_deg": (minor_ang or {}).get("dip_deg"),
        "moi_strength": cand.get("moi_strength"),
        "ellipsoid_kind": "variogram_continuity",
    }

    n_valid = best.get("n_valid")
    if n_valid is None and best_trial_row:
        n_valid = best_trial_row.get("n_valid")
    n_tgt = best.get("n_tgt")
    if n_tgt is None and best_trial_row:
        n_tgt = best_trial_row.get("n_tgt")
    n_invalid = best.get("n_invalid")
    if n_invalid is None and best_trial_row:
        n_invalid = best_trial_row.get("n_invalid")

    sector = _sector_fields(best)
    best_params: dict[str, Any] = {
        "R_major": best.get("R_major"),
        "R_inter": best.get("R_inter"),
        "R_minor": best.get("R_minor"),
        "K_inter": best.get("K_inter"),
        "K_minor": best.get("K_minor"),
        "Nmin": best.get("Nmin"),
        "Nmax": best.get("Nmax"),
        **sector,
    }
    if method == "anisotropic_idw":
        best_params["power"] = best.get("power")
        best_params["mean_n_used"] = best.get("mean_n_used")
        if best.get("J") is not None:
            best_params["J"] = best.get("J")

    coverage_fill = _build_coverage_fill_section(best, method)

    return {
        "run_id": run_id,
        "status": "completed" if best and result else "partial",
        "missing_artifacts": missing,
        "header": {
            "run_id": run_id,
            "filename": meta.get("original_filename"),
            "n_points": n_points,
            "has_hole_id": has_hole_id,
            "cv_method": _cv_label(cv),
            "cv": cv,
            "created_at": created_at,
            "method": method,
            "method_label": (
                "Анизотропный IDW"
                if method == "anisotropic_idw"
                else "Ordinary Kriging"
            ),
        },
        "kpi": {
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": best.get("prediction_coverage"),
            "J": best.get("J"),
            "coverage_fill_triggered": bool(
                (coverage_fill or {}).get("triggered")
            ),
            "n_passes": int((coverage_fill or {}).get("n_passes") or 1),
        },
        "spatial_structure": spatial,
        "search_params": {
            "method": method,
            "power": best.get("power"),
            "mean_n_used": best.get("mean_n_used"),
            "R_major": best.get("R_major"),
            "R_inter": best.get("R_inter"),
            "R_minor": best.get("R_minor"),
            "K_inter": best.get("K_inter"),
            "K_minor": best.get("K_minor"),
            "Nmin": best.get("Nmin"),
            "Nmax": best.get("Nmax"),
            **sector,
        },
        "coverage_fill": coverage_fill,
        "validation": {
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": coverage,
            "n_valid": n_valid,
            "n_tgt": n_tgt,
            "n_invalid": n_invalid,
            "oof_available": oof_available,
            "coverage_below_threshold": coverage_below,
            "coverage_threshold": COVERAGE_WARN,
            "coverage_fill_triggered": bool(
                (coverage_fill or {}).get("triggered")
            ),
        },
        "optimization": {
            "best_trial_number": best_trial_number,
            "n_trials": best.get("n_trials") or len(trials) or None,
            "objective": best.get("objective"),
            "loss": best.get("loss"),
            "best_params": best_params,
            "search_space": search_space,
            "proximity": proximity,
            "any_near_bound": any_near_bound,
            "trials": trials,
            "top10": top10,
        },
        "artifacts": {
            "images": images,
            "validation_images": validation_images,
            "files": extra_artifacts,
        },
        "conclusion_inputs": {
            "method": method,
            "power": best.get("power"),
            "mean_n_used": best.get("mean_n_used"),
            "anisotropy_type": spatial.get("anisotropy_type"),
            "variogram_model": spatial.get("variogram_model"),
            "nugget": spatial.get("nugget"),
            "sill": spatial.get("sill"),
            "range_major": spatial.get("range_major"),
            "range_intermediate": spatial.get("range_intermediate"),
            "range_minor": spatial.get("range_minor"),
            "major_azimuth_deg": spatial.get("major_azimuth_deg"),
            "major_dip_deg": spatial.get("major_dip_deg"),
            "intermediate_azimuth_deg": spatial.get("intermediate_azimuth_deg"),
            "intermediate_dip_deg": spatial.get("intermediate_dip_deg"),
            "minor_azimuth_deg": spatial.get("minor_azimuth_deg"),
            "minor_dip_deg": spatial.get("minor_dip_deg"),
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": coverage,
            "any_near_bound": any_near_bound,
            "coverage_below_threshold": coverage_below,
            "coverage_threshold": COVERAGE_WARN,
            "Nmin": best.get("Nmin"),
            "Nmax": best.get("Nmax"),
            "R_major": best.get("R_major"),
            "R_inter": best.get("R_inter"),
            "R_minor": best.get("R_minor"),
            **sector,
            "coverage_fill_triggered": bool(
                (coverage_fill or {}).get("triggered")
            ),
            "coverage_fill": coverage_fill,
        },
    }


def _pass_card(raw: dict[str, Any] | None, *, method: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {
        "pass": _safe_int(raw.get("pass")),
        "label": None,
        "role": None,
        "loss": raw.get("loss"),
        "power": _safe_float(raw.get("power")),
        "R_major": _safe_float(raw.get("R_major")),
        "R_inter": _safe_float(raw.get("R_inter")),
        "R_minor": _safe_float(raw.get("R_minor")),
        "K_inter": _safe_float(raw.get("K_inter")),
        "K_minor": _safe_float(raw.get("K_minor")),
        "Nmin": _safe_int(raw.get("Nmin")),
        "Nmax": _safe_int(raw.get("Nmax")),
        "CV_RMSE": _safe_float(
            raw.get("CV_RMSE")
            if raw.get("CV_RMSE") is not None
            else raw.get("CV_RMSE_all")
        ),
        "CV_MAE": _safe_float(
            raw.get("CV_MAE")
            if raw.get("CV_MAE") is not None
            else raw.get("CV_MAE_all")
        ),
        "local_CV_RMSE": _safe_float(raw.get("local_CV_RMSE")),
        "local_CV_MAE": _safe_float(raw.get("local_CV_MAE")),
        "prediction_coverage": _safe_float(raw.get("prediction_coverage")),
        "n_new": _safe_int(raw.get("n_new")),
        "n_trials": _safe_int(raw.get("n_trials")),
        **_sector_fields(raw),
    }
    pno = out.get("pass")
    if pno == 1:
        out["label"] = "Проход 1 — основной подбор"
        out["role"] = "primary"
        if method == "anisotropic_idw" and not out.get("loss"):
            out["loss"] = "j"
    elif pno == 2:
        out["label"] = "Проход 2 — дозаполнение покрытия"
        out["role"] = "fill"
        out["loss"] = out.get("loss") or "rmse"
    else:
        out["label"] = f"Проход {pno}" if pno is not None else "Проход"
        out["role"] = "other"
    return out


def _build_coverage_fill_section(
    best: dict[str, Any], method: str
) -> dict[str, Any]:
    """UI-friendly two-pass summary; safe when coverage_fill absent."""
    raw = best.get("coverage_fill")
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enabled", False))
    triggered = bool(raw.get("triggered", False))
    passes_raw = best.get("passes")
    passes: list[dict[str, Any]] = []
    if isinstance(passes_raw, list):
        for item in passes_raw:
            card = _pass_card(item if isinstance(item, dict) else None, method=method)
            if card:
                passes.append(card)
    if not passes and not triggered:
        # Single-pass legacy / fill skipped: synthesize pass-1 card from top-level
        passes.append(
            {
                "pass": 1,
                "label": "Проход 1 — основной подбор",
                "role": "primary",
                "loss": best.get("loss"),
                "power": _safe_float(best.get("power")),
                "R_major": _safe_float(best.get("R_major")),
                "R_inter": _safe_float(best.get("R_inter")),
                "R_minor": _safe_float(best.get("R_minor")),
                "K_inter": _safe_float(best.get("K_inter")),
                "K_minor": _safe_float(best.get("K_minor")),
                "Nmin": _safe_int(best.get("Nmin")),
                "Nmax": _safe_int(best.get("Nmax")),
                "CV_RMSE": _safe_float(best.get("CV_RMSE")),
                "CV_MAE": _safe_float(best.get("CV_MAE")),
                "local_CV_RMSE": None,
                "local_CV_MAE": None,
                "prediction_coverage": _safe_float(best.get("prediction_coverage")),
                "n_new": _safe_int(best.get("n_valid")),
                "n_trials": _safe_int(best.get("n_trials")),
                **_sector_fields(best),
            }
        )

    n_by_pass = raw.get("n_by_pass") if isinstance(raw.get("n_by_pass"), dict) else {}
    return {
        "enabled": enabled,
        "triggered": triggered,
        "n_passes": 2 if triggered else 1,
        "trigger_below": _safe_float(raw.get("trigger_below")),
        "pass1_coverage": _safe_float(raw.get("pass1_coverage")),
        "pass2_coverage": _safe_float(raw.get("pass2_coverage")),
        "pass2_coverage_min": _safe_float(raw.get("pass2_coverage_min")),
        "r_expand_factor": _safe_float(raw.get("r_expand_factor")),
        "n_by_pass": {
            "1": _safe_int(n_by_pass.get("1")),
            "2": _safe_int(n_by_pass.get("2")),
        },
        "residual_failure_reasons": raw.get("residual_failure_reasons")
        if isinstance(raw.get("residual_failure_reasons"), dict)
        else {},
        "passes": passes,
        "summary_ru": (
            "После первого прохода покрытие было неполным — выполнен второй "
            "проход с целевой функцией RMSE и требованием покрытия 100%. "
            "Точки, оценённые на проходе 1, не перезаписываются."
            if triggered
            else (
                "Второй проход не потребовался: покрытие после основного "
                "подбора уже достаточное."
                if enabled
                else None
            )
        ),
    }


def resolve_artifact(run_dir: Path, rel_path: str) -> Path:
    """Resolve artifact path under run_dir; raise ValueError if unsafe/missing."""
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path:
        raise ValueError("invalid artifact path")
    # normalize and reject traversal
    candidate = (run_dir / rel_path).resolve()
    root = run_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes run directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(str(rel_path))
    return candidate


def artifact_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"
