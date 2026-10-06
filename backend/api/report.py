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
    "group_kfold": "GroupKFold (по HoleID)",
    "spatial_block": "Spatial block CV",
    "kfold": "KFold",
}

ANISO_PNG_ORDER = (
    "omnidirectional_variogram.png",
    "principal_axes.png",
    "major_variogram.png",
    "intermediate_variogram.png",
    "minor_variogram.png",
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
                        "R_major": _safe_float(row.get("R_major")),
                        "K_inter": _safe_float(row.get("K_inter")),
                        "K_minor": _safe_float(row.get("K_minor")),
                        "R_inter": _safe_float(row.get("R_inter")),
                        "R_minor": _safe_float(row.get("R_minor")),
                        "Nmin": _safe_int(row.get("Nmin")),
                        "Nmax": _safe_int(row.get("Nmax")),
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
            "label": "Нет данных о диапазоне",
            "near_bound": None,
        }
    span = float(hi) - float(lo)
    if span <= 0:
        return {
            "status": "inside",
            "label": "Внутри диапазона",
            "near_bound": None,
        }
    margin = PROXIMITY_FRAC * span
    v = float(value)
    near_lo = v - float(lo) <= margin
    near_hi = float(hi) - v <= margin
    if near_lo or near_hi:
        return {
            "status": "near_bound",
            "label": "Близко к границе диапазона",
            "near_bound": "min" if near_lo and not near_hi else (
                "max" if near_hi and not near_lo else "both"
            ),
        }
    return {
        "status": "inside",
        "label": "Внутри диапазона",
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
        return f"{label}, {n_folds} folds"
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
                    "label": name.replace(".png", "").replace("_", " "),
                }
            )
            del found[name]
    for name in sorted(found):
        images.append(
            {
                "name": name,
                "path": f"anisotropy/{name}",
                "label": name.replace(".png", "").replace("_", " "),
            }
        )
    return images


def write_run_meta(
    run_dir: Path,
    *,
    original_filename: str | None,
    n_points: int,
    has_hole_id: bool,
) -> None:
    meta = {
        "original_filename": original_filename or None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_points": n_points,
        "has_hole_id": has_hole_id,
    }
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

    param_keys = (
        ("R_major", "R_major"),
        ("K_inter", "K_inter"),
        ("K_minor", "K_minor"),
        ("Nmin", "Nmin"),
        ("Nmax", "Nmax"),
    )
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

    ranked = sorted(
        [t for t in trials if t.get("objective") is not None],
        key=lambda t: float(t["objective"]),
    )
    top10 = ranked[:10]

    cv = best.get("cv") if isinstance(best.get("cv"), dict) else None
    coverage = _safe_float(best.get("prediction_coverage"))
    coverage_below = coverage is not None and coverage < COVERAGE_WARN

    images = _list_aniso_images(run_dir / "anisotropy")

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
        },
        "kpi": {
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": best.get("prediction_coverage"),
        },
        "spatial_structure": spatial,
        "search_params": {
            "R_major": best.get("R_major"),
            "R_inter": best.get("R_inter"),
            "R_minor": best.get("R_minor"),
            "K_inter": best.get("K_inter"),
            "K_minor": best.get("K_minor"),
            "Nmin": best.get("Nmin"),
            "Nmax": best.get("Nmax"),
        },
        "validation": {
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": coverage,
            "n_valid": (best_trial_row or {}).get("n_valid"),
            "n_tgt": (best_trial_row or {}).get("n_tgt"),
            "n_invalid": (best_trial_row or {}).get("n_invalid"),
            "oof_available": False,
            "coverage_below_threshold": coverage_below,
            "coverage_threshold": COVERAGE_WARN,
        },
        "optimization": {
            "best_trial_number": best_trial_number,
            "n_trials": best.get("n_trials") or len(trials) or None,
            "objective": best.get("objective"),
            "best_params": {
                "R_major": best.get("R_major"),
                "R_inter": best.get("R_inter"),
                "R_minor": best.get("R_minor"),
                "K_inter": best.get("K_inter"),
                "K_minor": best.get("K_minor"),
                "Nmin": best.get("Nmin"),
                "Nmax": best.get("Nmax"),
            },
            "search_space": search_space,
            "proximity": proximity,
            "any_near_bound": any_near_bound,
            "trials": trials,
            "top10": top10,
        },
        "artifacts": {"images": images},
        "conclusion_inputs": {
            "anisotropy_type": spatial.get("anisotropy_type"),
            "range_major": spatial.get("range_major"),
            "range_intermediate": spatial.get("range_intermediate"),
            "range_minor": spatial.get("range_minor"),
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
        },
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
