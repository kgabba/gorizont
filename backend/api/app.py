"""Thin FastAPI adapter over pipeline3d — no math changes."""

from __future__ import annotations

import csv
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from anisotropy3d.io import DataValidationError
from kriging3d_tuner.cv import DataError
from pipeline3d.pipeline import PipelineError, run_pipeline

logger = logging.getLogger("gorizo.api")
logging.basicConfig(level=logging.INFO)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(BACKEND_ROOT / "runs")))
ANISO_CONFIG = BACKEND_ROOT / "anisotropy3d" / "config.yaml"
TUNER_CONFIG = BACKEND_ROOT / "kriging3d_tuner" / "config.yaml"

app = FastAPI(
    title="gorizo",
    description="CSV → anisotropy3d → kriging3d_tuner",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _user_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    low = msg.lower()
    if "missing required column" in low or "missing column" in low:
        return "В CSV отсутствуют обязательные колонки X, Y, Z, Value."
    if "too few points" in low or "n_min" in low:
        return "Недостаточно точек для анализа."
    if "aniso_candidate" in low:
        return "Нет валидного aniso_candidate после этапа анизотропии."
    if "anisotropy result missing" in low:
        return "Этап анизотропии не создал result.json."
    if "invalid" in low and "csv" in low:
        return "Некорректный CSV."
    if isinstance(exc, PipelineError):
        return f"Ошибка пайплайна: {msg}"
    if isinstance(exc, (DataValidationError, DataError)):
        return msg
    return "Не удалось выполнить анализ. Проверьте данные и попробуйте снова."


def _inspect_csv(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid CSV encoding") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("invalid CSV: empty header")
    cols = {c.strip() for c in reader.fieldnames if c}
    required = {"X", "Y", "Z", "Value"}
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    rows = list(reader)
    return {
        "n_points": len(rows),
        "has_hole_id": "HoleID" in cols,
    }


def _build_response(run_id: str, run_dir: Path, input_meta: dict[str, Any]) -> dict[str, Any]:
    result_path = run_dir / "anisotropy" / "result.json"
    best_path = run_dir / "kriging_tuning" / "best_params.json"
    if not result_path.is_file():
        raise PipelineError("anisotropy result missing after pipeline")
    if not best_path.is_file():
        raise PipelineError("tuner best_params.json missing after pipeline")

    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    cand = result.get("aniso_candidate") or {}
    diagnostics = result.get("diagnostics") or {}
    moi_strength = cand.get("moi_strength")
    if moi_strength is None:
        moi_strength = diagnostics.get("moi_strength")

    return {
        "run_id": run_id,
        "status": "completed",
        "input": {
            "n_points": input_meta["n_points"],
            "has_hole_id": input_meta["has_hole_id"],
        },
        "anisotropy": {
            "type": result.get("type"),
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
            "major_axis_xyz": result.get("major_axis_xyz"),
            "intermediate_axis_xyz": result.get("intermediate_axis_xyz"),
            "minor_axis_xyz": result.get("minor_axis_xyz"),
            "moi_strength": moi_strength,
        },
        "optimization": {
            "R_major": best.get("R_major"),
            "R_inter": best.get("R_inter"),
            "R_minor": best.get("R_minor"),
            "K_inter": best.get("K_inter"),
            "K_minor": best.get("K_minor"),
            "Nmin": best.get("Nmin"),
            "Nmax": best.get("Nmax"),
            "CV_RMSE": best.get("CV_RMSE"),
            "CV_MAE": best.get("CV_MAE"),
            "prediction_coverage": best.get("prediction_coverage"),
            "best_trial_number": best.get("best_trial_number"),
        },
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    trials: int | None = Form(None),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    try:
        input_meta = _inspect_csv(raw)
    except ValueError as exc:
        logger.info("CSV validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=_user_error(exc)) from exc

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "input.csv"
    csv_path.write_bytes(raw)

    try:
        run_pipeline(
            data_path=csv_path,
            out_dir=run_dir,
            aniso_config=ANISO_CONFIG,
            tuner_config=TUNER_CONFIG,
            n_trials=trials,
        )
        return _build_response(run_id, run_dir, input_meta)
    except (PipelineError, DataValidationError, DataError) as exc:
        logger.exception("analyze failed for run_id=%s", run_id)
        raise HTTPException(status_code=400, detail=_user_error(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("unexpected analyze failure for run_id=%s", run_id)
        raise HTTPException(
            status_code=400,
            detail=_user_error(exc),
        ) from exc
