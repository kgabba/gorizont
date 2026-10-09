"""Thin FastAPI adapter over pipeline3d — no math changes."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from anisotropy3d.io import DataValidationError
from kriging3d_tuner.cv import DataError
from pipeline3d.pipeline import PipelineError, run_pipeline, write_progress

from .column_map import apply_mapping, inspect_csv_bytes
from .report import (
    artifact_media_type,
    build_report,
    resolve_artifact,
    write_run_meta,
)

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
    if "missing mapped column" in low:
        return "Некорректный выбор колонок. Проверьте соответствие полей."
    if "too few points" in low or "n_min" in low:
        return "Недостаточно точек для анализа."
    if "aniso_candidate" in low:
        return "Нет валидного aniso_candidate после этапа анизотропии."
    if "anisotropy result missing" in low:
        return "Этап анизотропии не создал result.json."
    if "invalid" in low and "csv" in low:
        return "Некорректный CSV."
    if "no data rows" in low:
        return "После выбора колонок не осталось строк с Value."
    if isinstance(exc, PipelineError):
        return f"Ошибка пайплайна: {msg}"
    if isinstance(exc, (DataValidationError, DataError)):
        return msg
    return "Не удалось выполнить анализ. Проверьте данные и попробуйте снова."


def _canonical_or_mapped(
    raw: bytes,
    *,
    col_x: str | None,
    col_y: str | None,
    col_z: str | None,
    col_value: str | None,
    col_holeid: str | None,
) -> tuple[bytes, dict[str, Any]]:
    """Prefer explicit mapping; else auto-suggest; else accept already-canonical CSV."""
    if col_x and col_y and col_z and col_value:
        return apply_mapping(
            raw,
            col_x=col_x,
            col_y=col_y,
            col_z=col_z,
            col_value=col_value,
            col_holeid=col_holeid or None,
        )

    suggestion = inspect_csv_bytes(raw)
    mapping = suggestion["mapping"]
    if suggestion["ready"]:
        return apply_mapping(
            raw,
            col_x=mapping["X"],
            col_y=mapping["Y"],
            col_z=mapping["Z"],
            col_value=mapping["Value"],
            col_holeid=mapping.get("HoleID"),
        )

    # Legacy: exact X,Y,Z,Value already present
    cols = set(suggestion["columns"])
    if {"X", "Y", "Z", "Value"} <= cols:
        return apply_mapping(
            raw,
            col_x="X",
            col_y="Y",
            col_z="Z",
            col_value="Value",
            col_holeid="HoleID" if "HoleID" in cols else None,
        )

    raise ValueError(
        "missing required columns: cannot map to X, Y, Z, Value — "
        f"have {suggestion['columns']}"
    )


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
            # Axes must match aniso_candidate (what tuner used), not standalone
            # top-level result — when ISO was forced, top-level axes are I.
            "major_axis_xyz": cand.get(
                "major_axis_xyz", result.get("major_axis_xyz")
            ),
            "intermediate_axis_xyz": cand.get(
                "intermediate_axis_xyz", result.get("intermediate_axis_xyz")
            ),
            "minor_axis_xyz": cand.get(
                "minor_axis_xyz", result.get("minor_axis_xyz")
            ),
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


def _run_dir_or_404(run_id: str) -> Path:
    if not run_id or "/" in run_id or ".." in run_id or "\\" in run_id:
        raise HTTPException(status_code=404, detail="Прогон не найден.")
    run_dir = (RUNS_DIR / run_id).resolve()
    try:
        run_dir.relative_to(RUNS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Прогон не найден.") from exc
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Прогон не найден.")
    return run_dir


@app.get("/analysis/{run_id}")
def get_analysis(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir_or_404(run_id)
    try:
        return build_report(run_id, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("report build failed for run_id=%s", run_id)
        raise HTTPException(
            status_code=400,
            detail="Не удалось собрать отчёт по артефактам прогона.",
        ) from exc


@app.get("/analysis/{run_id}/artifacts/{artifact_path:path}")
def get_analysis_artifact(run_id: str, artifact_path: str) -> FileResponse:
    run_dir = _run_dir_or_404(run_id)
    try:
        path = resolve_artifact(run_dir, artifact_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Некорректный путь артефакта.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Артефакт не найден.") from exc
    return FileResponse(path, media_type=artifact_media_type(path))


@app.post("/inspect")
async def inspect_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл.")
    try:
        return inspect_csv_bytes(raw)
    except ValueError as exc:
        logger.info("CSV inspect failed: %s", exc)
        raise HTTPException(status_code=400, detail=_user_error(exc)) from exc


def _read_progress(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "progress.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _run_analyze_job(
    *,
    run_id: str,
    run_dir: Path,
    csv_path: Path,
    input_meta: dict[str, Any],
    original_filename: str | None,
    n_trials: int | None,
) -> None:
    try:
        write_progress(
            run_dir,
            status="running",
            stage="upload",
            stage_label="Загрузка данных",
            trial=None,
            n_trials=n_trials,
        )
        run_pipeline(
            data_path=csv_path,
            out_dir=run_dir,
            aniso_config=ANISO_CONFIG,
            tuner_config=TUNER_CONFIG,
            n_trials=n_trials,
        )
        write_run_meta(
            run_dir,
            original_filename=original_filename,
            n_points=int(input_meta["n_points"]),
            has_hole_id=bool(input_meta["has_hole_id"]),
        )
        # pipeline already marks completed; keep meta on progress
        prog = _read_progress(run_dir) or {}
        write_progress(
            run_dir,
            status="completed",
            stage="done",
            stage_label="Готово",
            trial=prog.get("trial"),
            n_trials=prog.get("n_trials") or n_trials,
            best_trial_number=prog.get("best_trial_number"),
            n_points=int(input_meta["n_points"]),
        )
    except (PipelineError, DataValidationError, DataError) as exc:
        logger.exception("analyze failed for run_id=%s", run_id)
        write_progress(
            run_dir,
            status="failed",
            stage="error",
            stage_label="Ошибка",
            error=_user_error(exc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("unexpected analyze failure for run_id=%s", run_id)
        write_progress(
            run_dir,
            status="failed",
            stage="error",
            stage_label="Ошибка",
            error=_user_error(exc),
        )


@app.get("/analysis/{run_id}/status")
def get_analysis_status(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir_or_404(run_id)
    prog = _read_progress(run_dir) or {}
    status = str(prog.get("status") or "running")
    best_path = run_dir / "kriging_tuning" / "best_params.json"
    if status != "failed" and best_path.is_file() and (run_dir / "pipeline_summary.json").is_file():
        status = "completed"
    out: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "stage": prog.get("stage"),
        "stage_label": prog.get("stage_label"),
        "trial": prog.get("trial"),
        "n_trials": prog.get("n_trials"),
        "best_objective": prog.get("best_objective"),
        "error": prog.get("error"),
    }
    return out


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    trials: int | None = Form(None),
    col_x: str | None = Form(None),
    col_y: str | None = Form(None),
    col_z: str | None = Form(None),
    col_value: str | None = Form(None),
    col_holeid: str | None = Form(None),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    try:
        canonical, input_meta = _canonical_or_mapped(
            raw,
            col_x=col_x,
            col_y=col_y,
            col_z=col_z,
            col_value=col_value,
            col_holeid=col_holeid if col_holeid else None,
        )
    except ValueError as exc:
        logger.info("CSV mapping failed: %s", exc)
        raise HTTPException(status_code=400, detail=_user_error(exc)) from exc

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "input.csv"
    csv_path.write_bytes(canonical)
    write_run_meta(
        run_dir,
        original_filename=file.filename,
        n_points=int(input_meta["n_points"]),
        has_hole_id=bool(input_meta["has_hole_id"]),
    )
    write_progress(
        run_dir,
        status="running",
        stage="upload",
        stage_label="Загрузка данных",
        trial=None,
        n_trials=trials,
        n_points=int(input_meta["n_points"]),
    )

    thread = threading.Thread(
        target=_run_analyze_job,
        kwargs={
            "run_id": run_id,
            "run_dir": run_dir,
            "csv_path": csv_path,
            "input_meta": input_meta,
            "original_filename": file.filename,
            "n_trials": trials,
        },
        name=f"analyze-{run_id[:8]}",
        daemon=True,
    )
    thread.start()

    return {
        "run_id": run_id,
        "status": "running",
        "input": {
            "n_points": int(input_meta["n_points"]),
            "has_hole_id": bool(input_meta["has_hole_id"]),
        },
    }
