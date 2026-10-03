"""FastAPI surface: /health + sync /tune (no predict)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .pipeline import deep_merge, load_config, run_pipeline, run_pipeline_from_records

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "spatial_block": PACKAGE_ROOT / "configs" / "default_spatial_block.yaml",
    "buffered_delete_d": PACKAGE_ROOT / "configs" / "default_buffered_delete_d.yaml",
}

app = FastAPI(
    title="aniso_ok_tuner_3d",
    description="3D anisotropic Ordinary Kriging Optuna tuner (tune + report only).",
    version="0.1.0",
)


class Point3D(BaseModel):
    X: float
    Y: float
    Z: float
    Grade: float
    Domain: str | None = None


class TuneJsonRequest(BaseModel):
    points: list[Point3D] = Field(..., min_length=3)
    cv_method: Literal["spatial_block", "buffered_delete_d"] = "spatial_block"
    n_trials: int | None = None
    out_dir: str | None = None
    config_overrides: dict[str, Any] | None = None
    isotropic: bool | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _resolve_config(cv_method: str, overrides: dict[str, Any] | None) -> tuple[Path | None, dict]:
    base_path = DEFAULT_CONFIGS.get(cv_method)
    cfg: dict[str, Any] = load_config(base_path) if base_path and base_path.exists() else {}
    cfg.setdefault("cv", {})
    cfg["cv"]["method"] = cv_method
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return base_path if base_path and base_path.exists() else None, cfg


@app.post("/tune")
async def tune_json(body: TuneJsonRequest) -> dict[str, Any]:
    """Tune from JSON body; returns summary + report_markdown."""
    overrides: dict[str, Any] = dict(body.config_overrides or {})
    if body.n_trials is not None:
        overrides.setdefault("optuna", {})
        overrides["optuna"]["n_trials"] = int(body.n_trials)
    if body.isotropic is not None:
        overrides["isotropic"] = bool(body.isotropic)
    overrides.setdefault("cv", {})
    overrides["cv"]["method"] = body.cv_method

    _, cfg = _resolve_config(body.cv_method, overrides)
    out = Path(body.out_dir) if body.out_dir else Path(tempfile.mkdtemp(prefix="ok3d_"))
    records = [p.model_dump(exclude_none=True) for p in body.points]
    try:
        # Write ephemeral merged config
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "_api_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return run_pipeline_from_records(records, out, config_path=cfg_path)
    except Exception as exc:  # noqa: BLE001 — surface to client
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tune/csv")
async def tune_csv(
    file: UploadFile = File(...),
    cv_method: Literal["spatial_block", "buffered_delete_d"] = Form("spatial_block"),
    n_trials: int | None = Form(None),
    out_dir: str | None = Form(None),
    isotropic: bool | None = Form(None),
) -> dict[str, Any]:
    """Tune from uploaded CSV (X,Y,Z,Grade[,Domain])."""
    overrides: dict[str, Any] = {}
    if n_trials is not None:
        overrides.setdefault("optuna", {})
        overrides["optuna"]["n_trials"] = int(n_trials)
    if isotropic is not None:
        overrides["isotropic"] = bool(isotropic)
    overrides.setdefault("cv", {})
    overrides["cv"]["method"] = cv_method

    _, cfg = _resolve_config(cv_method, overrides)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="ok3d_"))
    out.mkdir(parents=True, exist_ok=True)
    raw = await file.read()
    csv_path = out / "upload.csv"
    csv_path.write_bytes(raw)
    cfg_path = out / "_api_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        return run_pipeline(csv_path, out, cfg_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
