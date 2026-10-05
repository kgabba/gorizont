"""Orchestrate anisotropy3d then kriging3d_tuner without mixing their math."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anisotropy3d.pipeline import run_anisotropy3d
from kriging3d_tuner.pipeline import run_tuner

# Fields required by kriging3d_tuner.load_variogram_json for aniso_candidate
_REQUIRED_ANISO_FIELDS = (
    "variogram_model",
    "nugget",
    "sill",
    "range_major",
    "range_intermediate",
    "range_minor",
    "orientation_matrix",
)


class PipelineError(RuntimeError):
    """Orchestration failure (gate / missing artifacts)."""


def validate_aniso_candidate(result_path: Path) -> dict[str, Any]:
    """Require a valid top-level ``aniso_candidate`` in anisotropy result.json.

    Standalone ISO/ANISO type verdict is ignored — only the candidate matters.
    """
    if not result_path.is_file():
        raise PipelineError(f"anisotropy result missing: {result_path}")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    cand = data.get("aniso_candidate")
    if not isinstance(cand, dict):
        raise PipelineError(
            "aniso_candidate missing or not an object; aborting kriging stage"
        )
    missing = [k for k in _REQUIRED_ANISO_FIELDS if k not in cand]
    if missing:
        raise PipelineError(
            f"aniso_candidate incomplete, missing fields: {missing}"
        )
    Q = cand["orientation_matrix"]
    if not (
        isinstance(Q, list)
        and len(Q) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in Q)
    ):
        raise PipelineError("aniso_candidate.orientation_matrix must be 3x3")
    return cand


def run_pipeline(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    aniso_config: str | Path,
    tuner_config: str | Path,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """CSV → anisotropy3d → gate aniso_candidate → kriging3d_tuner."""
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    aniso_config = Path(aniso_config)
    tuner_config = Path(tuner_config)

    if not data_path.is_file():
        raise PipelineError(f"data CSV not found: {data_path}")
    if not aniso_config.is_file():
        raise PipelineError(f"aniso config not found: {aniso_config}")
    if not tuner_config.is_file():
        raise PipelineError(f"tuner config not found: {tuner_config}")

    aniso_out = out_dir / "anisotropy"
    tune_out = out_dir / "kriging_tuning"
    aniso_out.mkdir(parents=True, exist_ok=True)
    # do not create tune_out until gate passes

    print(f"[pipeline3d] stage1 anisotropy3d → {aniso_out}", flush=True)
    run_anisotropy3d(data_path, aniso_out, aniso_config)

    result_json = aniso_out / "result.json"
    print(f"[pipeline3d] gate aniso_candidate @ {result_json}", flush=True)
    cand = validate_aniso_candidate(result_json)

    tune_out.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline3d] stage2 kriging3d_tuner → {tune_out}", flush=True)
    tune_result = run_tuner(
        data_path=data_path,
        anisotropy_path=result_json,
        out_dir=tune_out,
        config_path=tuner_config,
        n_trials=n_trials,
    )

    summary = {
        "data": str(data_path),
        "out_dir": str(out_dir),
        "aniso_config": str(aniso_config),
        "tuner_config": str(tuner_config),
        "anisotropy_dir": str(aniso_out),
        "kriging_tuning_dir": str(tune_out),
        "result_json": str(result_json),
        "aniso_candidate_ok": True,
        "aniso_candidate_ranges": {
            "range_major": cand.get("range_major"),
            "range_intermediate": cand.get("range_intermediate"),
            "range_minor": cand.get("range_minor"),
        },
        "n_trials": n_trials,
        "tuner_best": {
            "CV_RMSE": tune_result.get("CV_RMSE"),
            "CV_MAE": tune_result.get("CV_MAE"),
            "prediction_coverage": tune_result.get("prediction_coverage"),
            "best_trial_number": tune_result.get("best_trial_number"),
            "R_major": tune_result.get("R_major"),
            "K_inter": tune_result.get("K_inter"),
            "K_minor": tune_result.get("K_minor"),
            "Nmin": tune_result.get("Nmin"),
            "Nmax": tune_result.get("Nmax"),
        },
    }
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[pipeline3d] done → {summary_path}", flush=True)
    return summary
