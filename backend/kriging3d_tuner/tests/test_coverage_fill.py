"""Coverage-fill second pass (OK + IDW)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kriging3d_tuner.coverage_fill import _expand_space, maybe_run_coverage_fill
from kriging3d_tuner.models import SearchSpaceConfig
from kriging3d_tuner.pipeline import run_tuner


def _tiny_aniso(path: Path, r: float = 30.0) -> None:
    Q = np.eye(3).tolist()
    path.write_text(
        json.dumps(
            {
                "aniso_candidate": {
                    "variogram_model": "spherical",
                    "nugget": 0.1,
                    "sill": 1.0,
                    "range_major": r,
                    "range_intermediate": r,
                    "range_minor": r * 0.5,
                    "orientation_matrix": Q,
                }
            }
        ),
        encoding="utf-8",
    )


def _clustered_csv(path: Path, n_holes: int = 6, per_hole: int = 8) -> None:
    """Two spatial clusters so a tiny search leaves holes on GroupKFold."""
    rows = []
    rng = np.random.default_rng(0)
    for h in range(n_holes):
        # holes 0..2 near origin, 3..5 far away
        base = np.array([0.0, 0.0, 0.0]) if h < n_holes // 2 else np.array([500.0, 0.0, 0.0])
        for i in range(per_hole):
            xyz = base + rng.normal(0, 2.0, size=3)
            rows.append(
                {
                    "HoleID": h + 1,
                    "X": float(xyz[0]),
                    "Y": float(xyz[1]),
                    "Z": float(xyz[2]),
                    "Value": float(10.0 + 0.1 * h + rng.normal()),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_expand_space_scales_r_major_max_only():
    space = SearchSpaceConfig(
        r_major_min=10.0,
        r_major_max=50.0,
        k_min=0.05,
        nmin_lo=4,
        nmin_hi=8,
        nmax_lo=4,
        nmax_hi=16,
    )
    out = _expand_space(space, 2.0)
    assert out.r_major_min == 10.0
    assert out.r_major_max == pytest.approx(100.0)
    assert out.nmax_hi == 16


@pytest.mark.parametrize("method", ["ordinary_kriging", "anisotropic_idw"])
def test_coverage_fill_triggers_when_pass1_incomplete(tmp_path: Path, method: str):
    data = tmp_path / "data.csv"
    aniso = tmp_path / "aniso.json"
    _clustered_csv(data)
    _tiny_aniso(aniso, r=25.0)

    cfg = {
        "n_min_points": 10,
        "seed": 7,
        "mode": "single",
        "method": method,
        "cv": {"n_splits": 3},
        "objective": {
            "loss": "rmse" if method == "ordinary_kriging" else "j",
            "coverage_min": 0.0,
            "coverage_penalty": 1.0e6,
            "empty_rmse_fallback": 1.0e6,
        },
        "search_space": {
            "r_lo": 0.3,
            "r_hi": 0.8,  # deliberately tight vs cluster gap
            "k_min": 0.2,
            "nmin_lo": 2,
            "nmin_hi": 6,
            "nmax_lo": 2,
            "nmax_hi": 12,
        },
        "optuna": {
            "n_trials": 6,
            "n_startup_trials": 3,
            "n_ei_candidates": 8,
            "flush_every": 10,
        },
        "idw": {"power_min": 1.0, "power_max": 3.0},
        "octants": {"force_octant_search": False},
        "coverage_fill": {
            "enabled": True,
            "trigger_below": 1.0,
            "pass2_coverage_min": 1.0,
            "r_expand_factor": 8.0,
            "n_trials": 8,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    result = run_tuner(
        data, aniso, out, cfg_path, n_trials=6, method=method, mode="single"
    )

    assert "coverage_fill" in result
    assert result["coverage_fill"]["enabled"] is True
    # With two far clusters + tiny pass1 search, fill should usually trigger.
    # If pass1 already hit 100%, skip is also valid — assert schema either way.
    if result["coverage_fill"]["triggered"]:
        assert result["coverage_fill"]["pass1_coverage"] < 1.0 - 1e-12
        assert len(result["passes"]) == 2
        assert (out / "pass_2_coverage_fill" / "best_params.json").is_file()
        oof = pd.read_csv(out / "oof_predictions.csv")
        assert "pass_id" in oof.columns
        assert set(oof["pass_id"].unique()).issubset({1, 2})
        assert result["prediction_coverage"] >= result["coverage_fill"]["pass1_coverage"]
    else:
        assert result["coverage_fill"]["pass1_coverage"] >= 1.0 - 1e-12


def test_coverage_fill_disabled_by_default_key_absent(tmp_path: Path):
    data = tmp_path / "data.csv"
    aniso = tmp_path / "aniso.json"
    _clustered_csv(data, n_holes=4, per_hole=5)
    _tiny_aniso(aniso)
    cfg = {
        "n_min_points": 8,
        "seed": 1,
        "mode": "single",
        "method": "ordinary_kriging",
        "cv": {"n_splits": 2},
        "objective": {
            "loss": "rmse",
            "coverage_min": 0.0,
            "coverage_penalty": 1e6,
            "empty_rmse_fallback": 1e6,
        },
        "search_space": {
            "r_lo": 0.5,
            "r_hi": 3.0,
            "k_min": 0.2,
            "nmin_lo": 2,
            "nmin_hi": 6,
            "nmax_lo": 2,
            "nmax_hi": 10,
        },
        "optuna": {
            "n_trials": 4,
            "n_startup_trials": 2,
            "n_ei_candidates": 8,
            "flush_every": 10,
        },
        "octants": {"force_octant_search": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_tuner(
        data,
        aniso,
        tmp_path / "out",
        cfg_path,
        n_trials=4,
        method="ordinary_kriging",
    )
    assert result["coverage_fill"]["enabled"] is False
    assert result["coverage_fill"]["triggered"] is False
