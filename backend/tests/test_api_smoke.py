"""API + end-to-end smoke on synthetic 3D data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aniso_ok_tuner_3d.api import app
from aniso_ok_tuner_3d.pipeline import run_pipeline
from aniso_ok_tuner_3d.report import mark_boundaries


@pytest.fixture
def synth_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 36
    xyz = rng.uniform(0, 20, size=(n, 3))
    # mild trend + noise
    grade = 2.0 * xyz[:, 0] + 0.5 * xyz[:, 2] + rng.normal(0, 1.0, size=n)
    domain = np.where(xyz[:, 0] < 10, "A", "B")
    df = pd.DataFrame(
        {
            "X": xyz[:, 0],
            "Y": xyz[:, 1],
            "Z": xyz[:, 2],
            "Grade": grade,
            "Domain": domain,
        }
    )
    path = tmp_path / "synth3d.csv"
    df.to_csv(path, index=False)
    return path


def test_pipeline_smoke(synth_csv: Path, tmp_path: Path):
    cfg = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    out = tmp_path / "out"
    summary = run_pipeline(synth_csv, out, cfg)
    assert summary["n_domains"] == 2
    assert (out / "report.md").exists()
    assert (out / "summary.json").exists()
    assert "report_markdown" in summary
    # at least one domain should have optimization attempt
    statuses = [d["optimization"]["status"] for d in summary["domains"]]
    assert any(s in ("SUCCESS", "FAILED", "SKIPPED") for s in statuses)


def test_mark_boundaries():
    theta = {"R_major": 10.0, "N_max": 5, "nugget": 1.0, "range_scale": 1.2}
    space = {
        "R_major": [8.0, 12.0],
        "N_max": [5, 40],
        "nugget": [0.5, 1.5],
        "range_scale": [0.8, 1.2],
    }
    flags = mark_boundaries(theta, space)
    assert flags["params"]["N_max"] == "min"
    assert flags["params"]["range_scale"] == "max"
    assert flags["any_hit"] is True


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tune_json_smoke(tmp_path: Path):
    rng = np.random.default_rng(0)
    points = []
    for _ in range(24):
        x, y, z = rng.uniform(0, 10, size=3)
        points.append(
            {
                "X": float(x),
                "Y": float(y),
                "Z": float(z),
                "Grade": float(x + 0.2 * z + rng.normal(0, 0.5)),
            }
        )
    client = TestClient(app)
    # Use smoke-like overrides for speed
    r = client.post(
        "/tune",
        json={
            "points": points,
            "cv_method": "spatial_block",
            "n_trials": 3,
            "out_dir": str(tmp_path / "api_out"),
            "config_overrides": {
                "n_min_points": 8,
                "moi": {
                    "azimuth_step_deg": 90.0,
                    "dip_step_deg": 45.0,
                    "dip_min": -45.0,
                    "dip_max": 45.0,
                },
                "directional": {"min_pairs": 2, "n_lags": 5},
                "cv": {"grid_nx": 2, "grid_ny": 2, "grid_nz": 1},
                "optuna": {"n_startup_trials": 2, "n_ei_candidates": 4},
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "report_markdown" in body
    assert body["cv_method"] == "spatial_block"
    assert body["n_domains"] >= 1
