# Anisotropic Ordinary Kriging Tuner (3D)

Standalone port of `aniso_ok_tuner` to **X,Y,Z**. Per-**Domain** MOI (azimuth × dip
grid) → directional ranges → Optuna TPE on neighbourhood / nugget / `range_scale`.

CV choice (required): **`spatial_block`** (3D grid) or **`buffered_delete_d`**
(leave-d + distance buffer in the OK anisotropy metric).

FastAPI v1: sync **`POST /tune`** (+ `/tune/csv`) → JSON summary + `report_markdown`.
No `/predict`.

## Orientation convention

- `azimuth_deg` — major in plan, **math CCW from +X** (same as 2D `alpha_deg`)
- `dip_deg` — elevation of major from horizontal (−90…+90); positive → +Z
- **rake = 0** — intermediate axis is horizontal, 90° CCW from azimuth
- minor = major × intermediate (right-handed)

Metric: \(h'=\sqrt{u_1^2+(u_2/K_2)^2+(u_3/K_3)^2}\) with \(K_2=a_{\mathrm{inter}}/a_{\mathrm{major}}\),
\(K_3=a_{\mathrm{minor}}/a_{\mathrm{major}}\); spherical range \(= a_{\mathrm{major}}\cdot\mathrm{range\_scale}\).

## CSV schema

`X,Y,Z,Grade` + optional `Domain`. Missing / single Domain → one independent run.
Multiple Domains → separate MOI + Optuna (no cross-domain neighbours).

## Install

```bash
cd aniso_ok_tuner_3d
pip install -e ".[dev]"
```

## CLI

```bash
python tune.py --data path/to/points.csv \
  --config configs/smoke.yaml \
  --out-dir results/smoke \
  --cv spatial_block
```

## FastAPI

```bash
uvicorn aniso_ok_tuner_3d.api:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /tune` — JSON `{ "points": [...], "cv_method": "spatial_block", "n_trials": 20 }`
- `POST /tune/csv` — multipart CSV + form fields `cv_method`, `n_trials`

Primary score in the report is **CV RMSE** of the chosen method (no external holdout/LB).

## What Optuna tunes

| Param | Window |
|-------|--------|
| `R_major`, `R_inter`, `R_minor` | ±20% of directional ranges |
| `N_max` | config box |
| `nugget` | ±20% of fitted C₀ |
| `range_scale` | config box (default 0.8–1.2) |

Fixed: `azimuth`, `dip`, `a_major` / `a_inter` / `a_minor`, sill total.

## Outputs per domain

`out/<domain>/trials.csv`, `best_params.json`, `variogram_{major,inter,minor}.png`,
plus root `summary.json` and `report.md` (experiment-report format + 3D fields + `!` bounds).
