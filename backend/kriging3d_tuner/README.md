# kriging3d_tuner

3D **ANISO Ordinary Kriging / Anisotropic IDW + Optuna**: fixed orientation
(and OK variogram) from [`anisotropy3d`](../anisotropy3d/), tune **hard search
neighbourhood** (and IDW power).

Optuna chooses `octant_search` true/false for Ordinary Kriging (8 Datamine
octants). For anisotropic IDW it also suggests Micromine-style
`n_sectors ∈ {4, 8, 16}` when sector search is on. With sectors/octants on,
both `MAXPEROC` (per sector) and global `Nmax` apply; see
[`OCTANT_NOTES.md`](OCTANT_NOTES.md). OK objective is coverage-gated RMSE
(`coverage_min` default 0.97); IDW default loss is pooled J.

See [`DESIGN.md`](DESIGN.md) for the full specification.

## Install

```bash
cd kriging3d_tuner
pip install -e .
```

## Run

```bash
python -m kriging3d_tuner \
  --data points.csv \
  --anisotropy path/to/anisotropy3d/result.json \
  --config config.yaml \
  --out-dir results/run \
  --trials 200
```

Anisotropic IDW (single-pass; Optuna: `power`, `R_major`, `K_*`, `Nmin`,
`Nmax`, conditional `octant_search` / `n_sectors` / `MINOCT` / `MINPEROC` /
`MAXPEROC`):

```bash
python -m kriging3d_tuner ... --method anisotropic_idw --trials 200
```

Three-pass mode (cumulative coverage 30% → 60% → 100%):

```bash
python -m kriging3d_tuner ... --mode multipass --trials 200
```

Experimental sequential multipass minimizing loss **J** (RMSE² on estimated OOF
+ fold train variance λ_f on unevaluated; max 3 passes, accept only if J improves):

```bash
python -m kriging3d_tuner ... --mode sequential_multipass_optuna_J --trials 200
```

Outputs (single OK): `best_params.json`, `trials.csv`.  
Outputs (IDW): `best_params.json`, `trials.csv`, `oof_predictions.csv`, `REPORT_idw.md`.  
Outputs (multipass): `pass_*/`, `multipass_result.json`, `oof_predictions.csv`, `REPORT_multipass.md`.  
Outputs (J-mode): `pass_*/`, `multipass_j_result.json`, `oof_predictions.csv`, `REPORT_multipass_j.md`.
