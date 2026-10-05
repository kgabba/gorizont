# kriging3d_tuner

3D **ANISO Ordinary Kriging + Optuna**: fixed variogram ellipsoid from
[`anisotropy3d`](../anisotropy3d/), tune **hard search neighbourhood** only.

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

Outputs: `best_params.json`, `trials.csv`.
