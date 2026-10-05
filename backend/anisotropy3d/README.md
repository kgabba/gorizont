# anisotropy3d

Estimate **3D spatial continuity anisotropy** (variogram ellipsoid) from
`X, Y, Z, Value`. Orientation source of truth: **`orientation_matrix`** (3×3).
Azimuth/dip are report-only.

See [`DESIGN.md`](DESIGN.md) for math and fallback rules.

## Install

```bash
cd anisotropy3d
pip install -e .
```

## Run

```bash
anisotropy3d --data path/to/points.csv --out-dir results/run --config config.yaml
# or
python -m anisotropy3d.cli --data path/to/points.csv --out-dir results/run
```

CSV columns: `X`, `Y`, `Z`, `Value` (or `Grade`).

## Outputs

- `result.json` — full `Anisotropy3DResult` (incl. pair-sampling + coverage diagnostics)
- `directional_scan.csv` — hemisphere ranges + `range_boundary_hit`
- PNGs: omni VG, directional range sphere, principal axes, maj/int/min VG

`max_dist` is applied once when building the pair cloud; omni/directional reuse that cloud.
If pair sampling runs, the CLI logs `Pair sampling applied: ...`.

## Non-goals (v1)

No Optuna, kriging, CV, search radius, or 2D MOI port.
