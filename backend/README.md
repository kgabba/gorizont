# gorizont3d

End-to-end 3D ANISO pipeline. Two **independent** packages are orchestrated only —
their math is not copied or merged.

```text
DATA (CSV: X,Y,Z,Value)
  ↓
anisotropy3d          →  runs/<run>/anisotropy/result.json
  ↓                        (requires aniso_candidate)
kriging3d_tuner       →  runs/<run>/kriging_tuning/best_params.json
                         + trials.csv
```

## Layout

```text
gorizont3d/
  anisotropy3d/       # unchanged child project
  kriging3d_tuner/    # unchanged child project
  pipeline3d/         # thin orchestration only
  README.md
  pyproject.toml
```

Each child keeps its **own** `config.yaml`. No shared mega-config.

## Install

```bash
cd gorizont3d
pip install -e ./anisotropy3d
pip install -e ./kriging3d_tuner
pip install -e .
```

## Run

```bash
python -m pipeline3d \
  --data data.csv \
  --aniso-config anisotropy3d/config.yaml \
  --tuner-config kriging3d_tuner/config.yaml \
  --out-dir runs/test01 \
  --trials 200
```

## Outputs

```text
runs/test01/
  anisotropy/
    result.json
    diagnostics...
  kriging_tuning/
    best_params.json
    trials.csv
  pipeline_summary.json
```

Stage 2 runs **only** if `anisotropy/result.json` contains a valid
`aniso_candidate` (variogram fields + `orientation_matrix`). Standalone
ISO/ANISO verdict from anisotropy3d is ignored for this pipeline.

## Out of scope (v0)

- ISO vs ANISO comparison
- DomainID loop
- block model / grid prediction
- API / UI
- refactoring child packages
