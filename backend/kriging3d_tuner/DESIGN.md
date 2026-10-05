# kriging3d_tuner — Design

Independent project: **ANISO Ordinary Kriging + Optuna** in 3D.

Input: drillhole / sample points `X,Y,Z,Value` plus a **fixed** continuity
ellipsoid from [`anisotropy3d`](../anisotropy3d/) (`result.json` /
`aniso_candidate`). Output: best **search neighbourhood** hyperparameters and
CV metrics.

This document is the source of truth for v0. Implementation follows it
mechanically; do not invent soft rescues, ISO bake-offs, or VG re-tuning.

---

## 1. Goal / non-goals

### Goal

```text
CSV (X,Y,Z,Value [+HoleID]) + fixed anisotropy ellipsoid
  → build CV folds once
  → Optuna over search neighbourhood only
  → best ANISO OK configuration
  → best_params.json + trials.csv
```

### Non-goals (v0)

- Do **not** modify `anisotropy3d`.
- Do **not** re-estimate variogram model / nugget / sill / three ranges / orientation.
- Do **not** compare ISO vs ANISO candidates (that comes later in a separate pipeline).
- Do **not** run kriging on a prediction grid (CV OOF only).
- Do **not** soft-fill empty search neighbourhoods with nearest neighbours outside the ellipsoid.
- Do **not** use PyKrige / GSTools; own NumPy OK.

---

## 2. Fixed vs tuned

| Fixed (from anisotropy3d) | Tuned (Optuna) |
|---------------------------|----------------|
| `variogram_model` (v0: **spherical** only) | `R_major` |
| `nugget`, `sill` (partial sill) | `K_inter`, `K_minor` |
| `range_major`, `range_intermediate`, `range_minor` | `Nmin`, `Nmax` |
| `orientation_matrix` \(3\times 3\) (columns maj, int, min) | derived: `R_inter`, `R_minor` |

Derived search radii:

\[
R_{\mathrm{inter}} = R_{\mathrm{major}} \cdot K_{\mathrm{inter}},\qquad
R_{\mathrm{minor}} = R_{\mathrm{major}} \cdot K_{\mathrm{minor}}.
\]

Hard constraints:

\[
1 \ge K_{\mathrm{inter}} \ge K_{\mathrm{minor}} > 0,\qquad
N_{\min} \le N_{\max}.
\]

**Variogram ellipsoid ≠ search ellipsoid.** Same orientation; different semi-axes.
Search radii are **not** forced equal to variogram ranges.

---

## 3. Input schemas

### 3.1 Points CSV

Required columns (case-sensitive):

| Column | Type | Role |
|--------|------|------|
| `X`, `Y`, `Z` | float | Cartesian coordinates |
| `Value` | float | grade / target |

Optional:

| Column | Role |
|--------|------|
| `HoleID` | group id for GroupKFold / leave-holes-out |
| `DomainID` | reserved; **ignored in v0** (no domain loop) |

Drop rows with non-finite coords or value. Require `n ≥ n_min_points` (default 30).

### 3.2 Anisotropy JSON

Accept anisotropy3d `result.json` (or a slim equivalent).

**Priority:**

1. If top-level `aniso_candidate` exists and contains all required fields → use it.
2. Else use top-level fields of the same names.

Required fields:

| Field | Shape | Notes |
|-------|-------|-------|
| `variogram_model` | str | v0: must be `"spherical"` (else error) |
| `nugget` | float | ≥ 0 |
| `sill` | float | partial sill > 0 |
| `range_major` | float | \(a_{\mathrm{maj}} > 0\) |
| `range_intermediate` | float | \(a_{\mathrm{int}} > 0\) |
| `range_minor` | float | \(a_{\mathrm{min}} > 0\) |
| `orientation_matrix` | \(3\times 3\) list | columns = maj, int, min |

Optional echo (not required for OK): `major_axis_xyz`, etc.

Assert \(a_{\mathrm{maj}} \ge a_{\mathrm{int}} \ge a_{\mathrm{min}} > 0\) within a small tolerance;
if mildly unsorted, sort **linked** (range, column) together (same rule as anisotropy3d).

---

## 4. Coordinate / matrix conventions

Match anisotropy3d:

- Internal coordinates: Cartesian `X,Y,Z`.
- `orientation_matrix` \(Q\): columns \(\mathbf{q}_{\mathrm{maj}},\mathbf{q}_{\mathrm{int}},\mathbf{q}_{\mathrm{min}}\) are orthonormal, right-handed
  (if \(\det Q < 0\), flip minor column).
- Local principal coordinates for a lag \(\Delta\mathbf{x} = \mathbf{x}_j - \mathbf{x}_0\):

\[
\mathbf{x}' = Q^\top \Delta\mathbf{x} = (x', y', z')^\top.
\]

- Reporting azimuth/dip (if ever printed): azimuth 0°=+X, 90°=+Y CCW; dip 0°=horizontal, +90°=+Z.
  Not used inside OK / Optuna.

---

## 5. Two ellipsoids (must not mix metrics)

```text
neighbours ──► hard SEARCH ellipsoid (R_maj, R_int, R_min)
                    │
                    │ keep inside; cap to Nmax by d_search
                    ▼
               Ordinary Kriging
                    │
                    │ cov / γ distance via VARIOGRAM ellipsoid
                    ▼
            (a_maj, a_int, a_min) + nugget/sill
```

### 5.1 Search ellipsoid (HARD)

Point \(j\) is inside the search neighbourhood of target \(0\) iff

\[
\left(\frac{x'}{R_{\mathrm{major}}}\right)^2
+ \left(\frac{y'}{R_{\mathrm{inter}}}\right)^2
+ \left(\frac{z'}{R_{\mathrm{minor}}}\right)^2
\le 1.
\]

Anisotropic **search distance** (for ranking when capping `Nmax`):

\[
d_{\mathrm{search}}
= \sqrt{
  \left(\frac{x'}{R_{\mathrm{major}}}\right)^2
+ \left(\frac{y'}{R_{\mathrm{inter}}}\right)^2
+ \left(\frac{z'}{R_{\mathrm{minor}}}\right)^2
}.
\]

### 5.2 Variogram / covariance metric

Same \(Q\); lag for the model uses **variogram** semi-axes:

\[
h'
= \sqrt{
  \left(\frac{x'}{a_{\mathrm{maj}}}\right)^2
+ \left(\frac{y'}{a_{\mathrm{int}}}\right)^2
+ \left(\frac{z'}{a_{\mathrm{min}}}\right)^2
}.
\]

Spherical variogram (partial sill \(c\), range parameter \(1\) in \(h'\)-units):

\[
\gamma(h') =
\begin{cases}
0, & h' = 0, \\
c\,\bigl(\tfrac{3}{2}h' - \tfrac{1}{2}(h')^3\bigr), & 0 < h' < 1, \\
c, & h' \ge 1.
\end{cases}
\]

Covariance used in OK (stationary):

\[
C(h') = c - \gamma(h')
\quad\text{(off-diagonal)},\qquad
C(0) = c
\quad\text{on the covariogram; nugget handled as below}.
\]

**Nugget:** OK covariance matrix between distinct points \(i\neq j\):

\[
K_{ij} = C(h'_{ij}),
\qquad
K_{ii} = c + n_0
\]

where \(n_0 =\) `nugget`, \(c =\) `sill` (partial). Cross-covariances target↔data use \(C(h')\) without adding nugget to off-diagonals (standard OK).

---

## 6. Hard neighbourhood rules

For each prediction location:

1. Collect all **training** points with \(d_{\mathrm{search}}^2 \le 1\).
2. If count \(< N_{\min}\): prediction = **invalid** (`NaN`). **No** fallback to nearest points outside the ellipsoid.
3. If count \(> N_{\max}\): keep the \(N_{\max}\) smallest \(d_{\mathrm{search}}\) (stable tie-break: lower original index).
4. Else: use all inside points.

Never expand the ellipsoid, never switch to Euclidean kNN outside it.

---

## 7. Ordinary Kriging (3D)

Classical OK with Lagrange multiplier \(\mu\).

Let \(n\) = neighbourhood size after hard mask + `Nmax` cap. Build

\[
\begin{bmatrix}
K & \mathbf{1} \\
\mathbf{1}^\top & 0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\lambda} \\
\mu
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{k} \\
1
\end{bmatrix},
\]

where \(K_{ij}\) as in §5.2, \(k_i = C(h'_{i0})\).

Solve with `numpy.linalg.solve`; on failure use `lstsq`. Predict \(\hat{z} = \boldsymbol{\lambda}^\top \mathbf{z}_{\mathrm{nb}}\).

If \(n < N_{\min}\) or matrix is unusable → `NaN`.

Implementation note: same spirit as `aniso_ok_tuner` own NumPy OK, extended to 3D dual metrics (search vs γ). No PyKrige/GSTools.

---

## 8. Cross-validation

### 8.1 Fold construction (once, before Optuna)

All trials reuse the **same** fold list.

**If `HoleID` is present:**

- Groups = unique `HoleID`.
- Use `sklearn.model_selection.GroupKFold` with `n_splits` from config (default **5**).
- If `n_holes < n_splits`: set `n_splits = n_holes` (leave-one-hole-out when `n_holes` small). Require `n_holes ≥ 2`.
- Each fold: train = all points whose hole ∉ test holes; test = points of held-out holes.

**If `HoleID` is absent:**

- **3D spatial block CV** on the axis-aligned bounding box of `(X,Y,Z)`.
- Partition into `grid_nx × grid_ny × grid_nz` equal cells (defaults **3 × 3 × 2**).
- Leave-one-block-out: each non-empty cell is a test fold once; empty cells skipped.
- If fewer than 2 non-empty blocks → error (data too degenerate for block CV).

### 8.2 Metrics (pooled, not mean-of-folds)

Over all OOF targets across folds:

| Symbol | Definition |
|--------|------------|
| \(n_{\mathrm{tgt}}\) | number of OOF target points |
| \(n_{\mathrm{valid}}\) | number with finite prediction |
| \(n_{\mathrm{invalid}}\) | \(n_{\mathrm{tgt}} - n_{\mathrm{valid}}\) |
| `prediction_coverage` | \(n_{\mathrm{valid}} / n_{\mathrm{tgt}}\) |
| `CV_RMSE` | \(\sqrt{\mathrm{mean}(( \hat{z}-z)^2)}\) over **valid** only |
| `CV_MAE` | \(\mathrm{mean}|\hat{z}-z|\) over **valid** only |

**Do not** optimize the average of per-fold RMSE. Always pool residuals first.

### 8.3 Optuna objective (finite, coverage-penalized)

Defaults (config names in parentheses):

| Name | Default | Role |
|------|---------|------|
| `coverage_min` | `0.90` | soft target coverage |
| `coverage_penalty` | `1.0e6` | weight on coverage shortfall |
| `empty_rmse_fallback` | `1.0e6` | if \(n_{\mathrm{valid}} = 0\) |

\[
\mathrm{objective} =
\begin{cases}
\texttt{empty\_rmse\_fallback} + \texttt{coverage\_penalty}, & n_{\mathrm{valid}}=0, \\[4pt]
\mathrm{CV\_RMSE}
+ \texttt{coverage\_penalty}\cdot\max(0,\; \texttt{coverage\_min} - \mathrm{coverage}),
& \text{otherwise.}
\end{cases}
\]

Always return a finite float so TPE keeps learning. Log `CV_RMSE`, `CV_MAE`, coverage, \(n_{\mathrm{invalid}}\) per trial even when penalized.

Best trial for reporting = trial with **minimal objective**. Reported `CV_RMSE` / `CV_MAE` / `prediction_coverage` in `best_params.json` are the **unpenalized** pooled metrics of that trial (not the objective scalar).

---

## 9. Optuna search space

Sampler: `TPESampler` (seed from config). `n_trials` from CLI/config.

### 9.1 Continuous / integer suggestions

Let \(R_0 = a_{\mathrm{maj}}\) (variogram `range_major`).

| Parameter | Type | Default bounds |
|-----------|------|----------------|
| `R_major` | float | \([R_0\cdot r_{\mathrm{lo}},\; R_0\cdot r_{\mathrm{hi}}]\) with `r_lo=0.5`, `r_hi=3.0` |
| `K_inter` | float | \([k_{\min},\; 1]\) with `k_min=0.05` |
| `K_minor` | float | \([k_{\min},\; 1]\) |
| `Nmin` | int | `[nmin_lo, nmin_hi]` default `[4, 12]` |
| `Nmax` | int | `[nmax_lo, nmax_hi]` default `[8, 64]` |

Enforce constraints **inside** the trial:

1. Suggest `K_inter`, then suggest `K_minor` in `[k_min, K_inter]` (or suggest both and reject / prune if `K_minor > K_inter`).
2. Suggest `Nmax`, then `Nmin` in `[nmin_lo, min(nmin_hi, Nmax)]` (or prune if `Nmin > Nmax`).

Preferred: **ordered sampling** so every trial is feasible (no waste).

Derived for evaluation and logging:

\[
R_{\mathrm{inter}} = R_{\mathrm{major}} K_{\mathrm{inter}},\qquad
R_{\mathrm{minor}} = R_{\mathrm{major}} K_{\mathrm{minor}}.
\]

### 9.2 What is NOT tuned

`variogram_model`, `nugget`, `sill`, `range_*`, `orientation_matrix` — frozen for the whole study.

---

## 10. Outputs

### 10.1 `best_params.json`

```json
{
  "R_major": 0.0,
  "R_inter": 0.0,
  "R_minor": 0.0,
  "K_inter": 0.0,
  "K_minor": 0.0,
  "Nmin": 0,
  "Nmax": 0,
  "CV_RMSE": 0.0,
  "CV_MAE": 0.0,
  "prediction_coverage": 0.0,
  "objective": 0.0,
  "n_trials": 0,
  "best_trial_number": 0,
  "cv": {
    "method": "group_kfold | spatial_block_3d",
    "n_folds": 0,
    "n_splits_requested": 0
  },
  "variogram_fixed": {
    "variogram_model": "spherical",
    "nugget": 0.0,
    "sill": 0.0,
    "range_major": 0.0,
    "range_intermediate": 0.0,
    "range_minor": 0.0,
    "orientation_matrix": [[], [], []]
  },
  "search_space": {},
  "seed": 0
}
```

### 10.2 `trials.csv`

One row per trial. Minimum columns:

`trial`, `R_major`, `K_inter`, `K_minor`, `R_inter`, `R_minor`, `Nmin`, `Nmax`,
`objective`, `CV_RMSE`, `CV_MAE`, `prediction_coverage`, `n_valid`, `n_invalid`, `n_tgt`.

Flush periodically (e.g. every 25 trials) so long runs are resumable for inspection.

### 10.3 Optional (v0 allowed but not required)

- `_run_config.yaml` copy of resolved config
- console log of best-so-far

---

## 11. Module map (future implementation)

```text
kriging3d_tuner/
  DESIGN.md                 ← this document
  pyproject.toml            (later)
  config.yaml               (later)
  README.md                 (later)
  src/kriging3d_tuner/
    models.py               # dataclasses: AnisotropyFixed, SearchParams, Fold, ...
    geometry.py             # Qᵀ Δx; hard ellipsoid mask; d_search; h'
    kriging.py              # 3D OK given fixed VG + SearchParams
    cv.py                   # build_folds(...); pooled_oof_metrics(...)
    search_space.py         # suggest feasible (R_major, K_*, N*)
    objective.py            # trial → objective + metrics dict
    pipeline.py             # load → folds → Optuna study → write artifacts
    cli.py                  # --data --anisotropy --config --out-dir --trials
```

### Future CLI sketch

```bash
python -m kriging3d_tuner \
  --data points.csv \
  --anisotropy path/to/anisotropy3d/result.json \
  --config config.yaml \
  --out-dir results/run \
  --trials 200
```

---

## 12. Default config knobs (names locked)

```yaml
n_min_points: 30
seed: 42

cv:
  n_splits: 5                 # GroupKFold when HoleID present
  grid_nx: 3                  # spatial_block_3d
  grid_ny: 3
  grid_nz: 2

objective:
  coverage_min: 0.90
  coverage_penalty: 1.0e6
  empty_rmse_fallback: 1.0e6

search_space:
  r_lo: 0.5                   # R_major / range_major
  r_hi: 3.0
  k_min: 0.05
  nmin_lo: 4
  nmin_hi: 12
  nmax_lo: 8
  nmax_hi: 64

optuna:
  n_trials: 200
  n_startup_trials: null      # null → max(10, 0.3 * n_trials)
```

---

## 13. Out of scope for v0 (explicit)

- ISO Ordinary Kriging path and ISO/ANISO holdout comparison
- Tuning nugget / sill / ranges / orientation
- Soft geometric anisotropy “compress one axis” shortcuts that collapse search and γ into one metric
- Empty-ellipsoid rescue (4-nearest, etc.) as in 2D `aniso_ok_tuner`
- Domain-wise loops on `DomainID`
- Universal / indicator kriging, trend UK
- Parallel fold evaluation (optional later)
- Writing prediction grids / block models

---

## 14. Acceptance checklist (when code lands)

1. Search uses hard ellipsoid only; `< Nmin` → NaN; no outside-kNN fill.
2. OK lag uses variogram \((a_{\mathrm{maj}},a_{\mathrm{int}},a_{\mathrm{min}})\); neighbour mask uses \((R_{\mathrm{maj}},R_{\mathrm{int}},R_{\mathrm{min}})\).
3. Orientation frozen from input `orientation_matrix`.
4. Folds built once; all trials share them.
5. Objective = pooled RMSE + coverage penalty (not mean fold RMSE).
6. `best_params.json` and `trials.csv` match §10.
7. `anisotropy3d` tree untouched.
