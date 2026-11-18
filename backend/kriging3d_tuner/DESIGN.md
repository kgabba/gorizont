# kriging3d_tuner — Design

Independent project: **ANISO Ordinary Kriging / Anisotropic IDW + Optuna** in 3D.

Input: drillhole / sample points `X,Y,Z,Value` plus a **fixed** continuity
ellipsoid from [`anisotropy3d`](../anisotropy3d/) (`result.json` /
`aniso_candidate`). Output: best **search neighbourhood** hyperparameters
(and IDW power), CV / OOF metrics, and method-specific reports.

This document is the source of truth for the **current** product path. Detail
on sector / octant neighbour rules also lives in [`OCTANT_NOTES.md`](OCTANT_NOTES.md).

---

## 1. Goal / non-goals

### Goal

```text
CSV (X,Y,Z,Value [+HoleID]) + fixed anisotropy ellipsoid
  → build CV folds once
  → Optuna over search neighbourhood (+ IDW power; optional sectors/octants)
  → best configuration for method ∈ {ordinary_kriging, anisotropic_idw}
  → best_params.json + trials.csv (+ IDW REPORT_idw.md / OOF artifacts)
```

Production API uses:

- OK → [`config.yaml`](config.yaml) (`ordinary_kriging`, coverage-gated RMSE)
- IDW → [`config_idw.yaml`](config_idw.yaml) (`anisotropic_idw`, loss `j`, Micromine
  sector search **auto** in Optuna)

### Non-goals

- Do **not** re-estimate variogram model / nugget / sill / three ranges / orientation
  inside this tuner (they stay frozen from anisotropy3d).
- Do **not** soft-fill empty search neighbourhoods with nearest neighbours
  **outside** the ellipsoid (NaN / unevaluated OOF stays missing).
- Do **not** use PyKrige / GSTools; own NumPy OK and IDW.
- Optional multipass / sequential-J modes exist in code for experiments; the
  production IDW path is **single-pass** (+ optional `coverage_fill` second pass).

---

## 2. Fixed vs tuned

| Fixed (from anisotropy3d) | Tuned (Optuna) |
|---------------------------|----------------|
| `variogram_model` (product: **spherical**) | `R_major` |
| `nugget`, `sill` (partial sill) — used by OK | `K_inter`, `K_minor` |
| `range_major`, `range_intermediate`, `range_minor` | see method branches below |
| `orientation_matrix` \(3\times 3\) (columns maj, int, min) | derived: `R_inter`, `R_minor` |

Derived search radii:

\[
R_{\mathrm{inter}} = R_{\mathrm{major}} \cdot K_{\mathrm{inter}},\qquad
R_{\mathrm{minor}} = R_{\mathrm{major}} \cdot K_{\mathrm{minor}}.
\]

Hard constraints:

\[
1 \ge K_{\mathrm{inter}} \ge K_{\mathrm{minor}} > 0.
\]

**Variogram ellipsoid ≠ search ellipsoid.** Same orientation; different semi-axes.
Search radii are **not** forced equal to variogram ranges.

### 2.1 Method branches (what Optuna suggests)

| Branch | Parameters |
|--------|------------|
| `octant_search` / `sector_search` = **false** | `R_major`, `K_*`, **`Nmin`**, **`Nmax`** (+ `power` for IDW) |
| OK + octants **true** | `R_major`, `K_*`, `MINOCT`, `MINPEROC`, `MAXPEROC` (`n_sectors=8` fixed, Datamine-style). **`Nmin`/`Nmax` not suggested** — implied |
| IDW + sectors **true** | same + **`n_sectors ∈ {4, 8, 16}`**, Micromine `discard_underfilled`. **`Nmin`/`Nmax` not suggested** — implied |

Implied when sector/octant search is on:

\[
N_{\min} = \mathrm{MINOCT}\times\mathrm{MINPEROC},\qquad
N_{\max} = n_{\mathrm{sectors}}\times\mathrm{MAXPEROC}.
\]

IDW also suggests continuous `power` \(p\) in `[power_min, power_max]` (prod default `[0.5, 6]`).

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
| `DomainID` | reserved; ignored in the default path |

Drop rows with non-finite coords or value. Require `n ≥ n_min_points` (default 30).

### 3.2 Anisotropy JSON

Accept anisotropy3d `result.json` (or a slim equivalent).

**Priority:**

1. If top-level `aniso_candidate` exists and contains all required fields → use it.
2. Else use top-level fields of the same names.

Required fields:

| Field | Shape | Notes |
|-------|-------|-------|
| `variogram_model` | str | product path: `"spherical"` |
| `nugget` | float | ≥ 0 (OK) |
| `sill` | float | partial sill > 0 (OK) |
| `range_major` | float | \(a_{\mathrm{maj}} > 0\) |
| `range_intermediate` | float | \(a_{\mathrm{int}} > 0\) |
| `range_minor` | float | \(a_{\mathrm{min}} > 0\) |
| `orientation_matrix` | \(3\times 3\) list | columns = maj, int, min |

IDW uses orientation + search geometry; variogram ranges still set the Optuna
`R_major` scale via `r_lo`/`r_hi`. Nugget/sill are unused by IDW weights.

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

- Reporting azimuth/dip (if printed): azimuth 0°=+X, 90°=+Y CCW; dip 0°=horizontal, +90°=+Z.
  Not used inside OK / IDW / Optuna.

---

## 5. Two ellipsoids (must not mix metrics)

```text
neighbours ──► hard SEARCH ellipsoid (R_maj, R_int, R_min)
                    │
                    │ keep inside; optional sector/octant filter; Nmax
                    ▼
          Ordinary Kriging  OR  Anisotropic IDW
                    │
         OK: cov / γ via VARIOGRAM ellipsoid
         IDW: weights from d_search (power p)
```

### 5.1 Search ellipsoid (HARD)

Point \(j\) is inside the search neighbourhood of target \(0\) iff

\[
\left(\frac{x'}{R_{\mathrm{major}}}\right)^2
+ \left(\frac{y'}{R_{\mathrm{inter}}}\right)^2
+ \left(\frac{z'}{R_{\mathrm{minor}}}\right)^2
\le 1.
\]

Anisotropic **search distance** (ranking / IDW weights):

\[
d_{\mathrm{search}}
= \sqrt{
  \left(\frac{x'}{R_{\mathrm{major}}}\right)^2
+ \left(\frac{y'}{R_{\mathrm{inter}}}\right)^2
+ \left(\frac{z'}{R_{\mathrm{minor}}}\right)^2
}.
\]

### 5.2 Variogram / covariance metric (OK only)

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

\[
C(h') = c - \gamma(h')
\quad\text{(off-diagonal)},\qquad
K_{ii} = c + n_0,\quad n_0=\texttt{nugget}.
\]

---

## 6. Hard neighbourhood rules

For each prediction location:

1. Collect all **training** points with \(d_{\mathrm{search}}^2 \le 1\).
2. If `octant_search` / sector search is **off**:
   - If count \(< N_{\min}\): prediction = **invalid** (`NaN`). **No** fallback outside the ellipsoid.
   - If count \(> N_{\max}\): keep the \(N_{\max}\) smallest \(d_{\mathrm{search}}\) (stable tie-break: lower original index).
3. If sector/octant search is **on**:
   - Assign sector ids in the principal frame (`n_sectors` ∈ {4, 8, 16}; see OCTANT_NOTES).
   - Apply `MINOCT` / `MINPEROC` / `MAXPEROC`, then global implied `Nmax` with MINPEROC-protected trim.
   - Micromine IDW: underfilled sectors (`< MINPEROC`) contribute **no** samples.
   - Datamine OK octants: underfilled octants may still contribute up to `MAXPEROC`.
   - If constraints cannot be met → empty neighbourhood → `NaN`.
4. Else: use selected points.

Never expand the ellipsoid in the single-pass path; never switch to Euclidean kNN outside it.
(`coverage_fill` may re-tune a **second** search with an expanded Optuna `R` range for remaining OOF holes only.)

---

## 7. Estimators

### 7.1 Ordinary Kriging (3D)

Classical OK with Lagrange multiplier \(\mu\).

Let \(n\) = neighbourhood size after hard mask (+ sectors) + `Nmax` cap. Build

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

### 7.2 Anisotropic IDW

Same hard search neighbourhood (and optional Micromine sectors). Weights:

\[
w_i \propto d_{\mathrm{search},i}^{-p}
\quad\text{(with a small \(\varepsilon\) floor on \(d\))},\qquad
\hat{z} = \frac{\sum_i w_i z_i}{\sum_i w_i}.
\]

No variogram system. `power` \(p\) is Optuna-tuned. Same `Nmin` / empty → `NaN` rule.

---

## 8. Cross-validation

### 8.1 Fold construction (once, before Optuna)

All trials reuse the **same** fold list.

**If `HoleID` is present:**

- Groups = unique `HoleID`.
- Use `sklearn.model_selection.GroupKFold` with `n_splits` from config (default **5**).
- If `n_holes < n_splits`: set `n_splits = n_holes`. Require `n_holes ≥ 2`.
- Each fold: train = all points whose hole ∉ test holes; test = points of held-out holes.

**If `HoleID` is absent:**

- **3D spatial block CV** on the axis-aligned bounding box of `(X,Y,Z)`.
- Partition into `grid_nx × grid_ny × grid_nz` equal cells (defaults **3 × 3 × 2**).
- Leave-one-block-out: each non-empty cell is a test fold once; empty cells skipped.
- If fewer than 2 non-empty blocks → error.

### 8.2 Metrics (pooled, not mean-of-folds)

| Symbol | Definition |
|--------|------------|
| \(n_{\mathrm{tgt}}\) | number of OOF target points |
| \(n_{\mathrm{valid}}\) | number with finite prediction |
| \(n_{\mathrm{invalid}}\) | \(n_{\mathrm{tgt}} - n_{\mathrm{valid}}\) |
| `prediction_coverage` | \(n_{\mathrm{valid}} / n_{\mathrm{tgt}}\) |
| `CV_RMSE` | \(\sqrt{\mathrm{mean}(( \hat{z}-z)^2)}\) over **valid** only |
| `CV_MAE` | \(\mathrm{mean}|\hat{z}-z|\) over **valid** only |

**Do not** optimize the average of per-fold RMSE. Always pool residuals first.
Missing forecasts stay **NaN** (never zero-filled).

### 8.3 Objectives

**Ordinary Kriging (default `config.yaml`):** coverage-gated RMSE.

| Name | Prod default | Role |
|------|--------------|------|
| `coverage_min` | `0.97` | soft target coverage |
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

**Anisotropic IDW (prod `config_idw.yaml`):** minimize pooled **J** (loss `j`):
squared error on estimated OOF points + fold λ penalties on unevaluated points
(see `multipass_j` helpers / objective). Coverage gate is **off** for that loss
(`coverage_min=0`, `coverage_penalty=0`); coverage is still logged.

Best trial = minimal objective. `best_params.json` stores unpenalized pooled
`CV_RMSE` / `CV_MAE` / `prediction_coverage` of that trial plus search / sector fields.

### 8.4 Optional `coverage_fill` (second pass)

If enabled and pass-1 OOF coverage `< trigger_below`, run a second Optuna study
(RMSE + high coverage gate, optionally expanded `R` bounds) that only fills
still-missing OOF targets. Pass-1 estimates are **not** overwritten. Reported in
API/UI as coverage-fill passes.

---

## 9. Optuna search space

Sampler: `TPESampler` (seed from config). `n_trials` from CLI/config.
`octants.force_octant_search` may lock the sector/octant categorical; absent → auto.

### 9.1 Continuous / integer suggestions

Let \(R_0 = a_{\mathrm{maj}}\) (variogram `range_major`).

| Parameter | Type | Default bounds |
|-----------|------|----------------|
| `R_major` | float | \([R_0\cdot r_{\mathrm{lo}},\; R_0\cdot r_{\mathrm{hi}}]\) with `r_lo=0.5`, `r_hi=3.0` |
| `K_inter` | float | \([k_{\min},\; 1]\) with `k_min=0.05` |
| `K_minor` | float | \([k_{\min},\; K_{\mathrm{inter}}]\) |
| `Nmin` / `Nmax` | int | only when sector/octant search is **off** |
| `MINOCT`, `MINPEROC`, `MAXPEROC` | int | when sector/octant search is **on** |
| `n_sectors` | categorical `{4,8,16}` | IDW sector branch only |
| `power` | float | IDW only |

Preferred: **ordered / conditional sampling** so every trial is feasible.

### 9.2 What is NOT tuned

`variogram_model`, `nugget`, `sill`, `range_*`, `orientation_matrix` — frozen for the whole study.

---

## 10. Outputs

### 10.1 `best_params.json`

Includes method, radii, `Nmin`/`Nmax` (suggested or implied), optional
`octant_search` / `sector_search`, `n_sectors`, `MINOCT`/`MINPEROC`/`MAXPEROC`,
IDW `power`, metrics, `variogram_fixed`, `search_space`, seed, and coverage_fill
summary when used.

### 10.2 `trials.csv`

One row per trial. Includes at least:

`trial`, `method`, `power` (IDW), `octant_search` / `sector_search`, `n_sectors`,
`R_major`, `K_inter`, `K_minor`, `R_inter`, `R_minor`, `Nmin`, `Nmax`,
`MINOCT`, `MINPEROC`, `MAXPEROC`, `objective`, `CV_RMSE`, `CV_MAE`,
`prediction_coverage`, `n_valid`, `n_invalid`, `n_tgt`.

When sectors are on, `Nmin`/`Nmax` columns are the **implied** values (also in
trial user_attrs), not independent Optuna params.

### 10.3 Method reports

- IDW: `REPORT_idw.md` with an explicit **Sector search ON/OFF** block.
- Multipass / J experimental modes: their own `REPORT_*.md` under the run dir.
- Product UI (`backend/api/report.py` + frontend) surfaces sectorality when enabled.

---

## 11. Module map

```text
kriging3d_tuner/
  DESIGN.md
  OCTANT_NOTES.md
  config.yaml / config_idw.yaml
  src/kriging3d_tuner/
    models.py          # SearchParams (+ n_sectors, discard_underfilled), …
    geometry.py        # Qᵀ Δx; hard ellipsoid; d_search; h'
    kriging.py         # OK + sector/octant neighbour select
    idw.py             # anisotropic IDW
    cv.py / oof_store.py / diagnostics.py
    search_space.py    # conditional Optuna suggest (+ implied Nmin/Nmax)
    objective.py       # RMSE gate or J; IDW power; variable sectors
    coverage_fill.py   # optional pass-2 fill
    multipass.py / multipass_j.py   # experimental modes
    pipeline.py / cli.py
```

### CLI

```bash
python -m kriging3d_tuner \
  --data points.csv \
  --anisotropy path/to/anisotropy3d/result.json \
  --config config.yaml \
  --out-dir results/run \
  --trials 200

python -m kriging3d_tuner ... --method anisotropic_idw --config config_idw.yaml
```

---

## 12. Default config knobs (names locked)

OK (`config.yaml`) — coverage-gated RMSE; octant search auto (8 only when on):

```yaml
n_min_points: 30
seed: 42
method: ordinary_kriging
cv: { n_splits: 5, grid_nx: 3, grid_ny: 3, grid_nz: 2 }
objective:
  coverage_min: 0.97
  coverage_penalty: 1.0e6
  empty_rmse_fallback: 1.0e6
search_space:
  r_lo: 0.5
  r_hi: 3.0
  k_min: 0.05
  nmin_lo: 4
  nmin_hi: 12
  nmax_lo: 4
  nmax_hi: 64
  minoct_lo: 1
  minoct_hi: 4
  minperoc_lo: 1
  minperoc_hi: 3
  maxperoc_lo: 1
  maxperoc_hi: 4
```

IDW (`config_idw.yaml`) — loss `j`; sector search auto `{4,8,16}`;
`coverage_fill.enabled: true` for optional pass-2.

---

## 13. Out of scope (still)

- ISO Ordinary Kriging path and ISO/ANISO holdout bake-offs inside this package
- Tuning nugget / sill / ranges / orientation
- Soft geometric anisotropy that collapses search and γ into one metric
- Empty-ellipsoid rescue (kNN outside the ellipsoid)
- Domain-wise loops on `DomainID`
- Universal / indicator kriging, trend UK
- Claiming bit-for-bit Micromine / Datamine ESTIMA parity (see OCTANT_NOTES)

---

## 14. Acceptance checklist

1. Search uses hard ellipsoid only; `< Nmin` → NaN; no outside-kNN fill.
2. OK lag uses variogram \((a_{\mathrm{maj}},a_{\mathrm{int}},a_{\mathrm{min}})\); neighbour mask uses \((R_{\mathrm{maj}},R_{\mathrm{int}},R_{\mathrm{min}})\).
3. IDW uses \(d_{\mathrm{search}}\) + power; same search / sector plumbing as OK.
4. When sector search is on, Optuna does **not** sample `Nmin`/`Nmax`; they are implied.
5. IDW may choose `n_sectors ∈ {4,8,16}`; OK octants stay 8.
6. Orientation frozen from input `orientation_matrix`.
7. Folds built once; all trials share them.
8. Objectives match §8.3; missing OOF stay NaN.
9. Reports / API show sector ON/OFF and sector params when enabled.
