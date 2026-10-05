# anisotropy3d — Design

Independent module: estimate **3D spatial continuity anisotropy** (variogram ellipsoid)
from `X, Y, Z, Value`. No kriging, Optuna, CV, or search radius.

## 1. Mathematical pipeline

1. **Validate** points (NaN/inf, min n, dups, near-zero variance, planar/linear degeneracy).
2. **Pair cloud** — unordered pairs with a **single** `max_dist` cutoff; optional deterministic
   sampling when \(n(n-1)/2 >\) `max_pairs` (no full O(n²) index build). Log sampling when applied.
3. **Omnidirectional variogram** — weighted spherical fit on that same pair cloud
   (does **not** re-cut distance).
4. **Hemisphere directional scan** — Fibonacci directions; cone (+ optional bandwidth);
   per direction fit **range only** (nugget/sill fixed near omni). Flag
   `range_boundary_hit` if range ≥ 0.98 × upper bound.
5. **SPD ellipsoid fit** on valid directional ranges **excluding** boundary hits.
6. **Refine** along principal axes; joint nugget/sill + three ranges; optional local SO(3).
7. **Classify** ISO / AXIAL_PROLATE / AXIAL_OBLATE / TRIAXIAL.
8. **Confidence / fallback** — geometric coverage (SVD of valid directions) + residual checks;
   high nugget affects diagnostics only (not auto-ISO). Prefer safe sphere when 3D underdetermined.
9. **Report** JSON, CSV (incl. `range_boundary_hit`), diagnostic PNGs.

## 2. Ellipsoid fit formula

Continuity range in direction unit vector \(\mathbf{u}\):

\[
\frac{1}{r(\mathbf{u})^2} = \mathbf{u}^\top A \mathbf{u},\quad
A = A^\top \succ 0.
\]

Parametrize \(A = L L^\top\) (Cholesky, \(L\) lower-triangular with positive diagonal).

Weighted least squares:

\[
\min_L \sum_i w_i \bigl(\mathbf{u}_i^\top A \mathbf{u}_i - 1/r_i^2\bigr)^2.
\]

Weights (config knobs \(p,q,\varepsilon\)):

\[
w_i = n_{\mathrm{pairs},i}^{p}\cdot n_{\mathrm{lags},i}^{q}
\cdot \frac{1}{\mathrm{SSE}_i + \varepsilon}.
\]

Eigen: \(A = Q\Lambda Q^\top\). Ranges \(a_k = 1/\sqrt{\lambda_k}\) (clamp \(\lambda_k \ge \lambda_{\min}\)).

**Joint sort:** pairs \((a_k, \mathbf{q}_k)\) sorted so
\(a_{\mathrm{maj}} \ge a_{\mathrm{int}} \ge a_{\mathrm{min}}\).
`orientation_matrix` columns = those unit eigenvectors (right-handed: minor ← major × intermediate if needed).

Undirected axes: flip each vector so first nonzero component is positive (prefer \(z\ge 0\)).

**Forbidden:** sequential “max range → second → cross-product” axis picking.

## 3. Data structures

| Type | Role |
|------|------|
| `PairCloud3D` | arrays `dx,dy,dz,h,gamma,ux,uy,uz` + `max_dist` |
| `VariogramFit` | lag, gamma, n_pairs, nugget, sill, range_, sse, boundary flags |
| `DirectionalEstimate` | u, range_, sse, n_pairs, n_valid_lags, valid, `range_boundary_hit` |
| `Anisotropy3DResult` | type, 3 ranges, 3 axes, orientation_matrix, ratios, nugget/sill, confidence, diagnostics |

Source of truth for orientation: **`orientation_matrix`** (and axis vectors). Angles are reporting only.

## 4. Conventions

- Internal: Cartesian `X,Y,Z`.
- Spatial lag undirected: \(\mathbf{u} \equiv -\mathbf{u}\).
- Reporting angles (derived, not used in fit):
  - **azimuth**: 0° = +X, 90° = +Y, CCW in XY plane
  - **dip**: 0° = horizontal, +90° = +Z

## 5. Classification

Ratios \(R_{mi}=a_{\mathrm{maj}}/a_{\mathrm{int}}\), \(R_{mn}=a_{\mathrm{maj}}/a_{\mathrm{min}}\),
\(R_{in}=a_{\mathrm{int}}/a_{\mathrm{min}}\) vs config thresholds:

| Type | Condition (defaults) |
|------|----------------------|
| `ISOTROPIC` | \(R_{mn} < 1.3\) |
| `AXIAL_PROLATE` | \(R_{mi} \ge 1.3\) and \(R_{in} < 1.3\) |
| `AXIAL_OBLATE` | \(R_{mi} < 1.3\) and \(R_{in} \ge 1.3\) |
| `TRIAXIAL` | else |

Near-equal axes → diagnostic “orientation ill-defined in that plane”.

## 6. Fallback / confidence

**Technical sphere fallback** (safe default, not a claim of physical isotropy) if:

- planar/linear spatial dimensionality →
  `fallback_reason = "3D anisotropy underdetermined due to spatial dimensionality"`
- too few valid directions / poor count fraction
- valid directions do not span 3D (`directional_coverage_ok=False`; SVD rank / σ ratio)
- unstable/poor ellipsoid residual
- classification weak anisotropy (ratios ≈ 1)

**Not** auto-ISO: high `nugget/sill` alone — recorded in `confidence_reasons` / diagnostics only.

Fallback result:

- `type = ISOTROPIC` (technical sphere)
- `range_* = omni_range`
- `orientation_matrix = I`
- `fallback_reason` set

Diagnostics include: `directional_coverage_ok`, `total_possible_pairs`, `used_pairs`,
`pair_sampling_applied`, `pair_sampling_fraction`.

Confidence: `HIGH` / `MEDIUM` / `LOW` + `confidence_reasons[]` (thresholds in `config.yaml`).

## 7. Modules

| Module | Responsibility |
|--------|----------------|
| `models.py` | dataclasses |
| `io.py` | load + validate |
| `pairs.py` | pair cloud |
| `variogram.py` | experimental bins + weighted spherical fit |
| `directions.py` | Fibonacci hemisphere + cone filter |
| `ellipsoid.py` | SPD fit + eigen sort |
| `refinement.py` | axis refine + joint params + local SO(3) |
| `classification.py` | type labels |
| `confidence.py` | confidence + fallback decision |
| `pipeline.py` | orchestration only |
| `report.py` | JSON/CSV/PNG |
| `cli.py` | CLI entry |

## Non-goals

No Optuna, kriging, CV, search ellipsoid, synthetic tests, 2D MOI tensor.
