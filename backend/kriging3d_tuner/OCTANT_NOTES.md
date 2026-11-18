# Octant / sector search — Datamine OK + Micromine IDW

Ordinary Kriging follows the public description in
[Grade Estimation Octants](https://docs.dataminesoftware.com/StudioRM/2.2/STUDIO_RM/Grade%20Estimation%20Octants.htm)
(Studio RM 2.2) — **8 octants**, Datamine-style.

Anisotropic IDW also supports Micromine-style sector search
([Search Definition](https://webhelp.micromine.com/mm/latest/English/Content/mmmodel/IDH_DATA_SEARCH_DEFINITION.htm)):
**4 / 8 / 16** sectors inside the search ellipsoid, with underfilled sectors
emptied.

**This module does not claim bit-for-bit equivalence with Datamine ESTIMA or
Micromine.** Behaviour below documents our interpretation where the docs are
incomplete.

## What we implement

| Control | Meaning |
|---------|---------|
| `octant_search` / `sector_search` | Off → legacy hard ellipsoid + `Nmax`. On → sector limits. |
| `n_sectors` | `4` plan quadrants; `8` octants; `16` = 8 plan wedges × upper/lower. OK always uses 8. |
| `MINOCT` | Minimum number of *filled* sectors before estimation. |
| `MINPEROC` | Minimum samples in a sector for it to count as filled. |
| `MAXPEROC` | Maximum samples taken from one sector (nearest by anisotropic distance). |
| `discard_underfilled` | Micromine (IDW): drop all samples from sectors with `< MINPEROC`. Datamine (OK): underfilled sectors still contribute up to `MAXPEROC`. |
| `Nmin` / `Nmax` | Apply after sector filtering (`Nmax` is global). |

Algorithm order (sectors on):

1. Keep samples inside the oriented search ellipsoid (`d_search ≤ 1`).
2. Assign sector ids from principal-frame offsets `xp = Qᵀ(x − x₀)` (`n_sectors` ∈ {4, 8, 16}).
3. Per sector: filled if count ≥ `MINPEROC`; keep up to `MAXPEROC` nearest (Micromine: discard underfilled entirely).
4. If filled sectors `< MINOCT` → no estimate.
5. If selected count `< Nmin` → no estimate (returns NaN).
6. If selected count `> Nmax`: drop furthest by `d_search`, but **do not** drop a sample if that would leave its sector with fewer than `MINPEROC` samples. If `Nmax` cannot be met under that rule → no estimate.

Natural upper bound before the global trim is `≤ n_sectors × MAXPEROC`; Optuna also enforces `Nmax ≤ n_sectors × MAXPEROC` and `Nmax ≥ MINOCT × MINPEROC` when sampling.

When sectors are off, behaviour matches the pre-octant legacy path
(hard ellipsoid + nearest `Nmax`).

## Optuna conditional space

### Ordinary Kriging

Optuna suggests `octant_search ∈ {false, true}` (or a forced mode via
`octants.force_octant_search`):

| Branch | Suggested parameters |
|--------|----------------------|
| `false` | `R_major`, `K_inter`, `K_minor`, `Nmin`, `Nmax` |
| `true` | `R_major`, `K_inter`, `K_minor`, `MINOCT`, `MINPEROC`, `MAXPEROC` (`n_sectors=8` fixed) |

### Anisotropic IDW

Same `octant_search` bool, plus when true:

| Extra | Values |
|-------|--------|
| `n_sectors` | `{4, 8, 16}` |
| Micromine discard | `discard_underfilled=True` |

When sector search is on, Optuna **does not** suggest global `Nmin` / `Nmax`.
They are implied to remove redundant / conflicting combinations:

* `Nmin = MINOCT × MINPEROC`
* `Nmax = n_sectors × MAXPEROC`

Hard constraints when sampling the sector branch:

* `MINPEROC ≤ MAXPEROC`
* `MINOCT ≤ n_sectors`

Default compact ranges (see `config.yaml` / `config_idw.yaml`): when sectors
off, `Nmax ∈ [4, 64]`; when on, `MINOCT ∈ [1, 4]`, `MINPEROC ∈ [1, 3]`,
`MAXPEROC ∈ [1, 4]`.

## Coverage-gated objective

Missing OOF forecasts stay **NaN** (never zero-filled). Metrics:

* `CV_RMSE` — over successfully estimated control points only
* `coverage = n_valid / n_tgt`
* `n_invalid = n_tgt − n_valid`

Objective (minimize), default `coverage_min = 0.97`, `coverage_penalty = 1e6`:

```text
if n_valid == 0:
    objective = empty_rmse_fallback + coverage_penalty
else:
    objective = CV_RMSE + coverage_penalty * max(0, coverage_min − coverage)
```

Any trial below `coverage_min` is ranked worse than every trial that meets the
gate with a realistic RMSE (`≪ 1e6`). This stops Optuna from gaming RMSE by
estimating only an easy subset.

## Ambiguities / deliberate choices

1. **Octant numbering (1–8, “NE top”, etc.)**  
   ESTIMA labels octants for diagrams. Estimation only needs the *count* of filled octants, not labels. We use bits from signs of `(x', y', z')` in the ellipsoid frame (`0…7`). Relabeling would not change selected sets under the same sign rule.

2. **Samples exactly on a dividing plane**  
   Documentation does not specify the on-plane rule. We assign coordinates with `≥ 0` to the positive half-space. Other conventions could differ on a measure-zero set.

3. **“Transformed distance of the shrinking ellipsoid method”**  
   We use the existing anisotropic search distance `d_search` in the principal frame (same metric as the hard ellipsoid). We do **not** reproduce ESTIMA’s iterative search-volume shrinking cycle.

4. **Order of `MAXPEROC` vs `Nmax`**  
   Docs say both apply and describe the protected removal rule for `MAXNUMn`. We apply per-octant `MAXPEROC` first, then global `Nmax` with `MINPEROC` protection.

5. **Filled octant vs samples kept after `MAXPEROC`**  
   “Filled” is evaluated on the count of ellipsoid-inside samples in the octant *before* the `MAXPEROC` cap. Caps then choose which of those enter the kriging system.

6. **No Studio regression suite**  
   Without side-by-side ESTIMA runs on shared fixtures, full numerical parity is unverified.

## Config

See `config.yaml`: `objective.coverage_min`, `search_space.nmax_*` /
`minoct_*` / `minperoc_*` / `maxperoc_*`. Optional
`octants.force_octant_search` locks the Optuna branch for A/B studies.
