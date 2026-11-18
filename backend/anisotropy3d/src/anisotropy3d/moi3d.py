"""3D moment-of-inertia detector on radial covariance volume."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .directions import fibonacci_hemisphere
from .ellipsoid import stabilize_axis
from .models import PairCloud3D, VariogramFit
from .variogram import experimental_directional, fit_spherical_weighted


def assemble_partial_ranges(
    ranges_moi: np.ndarray,
    axes_moi: np.ndarray,
    statuses: Sequence[str],
    fits_moi: Sequence[VariogramFit],
    omni_range: float,
) -> dict[str, Any]:
    """Keep reliable axis ranges; omni-fill the rest; sort by hybrid length.

    Sorting uses hybrid values (not raw extrapolated fits), so boundary-hit
    axes cannot steal major/intermediate slots via artificial upper bounds.
    """
    ranges_moi = np.asarray(ranges_moi, dtype=float)
    axes_moi = np.asarray(axes_moi, dtype=float)
    r_hybrid = np.empty(3, dtype=float)
    origin: list[str] = []
    for i in range(3):
        if (
            statuses[i] == "reliable"
            and np.isfinite(ranges_moi[i])
            and float(ranges_moi[i]) > 0
        ):
            r_hybrid[i] = float(ranges_moi[i])
            origin.append("fitted")
        else:
            r_hybrid[i] = float(omni_range)
            origin.append("omni_fill")
    order = np.argsort(r_hybrid)[::-1]
    r_out = r_hybrid[order].copy()
    ax_out = axes_moi[:, order].copy()
    ax_out[:, 2] = stabilize_axis(np.cross(ax_out[:, 0], ax_out[:, 1]))
    ax_out[:, 1] = stabilize_axis(np.cross(ax_out[:, 2], ax_out[:, 0]))
    statuses_out = [str(statuses[int(i)]) for i in order]
    fits_out = [fits_moi[int(i)] for i in order]
    origin_out = [origin[int(i)] for i in order]
    dir_fits: list[dict[str, Any]] = []
    for j, fit in enumerate(fits_out):
        d = fit.as_dict()
        d["range"] = float(fit.range_) if np.isfinite(fit.range_) else None
        d["range_used"] = float(r_out[j])
        d["range_origin"] = origin_out[j]
        d["axis_index"] = int(j)
        dir_fits.append(d)
    return {
        "ranges": r_out,
        "axes": ax_out,
        "axis_fits": fits_out,
        "axis_range_status": statuses_out,
        "range_origin": origin_out,
        "directional_fits": dir_fits,
        "order": [int(i) for i in order],
        "n_reliable_kept": sum(1 for o in origin_out if o == "fitted"),
    }


def build_radial_covariance_samples(
    pairs: PairCloud3D,
    *,
    omni_nugget: float,
    omni_sill: float,
    max_dist: float,
    cfg: dict[str, Any],
) -> list[dict[str, float]]:
    """Experimental directional γ → standardized C on Fibonacci hemisphere.

    Does **not** fit a per-direction range. ``n_pairs`` gates lag validity only;
    it is not used as MOI mass weight.
    """
    dcfg = cfg.get("directions") or {}
    vcfg = cfg.get("variogram") or {}
    n_dir = int(dcfg.get("n_directions", 96))
    tol = float(dcfg.get("angular_tolerance_deg", 22.5))
    bandwidth = dcfg.get("bandwidth", None)
    if bandwidth is None and dcfg.get("bandwidth_frac_max_dist") is not None:
        bandwidth = float(dcfg["bandwidth_frac_max_dist"]) * float(max_dist)
    n_lags = int(vcfg.get("n_lags", 12))
    min_pairs_lag = int(vcfg.get("min_pairs_per_lag", 8))

    total_sill = float(omni_nugget) + float(omni_sill)
    if not np.isfinite(total_sill) or total_sill <= 1e-18:
        total_sill = 1e-18

    dist_pow = float((cfg.get("moi") or {}).get("distance_power", 2.5))
    mcfg = cfg.get("moi") or {}
    lag_lo = float(mcfg.get("lag_frac_of_max_dist_min", 0.2)) * float(max_dist)
    lag_hi = float(mcfg.get("lag_frac_of_max_dist_max", 0.9)) * float(max_dist)
    if lag_hi <= lag_lo:
        lag_lo, lag_hi = 0.0, float(max_dist)

    dirs = fibonacci_hemisphere(n_dir)
    rows: list[dict[str, float]] = []
    for u in dirs:
        u = np.asarray(u, dtype=float).reshape(3)
        u = u / (np.linalg.norm(u) + 1e-18)
        lag, gamma, counts, _n_cone = experimental_directional(
            pairs,
            u,
            angular_tolerance_deg=tol,
            bandwidth=float(bandwidth) if bandwidth is not None else None,
            n_lags=n_lags,
            max_dist=max_dist,
            min_pairs=min_pairs_lag,
        )
        for h, g, npairs in zip(lag, gamma, counts):
            h = float(h)
            if not np.isfinite(h) or h <= 1e-18:
                continue
            if h < lag_lo or h > lag_hi:
                continue
            c = 1.0 - float(g) / total_sill
            c = float(np.clip(c, 0.0, 1.0))
            mass = c / (h**dist_pow)
            if not np.isfinite(mass) or mass <= 0.0:
                continue
            rows.append(
                {
                    "ux": float(u[0]),
                    "uy": float(u[1]),
                    "uz": float(u[2]),
                    "lag": h,
                    "gamma": float(g),
                    "covariance": c,
                    "n_pairs": float(npairs),
                    "mass": float(mass),
                }
            )
    return rows


def compute_moi3d(
    samples: list[dict[str, float]],
) -> dict[str, Any]:
    """Accumulate 3D inertia tensor of covariance samples; eigen → axes.

    Smallest eigenvalue → major continuity axis; largest → minor.
    """
    if len(samples) < 3:
        return {
            "ok": False,
            "reason": f"too_few_moi_samples ({len(samples)})",
            "n_moi_samples": len(samples),
        }

    I = np.zeros((3, 3), dtype=float)
    for s in samples:
        u = np.array([s["ux"], s["uy"], s["uz"]], dtype=float)
        d = float(s["lag"])
        mass = float(s["mass"])
        r = d * u
        I += mass * (d * d * np.eye(3) - np.outer(r, r))

    I = 0.5 * (I + I.T)
    eigvals, eigvecs = np.linalg.eigh(I)  # ascending eigenvalues
    # major = smallest eig, minor = largest
    order = np.argsort(eigvals)  # already ascending, but explicit
    eigvals = eigvals[order]
    axes = eigvecs[:, order].copy()  # cols: maj, int, min
    for j in range(3):
        axes[:, j] = stabilize_axis(axes[:, j])
    axes[:, 2] = stabilize_axis(np.cross(axes[:, 0], axes[:, 1]))
    # re-orthonormalize int via RH
    axes[:, 1] = stabilize_axis(np.cross(axes[:, 2], axes[:, 0]))
    axes[:, 2] = stabilize_axis(np.cross(axes[:, 0], axes[:, 1]))

    ev = np.asarray(eigvals, dtype=float)
    emin = float(np.min(ev))
    emax = float(np.max(ev))
    if emin <= 1e-18:
        strength = float("inf") if emax > 1e-18 else 1.0
    else:
        strength = float(emax / emin)

    return {
        "ok": True,
        "I": I,
        "moi_eigenvalues": ev.copy(),  # maj, int, min order
        "moi_strength": strength,
        "axes": axes,
        "orientation_matrix": axes.copy(),
        "n_moi_samples": len(samples),
    }


def _sill_approach_ok(
    lag: np.ndarray,
    gamma: np.ndarray,
    *,
    nugget: float,
    partial_sill: float,
    frac: float,
) -> bool:
    """Late-lag mean γ approaches total sill (engineering heuristic)."""
    if lag.size < 2:
        return False
    total = float(nugget) + float(partial_sill)
    if total <= 1e-18:
        return False
    # last third of lags by distance
    order = np.argsort(lag)
    k = max(1, int(np.ceil(lag.size / 3.0)))
    late = gamma[order[-k:]]
    return bool(float(np.mean(late)) >= float(frac) * total)


def _fit_one_axis(
    pairs: PairCloud3D,
    axis: np.ndarray,
    *,
    omni_nugget: float,
    omni_sill: float,
    max_dist: float,
    cfg: dict[str, Any],
) -> VariogramFit:
    """Adaptive-tol range-only fit on one undirected axis with identifiability gates."""
    rfcfg = cfg.get("range_fit") or {}
    rcfg = cfg.get("refinement") or {}
    vcfg = cfg.get("variogram") or {}
    dcfg = cfg.get("directions") or {}

    n_lags = int(rfcfg.get("n_lags", rcfg.get("n_lags", 16)))
    min_lags = int(rfcfg.get("min_valid_lags", 5))
    min_pairs_axis = int(rfcfg.get("min_pairs_per_axis", 80))
    min_pairs_lag = int(
        rfcfg.get("min_pairs_per_lag", vcfg.get("min_pairs_per_lag", 8))
    )
    boundary_frac = float(
        rfcfg.get("boundary_frac", dcfg.get("range_boundary_frac", 0.98))
    )
    upper_factor = float(rfcfg.get("range_upper_factor", 2.5))
    range_upper = float(max_dist) * upper_factor
    require_sill = bool(rfcfg.get("require_sill_approach", True))
    sill_frac = float(rfcfg.get("sill_approach_frac", 0.8))

    tol_list = rfcfg.get("adaptive_angular_tol_deg")
    if not tol_list:
        tol_list = [float(rcfg.get("angular_tolerance_deg", 18.0))]
    tols = sorted({float(t) for t in tol_list})
    if not bool(rfcfg.get("prefer_smallest_tol", True)):
        tols = list(reversed(tols))

    bw_raw = rcfg.get("bandwidth", None)
    bandwidth = float(bw_raw) if bw_raw is not None else None
    ax = stabilize_axis(np.asarray(axis, dtype=float).reshape(3))

    best_fit: VariogramFit | None = None
    last_lag = np.array([], dtype=float)
    last_gamma = np.array([], dtype=float)
    last_counts = np.array([], dtype=float)
    last_tol = float(tols[-1]) if tols else 22.5
    last_n_cone = 0

    for tol in tols:
        last_tol = float(tol)
        lag, gamma, counts, n_cone = experimental_directional(
            pairs,
            ax,
            angular_tolerance_deg=float(tol),
            bandwidth=bandwidth,
            n_lags=n_lags,
            max_dist=max_dist,
            min_pairs=min_pairs_lag,
        )
        last_lag, last_gamma, last_counts, last_n_cone = lag, gamma, counts, n_cone
        n_pairs_total = int(np.sum(counts)) if counts.size else 0
        if int(lag.size) < min_lags or n_pairs_total < min_pairs_axis:
            continue
        fit = fit_spherical_weighted(
            lag,
            gamma,
            counts,
            nugget=float(omni_nugget),
            sill=float(omni_sill),
            max_range=range_upper,
            boundary_frac=boundary_frac,
            min_valid_lags=min_lags,
        )
        fit.angular_tolerance_deg_used = float(tol)
        fit.n_pairs_total = n_pairs_total
        fit.max_observed_lag = float(np.max(lag)) if lag.size else None
        fit.n_valid_lags = int(lag.size)
        fit.range_upper_bound = range_upper
        approach = _sill_approach_ok(
            lag,
            gamma,
            nugget=float(omni_nugget),
            partial_sill=float(omni_sill),
            frac=sill_frac,
        )
        fit.sill_approach_ok = approach
        reasons: list[str] = []
        status = "reliable"
        if fit.range_boundary_hit:
            status = "not_identifiable"
            reasons.append(
                "range_boundary_hit: optimizer at artificial upper bound "
                f"{range_upper:.4g} (factor={upper_factor}×max_dist); "
                "not a geological range estimate"
            )
        if require_sill and not approach:
            if status == "reliable":
                status = "unreliable"
            reasons.append(
                f"sill_approach_failed (late-lag γ < {sill_frac}×total_sill)"
            )
        if not np.isfinite(fit.range_) or float(fit.range_) <= 0:
            status = "not_identifiable"
            reasons.append("non_finite_or_non_positive_range")
        fit.status = status
        fit.status_reasons = reasons
        best_fit = fit
        break  # smallest tol that meets lag/pair gates

    if best_fit is not None:
        return best_fit

    # Insufficient support even at widest tolerance
    reasons = [
        f"insufficient_directional_support at tol<={last_tol}° "
        f"(lags={int(last_lag.size)}<{min_lags} or "
        f"pairs={int(np.sum(last_counts) if last_counts.size else 0)}"
        f"<{min_pairs_axis}; n_cone={last_n_cone})"
    ]
    return VariogramFit(
        lag=last_lag,
        gamma=last_gamma,
        n_pairs=last_counts,
        nugget=float(omni_nugget),
        sill=float(omni_sill),
        range_=float("nan"),
        sse=float("inf"),
        n_valid_lags=int(last_lag.size),
        range_upper_bound=range_upper,
        range_boundary_hit=False,
        status="not_identifiable",
        angular_tolerance_deg_used=last_tol,
        n_pairs_total=int(np.sum(last_counts) if last_counts.size else 0),
        max_observed_lag=float(np.max(last_lag)) if last_lag.size else None,
        sill_approach_ok=False,
        status_reasons=reasons,
    )


def fit_ranges_on_axes(
    pairs: PairCloud3D,
    axes: np.ndarray,
    *,
    omni_nugget: float,
    omni_sill: float,
    max_dist: float,
    cfg: dict[str, Any],
    sort_by_range: bool = True,
) -> dict[str, Any]:
    """Fit range-only directional VG on maj/int/min axes with identifiability.

    If ``sort_by_range``, sort linked (range, axis, fit) together by range desc
    among finite ranges only (non-finite stay at end).
    """
    axes = np.asarray(axes, dtype=float).copy()
    packed: list[tuple[float, np.ndarray, VariogramFit]] = []
    for j in range(3):
        ax = stabilize_axis(axes[:, j])
        fit = _fit_one_axis(
            pairs,
            ax,
            omni_nugget=omni_nugget,
            omni_sill=omni_sill,
            max_dist=max_dist,
            cfg=cfg,
        )
        r = float(fit.range_) if np.isfinite(fit.range_) else float("nan")
        packed.append((r, ax, fit))

    if sort_by_range:
        def _key(item: tuple[float, np.ndarray, VariogramFit]) -> float:
            r = item[0]
            return -r if np.isfinite(r) else float("inf")

        packed.sort(key=_key)

    ranges = np.array([p[0] for p in packed], dtype=float)
    axes_out = np.column_stack([p[1] for p in packed])
    # Keep RH triad when possible
    axes_out[:, 2] = stabilize_axis(np.cross(axes_out[:, 0], axes_out[:, 1]))
    axes_out[:, 1] = stabilize_axis(np.cross(axes_out[:, 2], axes_out[:, 0]))
    fits = [p[2] for p in packed]
    for j, fit in enumerate(fits):
        if np.isfinite(ranges[j]):
            fit.range_ = float(ranges[j])
    statuses = [f.status for f in fits]
    return {
        "ranges": ranges,
        "axes": axes_out,
        "axis_fits": fits,
        "axis_range_status": statuses,
        "n_reliable": sum(1 for s in statuses if s == "reliable"),
        "n_unreliable": sum(1 for s in statuses if s == "unreliable"),
        "n_not_identifiable": sum(1 for s in statuses if s == "not_identifiable"),
    }


def _pair_vertical_fraction(pairs: PairCloud3D, dip_thresh_deg: float = 60.0) -> float:
    """Fraction of pairs with |uz| high (proxy for inside-hole pairs)."""
    if pairs.n == 0:
        return 0.0
    ang_from_horiz = np.degrees(np.arcsin(np.clip(np.abs(pairs.uz), 0.0, 1.0)))
    return float(np.mean(ang_from_horiz >= float(dip_thresh_deg)))


def run_moi3d_detector(
    pairs: PairCloud3D,
    *,
    omni_nugget: float,
    omni_sill: float,
    omni_range: float,
    max_dist: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """MOI orientation estimator + dual ISO/ANISO candidates.

    MOI sets principal orientation; directional VG checks range identifiability.
    Three MOI eigenvectors ≠ three reliable ranges.
    """
    mcfg = cfg.get("moi") or {}
    ccfg = cfg.get("confidence") or {}
    iso_ratio_max = float(mcfg.get("isotropy_ratio_max", 1.30))
    orient_min_strength = float(ccfg.get("orientation_min_strength", 1.15))
    orient_min_samples = int(ccfg.get("orientation_min_moi_samples", 24))
    class_cfg = cfg.get("classification") or {}
    ratio_iso_max = float(class_cfg.get("ratio_iso_max", 1.3))
    vg_model = str((cfg.get("variogram") or {}).get("model", "spherical"))

    samples = build_radial_covariance_samples(
        pairs,
        omni_nugget=omni_nugget,
        omni_sill=omni_sill,
        max_dist=max_dist,
        cfg=cfg,
    )
    moi = compute_moi3d(samples)
    vert_frac = _pair_vertical_fraction(pairs)

    iso_candidate: dict[str, Any] = {
        "type": "ISOTROPIC",
        "range_major": float(omni_range),
        "range_intermediate": float(omni_range),
        "range_minor": float(omni_range),
        "orientation_matrix": np.eye(3).tolist(),
        "nugget": float(omni_nugget),
        "sill": float(omni_sill),
        "partial_sill": float(omni_sill),
        "total_sill": float(omni_nugget) + float(omni_sill),
        "variogram_model": vg_model,
        "technical_fallback": True,
        "geological_isotropy_claimed": False,
    }

    base: dict[str, Any] = {
        "samples": samples,
        "n_moi_samples": len(samples),
        "moi_eigenvalues": None,
        "moi_strength": None,
        "moi_axes": None,
        "axis_fits": None,
        "axis_range_status": None,
        "iso_candidate": iso_candidate,
        "aniso_candidate": None,
        "pair_vertical_fraction": vert_frac,
        "anisotropy_evidence": {
            "moi_ok": bool(moi.get("ok")),
            "standalone_selected": "ISO",
            "standalone_reason": moi.get("reason", "moi_failed"),
            "anisotropy_identifiability": "NOT_IDENTIFIABLE",
        },
    }
    if not moi.get("ok"):
        # Technical sphere for downstream; not a geological ISO claim
        sphere = {
            **iso_candidate,
            "fallback_kind": "technical_sphere_moi_failed",
            "fallback_reason": moi.get("reason", "moi_failed"),
        }
        base["aniso_candidate"] = sphere
        return {
            **base,
            "ok": False,
            "force_isotropic": True,
            "orientation_usable": False,
            "ranges_reliable": False,
            "anisotropy_identifiability": "NOT_IDENTIFIABLE",
            "n_reliable_axes": 0,
            "reason": moi.get("reason", "moi_failed"),
            "ranges": np.array([omni_range, omni_range, omni_range], dtype=float),
            "axes": np.eye(3),
        }

    ev = np.asarray(moi["moi_eigenvalues"], dtype=float)
    strength = float(moi["moi_strength"])
    axes = np.asarray(moi["axes"], dtype=float)
    n_samples = int(moi["n_moi_samples"])
    orientation_usable = bool(
        strength >= orient_min_strength and n_samples >= orient_min_samples
    )
    base.update(
        {
            "moi_eigenvalues": ev.tolist(),
            "moi_strength": strength,
            "moi_axes": axes.tolist(),
            "n_moi_samples": n_samples,
            "orientation_usable": orientation_usable,
        }
    )
    if vert_frac >= 0.55:
        base["orientation_warning"] = (
            f"high_vertical_pair_fraction={vert_frac:.2f}: "
            "inside-hole pairs may bias MOI orientation"
        )

    fitted_moi = fit_ranges_on_axes(
        pairs,
        axes,
        omni_nugget=omni_nugget,
        omni_sill=omni_sill,
        max_dist=max_dist,
        cfg=cfg,
        sort_by_range=False,
    )
    ranges_moi = np.asarray(fitted_moi["ranges"], dtype=float)
    axes_moi = np.asarray(fitted_moi["axes"], dtype=float)
    fits_moi = fitted_moi["axis_fits"]
    statuses = list(fitted_moi["axis_range_status"])
    n_reliable = int(fitted_moi["n_reliable"])
    base["axis_fits"] = fits_moi
    base["axis_range_status"] = statuses
    base["n_reliable_axes"] = n_reliable

    # Sort by fitted range among finite values for reporting order
    finite_mask = np.isfinite(ranges_moi) & (ranges_moi > 0)
    if int(finite_mask.sum()) == 3:
        order = np.argsort(ranges_moi)[::-1]
    else:
        order = np.arange(3)
    ranges_sorted = ranges_moi[order].copy()
    axes_sorted = axes_moi[:, order].copy()
    axes_sorted[:, 2] = stabilize_axis(np.cross(axes_sorted[:, 0], axes_sorted[:, 1]))
    axes_sorted[:, 1] = stabilize_axis(np.cross(axes_sorted[:, 2], axes_sorted[:, 0]))
    fits_sorted = [fits_moi[int(i)] for i in order]
    statuses_sorted = [statuses[int(i)] for i in order]
    for j, fit in enumerate(fits_sorted):
        if np.isfinite(ranges_sorted[j]):
            fit.range_ = float(ranges_sorted[j])

    dir_fits = []
    for j, fit in enumerate(fits_sorted):
        d = fit.as_dict()
        d["range"] = float(ranges_sorted[j]) if np.isfinite(ranges_sorted[j]) else None
        d["axis_index"] = int(j)
        dir_fits.append(d)

    all_reliable = n_reliable == 3
    ranges_reliable = all_reliable
    strength_gate = bool(strength > iso_ratio_max)
    allow_partial = bool(mcfg.get("allow_partial_ranges", True))

    if all_reliable:
        ident = "IDENTIFIED"
        r_out = ranges_sorted.copy()
        ax_out = axes_sorted.copy()
        technical = False
        fallback_kind = None
        maj_min = float(r_out[0] / max(float(r_out[2]), 1e-12))
        ranges_gate = maj_min >= ratio_iso_max
        force_iso = not (strength_gate and ranges_gate)
        reason = None
        if force_iso:
            if not strength_gate:
                reason = f"moi_weak_orientation (strength={strength:.4g}<={iso_ratio_max})"
            else:
                reason = (
                    f"ranges_isotropic (maj/min={maj_min:.4g}<{ratio_iso_max})"
                )
            # Still keep identified anisotropic candidate numbers; standalone may prefer ISO
        aniso_candidate = {
            "orientation_matrix": ax_out.tolist(),
            "major_axis_xyz": ax_out[:, 0].tolist(),
            "intermediate_axis_xyz": ax_out[:, 1].tolist(),
            "minor_axis_xyz": ax_out[:, 2].tolist(),
            "range_major": float(r_out[0]),
            "range_intermediate": float(r_out[1]),
            "range_minor": float(r_out[2]),
            "major_minor_ratio": float(maj_min),
            "directional_fits": dir_fits,
            "axis_range_status": statuses_sorted,
            "moi_strength": strength,
            "moi_eigenvalues": ev.tolist(),
            "nugget": float(omni_nugget),
            "sill": float(omni_sill),
            "partial_sill": float(omni_sill),
            "total_sill": float(omni_nugget) + float(omni_sill),
            "variogram_model": vg_model,
            "ranges_reliable": True,
            "orientation_usable_for_aniso_search": True,
            "technical_fallback": False,
            "anisotropy_identifiability": "IDENTIFIED",
        }
    elif orientation_usable and n_reliable >= 1 and allow_partial:
        # Sparse-data degradation: keep confirmed axis ranges; omni-fill the rest.
        # Do not claim full IDENTIFIED / geological classification.
        ident = "PARTIAL_RANGES"
        partial = assemble_partial_ranges(
            ranges_moi, axes_moi, statuses, fits_moi, float(omni_range)
        )
        r_out = np.asarray(partial["ranges"], dtype=float)
        ax_out = np.asarray(partial["axes"], dtype=float)
        statuses_sorted = list(partial["axis_range_status"])
        fits_sorted = list(partial["axis_fits"])
        dir_fits = list(partial["directional_fits"])
        technical = True
        fallback_kind = "partial_reliable_ranges_omni_fill"
        force_iso = False
        reason = (
            "partial_ranges: keeping reliable axis ranges, omni-fill elsewhere "
            f"(reliable_axes={n_reliable}/3; statuses={statuses}; "
            f"range_origin={partial['range_origin']})"
        )
        maj_min = float(r_out[0] / max(float(r_out[2]), 1e-12))
        aniso_candidate = {
            "orientation_matrix": ax_out.tolist(),
            "major_axis_xyz": ax_out[:, 0].tolist(),
            "intermediate_axis_xyz": ax_out[:, 1].tolist(),
            "minor_axis_xyz": ax_out[:, 2].tolist(),
            "range_major": float(r_out[0]),
            "range_intermediate": float(r_out[1]),
            "range_minor": float(r_out[2]),
            "major_minor_ratio": float(maj_min),
            "directional_fits": dir_fits,
            "axis_range_status": statuses_sorted,
            "range_origin": list(partial["range_origin"]),
            "fitted_ranges_moi_order": [
                float(x) if np.isfinite(x) else None for x in ranges_moi.tolist()
            ],
            "moi_strength": strength,
            "moi_eigenvalues": ev.tolist(),
            "nugget": float(omni_nugget),
            "sill": float(omni_sill),
            "partial_sill": float(omni_sill),
            "total_sill": float(omni_nugget) + float(omni_sill),
            "variogram_model": vg_model,
            "ranges_reliable": False,
            "orientation_usable_for_aniso_search": True,
            "technical_fallback": True,
            "fallback_kind": fallback_kind,
            "fallback_reason": reason,
            "anisotropy_identifiability": "PARTIAL_RANGES",
            "geological_isotropy_claimed": False,
        }
    elif orientation_usable:
        ident = "ORIENTATION_ONLY"
        # Downstream-safe: omni ranges + MOI orientation (covariance isotropic in Q)
        r_out = np.array([omni_range, omni_range, omni_range], dtype=float)
        ax_out = axes.copy()
        technical = True
        fallback_kind = "omni_ranges_with_moi_orientation"
        force_iso = False  # orientation usable; not a silent ISO claim
        reason = (
            "ranges_not_identifiable; keeping MOI orientation with omni ranges "
            f"(reliable_axes={n_reliable}/3; statuses={statuses})"
        )
        aniso_candidate = {
            "orientation_matrix": ax_out.tolist(),
            "major_axis_xyz": ax_out[:, 0].tolist(),
            "intermediate_axis_xyz": ax_out[:, 1].tolist(),
            "minor_axis_xyz": ax_out[:, 2].tolist(),
            "range_major": float(omni_range),
            "range_intermediate": float(omni_range),
            "range_minor": float(omni_range),
            "major_minor_ratio": 1.0,
            "directional_fits": dir_fits,
            "axis_range_status": statuses_sorted,
            "fitted_ranges_moi_order": [
                float(x) if np.isfinite(x) else None for x in ranges_moi.tolist()
            ],
            "moi_strength": strength,
            "moi_eigenvalues": ev.tolist(),
            "nugget": float(omni_nugget),
            "sill": float(omni_sill),
            "partial_sill": float(omni_sill),
            "total_sill": float(omni_nugget) + float(omni_sill),
            "variogram_model": vg_model,
            "ranges_reliable": False,
            "orientation_usable_for_aniso_search": True,
            "technical_fallback": True,
            "fallback_kind": fallback_kind,
            "fallback_reason": reason,
            "anisotropy_identifiability": "ORIENTATION_ONLY",
            "geological_isotropy_claimed": False,
        }
    else:
        ident = "NOT_IDENTIFIABLE"
        r_out = np.array([omni_range, omni_range, omni_range], dtype=float)
        ax_out = np.eye(3)
        technical = True
        fallback_kind = "technical_sphere_not_identifiable"
        force_iso = True
        reason = (
            f"anisotropy_not_identifiable (orientation_usable={orientation_usable}, "
            f"reliable_axes={n_reliable}/3, strength={strength:.4g})"
        )
        aniso_candidate = {
            **iso_candidate,
            "directional_fits": dir_fits,
            "axis_range_status": statuses_sorted,
            "fitted_ranges_moi_order": [
                float(x) if np.isfinite(x) else None for x in ranges_moi.tolist()
            ],
            "moi_strength": strength,
            "fallback_kind": fallback_kind,
            "fallback_reason": reason,
            "ranges_reliable": False,
            "orientation_usable_for_aniso_search": False,
            "anisotropy_identifiability": "NOT_IDENTIFIABLE",
            "geological_isotropy_claimed": False,
        }

    base["aniso_candidate"] = aniso_candidate
    evidence = {
        "moi_ok": True,
        "moi_strength": strength,
        "moi_eigenvalues": ev.tolist(),
        "moi_isotropy_ratio_max": iso_ratio_max,
        "orientation_usable": orientation_usable,
        "n_reliable_axes": n_reliable,
        "axis_range_status": statuses_sorted,
        "anisotropy_identifiability": ident,
        "ranges_reliable": ranges_reliable,
        "aniso_candidate_available": True,
        "standalone_selected": (
            "ANISO"
            if ident == "IDENTIFIED" and not force_iso
            else (
                "PARTIAL_RANGES"
                if ident == "PARTIAL_RANGES"
                else ("ORIENTATION_ONLY" if ident == "ORIENTATION_ONLY" else "ISO")
            )
        ),
        "standalone_reason": reason,
        "pair_vertical_fraction": vert_frac,
        "technical_fallback": technical,
        "fallback_kind": fallback_kind,
        "geological_isotropy_claimed": bool(
            ident == "IDENTIFIED" and force_iso and ranges_reliable
        ),
    }
    base["anisotropy_evidence"] = evidence

    keep_oriented = ident in ("ORIENTATION_ONLY", "PARTIAL_RANGES")
    return {
        **base,
        "ok": True,
        "force_isotropic": bool(force_iso and not keep_oriented),
        "orientation_usable": orientation_usable,
        "ranges_reliable": ranges_reliable,
        "anisotropy_identifiability": ident,
        "n_reliable_axes": n_reliable,
        "reason": reason,
        "ranges": np.asarray(r_out, dtype=float).copy(),
        "axes": np.asarray(ax_out, dtype=float).copy(),
        "axis_fits": fits_sorted,
        "axis_range_status": statuses_sorted,
        "moi_strength": strength,
        "ranges_pre_accept": ranges_moi.tolist(),
    }
