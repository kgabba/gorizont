"""3D moment-of-inertia detector on radial covariance volume."""

from __future__ import annotations

from typing import Any

import numpy as np

from .directions import fibonacci_hemisphere
from .ellipsoid import stabilize_axis
from .models import PairCloud3D, VariogramFit
from .variogram import experimental_directional, fit_spherical_weighted


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
    """Fit range-only directional VG on maj/int/min axes.

    If ``sort_by_range``, sort linked (range, axis, fit) together by range desc.
    """
    rcfg = cfg.get("refinement") or {}
    dcfg = cfg.get("directions") or {}
    vcfg = cfg.get("variogram") or {}
    n_lags = int(rcfg.get("n_lags", 16))
    tol = float(rcfg.get("angular_tolerance_deg", 18.0))
    bw_raw = rcfg.get("bandwidth", None)
    bandwidth = float(bw_raw) if bw_raw is not None else None
    min_pairs_lag = int(vcfg.get("min_pairs_per_lag", 8))
    boundary_frac = float(dcfg.get("range_boundary_frac", 0.98))
    range_upper = float(max_dist) * 2.5

    axes = np.asarray(axes, dtype=float).copy()
    packed: list[tuple[float, np.ndarray, VariogramFit]] = []
    for j in range(3):
        ax = stabilize_axis(axes[:, j])
        lag, gamma, counts, _ = experimental_directional(
            pairs,
            ax,
            angular_tolerance_deg=tol,
            bandwidth=bandwidth,
            n_lags=n_lags,
            max_dist=max_dist,
            min_pairs=min_pairs_lag,
        )
        if lag.size < 2:
            fit = VariogramFit(
                lag=lag,
                gamma=gamma,
                n_pairs=counts,
                nugget=float(omni_nugget),
                sill=float(omni_sill),
                range_=float("nan"),
                sse=float("inf"),
                n_valid_lags=int(lag.size),
            )
            packed.append((float("nan"), ax, fit))
            continue
        fit = fit_spherical_weighted(
            lag,
            gamma,
            counts,
            nugget=float(omni_nugget),
            sill=float(omni_sill),
            max_range=range_upper,
            boundary_frac=boundary_frac,
        )
        packed.append((float(fit.range_), ax, fit))

    if sort_by_range:
        def _key(item: tuple[float, np.ndarray, VariogramFit]) -> float:
            r = item[0]
            return -r if np.isfinite(r) else float("inf")

        packed.sort(key=_key)

    ranges = np.array([p[0] for p in packed], dtype=float)
    axes_out = np.column_stack([p[1] for p in packed])
    axes_out[:, 2] = stabilize_axis(np.cross(axes_out[:, 0], axes_out[:, 1]))
    fits = [p[2] for p in packed]
    for j, fit in enumerate(fits):
        if np.isfinite(ranges[j]):
            fit.range_ = float(ranges[j])
    return {"ranges": ranges, "axes": axes_out, "axis_fits": fits}


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

    MOI is used for principal orientation, not as a hard existence test.
    Conservative gates still choose the standalone output, but never drop
    a technically valid ANISO candidate from the result payload.
    """
    mcfg = cfg.get("moi") or {}
    iso_ratio_max = float(mcfg.get("isotropy_ratio_max", 1.15))
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

    iso_candidate: dict[str, Any] = {
        "type": "ISOTROPIC",
        "range_major": float(omni_range),
        "range_intermediate": float(omni_range),
        "range_minor": float(omni_range),
        "orientation_matrix": np.eye(3).tolist(),
        "nugget": float(omni_nugget),
        "sill": float(omni_sill),
        "variogram_model": vg_model,
    }

    base: dict[str, Any] = {
        "samples": samples,
        "n_moi_samples": len(samples),
        "moi_eigenvalues": None,
        "moi_strength": None,
        "moi_axes": None,
        "axis_fits": None,
        "iso_candidate": iso_candidate,
        "aniso_candidate": None,
        "anisotropy_evidence": {
            "moi_ok": bool(moi.get("ok")),
            "standalone_selected": "ISO",
            "standalone_reason": moi.get("reason", "moi_failed"),
        },
    }
    if not moi.get("ok"):
        return {
            **base,
            "ok": False,
            "force_isotropic": True,
            "reason": moi.get("reason", "moi_failed"),
            "ranges": np.array([omni_range, omni_range, omni_range], dtype=float),
            "axes": np.eye(3),
        }

    ev = np.asarray(moi["moi_eigenvalues"], dtype=float)
    strength = float(moi["moi_strength"])
    axes = np.asarray(moi["axes"], dtype=float)
    base.update(
        {
            "moi_eigenvalues": ev.tolist(),
            "moi_strength": strength,
            "moi_axes": axes.tolist(),
            "n_moi_samples": int(moi["n_moi_samples"]),
        }
    )

    # Always fit directional ranges on MOI axes when MOI tensor is valid
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
    base["axis_fits"] = fits_moi

    aniso_candidate: dict[str, Any] | None = None
    ranges_ok = bool(np.all(np.isfinite(ranges_moi)) and float(np.min(ranges_moi)) > 0)
    ranges_sorted = None
    axes_sorted = None
    order = None
    if ranges_ok:
        order = np.argsort(ranges_moi)[::-1]
        ranges_sorted = ranges_moi[order]
        axes_sorted = axes_moi[:, order].copy()
        axes_sorted[:, 2] = stabilize_axis(
            np.cross(axes_sorted[:, 0], axes_sorted[:, 1])
        )
        dir_fits = []
        for j, idx in enumerate(order):
            d = fits_moi[int(idx)].as_dict()
            d["range"] = float(ranges_sorted[j])
            dir_fits.append(d)
        aniso_candidate = {
            "orientation_matrix": axes_sorted.tolist(),
            "major_axis_xyz": axes_sorted[:, 0].tolist(),
            "intermediate_axis_xyz": axes_sorted[:, 1].tolist(),
            "minor_axis_xyz": axes_sorted[:, 2].tolist(),
            "range_major": float(ranges_sorted[0]),
            "range_intermediate": float(ranges_sorted[1]),
            "range_minor": float(ranges_sorted[2]),
            "major_minor_ratio": float(
                ranges_sorted[0] / max(float(ranges_sorted[2]), 1e-12)
            ),
            "directional_fits": dir_fits,
            "moi_strength": strength,
            "moi_eigenvalues": ev.tolist(),
            "moi_axes": axes.tolist(),
            "ranges_moi_order": ranges_moi.tolist(),
            "nugget": float(omni_nugget),
            "sill": float(omni_sill),
            "variogram_model": vg_model,
        }
        base["aniso_candidate"] = aniso_candidate

    r_maj = float(ranges_moi[0]) if ranges_ok else float("nan")
    r_min = float(ranges_moi[2]) if ranges_ok else float("nan")
    maj_min = r_maj / max(r_min, 1e-12) if ranges_ok else float("nan")
    moi_range_consistent = bool(ranges_ok and r_maj >= r_min)
    strength_gate = bool(strength > iso_ratio_max)
    ranges_gate = bool(ranges_ok and maj_min >= ratio_iso_max)

    # Conservative standalone selection (does not drop aniso_candidate)
    force_iso = True
    reason: str | None
    if not ranges_ok:
        reason = "axis_range_fit_failed"
    elif not strength_gate:
        reason = f"moi_isotropic (strength={strength:.4g}<={iso_ratio_max})"
    elif not moi_range_consistent:
        reason = (
            f"moi_range_inconsistent (r_maj={r_maj:.4g}<r_min={r_min:.4g}; "
            f"moi_strength={strength:.4g})"
        )
    elif not ranges_gate:
        reason = (
            f"ranges_isotropic (maj/min={maj_min:.4g}<{ratio_iso_max}; "
            f"moi_strength={strength:.4g})"
        )
    else:
        force_iso = False
        reason = None

    evidence = {
        "moi_ok": True,
        "moi_strength": strength,
        "moi_eigenvalues": ev.tolist(),
        "moi_isotropy_ratio_max": iso_ratio_max,
        "moi_passes_strength_gate": strength_gate,
        "range_major_minor_ratio_moi_order": (
            float(maj_min) if np.isfinite(maj_min) else None
        ),
        "ratio_iso_max": ratio_iso_max,
        "ranges_pass_aniso_gate": ranges_gate,
        "moi_range_consistent": moi_range_consistent,
        "aniso_candidate_available": aniso_candidate is not None,
        "standalone_selected": "ISO" if force_iso else "ANISO",
        "standalone_reason": reason,
    }
    base["anisotropy_evidence"] = evidence

    if force_iso or aniso_candidate is None:
        return {
            **base,
            "ok": True,
            "force_isotropic": True,
            "reason": reason or "force_isotropic",
            "ranges": np.array([omni_range, omni_range, omni_range], dtype=float),
            "axes": np.eye(3),
            "ranges_pre_accept": ranges_moi.tolist() if ranges_ok else None,
        }

    assert order is not None and ranges_sorted is not None and axes_sorted is not None
    fits_sorted = [fits_moi[int(i)] for i in order]
    for j, fit in enumerate(fits_sorted):
        fit.range_ = float(ranges_sorted[j])

    return {
        **base,
        "ok": True,
        "force_isotropic": False,
        "reason": None,
        "ranges": np.asarray(ranges_sorted, dtype=float).copy(),
        "axes": np.asarray(axes_sorted, dtype=float).copy(),
        "axis_fits": fits_sorted,
        "moi_strength": strength,
        "ranges_pre_accept": ranges_moi.tolist(),
    }
