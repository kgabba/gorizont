"""Orchestration: validate → pairs → omni → detector → classify → conf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .classification import classify_anisotropy
from .confidence import decide_confidence_and_fallback
from .directions import directional_coverage_ok, scan_directions
from .ellipsoid import fit_ellipsoid
from .io import DataValidationError, load_points, validate_points
from .models import Anisotropy3DResult, DirectionalEstimate
from .moi3d import run_moi3d_detector
from .pairs import compute_pairs
from .refinement import refine_principal_axes
from .report import vector_to_azimuth_dip, write_report
from .variogram import fit_omni


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_legacy_range_ellipsoid(
    *,
    pairs,
    omni,
    max_dist: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Legacy detector: Fibonacci ranges → SPD ellipsoid (kept for A/B)."""
    estimates = scan_directions(
        pairs,
        omni_nugget=omni.nugget,
        omni_sill=omni.sill,
        max_dist=max_dist,
        cfg=cfg,
    )
    n_valid = sum(1 for e in estimates if e.valid)
    n_boundary = sum(1 for e in estimates if e.valid and e.range_boundary_hit)
    n_usable = sum(
        1
        for e in estimates
        if e.valid
        and not e.range_boundary_hit
        and np.isfinite(e.range_)
        and e.range_ > 0
    )
    cov_ok, cov_diag = directional_coverage_ok(estimates, cfg)
    print(
        f"[anisotropy3d] legacy scan: {n_valid}/{len(estimates)} valid "
        f"(usable={n_usable}, boundary_hit={n_boundary}) coverage_ok={cov_ok}",
        flush=True,
    )

    ell = fit_ellipsoid(estimates, cfg)
    axis_fits = None
    if not ell.get("ok"):
        ranges = np.array([omni.range_, omni.range_, omni.range_])
        axes = np.eye(3)
        ell_rel = None
        ell_reason = ell.get("reason", "ellipsoid_fit_failed")
    else:
        refined = refine_principal_axes(
            pairs,
            ell["axes"],
            ell["ranges"],
            omni_nugget=omni.nugget,
            omni_sill=omni.sill,
            max_dist=max_dist,
            cfg=cfg,
        )
        ranges = np.asarray(ell["ranges"], dtype=float).copy()
        axes = np.asarray(ell["axes"], dtype=float).copy()
        axis_fits = refined["axis_fits"]
        ell_rel = float(ell["rmse_inv_r2_rel"])
        ell_reason = None
        print(
            f"[anisotropy3d] ellipsoid rel_rmse={ell_rel:.4f} "
            f"ranges={ranges[0]:.4g}/{ranges[1]:.4g}/{ranges[2]:.4g}",
            flush=True,
        )

    return {
        "estimates": estimates,
        "ranges": ranges,
        "axes": axes,
        "nugget": float(omni.nugget),
        "sill": float(omni.sill),
        "axis_fits": axis_fits,
        "ell_rel": ell_rel,
        "ell_reason": ell_reason,
        "n_usable": n_usable,
        "n_boundary": n_boundary,
        "cov_ok": cov_ok,
        "cov_diag": cov_diag,
        "ell": ell,
        "moi_samples": None,
        "extra": {
            "detector_mode": "legacy_range_ellipsoid",
            "ellipsoid_ok": bool(ell.get("ok")),
            "ellipsoid_rmse_inv_r2": ell.get("rmse_inv_r2"),
            "n_boundary_excluded": ell.get("n_boundary_excluded", n_boundary),
            **cov_diag,
        },
    }


def _run_moi3d(
    *,
    pairs,
    omni,
    max_dist: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Primary detector: radial covariance → MOI axes → 3 directional ranges."""
    det = run_moi3d_detector(
        pairs,
        omni_nugget=omni.nugget,
        omni_sill=omni.sill,
        omni_range=float(omni.range_),
        max_dist=max_dist,
        cfg=cfg,
    )
    ranges = np.asarray(det["ranges"], dtype=float)
    axes = np.asarray(det["axes"], dtype=float)
    axis_fits = det.get("axis_fits")
    strength = det.get("moi_strength")
    print(
        f"[anisotropy3d] moi3d n_samples={det.get('n_moi_samples')} "
        f"strength={strength if strength is not None else 'n/a'} "
        f"force_iso={det.get('force_isotropic')} "
        f"reason={det.get('reason')!r} "
        f"ranges={ranges[0]:.4g}/{ranges[1]:.4g}/{ranges[2]:.4g}",
        flush=True,
    )

    # If MOI accepted anisotropy but axis_fits missing, build diagnostic VG
    if axis_fits is None and not det.get("force_isotropic"):
        refined = refine_principal_axes(
            pairs,
            axes,
            ranges,
            omni_nugget=omni.nugget,
            omni_sill=omni.sill,
            max_dist=max_dist,
            cfg=cfg,
        )
        axis_fits = refined["axis_fits"]
    elif axis_fits is None and det.get("force_isotropic"):
        # still useful plots along omni sphere axes
        refined = refine_principal_axes(
            pairs,
            np.eye(3),
            ranges,
            omni_nugget=omni.nugget,
            omni_sill=omni.sill,
            max_dist=max_dist,
            cfg=cfg,
        )
        axis_fits = refined["axis_fits"]

    n_samples = int(det.get("n_moi_samples") or 0)
    n_reliable = int(det.get("n_reliable_axes") or 0)
    # Coverage for MOI: orientation usable — NOT raw sample count as "directions"
    cov_ok = bool(det.get("orientation_usable"))
    n_boundary = 0
    if axis_fits:
        n_boundary = sum(
            1 for f in axis_fits if bool(getattr(f, "range_boundary_hit", False))
        )

    return {
        "estimates": [],  # no legacy per-direction range scan
        "ranges": ranges,
        "axes": axes,
        "nugget": float(omni.nugget),
        "sill": float(omni.sill),
        "axis_fits": axis_fits,
        "ell_rel": None,
        "ell_reason": det.get("reason") if det.get("force_isotropic") else None,
        # Do NOT pass n_moi_samples as n_usable / n_valid_directions
        "n_usable": n_reliable,
        "n_boundary": n_boundary,
        "cov_ok": cov_ok,
        "cov_diag": {
            "n_valid_for_coverage": n_reliable,
            "n_moi_samples_not_used_as_directions": n_samples,
            "coverage_singular_values": None,
            "coverage_sv_ratio": None,
            "coverage_rank": 3 if cov_ok else 0,
        },
        "ell": {"ok": False},
        "moi_samples": det.get("samples") or [],
        "extra": {
            "detector_mode": "moi3d",
            "moi_ok": bool(det.get("ok")),
            "moi_eigenvalues": det.get("moi_eigenvalues"),
            "moi_strength": det.get("moi_strength"),
            "moi_axes": det.get("moi_axes"),
            "n_moi_samples": n_samples,
            "orientation_usable": bool(det.get("orientation_usable")),
            "ranges_reliable": bool(det.get("ranges_reliable")),
            "anisotropy_identifiability": det.get("anisotropy_identifiability"),
            "axis_range_status": det.get("axis_range_status"),
            "n_reliable_axes": n_reliable,
            "moi_force_isotropic": bool(det.get("force_isotropic")),
            "moi_reason": det.get("reason"),
            "orientation_warning": det.get("orientation_warning"),
            "pair_vertical_fraction": det.get("pair_vertical_fraction"),
            "ranges_pre_accept": det.get("ranges_pre_accept"),
            "ellipsoid_ok": False,
            "iso_candidate": det.get("iso_candidate"),
            "aniso_candidate": det.get("aniso_candidate"),
            "anisotropy_evidence": det.get("anisotropy_evidence"),
        },
    }


def run_anisotropy3d(
    data_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
) -> Anisotropy3DResult:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_points(data_path)
    try:
        pts = validate_points(df, cfg)
    except DataValidationError:
        raise

    pairs, pair_meta = compute_pairs(pts.xyz, pts.values, cfg)
    if pairs.n < 50:
        raise DataValidationError(f"too few pairs after filters: {pairs.n}")

    omni, max_dist = fit_omni(pairs, cfg)
    print(
        f"[anisotropy3d] n={pts.diagnostics['n_points']} pairs={pairs.n} "
        f"max_dist={max_dist:.4g} omni range={omni.range_:.4g} "
        f"nugget={omni.nugget:.4g} sill={omni.sill:.4g}",
        flush=True,
    )

    pair_diag = {
        "total_possible_pairs": pair_meta["total_possible_pairs"],
        "used_pairs": pair_meta["used_pairs"],
        "pair_sampling_applied": pair_meta["pair_sampling_applied"],
        "pair_sampling_fraction": pair_meta["pair_sampling_fraction"],
        "max_dist": pair_meta["max_dist"],
        "n_pairs": pairs.n,
    }

    deg = pts.diagnostics.get("spatial_degeneracy")
    if deg in ("linear", "planar"):
        estimates: list[DirectionalEstimate] = []
        # Hard geometric degeneracy only (true line / zero-thickness sheet).
        # Pancake drillhole clouds are NOT planar here — see io.validate_points.
        vg_model = str((cfg.get("variogram") or {}).get("model", "spherical"))
        r_iso = float(max(omni.range_, 1e-12))
        nug = float(max(omni.nugget, 0.0))
        sil = float(omni.sill)
        if not np.isfinite(sil) or sil <= 1e-9:
            # Pure-nugget omni fit: keep a tiny positive partial sill for OK.
            sil = max(1e-6, 1e-6 * max(nug, 1.0))
        Q = np.eye(3)
        sphere_candidate = {
            "variogram_model": vg_model,
            "nugget": nug,
            "sill": sil,
            "partial_sill": sil,
            "total_sill": nug + sil,
            "range_major": r_iso,
            "range_intermediate": r_iso,
            "range_minor": r_iso,
            "orientation_matrix": Q.tolist(),
            "major_axis_xyz": Q[:, 0].tolist(),
            "intermediate_axis_xyz": Q[:, 1].tolist(),
            "minor_axis_xyz": Q[:, 2].tolist(),
            "major_minor_ratio": 1.0,
            "technical_fallback": True,
            "geological_isotropy_claimed": False,
            "anisotropy_identifiability": "NOT_IDENTIFIABLE",
            "ranges_reliable": False,
            "orientation_usable_for_aniso_search": False,
            "fallback_kind": "technical_sphere_spatial_degeneracy",
            "fallback_reason": (
                f"spatial_degeneracy={deg}: 3D anisotropy underdetermined; "
                "technical omni-sphere for kriging (not a geological ISO claim)"
            ),
        }
        result = decide_confidence_and_fallback(
            aniso_type="ISOTROPIC",
            ranges=np.array([r_iso, r_iso, r_iso]),
            axes=Q,
            nugget=nug,
            sill=sil,
            n_points=int(pts.diagnostics["n_points"]),
            n_directions=0,
            n_valid_directions=0,
            ellipsoid_rmse_rel=None,
            spatial_degeneracy=deg,
            class_notes=[
                f"technical_sphere_candidate_due_to_{deg}_geometry"
            ],
            cfg=cfg,
            omni_range=r_iso,
            variogram_model=vg_model,
            anisotropy_identifiability="NOT_IDENTIFIABLE",
            geological_isotropy_claimed=False,
            force_technical_sphere=True,
            orientation_confidence="LOW",
            range_confidence="LOW",
            extra_diagnostics={
                **pts.diagnostics,
                **pair_diag,
                "omni": omni.as_dict(),
                "early_fallback": True,
                "directional_coverage_ok": False,
                "detector_mode": (cfg.get("detector") or {}).get("mode", "moi3d"),
                "aniso_candidate": sphere_candidate,
                "iso_candidate": dict(sphere_candidate),
                "anisotropy_evidence": {
                    "moi_ok": False,
                    "aniso_candidate_available": True,
                    "standalone_selected": "NOT_IDENTIFIABLE",
                    "standalone_reason": f"spatial_degeneracy={deg}",
                    "anisotropy_identifiability": "NOT_IDENTIFIABLE",
                    "technical_sphere_for_downstream": True,
                    "geological_isotropy_claimed": False,
                },
            },
            directional_coverage_ok=False,
        )
        for name, vec in (
            ("major", result.major_axis_xyz),
            ("intermediate", result.intermediate_axis_xyz),
            ("minor", result.minor_axis_xyz),
        ):
            az, dip = vector_to_azimuth_dip(vec)
            setattr(result, f"{name}_azimuth_deg", az)
            setattr(result, f"{name}_dip_deg", dip)
        write_report(
            out,
            result=result,
            estimates=estimates,
            omni=omni,
            axis_fits=None,
            cfg=cfg,
            moi_samples=None,
        )
        print(
            f"[anisotropy3d] early sphere fallback ({deg}); "
            f"aniso_candidate=isotropic_sphere → {out}",
            flush=True,
        )
        return result

    mode = str((cfg.get("detector") or {}).get("mode", "moi3d")).strip().lower()
    if mode == "legacy_range_ellipsoid":
        det = _run_legacy_range_ellipsoid(
            pairs=pairs, omni=omni, max_dist=max_dist, cfg=cfg
        )
    elif mode == "moi3d":
        det = _run_moi3d(pairs=pairs, omni=omni, max_dist=max_dist, cfg=cfg)
    else:
        raise ValueError(
            f"unknown detector.mode={mode!r}; use moi3d|legacy_range_ellipsoid"
        )

    ranges = det["ranges"]
    axes = det["axes"]
    nugget, sill = det["nugget"], det["sill"]
    estimates = det["estimates"]
    axis_fits = det["axis_fits"]
    ell_rel = det["ell_rel"]
    ell_reason = det["ell_reason"]

    axis_statuses = det["extra"].get("axis_range_status")
    ranges_reliable = det["extra"].get("ranges_reliable")
    if mode == "legacy_range_ellipsoid":
        ranges_reliable = True  # legacy uses its own coverage gates
        axis_statuses = None

    typ, class_meta = classify_anisotropy(
        float(ranges[0]),
        float(ranges[1]),
        float(ranges[2]),
        cfg,
        axis_statuses=axis_statuses,
        ranges_reliable=ranges_reliable if mode == "moi3d" else True,
    )
    if ell_reason:
        class_meta["notes"] = list(class_meta.get("notes") or []) + [str(ell_reason)]

    ident = det["extra"].get("anisotropy_identifiability")
    if mode == "moi3d" and ident == "ORIENTATION_ONLY":
        typ = "UNCLASSIFIED"
        class_meta["notes"] = list(class_meta.get("notes") or []) + [
            "orientation_only_skip_ratio_classification"
        ]
    elif mode == "moi3d" and ident == "PARTIAL_RANGES":
        typ = "UNCLASSIFIED"
        class_meta["notes"] = list(class_meta.get("notes") or []) + [
            "partial_ranges_skip_ratio_classification"
        ]
    elif mode == "moi3d" and ident == "NOT_IDENTIFIABLE":
        typ = "ISOTROPIC"
        class_meta["notes"] = list(class_meta.get("notes") or []) + [
            "not_identifiable_technical_sphere_reporting_type"
        ]
    elif mode == "moi3d" and det["extra"].get("moi_force_isotropic"):
        # Reliable equal/weak ranges or weak MOI → report ISO, not random AXIAL noise
        typ = "ISOTROPIC"
        class_meta["notes"] = list(class_meta.get("notes") or []) + [
            "moi_acceptance_forced_isotropic"
        ]

    # Trend diagnostics (engineering heuristics; no auto-detrend)
    trend_notes: list[str] = []
    tcfg = cfg.get("trend_diagnostics") or {}
    if bool(tcfg.get("enabled", True)):
        corr_warn = float(tcfg.get("coord_corr_warn", 0.35))
        xyz = pts.xyz
        val = pts.values
        for i, name in enumerate(("X", "Y", "Z")):
            if float(np.std(xyz[:, i])) <= 1e-18 or float(np.std(val)) <= 1e-18:
                continue
            r = float(np.corrcoef(xyz[:, i], val)[0, 1])
            if abs(r) >= corr_warn:
                trend_notes.append(
                    f"possible_trend Value~{name} corr={r:.3f} (>={corr_warn})"
                )
        omni_sse = float(omni.sse) if np.isfinite(omni.sse) else float("inf")
        if omni.lag.size >= 4 and omni_sse > 0:
            # crude misfit: late residual sign pattern
            pred = omni.nugget + omni.sill  # total at large h if spherical reached
            late = omni.gamma[-max(1, omni.lag.size // 3) :]
            if float(np.mean(late)) > 1.15 * pred and pred > 0:
                trend_notes.append(
                    "omni_late_gamma_above_total_sill: single spherical may misfit"
                )

    n_dir_for_conf = (
        len(estimates) if mode == "legacy_range_ellipsoid" else max(int(det["n_usable"]), 1)
    )

    keep_oriented = ident in ("ORIENTATION_ONLY", "PARTIAL_RANGES")
    force_tech = bool(
        mode == "moi3d"
        and ident == "NOT_IDENTIFIABLE"
    ) or bool(
        mode == "moi3d"
        and det["extra"].get("moi_force_isotropic")
        and not keep_oriented
    )

    result = decide_confidence_and_fallback(
        aniso_type=typ,
        ranges=ranges,
        axes=axes,
        nugget=nugget,
        sill=sill,
        n_points=int(pts.diagnostics["n_points"]),
        n_directions=max(n_dir_for_conf, 1),
        n_valid_directions=int(det["n_usable"]),
        ellipsoid_rmse_rel=ell_rel,
        spatial_degeneracy=pts.diagnostics.get("spatial_degeneracy"),
        class_notes=list(class_meta.get("notes") or []) + trend_notes,
        cfg=cfg,
        omni_range=float(omni.range_),
        variogram_model=str((cfg.get("variogram") or {}).get("model", "spherical")),
        directional_coverage_ok=bool(det["cov_ok"]),
        anisotropy_identifiability=ident if mode == "moi3d" else None,
        geological_isotropy_claimed=(
            typ == "ISOTROPIC" and ident == "IDENTIFIED"
            if mode == "moi3d"
            else None
        ),
        axis_fits=axis_fits,
        force_technical_sphere=force_tech,
        extra_diagnostics={
            **pts.diagnostics,
            **pair_diag,
            **det["cov_diag"],
            **det["extra"],
            "omni": {
                **omni.as_dict(),
                "partial_sill": float(omni.sill),
                "total_sill": float(omni.nugget) + float(omni.sill),
            },
            "classification": typ,
            "trend_warnings": trend_notes,
            "ratios_pre_fallback": {
                "major_intermediate": class_meta["major_intermediate_ratio"],
                "major_minor": class_meta["major_minor_ratio"],
                "intermediate_minor": class_meta["intermediate_minor_ratio"],
            },
            "moi_eigenvalues": det["extra"].get("moi_eigenvalues"),
            "moi_strength": det["extra"].get("moi_strength"),
            "moi_axes": det["extra"].get("moi_axes"),
            "n_moi_samples": det["extra"].get("n_moi_samples"),
            "moi_ok": det["extra"].get("moi_ok", True),
            "orientation_usable": det["extra"].get("orientation_usable"),
        },
    )

    # Ensure aniso_candidate always present for pipeline3d gate
    if "aniso_candidate" not in result.diagnostics or not isinstance(
        result.diagnostics.get("aniso_candidate"), dict
    ):
        result.diagnostics["aniso_candidate"] = {
            "variogram_model": result.variogram_model,
            "nugget": result.nugget,
            "sill": result.sill,
            "partial_sill": result.sill,
            "total_sill": result.nugget + result.sill,
            "range_major": result.range_major,
            "range_intermediate": result.range_intermediate,
            "range_minor": result.range_minor,
            "orientation_matrix": result.orientation_matrix.tolist(),
            "major_axis_xyz": result.major_axis_xyz.tolist(),
            "intermediate_axis_xyz": result.intermediate_axis_xyz.tolist(),
            "minor_axis_xyz": result.minor_axis_xyz.tolist(),
            "technical_fallback": True,
            "anisotropy_identifiability": result.anisotropy_identifiability,
        }

    for name, vec in (
        ("major", result.major_axis_xyz),
        ("intermediate", result.intermediate_axis_xyz),
        ("minor", result.minor_axis_xyz),
    ):
        az, dip = vector_to_azimuth_dip(vec)
        setattr(result, f"{name}_azimuth_deg", az)
        setattr(result, f"{name}_dip_deg", dip)

    write_report(
        out,
        result=result,
        estimates=estimates,
        omni=omni,
        axis_fits=axis_fits,
        cfg=cfg,
        moi_samples=det.get("moi_samples"),
    )
    print(
        f"[anisotropy3d] type={result.type} confidence={result.confidence} "
        f"fallback={result.fallback_reason!r} → {out}",
        flush=True,
    )
    return result
