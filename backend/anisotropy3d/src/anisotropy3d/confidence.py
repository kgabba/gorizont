"""Confidence scoring with split orientation / range reliability."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .models import Anisotropy3DResult, VariogramFit

SPATIAL_DIM_FALLBACK = "3D anisotropy underdetermined due to spatial dimensionality"
NOT_IDENTIFIABLE = "anisotropy_not_identifiable"


def _rank_conf(level: str) -> int:
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(level, 1)


def _min_conf(a: str, b: str) -> str:
    return a if _rank_conf(a) <= _rank_conf(b) else b


def score_orientation(
    *,
    moi_ok: bool,
    moi_strength: float | None,
    n_moi_samples: int,
    orientation_usable: bool,
    spatial_degeneracy: str | None,
    cfg: dict[str, Any],
    extra_warnings: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Orientation confidence independent of range fits."""
    ccfg = cfg.get("confidence") or {}
    reasons: list[str] = []
    if spatial_degeneracy in ("linear", "planar"):
        return "LOW", [f"spatial_degeneracy={spatial_degeneracy}"]
    if not moi_ok:
        return "LOW", ["moi_failed"]
    min_s = int(ccfg.get("orientation_min_moi_samples", 24))
    min_str = float(ccfg.get("orientation_min_strength", 1.15))
    strength = float(moi_strength) if moi_strength is not None else 1.0
    if n_moi_samples < min_s:
        reasons.append(f"n_moi_samples={n_moi_samples}<{min_s}")
    if strength < min_str:
        reasons.append(f"moi_strength={strength:.4g}<{min_str}")
    if not orientation_usable:
        reasons.append("orientation_not_usable")
        return "LOW", reasons or ["orientation_not_usable"]
    if extra_warnings:
        reasons.extend(list(extra_warnings))
    if strength >= min_str * 1.5 and n_moi_samples >= min_s * 2 and not extra_warnings:
        reasons.append("strong_moi_orientation_support")
        return "HIGH", reasons
    reasons.append("moderate_moi_orientation_support")
    return "MEDIUM", reasons


def score_ranges(
    axis_fits: Sequence[VariogramFit] | None,
    axis_statuses: Sequence[str] | None,
    *,
    cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    """Range confidence from per-axis identifiability only."""
    ccfg = cfg.get("confidence") or {}
    reasons: list[str] = []
    if not axis_fits or len(axis_fits) != 3:
        return "LOW", ["missing_axis_fits"]
    statuses = list(axis_statuses) if axis_statuses is not None else [
        getattr(f, "status", "unknown") for f in axis_fits
    ]
    n_rel = sum(1 for s in statuses if s == "reliable")
    n_unrel = sum(1 for s in statuses if s == "unreliable")
    n_ni = sum(1 for s in statuses if s == "not_identifiable")
    any_boundary = any(bool(getattr(f, "range_boundary_hit", False)) for f in axis_fits)
    reasons.append(
        f"axis_status reliable={n_rel} unreliable={n_unrel} not_identifiable={n_ni}"
    )
    for i, f in enumerate(axis_fits):
        for r in getattr(f, "status_reasons", []) or []:
            reasons.append(f"axis{i}:{r}")
    if any_boundary:
        reasons.append("range_boundary_hit_present")
    require_all = bool(ccfg.get("range_high_requires_all_reliable", True))
    if n_rel == 3 and not any_boundary:
        reasons.append("all_three_ranges_reliable")
        return "HIGH", reasons
    if n_rel == 3 and any_boundary:
        # Should not happen if boundary → not_identifiable, but guard
        return "LOW", reasons
    if n_rel >= 1 and n_ni < 3:
        if require_all:
            reasons.append("not_all_ranges_reliable")
        return "MEDIUM" if n_rel >= 2 and not any_boundary else "LOW", reasons
    return "LOW", reasons or ["no_reliable_ranges"]


def decide_confidence_and_fallback(
    *,
    aniso_type: str,
    ranges: np.ndarray,
    axes: np.ndarray,
    nugget: float,
    sill: float,
    n_points: int,
    n_directions: int,
    n_valid_directions: int,
    ellipsoid_rmse_rel: float | None,
    spatial_degeneracy: str | None,
    class_notes: list[str],
    cfg: dict[str, Any],
    omni_range: float,
    variogram_model: str,
    extra_diagnostics: dict[str, Any],
    directional_coverage_ok: bool = True,
    # New optional inputs (MOI path)
    orientation_confidence: str | None = None,
    range_confidence: str | None = None,
    anisotropy_identifiability: str | None = None,
    geological_isotropy_claimed: bool | None = None,
    axis_fits: Sequence[VariogramFit] | None = None,
    force_technical_sphere: bool = False,
) -> Anisotropy3DResult:
    ccfg = cfg.get("confidence") or {}
    reasons: list[str] = []
    fallback_reason: str | None = None

    nug_frac = float(nugget) / max(float(nugget) + float(sill), 1e-12)
    nug_thr = float(ccfg.get("nugget_frac_high", 0.45))
    min_valid = int(ccfg.get("min_valid_directions", 12))
    min_frac = float(ccfg.get("min_valid_frac", 0.25))
    max_rel = float(ccfg.get("max_ellipsoid_rmse_rel", 0.35))

    if nug_frac >= nug_thr:
        reasons.append(
            f"high_nugget_frac={nug_frac:.3f}>={nug_thr} (noise; not auto-ISO)"
        )

    # --- orientation / range scores ---
    # Prefer explicit detector_mode; moi_strength key may be present-but-None from pipeline.
    use_moi_scores = (
        extra_diagnostics.get("detector_mode") == "moi3d"
        or extra_diagnostics.get("moi_strength") is not None
    )
    if use_moi_scores:
        o_conf, o_reasons = score_orientation(
            moi_ok=bool(
                extra_diagnostics.get(
                    "moi_ok",
                    int(extra_diagnostics.get("n_moi_samples") or 0) > 0,
                )
            ),
            moi_strength=extra_diagnostics.get("moi_strength"),
            n_moi_samples=int(extra_diagnostics.get("n_moi_samples") or 0),
            orientation_usable=bool(
                extra_diagnostics.get("orientation_usable", False)
            ),
            spatial_degeneracy=spatial_degeneracy,
            cfg=cfg,
            extra_warnings=(
                [extra_diagnostics["orientation_warning"]]
                if extra_diagnostics.get("orientation_warning")
                else None
            ),
        )
        statuses = extra_diagnostics.get("axis_range_status")
        r_conf, r_reasons = score_ranges(
            axis_fits,
            statuses,
            cfg=cfg,
        )
    else:
        # Legacy detector path: map directional coverage → both scores
        o_conf, o_reasons = "MEDIUM", ["legacy_detector_orientation_proxy"]
        r_conf, r_reasons = "MEDIUM", ["legacy_detector_range_proxy"]
        if spatial_degeneracy in ("linear", "planar"):
            o_conf = r_conf = "LOW"
            o_reasons = [f"spatial_degeneracy={spatial_degeneracy}"]
            r_reasons = list(o_reasons)
        elif not directional_coverage_ok or n_valid_directions < min_valid:
            o_conf = r_conf = "LOW"
            o_reasons.append("poor_directional_coverage")
            r_reasons.append("poor_directional_coverage")
        elif (
            directional_coverage_ok
            and n_valid_directions >= int(ccfg.get("high_min_valid_directions", 24))
            and (ellipsoid_rmse_rel is None or ellipsoid_rmse_rel <= float(ccfg.get("high_max_ellipsoid_rmse_rel", 0.15)))
        ):
            o_conf = r_conf = "HIGH"

    if orientation_confidence is not None:
        o_conf = orientation_confidence
    if range_confidence is not None:
        r_conf = range_confidence

    reasons.extend(o_reasons)
    reasons.extend(r_reasons)

    # Identifiability
    ident = anisotropy_identifiability
    if ident is None:
        ident = str(
            extra_diagnostics.get("anisotropy_identifiability")
            or "NOT_IDENTIFIABLE"
        )
    if spatial_degeneracy in ("linear", "planar"):
        ident = "NOT_IDENTIFIABLE"
        force_technical_sphere = True
        fallback_reason = SPATIAL_DIM_FALLBACK
        reasons.append(f"spatial_degeneracy={spatial_degeneracy}")

    # Legacy hard gates (still apply for legacy mode)
    if extra_diagnostics.get("detector_mode") == "legacy_range_ellipsoid":
        if n_valid_directions < min_valid:
            force_technical_sphere = True
            reasons.append(f"n_valid_directions={n_valid_directions}<{min_valid}")
            fallback_reason = fallback_reason or "too few valid directional ranges"
            ident = "NOT_IDENTIFIABLE"
        if n_directions > 0 and n_valid_directions / n_directions < min_frac:
            force_technical_sphere = True
            reasons.append("poor directional coverage fraction")
            fallback_reason = fallback_reason or "poor directional sphere coverage"
            ident = "NOT_IDENTIFIABLE"
        if not directional_coverage_ok:
            force_technical_sphere = True
            reasons.append("directional_coverage_ok=False")
            fallback_reason = fallback_reason or NOT_IDENTIFIABLE
            ident = "NOT_IDENTIFIABLE"

    if ellipsoid_rmse_rel is not None and ellipsoid_rmse_rel > max_rel:
        reasons.append(
            f"ellipsoid_rmse_rel={ellipsoid_rmse_rel:.3f}>{max_rel} "
            f"(diagnostic; not auto-ISO)"
        )

    # Aggregate confidence: never HIGH overall if ranges LOW
    conf = _min_conf(o_conf, r_conf)
    if force_technical_sphere and ident == "NOT_IDENTIFIABLE":
        conf = "LOW"
        reasons.append("technical_sphere_fallback")
        fallback_reason = fallback_reason or NOT_IDENTIFIABLE

    if ident == "ORIENTATION_ONLY":
        # Orientation may be MEDIUM/HIGH; ranges LOW → overall at most MEDIUM
        conf = _min_conf(conf, "MEDIUM")
        reasons.append("orientation_only_ranges_not_identifiable")
        fallback_reason = fallback_reason or str(
            extra_diagnostics.get("moi_reason")
            or "omni_ranges_with_moi_orientation"
        )
    elif ident == "PARTIAL_RANGES":
        # Some axes confirmed; never claim full HIGH overall without all three
        conf = _min_conf(conf, "MEDIUM")
        reasons.append("partial_ranges_some_axes_reliable")
        fallback_reason = fallback_reason or str(
            extra_diagnostics.get("moi_reason")
            or "partial_reliable_ranges_omni_fill"
        )

    geo_iso = geological_isotropy_claimed
    if geo_iso is None:
        geo_iso = bool(
            aniso_type == "ISOTROPIC"
            and ident == "IDENTIFIED"
            and not force_technical_sphere
        )

    # Classification ISO with IDENTIFIED reliable equal ranges → geological claim OK
    if aniso_type == "ISOTROPIC" and ident == "IDENTIFIED" and r_conf == "HIGH":
        geo_iso = True
        reasons.append("classification=ISOTROPIC_with_reliable_equal_ranges")
    elif aniso_type == "ISOTROPIC" and ident != "IDENTIFIED":
        geo_iso = False
        reasons.append("type_ISOTROPIC_is_reporting_only_not_geological_claim")

    reasons.extend(class_notes)

    # Apply technical sphere geometry only when forced AND not orientation-preserving
    out_type = aniso_type
    out_ranges = np.asarray(ranges, dtype=float).copy()
    out_axes = np.asarray(axes, dtype=float).copy()
    if force_technical_sphere and ident == "NOT_IDENTIFIABLE":
        r = float(max(omni_range, 1e-12))
        out_ranges = np.array([r, r, r], dtype=float)
        out_axes = np.eye(3)
        out_type = "ISOTROPIC"
        ratios = (1.0, 1.0, 1.0)
        geo_iso = False
    else:
        if ident in ("ORIENTATION_ONLY", "PARTIAL_RANGES"):
            # Keep MOI axes + detector ranges (omni or hybrid); skip ratio class
            out_type = "UNCLASSIFIED"
            geo_iso = False
        a, b, c = float(out_ranges[0]), float(out_ranges[1]), float(out_ranges[2])
        ratios = (a / max(b, 1e-12), a / max(c, 1e-12), b / max(c, 1e-12))
        if not force_technical_sphere and ident == "IDENTIFIED":
            fallback_reason = None

    diag = {
        **extra_diagnostics,
        "n_points": n_points,
        "n_directions": n_directions,
        "n_valid_directions": n_valid_directions,
        "nugget_frac": nug_frac,
        "ellipsoid_rmse_rel": ellipsoid_rmse_rel,
        "spatial_degeneracy": spatial_degeneracy,
        "directional_coverage_ok": bool(directional_coverage_ok),
        "forced_isotropic_sphere": bool(
            force_technical_sphere and ident == "NOT_IDENTIFIABLE"
        ),
        "sphere_is_technical_fallback": bool(
            force_technical_sphere and not geo_iso
        ),
        "cv_methodology_note": (
            "CV_RMSE characterizes neighbourhood tuning with anisotropy "
            "fixed from the full sample; it is not an independent test of "
            "the anisotropy estimation pipeline."
        ),
        "orientation_confidence": o_conf,
        "range_confidence": r_conf,
        "anisotropy_identifiability": ident,
        "geological_isotropy_claimed": bool(geo_iso),
        "partial_sill": float(sill),
        "total_sill": float(nugget) + float(sill),
    }

    return Anisotropy3DResult(
        type=out_type,
        range_major=float(out_ranges[0]),
        range_intermediate=float(out_ranges[1]),
        range_minor=float(out_ranges[2]),
        major_axis_xyz=out_axes[:, 0].copy(),
        intermediate_axis_xyz=out_axes[:, 1].copy(),
        minor_axis_xyz=out_axes[:, 2].copy(),
        orientation_matrix=out_axes.copy(),
        major_intermediate_ratio=float(ratios[0]),
        major_minor_ratio=float(ratios[1]),
        intermediate_minor_ratio=float(ratios[2]),
        nugget=float(nugget),
        sill=float(sill),
        variogram_model=variogram_model,
        confidence=conf,
        orientation_confidence=o_conf,
        range_confidence=r_conf,
        anisotropy_identifiability=ident,
        geological_isotropy_claimed=bool(geo_iso),
        confidence_reasons=reasons,
        fallback_reason=fallback_reason,
        diagnostics=diag,
    )
