"""Confidence scoring and isotropic fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import Anisotropy3DResult

# Canonical reason when 3D structure is underdetermined by geometry
SPATIAL_DIM_FALLBACK = "3D anisotropy underdetermined due to spatial dimensionality"


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
) -> Anisotropy3DResult:
    ccfg = cfg.get("confidence") or {}
    reasons: list[str] = []
    fallback_reason: str | None = None
    force_iso = False

    nug_frac = float(nugget) / max(float(nugget) + float(sill), 1e-12)
    nug_thr = float(ccfg.get("nugget_frac_high", 0.45))
    min_valid = int(ccfg.get("min_valid_directions", 12))
    min_frac = float(ccfg.get("min_valid_frac", 0.25))
    max_rel = float(ccfg.get("max_ellipsoid_rmse_rel", 0.35))

    # High nugget: degrade confidence messaging only — NOT automatic ISO fallback
    if nug_frac >= nug_thr:
        reasons.append(
            f"high_nugget_frac={nug_frac:.3f}>={nug_thr} (noise; not auto-ISO)"
        )

    if spatial_degeneracy in ("linear", "planar"):
        force_iso = True
        reasons.append(f"spatial_degeneracy={spatial_degeneracy}")
        fallback_reason = SPATIAL_DIM_FALLBACK

    if n_valid_directions < min_valid:
        force_iso = True
        reasons.append(f"n_valid_directions={n_valid_directions}<{min_valid}")
        fallback_reason = fallback_reason or "too few valid directional ranges"

    if n_directions > 0 and n_valid_directions / n_directions < min_frac:
        force_iso = True
        reasons.append("poor directional coverage fraction")
        fallback_reason = fallback_reason or "poor directional sphere coverage"

    if not directional_coverage_ok:
        force_iso = True
        reasons.append("directional_coverage_ok=False (directions not spanning 3D)")
        fallback_reason = (
            fallback_reason
            or "valid directions do not span 3D; SPD ellipsoid underdetermined"
        )

    # ellipsoid_rmse_rel: diagnostic / confidence only — NOT hard ISO fallback
    if ellipsoid_rmse_rel is not None and ellipsoid_rmse_rel > max_rel:
        reasons.append(
            f"ellipsoid_rmse_rel={ellipsoid_rmse_rel:.3f}>{max_rel} "
            f"(diagnostic; not auto-ISO)"
        )

    if aniso_type == "ISOTROPIC":
        force_iso = True
        reasons.append("classification=ISOTROPIC")
        fallback_reason = fallback_reason or "anisotropy not expressed (ratios ~ equal)"

    # confidence level
    high_min = int(ccfg.get("high_min_valid_directions", 24))
    high_rel = float(ccfg.get("high_max_ellipsoid_rmse_rel", 0.15))
    conf = "MEDIUM"
    if force_iso and fallback_reason:
        conf = "LOW"
        reasons.append("fallback_to_safe_sphere")
    elif (
        not force_iso
        and directional_coverage_ok
        and n_valid_directions >= high_min
        and (ellipsoid_rmse_rel is None or ellipsoid_rmse_rel <= high_rel)
        and nug_frac < nug_thr * 0.7
    ):
        conf = "HIGH"
        reasons.append("strong directional support and tight ellipsoid fit")
    else:
        reasons.append("partial support or moderate residuals")
        if nug_frac >= nug_thr and not force_iso:
            # keep MEDIUM at most when noisy but anisotropy still estimated
            conf = "MEDIUM"

    reasons.extend(class_notes)

    if force_iso:
        r = float(max(omni_range, 1e-12))
        ranges = np.array([r, r, r], dtype=float)
        axes = np.eye(3)
        aniso_type = "ISOTROPIC"
        ratios = (1.0, 1.0, 1.0)
    else:
        # Keep classification-driven ISO reason only when we actually fell back
        # for weak anisotropy; clear unrelated fallback if we did not force.
        fallback_reason = None
        a, b, c = float(ranges[0]), float(ranges[1]), float(ranges[2])
        ratios = (a / max(b, 1e-12), a / max(c, 1e-12), b / max(c, 1e-12))

    diag = {
        **extra_diagnostics,
        "n_points": n_points,
        "n_directions": n_directions,
        "n_valid_directions": n_valid_directions,
        "nugget_frac": nug_frac,
        "ellipsoid_rmse_rel": ellipsoid_rmse_rel,
        "spatial_degeneracy": spatial_degeneracy,
        "directional_coverage_ok": bool(directional_coverage_ok),
        "forced_isotropic_sphere": force_iso,
        # technical sphere ≠ claim of physical isotropy when dim-limited
        "sphere_is_technical_fallback": bool(
            force_iso and fallback_reason == SPATIAL_DIM_FALLBACK
        ),
    }

    return Anisotropy3DResult(
        type=aniso_type,
        range_major=float(ranges[0]),
        range_intermediate=float(ranges[1]),
        range_minor=float(ranges[2]),
        major_axis_xyz=axes[:, 0].copy(),
        intermediate_axis_xyz=axes[:, 1].copy(),
        minor_axis_xyz=axes[:, 2].copy(),
        orientation_matrix=axes.copy(),
        major_intermediate_ratio=float(ratios[0]),
        major_minor_ratio=float(ratios[1]),
        intermediate_minor_ratio=float(ratios[2]),
        nugget=float(nugget),
        sill=float(sill),
        variogram_model=variogram_model,
        confidence=conf,
        confidence_reasons=reasons,
        fallback_reason=fallback_reason,
        diagnostics=diag,
    )
