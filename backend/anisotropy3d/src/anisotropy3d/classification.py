"""ISO / AXIAL / TRIAXIAL classification."""

from __future__ import annotations

from typing import Any


def classify_anisotropy(
    range_major: float,
    range_intermediate: float,
    range_minor: float,
    cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    ccfg = cfg.get("classification") or {}
    iso_max = float(ccfg.get("ratio_iso_max", 1.3))
    eq_max = float(ccfg.get("ratio_equal_max", 1.3))

    a = max(float(range_major), 1e-12)
    b = max(float(range_intermediate), 1e-12)
    c = max(float(range_minor), 1e-12)
    r_mi = a / b
    r_mn = a / c
    r_in = b / c

    notes: list[str] = []
    if r_mi < eq_max:
        notes.append("maj≈int plane: orientation within plane ill-defined")
    if r_in < eq_max:
        notes.append("int≈min plane: orientation within plane ill-defined")
    if r_mn < eq_max:
        notes.append("maj≈min: nearly isotropic")

    if r_mn < iso_max:
        typ = "ISOTROPIC"
    elif r_mi >= eq_max and r_in < eq_max:
        typ = "AXIAL_PROLATE"
    elif r_mi < eq_max and r_in >= eq_max:
        typ = "AXIAL_OBLATE"
    else:
        typ = "TRIAXIAL"

    return typ, {
        "major_intermediate_ratio": r_mi,
        "major_minor_ratio": r_mn,
        "intermediate_minor_ratio": r_in,
        "notes": notes,
    }
