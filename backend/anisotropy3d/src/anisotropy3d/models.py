"""Dataclasses for anisotropy3d."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PairCloud3D:
    """Unordered lag pairs in 3D."""

    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    h: np.ndarray
    gamma: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    max_dist: float = 0.0  # single distance cutoff used when building this cloud

    @property
    def n(self) -> int:
        return int(self.h.size)

    def u_stack(self) -> np.ndarray:
        return np.column_stack([self.ux, self.uy, self.uz])


@dataclass
class VariogramFit:
    """Experimental + fitted 1D variogram."""

    lag: np.ndarray
    gamma: np.ndarray
    n_pairs: np.ndarray
    nugget: float
    sill: float  # partial sill
    range_: float
    sse: float
    model: str = "spherical"
    n_valid_lags: int = 0
    range_upper_bound: float | None = None
    range_boundary_hit: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "lag": self.lag.tolist(),
            "gamma": self.gamma.tolist(),
            "n_pairs": self.n_pairs.tolist(),
            "nugget": float(self.nugget),
            "sill": float(self.sill),
            "range": float(self.range_),
            "sse": float(self.sse),
            "model": self.model,
            "n_valid_lags": int(self.n_valid_lags),
            "range_upper_bound": self.range_upper_bound,
            "range_boundary_hit": bool(self.range_boundary_hit),
        }


@dataclass
class DirectionalEstimate:
    """One hemisphere direction after coarse scan."""

    u: np.ndarray  # shape (3,)
    range_: float
    sse: float
    n_pairs: int
    n_valid_lags: int
    valid: bool
    fit: VariogramFit | None = None
    range_boundary_hit: bool = False
    range_upper_bound: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ux": float(self.u[0]),
            "uy": float(self.u[1]),
            "uz": float(self.u[2]),
            "range": float(self.range_),
            "sse": float(self.sse),
            "n_pairs": int(self.n_pairs),
            "n_valid_lags": int(self.n_valid_lags),
            "valid": bool(self.valid),
            "range_boundary_hit": bool(self.range_boundary_hit),
            "range_upper_bound": self.range_upper_bound,
        }


@dataclass
class Anisotropy3DResult:
    """Final anisotropy estimate."""

    type: str  # ISOTROPIC | AXIAL_PROLATE | AXIAL_OBLATE | TRIAXIAL
    range_major: float
    range_intermediate: float
    range_minor: float
    major_axis_xyz: np.ndarray
    intermediate_axis_xyz: np.ndarray
    minor_axis_xyz: np.ndarray
    orientation_matrix: np.ndarray  # columns = maj, int, min
    major_intermediate_ratio: float
    major_minor_ratio: float
    intermediate_minor_ratio: float
    nugget: float
    sill: float
    variogram_model: str
    confidence: str  # HIGH | MEDIUM | LOW
    confidence_reasons: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # reporting angles (degrees); derived from vectors
    major_azimuth_deg: float | None = None
    major_dip_deg: float | None = None
    intermediate_azimuth_deg: float | None = None
    intermediate_dip_deg: float | None = None
    minor_azimuth_deg: float | None = None
    minor_dip_deg: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "type": self.type,
            "range_major": float(self.range_major),
            "range_intermediate": float(self.range_intermediate),
            "range_minor": float(self.range_minor),
            "major_axis_xyz": self.major_axis_xyz.tolist(),
            "intermediate_axis_xyz": self.intermediate_axis_xyz.tolist(),
            "minor_axis_xyz": self.minor_axis_xyz.tolist(),
            "orientation_matrix": self.orientation_matrix.tolist(),
            "major_intermediate_ratio": float(self.major_intermediate_ratio),
            "major_minor_ratio": float(self.major_minor_ratio),
            "intermediate_minor_ratio": float(self.intermediate_minor_ratio),
            "nugget": float(self.nugget),
            "sill": float(self.sill),
            "variogram_model": self.variogram_model,
            "confidence": self.confidence,
            "confidence_reasons": list(self.confidence_reasons),
            "fallback_reason": self.fallback_reason,
            "diagnostics": self.diagnostics,
            "major_azimuth_deg": self.major_azimuth_deg,
            "major_dip_deg": self.major_dip_deg,
            "intermediate_azimuth_deg": self.intermediate_azimuth_deg,
            "intermediate_dip_deg": self.intermediate_dip_deg,
            "minor_azimuth_deg": self.minor_azimuth_deg,
            "minor_dip_deg": self.minor_dip_deg,
        }
        # Dual candidates for later ISO vs ANISO CV selection (MOI path)
        for key in ("iso_candidate", "aniso_candidate", "anisotropy_evidence"):
            if key in self.diagnostics:
                d[key] = self.diagnostics[key]
        return d
