"""Principal-axis directional diagnostics (non-destructive w.r.t. ellipsoid)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .ellipsoid import stabilize_axis
from .models import PairCloud3D, VariogramFit
from .variogram import experimental_directional, spherical_model


def refine_principal_axes(
    pairs: PairCloud3D,
    axes: np.ndarray,
    ranges: np.ndarray,
    *,
    omni_nugget: float,
    omni_sill: float,
    max_dist: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Directional VG on maj/int/min for diagnostics/plots only.

    Does **not** overwrite ellipsoid axes/ranges. Nugget/sill stay at omni.
    Joint range / local orientation optimization is disabled for now.
    """
    rcfg = cfg.get("refinement") or {}
    n_lags = int(rcfg.get("n_lags", 16))
    tol = float(rcfg.get("angular_tolerance_deg", 18.0))
    bw_raw = rcfg.get("bandwidth", None)
    bandwidth = float(bw_raw) if bw_raw is not None else None
    min_pairs_lag = int((cfg.get("variogram") or {}).get("min_pairs_per_lag", 8))

    axes_out = np.asarray(axes, dtype=float).copy()
    for j in range(3):
        axes_out[:, j] = stabilize_axis(axes_out[:, j])
    axes_out[:, 2] = stabilize_axis(np.cross(axes_out[:, 0], axes_out[:, 1]))
    ranges_out = np.asarray(ranges, dtype=float).copy()
    nug = float(omni_nugget)
    sil = float(omni_sill)

    axis_fits: list[VariogramFit] = []
    joint_sse = 0.0
    for j in range(3):
        lag, gamma, counts, _ = experimental_directional(
            pairs,
            axes_out[:, j],
            angular_tolerance_deg=tol,
            bandwidth=bandwidth,
            n_lags=n_lags,
            max_dist=max_dist,
            min_pairs=min_pairs_lag,
        )
        pred = (
            spherical_model(lag, nug, sil, float(ranges_out[j]))
            if lag.size
            else np.array([])
        )
        w = np.maximum(counts.astype(float), 1.0) if counts.size else np.array([])
        sse = (
            float(np.sum(w * (gamma - pred) ** 2)) if lag.size else float("inf")
        )
        if np.isfinite(sse):
            joint_sse += sse
        axis_fits.append(
            VariogramFit(
                lag=lag,
                gamma=gamma,
                n_pairs=counts,
                nugget=nug,
                sill=sil,
                range_=float(ranges_out[j]),
                sse=sse,
                n_valid_lags=int(lag.size),
            )
        )

    return {
        "axes": axes_out,
        "ranges": ranges_out,
        "nugget": nug,
        "sill": sil,
        "axis_fits": axis_fits,
        "joint_sse": float(joint_sse),
    }
