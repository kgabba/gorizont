"""3D MOI via coarse azimuth × dip grid search on directional range."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .directional import fit_directional_variogram
from .geometry import az_dip_from_direction, ellipsoid_axes, order_ranges_descending
from .pairs import PairCloud3D


@dataclass(frozen=True)
class MOIResult3D:
    azimuth_deg: float
    dip_deg: float
    a_major: float
    a_inter: float
    a_minor: float
    nugget_fit: float
    sill_partial: float
    score_grid: list[dict]


def estimate_moi_and_ranges(
    pairs: PairCloud3D,
    *,
    azimuth_step_deg: float = 15.0,
    dip_step_deg: float = 10.0,
    dip_min: float = -80.0,
    dip_max: float = 80.0,
    tolerance_deg: float = 20.0,
    n_lags: int = 12,
    max_dist_percentile: float = 50.0,
    min_pairs: int = 8,
) -> MOIResult3D:
    """Grid-search major orientation; fit major / inter / minor spherical ranges."""
    dir_kw = dict(
        tolerance_deg=float(tolerance_deg),
        n_lags=int(n_lags),
        max_dist_percentile=float(max_dist_percentile),
        min_pairs=int(min_pairs),
    )

    azimuths = np.arange(0.0, 180.0, float(azimuth_step_deg))
    dips = np.arange(float(dip_min), float(dip_max) + 1e-9, float(dip_step_deg))

    best_range = -1.0
    best_az, best_dip = 0.0, 0.0
    best_nugget, best_sill = 0.0, 1.0
    score_grid: list[dict] = []

    for az in azimuths:
        for dip in dips:
            vg = fit_directional_variogram(pairs, float(az), float(dip), **dir_kw)
            score_grid.append(
                {
                    "azimuth_deg": float(az),
                    "dip_deg": float(dip),
                    "range": float(vg.range_),
                    "n_lag_bins": int(vg.lag.size),
                }
            )
            if vg.lag.size == 0:
                continue
            if vg.range_ > best_range:
                best_range = float(vg.range_)
                best_az, best_dip = float(az), float(dip)
                best_nugget = float(vg.nugget)
                best_sill = float(vg.sill)

    if best_range <= 0:
        # fallback isotropic-ish: use max pairwise distance fraction
        h_max = float(np.percentile(pairs.h, max_dist_percentile)) if pairs.h.size else 1.0
        return MOIResult3D(
            azimuth_deg=0.0,
            dip_deg=0.0,
            a_major=max(h_max, 1e-3),
            a_inter=max(h_max, 1e-3),
            a_minor=max(h_max, 1e-3),
            nugget_fit=0.0,
            sill_partial=max(float(np.mean(pairs.gamma)) if pairs.gamma.size else 1.0, 1e-6),
            score_grid=score_grid,
        )

    # Intermediate: horizontal, azimuth + 90°, dip 0
    inter_az = (best_az + 90.0) % 180.0
    inter_dip = 0.0
    # Minor: normal to major-inter plane (= e3 of ellipsoid)
    _, _, e3 = ellipsoid_axes(best_az, best_dip)
    minor_az, minor_dip = az_dip_from_direction(e3)

    major_vg = fit_directional_variogram(pairs, best_az, best_dip, **dir_kw)
    inter_vg = fit_directional_variogram(pairs, inter_az, inter_dip, **dir_kw)
    minor_vg = fit_directional_variogram(pairs, minor_az, minor_dip, **dir_kw)

    a1, a2, a3 = major_vg.range_, inter_vg.range_, minor_vg.range_
    a_maj, a_int, a_min, az, dip = order_ranges_descending(
        a1, a2, a3, best_az, best_dip
    )

    return MOIResult3D(
        azimuth_deg=az,
        dip_deg=dip,
        a_major=a_maj,
        a_inter=a_int,
        a_minor=a_min,
        nugget_fit=float(max(major_vg.nugget, best_nugget, 0.0)),
        sill_partial=float(max(major_vg.sill, best_sill, 1e-12)),
        score_grid=score_grid,
    )
