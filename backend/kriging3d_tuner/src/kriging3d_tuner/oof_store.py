"""Out-of-fold prediction store for single- and multi-pass OK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kriging import select_neighbour_indices
from .models import PointCloud, SearchParams, VariogramFixed


FAIL_EMPTY_NEIGHBOURHOOD = "empty_neighbourhood"
FAIL_BELOW_NMIN = "n_below_nmin"
FAIL_OCTANT_MINOCT = "octant_minoct"
FAIL_NMAX_CONFLICT = "nmax_minperoc_conflict"
FAIL_OTHER = "other"


@dataclass
class OOFStore:
    """Per-point OOF predictions; later passes never overwrite earlier ones."""

    n_tgt: int
    y_true: np.ndarray  # (n_points,) full cloud values; only OOF indices used
    pred: np.ndarray = field(init=False)
    pass_id: np.ndarray = field(init=False)  # 0 = unset, else 1..K
    n_used: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = int(self.y_true.shape[0])
        self.pred = np.full(n, np.nan, dtype=float)
        self.pass_id = np.zeros(n, dtype=int)
        self.n_used = np.full(n, -1, dtype=int)

    @property
    def estimated_mask(self) -> np.ndarray:
        return np.isfinite(self.pred) & (self.pass_id > 0)

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.estimated_mask))

    @property
    def n_invalid(self) -> int:
        return int(self.n_tgt) - self.n_valid

    @property
    def coverage(self) -> float:
        if self.n_tgt <= 0:
            return 0.0
        return float(self.n_valid / self.n_tgt)

    def candidate_mask(self, oof_indices: np.ndarray) -> np.ndarray:
        """Boolean mask over full cloud: OOF and not yet estimated."""
        m = np.zeros(self.y_true.shape[0], dtype=bool)
        for gi in np.asarray(oof_indices, dtype=int).ravel():
            if self.pass_id[gi] == 0:
                m[gi] = True
        return m

    def merge(
        self,
        indices: np.ndarray,
        preds: np.ndarray,
        n_used: np.ndarray,
        pass_no: int,
    ) -> int:
        """Write finite preds for unset indices only. Returns number newly set."""
        indices = np.asarray(indices, dtype=int).ravel()
        preds = np.asarray(preds, dtype=float).ravel()
        n_used = np.asarray(n_used, dtype=int).ravel()
        n_new = 0
        for gi, p, nu in zip(indices, preds, n_used):
            if self.pass_id[gi] != 0:
                continue
            if not np.isfinite(p):
                continue
            self.pred[gi] = float(p)
            self.pass_id[gi] = int(pass_no)
            self.n_used[gi] = int(nu)
            n_new += 1
        return n_new

    def rmse_mae_all(self, empty_fallback: float = 1.0e6) -> tuple[float, float]:
        m = self.estimated_mask
        if not np.any(m):
            return float(empty_fallback), float(empty_fallback)
        err = self.pred[m] - self.y_true[m]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        return rmse, mae

    def pass_local_metrics(
        self, pass_no: int, empty_fallback: float = 1.0e6
    ) -> dict[str, Any]:
        m = self.pass_id == int(pass_no)
        n_new = int(np.sum(m))
        if n_new == 0:
            return {
                "n_new": 0,
                "pct_new": 0.0,
                "CV_RMSE": float(empty_fallback),
                "CV_MAE": float(empty_fallback),
                "mean_n_used": float("nan"),
            }
        err = self.pred[m] - self.y_true[m]
        return {
            "n_new": n_new,
            "pct_new": 100.0 * n_new / max(self.n_tgt, 1),
            "CV_RMSE": float(np.sqrt(np.mean(err**2))),
            "CV_MAE": float(np.mean(np.abs(err))),
            "mean_n_used": float(np.mean(self.n_used[m])),
        }

    def to_predictions_frame(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for gi in np.where(self.estimated_mask)[0]:
            rows.append(
                {
                    "global_idx": int(gi),
                    "y_true": float(self.y_true[gi]),
                    "y_pred": float(self.pred[gi]),
                    "pass_id": int(self.pass_id[gi]),
                    "n_used": int(self.n_used[gi]),
                }
            )
        return rows


def classify_neighbour_failure(
    delta_xyz: np.ndarray,
    orientation: np.ndarray,
    search: SearchParams,
) -> str:
    """Best-effort reason when OK would return NaN for this target."""
    idx = select_neighbour_indices(delta_xyz, orientation, search)
    if idx.size == 0:
        if search.octant_search:
            # Distinguish MINOCT vs empty ellipsoid vs Nmax conflict:
            # empty ellipsoid → select returns empty before octant logic with no inside
            from .geometry import search_distance, to_principal

            xp = to_principal(delta_xyz, orientation)
            d = search_distance(xp, search.r_major, search.r_inter, search.r_minor)
            inside = np.where(d <= 1.0 + 1e-12)[0]
            if inside.size == 0:
                return FAIL_EMPTY_NEIGHBOURHOOD
            # Re-run without Nmax trim detection: if octants alone yield pool but
            # Nmax conflict empties — probe with huge n_max
            probe = SearchParams(
                r_major=search.r_major,
                k_inter=search.k_inter,
                k_minor=search.k_minor,
                n_min=search.n_min,
                n_max=10**9,
                octant_search=True,
                minoct=search.minoct,
                minperoc=search.minperoc,
                maxperoc=search.maxperoc,
                n_sectors=int(search.n_sectors),
                discard_underfilled=bool(search.discard_underfilled),
            )
            idx2 = select_neighbour_indices(delta_xyz, orientation, probe)
            if idx2.size == 0:
                return FAIL_OCTANT_MINOCT
            return FAIL_NMAX_CONFLICT
        return FAIL_EMPTY_NEIGHBOURHOOD
    if idx.size < search.n_min:
        return FAIL_BELOW_NMIN
    return FAIL_OTHER


def diagnose_residuals(
    cloud: PointCloud,
    plan,
    vg: VariogramFixed,
    store: OOFStore,
    search: SearchParams,
) -> dict[str, int]:
    """Count failure reasons for still-unevaluated OOF points under ``search``."""
    counts: dict[str, int] = {}
    Q = vg.orientation_matrix
    for fold in plan.folds:
        tr = fold.train_idx
        known = cloud.xyz[tr]
        for gi in fold.test_idx.tolist():
            if store.pass_id[gi] != 0:
                continue
            reason = classify_neighbour_failure(
                known - cloud.xyz[gi], Q, search
            )
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def risky_extrapolation(r_major: float, range_major: float, factor: float = 2.0) -> bool:
    return float(r_major) > float(factor) * float(range_major)


def lexico_coverage_rmse_objective(
    coverage: float,
    rmse_all: float,
    *,
    coverage_scale: float = 1.0e6,
) -> float:
    """Minimize: prefer higher coverage, then lower RMSE (pass 3)."""
    return float(-float(coverage) * float(coverage_scale) + float(rmse_all))
