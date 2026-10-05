"""3D unordered pair cloud."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import PairCloud3D


def _cfg(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _index_to_pair(idx: int, n: int) -> tuple[int, int]:
    """Map linear index in [0, n*(n-1)/2) → unordered pair (i, j), i < j."""
    # Row i of upper triangle has (n - i - 1) entries; starts at i*(2n-i-1)//2
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        start = mid * (2 * n - mid - 1) // 2
        if start <= idx:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    start = i * (2 * n - i - 1) // 2
    j = i + 1 + (idx - start)
    return int(i), int(j)


def _sample_pair_indices(
    n: int, n_sample: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique unordered pairs without building full O(n²) index arrays."""
    n_all = n * (n - 1) // 2
    n_sample = min(int(n_sample), int(n_all))
    if n_sample <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    chosen = rng.choice(n_all, size=n_sample, replace=False)
    i_idx = np.empty(n_sample, dtype=int)
    j_idx = np.empty(n_sample, dtype=int)
    for k, idx in enumerate(chosen):
        i_idx[k], j_idx[k] = _index_to_pair(int(idx), n)
    return i_idx, j_idx


def compute_pairs(
    xyz: np.ndarray,
    values: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[PairCloud3D, dict[str, Any]]:
    """Build unordered pairs with a single max_dist filter and optional sampling.

    Returns
    -------
    pairs, meta
        ``meta`` includes total_possible_pairs, used_pairs, pair_sampling_*, max_dist.
    """
    xyz = np.asarray(xyz, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    empty_meta = {
        "total_possible_pairs": 0,
        "used_pairs": 0,
        "pair_sampling_applied": False,
        "pair_sampling_fraction": 1.0,
        "max_dist": 0.0,
    }
    if n < 2:
        empty = np.array([], dtype=float)
        return (
            PairCloud3D(empty, empty, empty, empty, empty, empty, empty, empty),
            empty_meta,
        )

    max_pairs_cfg = _cfg(cfg, "pairs", "max_pairs", default=200_000)
    seed = int(_cfg(cfg, "pairs", "seed", default=42))
    max_dist_cfg = _cfg(cfg, "pairs", "max_dist", default=None)
    pct = float(_cfg(cfg, "pairs", "max_dist_percentile", default=50.0))
    rng = np.random.default_rng(seed)

    total_possible = int(n * (n - 1) // 2)
    sampling_needed = max_pairs_cfg is not None and total_possible > int(max_pairs_cfg)
    max_pairs = int(max_pairs_cfg) if max_pairs_cfg is not None else total_possible

    # --- resolve max_dist once (from config or percentile of a distance sample) ---
    if max_dist_cfg is not None:
        max_dist = float(max_dist_cfg)
    else:
        n_dist_sample = min(total_possible, max(10_000, max_pairs))
        if sampling_needed or total_possible > n_dist_sample:
            i_s, j_s = _sample_pair_indices(n, n_dist_sample, rng)
        else:
            i_s, j_s = np.triu_indices(n, k=1)
        h_s = np.linalg.norm(xyz[j_s] - xyz[i_s], axis=1)
        h_s = h_s[h_s > 1e-15]
        max_dist = float(np.percentile(h_s, pct)) if h_s.size else 0.0

    # --- build pair index sets ---
    if not sampling_needed:
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_sampling_applied = False
    else:
        # Oversample unique pairs, then keep those within max_dist (up to max_pairs)
        oversample = min(total_possible, max(max_pairs * 5, max_pairs))
        i_idx, j_idx = _sample_pair_indices(n, oversample, rng)
        pair_sampling_applied = True

    dxyz = xyz[j_idx] - xyz[i_idx]
    h = np.linalg.norm(dxyz, axis=1)
    nz = h > 1e-15
    dxyz, h, i_idx, j_idx = dxyz[nz], h[nz], i_idx[nz], j_idx[nz]

    keep = h <= max_dist
    dxyz, h, i_idx, j_idx = dxyz[keep], h[keep], i_idx[keep], j_idx[keep]

    if max_pairs_cfg is not None and h.size > max_pairs:
        # Final cap (deterministic via same rng stream)
        sel = rng.choice(h.size, size=max_pairs, replace=False)
        dxyz, h, i_idx, j_idx = dxyz[sel], h[sel], i_idx[sel], j_idx[sel]
        pair_sampling_applied = True

    gamma = 0.5 * (values[j_idx] - values[i_idx]) ** 2
    u = dxyz / h[:, None]
    used = int(h.size)
    frac = float(used) / float(total_possible) if total_possible > 0 else 1.0

    if pair_sampling_applied:
        print(
            f"Pair sampling applied: total_possible_pairs={total_possible}, "
            f"sampled_pairs={used}, sampling_fraction={frac:.6g}",
            flush=True,
        )

    meta = {
        "total_possible_pairs": total_possible,
        "used_pairs": used,
        "pair_sampling_applied": bool(pair_sampling_applied),
        "pair_sampling_fraction": frac,
        "max_dist": float(max_dist),
    }
    pairs = PairCloud3D(
        dx=dxyz[:, 0],
        dy=dxyz[:, 1],
        dz=dxyz[:, 2],
        h=h,
        gamma=gamma,
        ux=u[:, 0],
        uy=u[:, 1],
        uz=u[:, 2],
        max_dist=float(max_dist),
    )
    return pairs, meta
