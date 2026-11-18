"""3D Ordinary Kriging with hard search ellipsoid + fixed variogram ellipsoid."""

from __future__ import annotations

import numpy as np

from .geometry import search_distance, to_principal, variogram_hprime
from .models import SearchParams, VariogramFixed


def spherical_covariance(h_prime: np.ndarray, sill: float) -> np.ndarray:
    """Covariance C(h') with partial sill and range=1 in h'-units."""
    h = np.asarray(h_prime, dtype=float)
    sil = max(float(sill), 1e-18)
    gamma = np.zeros_like(h, dtype=float)
    m = (h > 0) & (h < 1.0)
    hr = h[m]
    gamma[m] = sil * (1.5 * hr - 0.5 * hr**3)
    gamma[h >= 1.0] = sil
    return sil - gamma


def _sector_ids(xp: np.ndarray, n_sectors: int) -> np.ndarray:
    """Map principal-frame offsets to sector ids ``0 .. n_sectors-1``.

    Planes through the estimate origin, parallel to search-ellipsoid axes.
    Half-spaces use ``>= 0`` for the positive side (see OCTANT_NOTES.md).

    * ``4`` — plan quadrants from signs of ``(x', y')`` (Z ignored).
    * ``8`` — Datamine/Micromine octants from signs of ``(x', y', z')``.
    * ``16`` — eight plan wedges (π/4, clockwise from +X / 12 o'clock) ×
      upper/lower by ``z'`` (Micromine SIXTEEN).
    """
    xp = np.asarray(xp, dtype=float)
    if xp.ndim == 1:
        xp = xp.reshape(1, 3)
    n_sec = int(n_sectors)
    if n_sec == 4:
        ox = (xp[:, 0] >= 0.0).astype(np.int64)
        oy = (xp[:, 1] >= 0.0).astype(np.int64)
        return ox + 2 * oy
    if n_sec == 8:
        ox = (xp[:, 0] >= 0.0).astype(np.int64)
        oy = (xp[:, 1] >= 0.0).astype(np.int64)
        oz = (xp[:, 2] >= 0.0).astype(np.int64)
        return ox + 2 * oy + 4 * oz
    if n_sec == 16:
        # Clockwise from +X (12 o'clock): negate atan2 CCW angle.
        ang = np.arctan2(xp[:, 1], xp[:, 0])
        ang_cw = (-ang) % (2.0 * np.pi)
        wedge = np.floor(ang_cw / (0.25 * np.pi)).astype(np.int64)
        wedge = np.clip(wedge, 0, 7)
        lower = (xp[:, 2] < 0.0).astype(np.int64)
        return wedge + 8 * lower
    raise ValueError(f"n_sectors must be 4, 8, or 16; got {n_sec}")


def _octant_ids(xp: np.ndarray) -> np.ndarray:
    """Map principal-frame offsets to octants 0..7 (alias for ``n_sectors=8``)."""
    return _sector_ids(xp, 8)


def _select_sector_neighbours(
    idx: np.ndarray,
    xp_in: np.ndarray,
    d_in: np.ndarray,
    search: SearchParams,
) -> np.ndarray:
    """Sector / octant filter on samples already inside the ellipsoid.

    Applies ``MINOCT`` / ``MINPEROC`` / ``MAXPEROC`` over ``n_sectors``, then a
    global ``Nmax`` trim (furthest first, without dropping a filled sector
    below ``MINPEROC``).

    When ``discard_underfilled`` is True (Micromine IDW), sectors with fewer
    than ``MINPEROC`` samples contribute nothing. When False (Datamine OK
    octants), underfilled sectors still contribute up to ``MAXPEROC``.
    """
    minoct = int(search.minoct)
    minperoc = int(search.minperoc)
    maxperoc = int(search.maxperoc)
    n_max = int(search.n_max)
    n_sectors = int(search.n_sectors)
    discard_underfilled = bool(search.discard_underfilled)

    if minoct < 1 or minperoc < 1 or maxperoc < 1 or n_sectors < 1:
        return np.array([], dtype=int)

    sec_id = _sector_ids(xp_in, n_sectors)
    selected_parts: list[np.ndarray] = []
    filled = 0

    for o in range(n_sectors):
        mask = sec_id == o
        if not np.any(mask):
            continue
        local_idx = idx[mask]
        local_d = d_in[mask]
        # nearest first; stable tie-break on original index
        order = np.lexsort((local_idx, local_d))
        sorted_local = local_idx[order]
        if sorted_local.size >= minperoc:
            filled += 1
            keep_n = min(int(sorted_local.size), maxperoc)
            selected_parts.append(sorted_local[:keep_n])
        elif not discard_underfilled:
            keep_n = min(int(sorted_local.size), maxperoc)
            selected_parts.append(sorted_local[:keep_n])

    if filled < minoct:
        return np.array([], dtype=int)
    if not selected_parts:
        return np.array([], dtype=int)

    pooled = np.concatenate(selected_parts)
    if pooled.size <= n_max:
        return pooled

    pos = {int(g): i for i, g in enumerate(idx.tolist())}
    sel = pooled.astype(int, copy=True)
    sel_d = np.array([d_in[pos[int(g)]] for g in sel], dtype=float)
    sel_sec = np.array([sec_id[pos[int(g)]] for g in sel], dtype=int)

    counts = np.zeros(n_sectors, dtype=int)
    for o in sel_sec:
        counts[int(o)] += 1

    rem_order = np.lexsort((-sel, -sel_d))
    keep = np.ones(sel.size, dtype=bool)
    n_keep = int(sel.size)

    for j in rem_order:
        if n_keep <= n_max:
            break
        if not keep[j]:
            continue
        o = int(sel_sec[j])
        if counts[o] - 1 < minperoc:
            continue
        keep[j] = False
        counts[o] -= 1
        n_keep -= 1

    if n_keep > n_max:
        # Impossible to satisfy Nmax together with MINPEROC protection
        return np.array([], dtype=int)

    return sel[keep]


def _select_octant_neighbours(
    idx: np.ndarray,
    xp_in: np.ndarray,
    d_in: np.ndarray,
    search: SearchParams,
) -> np.ndarray:
    """Backward-compatible alias for ``_select_sector_neighbours``."""
    return _select_sector_neighbours(idx, xp_in, d_in, search)


def select_neighbour_indices(
    delta_xyz: np.ndarray,
    orientation: np.ndarray,
    search: SearchParams,
) -> np.ndarray:
    """Hard search ellipsoid; optional sector / octant limits.

    * ``octant_search=False``: legacy path — nearest ``Nmax`` by ``d_search``.
    * ``octant_search=True``: ``MINOCT`` / ``MINPEROC`` / ``MAXPEROC`` over
      ``n_sectors`` ∈ {4, 8, 16}, then global ``Nmax`` with ``MINPEROC``-
      protected removal (see OCTANT_NOTES.md).

    Returns empty array when sector / ``Nmax`` constraints cannot be satisfied
    (caller treats that as no estimate / NaN). Missing predictions stay NaN —
    they are never replaced with zeros.
    """
    xp = to_principal(delta_xyz, orientation)
    d = search_distance(xp, search.r_major, search.r_inter, search.r_minor)
    inside = d <= 1.0 + 1e-12
    idx = np.where(inside)[0]
    if idx.size == 0:
        return idx

    if not search.octant_search:
        if idx.size > search.n_max:
            d_in = d[idx]
            order = np.lexsort((idx, d_in))
            idx = idx[order[: search.n_max]]
        return idx

    return _select_sector_neighbours(idx, xp[idx], d[idx], search)


def ordinary_kriging_at_points(
    known_xyz: np.ndarray,
    known_z: np.ndarray,
    pred_xyz: np.ndarray,
    vg: VariogramFixed,
    search: SearchParams,
) -> np.ndarray:
    """Predict at each row of pred_xyz; NaN if neighbourhood < Nmin."""
    known_xyz = np.asarray(known_xyz, dtype=float)
    known_z = np.asarray(known_z, dtype=float).reshape(-1)
    pred_xyz = np.asarray(pred_xyz, dtype=float)
    Q = vg.orientation_matrix
    n_known = known_xyz.shape[0]
    m = pred_xyz.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if n_known == 0:
        return np.full(m, np.nan)

    preds = np.full(m, np.nan, dtype=float)
    nug = max(float(vg.nugget), 0.0)
    sil = max(float(vg.sill), 1e-18)
    a_maj = float(vg.range_major)
    a_int = float(vg.range_intermediate)
    a_min = float(vg.range_minor)

    for i in range(m):
        delta = known_xyz - pred_xyz[i]
        idx = select_neighbour_indices(delta, Q, search)
        if idx.size < search.n_min:
            continue

        nb = known_xyz[idx]
        zz = known_z[idx]
        n = idx.size

        delta_nn = nb[:, None, :] - nb[None, :, :]
        xp_nn = to_principal(delta_nn.reshape(-1, 3), Q).reshape(n, n, 3)
        h_nn = variogram_hprime(
            xp_nn.reshape(-1, 3), a_maj, a_int, a_min
        ).reshape(n, n)

        delta_n0 = nb - pred_xyz[i]
        xp_n0 = to_principal(delta_n0, Q)
        h_n0 = variogram_hprime(xp_n0, a_maj, a_int, a_min)

        K = spherical_covariance(h_nn, sil)
        np.fill_diagonal(K, sil + nug)
        k0 = spherical_covariance(h_n0, sil)

        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = K
        A[n, n] = 0.0
        b = np.ones(n + 1, dtype=float)
        b[:n] = k0

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        preds[i] = float(np.dot(w[:n], zz))

    return preds
