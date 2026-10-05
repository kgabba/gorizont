"""JSON / CSV / PNG diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .models import Anisotropy3DResult, DirectionalEstimate, VariogramFit
from .variogram import spherical_model


def vector_to_azimuth_dip(v: np.ndarray) -> tuple[float, float]:
    """Reporting convention: az 0=+X CCW to +Y; dip 0=horizontal, 90=+Z."""
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-18)
    # undirected: prefer +z
    if v[2] < 0:
        v = -v
    horiz = np.hypot(v[0], v[1])
    dip = float(np.degrees(np.arctan2(v[2], horiz)))
    az = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    return az, dip


def write_report(
    out_dir: Path,
    *,
    result: Anisotropy3DResult,
    estimates: Sequence[DirectionalEstimate],
    omni: VariogramFit,
    axis_fits: Sequence[VariogramFit] | None,
    cfg: dict[str, Any],
    moi_samples: Sequence[dict[str, Any]] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "result.json").write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = [e.as_dict() for e in estimates]
    pd.DataFrame(rows).to_csv(out_dir / "directional_scan.csv", index=False)

    if moi_samples is not None:
        pd.DataFrame(list(moi_samples)).to_csv(
            out_dir / "moi_radial_samples.csv", index=False
        )

    if not bool((cfg.get("output") or {}).get("write_png", True)):
        return

    _plot_omni(out_dir / "omnidirectional_variogram.png", omni)
    if estimates:
        _plot_range_sphere(out_dir / "directional_range_sphere.png", estimates, result)
    _plot_axes(out_dir / "principal_axes.png", result)
    names = ("major_variogram.png", "intermediate_variogram.png", "minor_variogram.png")
    labels = ("major", "intermediate", "minor")
    if axis_fits and len(axis_fits) == 3:
        for fit, name, lab in zip(axis_fits, names, labels):
            _plot_omni(out_dir / name, fit, title=f"{lab} directional VG")


def _plot_omni(path: Path, fit: VariogramFit, title: str = "Omnidirectional variogram") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if fit.lag.size:
        ax.scatter(fit.lag, fit.gamma, c="C0", s=40, zorder=3, label="experimental")
        h = np.linspace(0, float(np.max(fit.lag)) * 1.05, 200)
        ax.plot(
            h,
            spherical_model(h, fit.nugget, fit.sill, fit.range_),
            "C1-",
            lw=2,
            label="spherical",
        )
        ax.axhline(fit.nugget + fit.sill, color="gray", ls="--", lw=1, label="sill")
        ax.legend(fontsize=8)
    ax.set_xlabel("h")
    ax.set_ylabel("γ(h)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_range_sphere(
    path: Path,
    estimates: Sequence[DirectionalEstimate],
    result: Anisotropy3DResult,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    valid = [e for e in estimates if e.valid and np.isfinite(e.range_)]
    if valid:
        U = np.stack([e.u for e in valid])
        R = np.array([e.range_ for e in valid])
        # show both hemispheres for undirected axes
        pts = np.vstack([U * R[:, None], -U * R[:, None]])
        rr = np.concatenate([R, R])
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=rr, cmap="viridis", s=20, alpha=0.8
        )
        fig.colorbar(sc, ax=ax, shrink=0.7, label="range")
    # principal axes
    colors = ("r", "g", "b")
    labels = ("maj", "int", "min")
    lengths = (
        result.range_major,
        result.range_intermediate,
        result.range_minor,
    )
    axes = (
        result.major_axis_xyz,
        result.intermediate_axis_xyz,
        result.minor_axis_xyz,
    )
    for v, L, c, lab in zip(axes, lengths, colors, labels):
        p = v * L
        ax.plot(
            [-p[0], p[0]],
            [-p[1], p[1]],
            [-p[2], p[2]],
            color=c,
            lw=2.5,
            label=lab,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Directional ranges + principal axes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_axes(path: Path, result: Anisotropy3DResult) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = ("r", "g", "b")
    labels = ("major", "intermediate", "minor")
    lengths = (
        result.range_major,
        result.range_intermediate,
        result.range_minor,
    )
    axes = (
        result.major_axis_xyz,
        result.intermediate_axis_xyz,
        result.minor_axis_xyz,
    )
    for v, L, c, lab in zip(axes, lengths, colors, labels):
        p = v * L
        ax.plot([-p[0], p[0]], [-p[1], p[1]], [-p[2], p[2]], color=c, lw=3, label=lab)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Principal axes ({result.type})")
    ax.legend()
    # equal aspect
    all_pts = np.vstack([*(np.vstack([-v * L, v * L]) for v, L in zip(axes, lengths))])
    if all_pts.size:
        c = all_pts.mean(axis=0)
        r = np.max(np.linalg.norm(all_pts - c, axis=1)) + 1e-9
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
