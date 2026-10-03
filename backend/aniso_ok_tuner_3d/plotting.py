"""Variogram figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .directional import DirectionalVariogram3D, spherical_model


def save_directional_variogram(
    vg: DirectionalVariogram3D,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if vg.lag.size:
        ax.scatter(vg.lag, vg.gamma, c="C0", s=28, label="experimental", zorder=3)
        h_line = np.linspace(0, float(vg.lag.max()) * 1.05, 80)
        ax.plot(
            h_line,
            spherical_model(h_line, vg.nugget, vg.sill, vg.range_),
            "C1-",
            label=(
                f"sph a={vg.range_:.3g} C0={vg.nugget:.3g} C={vg.sill:.3g}"
            ),
        )
    ax.set_xlabel("lag")
    ax.set_ylabel("γ")
    ax.set_title(
        f"{title}  az={vg.azimuth_deg:.1f}° dip={vg.dip_deg:.1f}°"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
