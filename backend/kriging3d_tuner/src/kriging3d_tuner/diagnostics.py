"""OOF quality diagnostics: scatter, QQ, correlation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def oof_quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Extra metrics beyond RMSE/MAE for estimated points only."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return {
            "n": 0,
            "R2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "bias_mean_error": float("nan"),
            "CV_RMSE": float("nan"),
            "CV_MAE": float("nan"),
        }
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    if yt.size >= 2 and float(np.std(yt)) > 0 and float(np.std(yp)) > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
        rt = yt.argsort().argsort().astype(float)
        rp = yp.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rt, rp)[0, 1])
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {
        "n": int(yt.size),
        "R2": r2,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "bias_mean_error": float(np.mean(err)),
        "CV_RMSE": float(np.sqrt(np.mean(err**2))),
        "CV_MAE": float(np.mean(np.abs(err))),
    }


def write_oof_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str | Path,
    *,
    pass_id: np.ndarray | None = None,
    title_prefix: str = "OOF",
) -> dict[str, Any]:
    """Write scatter.png, qq*.png and quality_metrics.json under ``out_dir``.

    ``y_true``, ``y_pred`` and optional ``pass_id`` must be the same length
    (typically rows from ``oof_predictions.csv`` — already estimated only).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    pid = None
    if pass_id is not None:
        pid_arr = np.asarray(pass_id, dtype=int).ravel()
        if pid_arr.size == m.size:
            pid = pid_arr[m]
        elif pid_arr.size == yt.size:
            pid = pid_arr

    metrics = oof_quality_metrics(yt, yp)

    # --- scatter (obs vs pred) ---
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    if pid is not None and pid.size == yt.size:
        for pno, color in ((1, "#1f77b4"), (2, "#ff7f0e"), (3, "#2ca02c")):
            sel = pid == pno
            if np.any(sel):
                ax.scatter(
                    yt[sel],
                    yp[sel],
                    s=14,
                    alpha=0.65,
                    c=color,
                    label=f"pass {pno} (n={int(sel.sum())})",
                    edgecolors="none",
                )
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.scatter(yt, yp, s=14, alpha=0.65, c="#1f77b4", edgecolors="none")

    if yt.size:
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted (OOF)")
    ax.set_title(
        f"{title_prefix} scatter  |  R²={metrics['R2']:.3f}  r={metrics['pearson_r']:.3f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    scatter_path = out / "scatter.png"
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)

    # --- QQ residuals vs normal ---
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if yt.size >= 2:
        resid = np.sort(yp - yt)
        n = resid.size
        ppos = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        theo = norm.ppf(ppos)
        ax.scatter(theo, resid, s=14, alpha=0.65, c="#1f77b4", edgecolors="none")
        q_t = np.quantile(theo, [0.25, 0.75])
        q_r = np.quantile(resid, [0.25, 0.75])
        slope = (q_r[1] - q_r[0]) / (q_t[1] - q_t[0] + 1e-12)
        intercept = q_r[0] - slope * q_t[0]
        xline = np.array([float(theo.min()), float(theo.max())])
        ax.plot(xline, intercept + slope * xline, "k--", lw=1)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Residual quantiles (pred − obs)")
    ax.set_title(
        f"{title_prefix} QQ residuals  |  bias={metrics['bias_mean_error']:.3g}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    qq_path = out / "qq.png"
    fig.savefig(qq_path, dpi=140)
    plt.close(fig)

    # --- QQ observed vs predicted quantiles ---
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if yt.size >= 2:
        qs = np.linspace(0.01, 0.99, min(99, yt.size))
        ax.scatter(
            np.quantile(yt, qs),
            np.quantile(yp, qs),
            s=14,
            alpha=0.75,
            c="#2ca02c",
            edgecolors="none",
        )
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Observed quantiles")
    ax.set_ylabel("Predicted quantiles")
    ax.set_title(f"{title_prefix} QQ obs vs pred")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    qq_op_path = out / "qq_obs_pred.png"
    fig.savefig(qq_op_path, dpi=140)
    plt.close(fig)

    metrics_out: dict[str, Any] = {
        **metrics,
        "files": {
            "scatter": scatter_path.name,
            "qq_residuals": qq_path.name,
            "qq_obs_pred": qq_op_path.name,
        },
    }
    (out / "quality_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )
    return metrics_out


def write_swath_plots(
    xyz: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str | Path,
    *,
    n_bins: int = 15,
    min_count: int = 3,
    title_prefix: str = "OOF",
) -> dict[str, Any]:
    """Mean observed vs predicted along X / Y / Z (swath plots)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xyz_a = np.asarray(xyz, dtype=float)
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if xyz_a.ndim != 2 or xyz_a.shape[1] != 3:
        raise ValueError("xyz must be (n, 3)")
    if xyz_a.shape[0] != yt.size or yt.size != yp.size:
        raise ValueError("xyz / y_true / y_pred length mismatch")

    m = np.isfinite(xyz_a).all(axis=1) & np.isfinite(yt) & np.isfinite(yp)
    xyz_a, yt, yp = xyz_a[m], yt[m], yp[m]

    files: dict[str, str] = {}
    axis_names = ("X", "Y", "Z")
    for axis_i, axis_name in enumerate(axis_names):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        coord = xyz_a[:, axis_i]
        if yt.size == 0 or float(np.ptp(coord)) <= 0:
            ax.set_title(f"{title_prefix} swath {axis_name} (нет данных)")
            ax.set_xlabel(axis_name)
            ax.set_ylabel("Mean grade")
        else:
            edges = np.linspace(float(coord.min()), float(coord.max()), int(n_bins) + 1)
            centers: list[float] = []
            mean_t: list[float] = []
            mean_p: list[float] = []
            counts: list[int] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                if hi == edges[-1]:
                    sel = (coord >= lo) & (coord <= hi)
                else:
                    sel = (coord >= lo) & (coord < hi)
                n = int(sel.sum())
                if n < int(min_count):
                    continue
                centers.append(0.5 * (float(lo) + float(hi)))
                mean_t.append(float(np.mean(yt[sel])))
                mean_p.append(float(np.mean(yp[sel])))
                counts.append(n)
            if centers:
                ax.plot(
                    centers,
                    mean_t,
                    "o-",
                    color="#1f77b4",
                    lw=1.5,
                    ms=5,
                    label="Observed (OOF)",
                )
                ax.plot(
                    centers,
                    mean_p,
                    "s--",
                    color="#d62728",
                    lw=1.5,
                    ms=5,
                    label="Predicted (OOF)",
                )
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel(axis_name)
            ax.set_ylabel("Mean grade")
            ax.set_title(
                f"{title_prefix} swath · {axis_name}  |  "
                f"bins≥{min_count}: {len(centers)}"
            )
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out / f"swath_{axis_name.lower()}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files[f"swath_{axis_name.lower()}"] = path.name

    meta = {
        "n": int(yt.size),
        "n_bins": int(n_bins),
        "min_count": int(min_count),
        "files": files,
    }
    (out / "swath_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
