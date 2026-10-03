"""Experiment report markdown + Optuna boundary markers (!)."""

from __future__ import annotations

from datetime import date
from typing import Any


def _at_boundary(value: float | int, lo: float, hi: float, *, is_int: bool) -> str | None:
    """Return 'min' / 'max' / None. Floats: within 2% of box width from edge."""
    lo_f, hi_f = float(lo), float(hi)
    width = hi_f - lo_f
    if is_int:
        v = int(round(float(value)))
        if v == int(round(lo_f)):
            return "min"
        if v == int(round(hi_f)):
            return "max"
        return None
    if width <= 0:
        return None
    tol = 0.02 * width
    v = float(value)
    if abs(v - lo_f) <= tol:
        return "min"
    if abs(v - hi_f) <= tol:
        return "max"
    return None


def mark_boundaries(best_theta: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    """Map tuned param → boundary side or None; also ``any_hit`` bool."""
    flags: dict[str, Any] = {"any_hit": False, "params": {}}
    mapping = [
        ("R_major", "R_major", False),
        ("R_inter", "R_inter", False),
        ("R_minor", "R_minor", False),
        ("R", "R", False),
        ("nugget", "nugget", False),
        ("range_scale", "range_scale", False),
        ("N_max", "N_max", True),
    ]
    for key, box_key, is_int in mapping:
        if key not in best_theta or box_key not in search_space:
            continue
        box = search_space[box_key]
        if not isinstance(box, (list, tuple)) or len(box) != 2:
            continue
        side = _at_boundary(best_theta[key], box[0], box[1], is_int=is_int)
        flags["params"][key] = side
        if side is not None:
            flags["any_hit"] = True
    return flags


def _fmt_val(v: Any, hit: str | None) -> str:
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    return f"{s}!" if hit else s


def build_report_markdown(
    domains: list[dict[str, Any]],
    *,
    data_label: str,
    out_dir: str,
    cv_method: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Markdown report matching docs/experiment_report_format.md (+ 3D fields).

    Primary score for API/CLI runs without external truth = CV RMSE of ``cv_method``.
    """
    lines: list[str] = []
    lines.append("# 3D OK tuner — experiment report")
    lines.append("")
    lines.append(f"- **Train / input:** `{data_label}`")
    lines.append("- **Truth / LB:** _(none — API/CLI tune only)_")
    lines.append(f"- **Primary score:** CV RMSE (`{cv_method}`)")
    lines.append(
        "- **`!`:** Optuna bound hit (floats: within 2% of box width; ints: exact)"
    )
    lines.append(f"- **Results:** `{out_dir}`")
    lines.append(f"- **Date:** {date.today().isoformat()}")
    lines.append("")

    # Summary table
    lines.append("| # | Domain | Method | Key params | CV RMSE | Notes |")
    lines.append("|---|--------|--------|------------|---------|-------|")

    best_rmse = None
    scored: list[tuple[int, float]] = []
    for i, d in enumerate(domains, start=1):
        opt = d.get("optimization") or {}
        rmse = opt.get("best_cv_rmse")
        if isinstance(rmse, (int, float)):
            scored.append((i, float(rmse)))
            if best_rmse is None or float(rmse) < best_rmse:
                best_rmse = float(rmse)

    for i, d in enumerate(domains, start=1):
        opt = d.get("optimization") or {}
        status = opt.get("status", "?")
        theta = opt.get("best_theta") or {}
        flags = (opt.get("boundary_flags") or {}).get("params") or {}
        fixed = d.get("fixed_model") or {}
        key_parts = []
        if theta:
            for k in ("R_major", "R_inter", "R_minor", "N_max", "nugget", "range_scale"):
                if k in theta:
                    key_parts.append(f"{k}={_fmt_val(theta[k], flags.get(k))}")
        key_parts.append(
            f"az={fixed.get('azimuth_deg', '—')}, dip={fixed.get('dip_deg', '—')}"
        )
        rmse = opt.get("best_cv_rmse")
        if isinstance(rmse, (int, float)) and best_rmse is not None and float(rmse) == best_rmse:
            rmse_s = f"**{float(rmse):.6g}**"
        elif isinstance(rmse, (int, float)):
            rmse_s = f"{float(rmse):.6g}"
        else:
            rmse_s = status
        notes = status if status != "SUCCESS" else cv_method
        lines.append(
            f"| {i} | {d.get('domain')} | OK3D | {', '.join(key_parts)} | {rmse_s} | {notes} |"
        )

    lines.append("")

    for i, d in enumerate(domains, start=1):
        opt = d.get("optimization") or {}
        fixed = d.get("fixed_model") or {}
        extent = d.get("extent") or {}
        theta = opt.get("best_theta") or {}
        space = opt.get("search_space") or {}
        flags = (opt.get("boundary_flags") or {}).get("params") or {}
        any_hit = bool((opt.get("boundary_flags") or {}).get("any_hit"))
        ev = opt.get("best_evaluation")
        n_trials = opt.get("n_trials")
        rmse = opt.get("best_cv_rmse")
        mae = opt.get("best_cv_mae")
        cv_details = opt.get("cv_details") or {}

        lines.append("---")
        lines.append("")
        lines.append(f"## Exp {i} — Domain `{d.get('domain')}`")
        lines.append("")
        lines.append(f"**Дата:** {date.today().isoformat()}  ")
        lines.append(f"**Results / code:** `{out_dir}/{d.get('domain')}/`  ")
        if ev is not None and rmse is not None:
            mae_s = f", MAE={float(mae):.6g}" if mae is not None else ""
            lines.append(
                f"**Best CV trial:** evaluation **{ev}/{n_trials}**, "
                f"CV RMSE=**{float(rmse):.6g}**{mae_s} (`{cv_method}`)"
            )
        else:
            lines.append(
                f"**Best CV trial:** _(skipped / failed)_ — {opt.get('status')} "
                f"{opt.get('reason', '')}"
            )
        lines.append("")
        lines.append("### Pipeline")
        lines.append("1. Per-domain 3D pair cloud")
        lines.append("2. MOI grid search (azimuth × dip) → directional spherical ranges")
        lines.append(
            f"3. Optuna TPE on R_major/inter/minor, N_max, nugget, range_scale "
            f"(CV=`{cv_method}`)"
        )
        lines.append("")
        lines.append("### 3D geometry")
        lines.append("")
        lines.append(
            f"- azimuth={fixed.get('azimuth_deg')}, dip={fixed.get('dip_deg')} "
            f"(rake=0)"
        )
        lines.append(
            f"- a_major/inter/minor="
            f"{fixed.get('a_major')}/{fixed.get('a_inter')}/{fixed.get('a_minor')}"
        )
        lines.append(
            f"- extent: ΔX={extent.get('x_span')}, ΔY={extent.get('y_span')}, "
            f"ΔZ={extent.get('z_span')} "
            f"(Z∈[{extent.get('z_min')}, {extent.get('z_max')}]), "
            f"n={d.get('n_points')}"
        )
        lines.append("")

        if theta and space:
            lines.append("### Tuned params vs search space")
            lines.append("")
            lines.append("| Param | Value | Search box | Boundary? |")
            lines.append("|-------|-------|------------|-----------|")
            for key in ("R_major", "R_inter", "R_minor", "R", "N_max", "nugget", "range_scale"):
                if key not in theta or key not in space:
                    continue
                box = space[key]
                hit = flags.get(key)
                bound = f"**!** ({hit})" if hit else ""
                lines.append(
                    f"| `{key}` | {_fmt_val(theta[key], None)} | "
                    f"[{box[0]}, {box[1]}] | {bound} |"
                )
            lines.append("")
            if not any_hit:
                lines.append(
                    "Ни один тюнимый параметр не на границе → **без `!`**."
                )
                lines.append("")
        else:
            lines.append("### Tuned params vs search space")
            lines.append("")
            lines.append("_(нет Optuna-результата)_")
            lines.append("")

        lines.append(
            f"**Fixed** (not tuned, no `!`): az={fixed.get('azimuth_deg')}, "
            f"dip={fixed.get('dip_deg')}, "
            f"a_maj/int/min={fixed.get('a_major')}/{fixed.get('a_inter')}/{fixed.get('a_minor')}, "
            f"nugget_fit={fixed.get('nugget_fit')}, sill_total={fixed.get('sill_total')}"
        )
        lines.append("")
        lines.append("### Scores")
        lines.append("")
        lines.append("| Metric | CV |")
        lines.append("|--------|----|")
        if rmse is not None:
            lines.append(f"| RMSE | **{float(rmse):.6g}** |")
        else:
            lines.append("| RMSE | — |")
        if mae is not None:
            lines.append(f"| MAE | {float(mae):.6g} |")
        lines.append("")
        if cv_details:
            lines.append(f"CV details: `{cv_details}`")
            lines.append("")
        lines.append("### Notes")
        lines.append(
            f"- Primary protocol for this run is **CV RMSE** (`{cv_method}`), "
            "not holdout/LB."
        )
        lines.append("")

    return "\n".join(lines)
