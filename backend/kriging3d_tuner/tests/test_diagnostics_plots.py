"""QQ / swath diagnostic plot writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kriging3d_tuner.diagnostics import (
    write_oof_diagnostic_plots,
    write_swath_plots,
)


def test_oof_and_swath_plots(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 80
    xyz = rng.uniform(0, 100, size=(n, 3))
    y_true = xyz[:, 0] * 0.02 + rng.normal(0, 0.3, size=n)
    y_pred = y_true + rng.normal(0, 0.2, size=n)

    diag = write_oof_diagnostic_plots(y_true, y_pred, tmp_path)
    sw = write_swath_plots(xyz, y_true, y_pred, tmp_path)

    for name in ("scatter.png", "qq.png", "qq_obs_pred.png"):
        assert (tmp_path / name).is_file()
    for name in ("swath_x.png", "swath_y.png", "swath_z.png"):
        assert (tmp_path / name).is_file()
    assert diag["n"] == n
    assert sw["n"] == n
