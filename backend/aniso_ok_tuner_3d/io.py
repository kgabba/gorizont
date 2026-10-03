"""CSV / dataframe loading for 3D points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("X", "Y", "Z", "Grade")


def load_points(csv_path: str | Path) -> pd.DataFrame:
    """Load samples: X,Y,Z,Grade + optional Domain (default DEFAULT)."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "Domain" not in df.columns:
        df = df.copy()
        df["Domain"] = "DEFAULT"

    out = df[["X", "Y", "Z", "Grade", "Domain"]].copy()
    out["Domain"] = out["Domain"].astype(str)
    out = out.dropna(subset=["X", "Y", "Z", "Grade"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows after dropping NaN in X/Y/Z/Grade")
    return out


def points_from_records(records: list[dict]) -> pd.DataFrame:
    """Build points frame from list of dicts (API JSON body)."""
    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"points missing required fields: {missing}")
    if "Domain" not in df.columns:
        df["Domain"] = "DEFAULT"
    out = df[["X", "Y", "Z", "Grade", "Domain"]].copy()
    out["Domain"] = out["Domain"].astype(str)
    out = out.dropna(subset=["X", "Y", "Z", "Grade"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid points after dropping NaN")
    return out
