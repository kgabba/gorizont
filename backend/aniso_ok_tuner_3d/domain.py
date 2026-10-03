"""Domain iteration helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def iter_domains(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (domain_name, subset) independently — no cross-domain pairs."""
    for domain in sorted(df["Domain"].unique()):
        subset = df.loc[df["Domain"] == domain].reset_index(drop=True)
        yield str(domain), subset


def safe_dirname(domain: str) -> str:
    """Filesystem-safe folder name for a domain label."""
    return (
        str(domain)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )
