"""CSV column heuristics for mapping to X/Y/Z/Value/HoleID — API only, no math."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

# Preferred assay / grade names when several candidates exist
_ASSAY_PRIORITY = (
    "au",
    "gold",
    "cu",
    "copper",
    "ag",
    "zn",
    "pb",
    "ni",
    "fe",
    "as",
    "mo",
    "co",
    "sn",
    "w",
    "u",
    "th",
    "grade",
    "value",
    "assay",
)

_ASSAY_TOKEN = re.compile(
    r"^(?:grade|value|assay|ore|metal|"
    r"au|ag|cu|zn|pb|ni|fe|as|mo|co|sn|w|u|th|gold|copper|silver|zinc|"
    r"lead|nickel|iron)(?:$|[^a-z0-9])",
    re.I,
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.strip().lower()) if t]


def _score_x(col: str) -> int:
    n = _norm(col)
    t = _tokens(col)
    if n in ("x", "east", "easting", "xc", "xcoord", "xcoordinate"):
        return 100
    if "easting" in n or n.endswith("east"):
        return 80
    if t and t[0] == "x":
        return 70
    return 0


def _score_y(col: str) -> int:
    n = _norm(col)
    t = _tokens(col)
    if n in ("y", "north", "northing", "yc", "ycoord", "ycoordinate"):
        return 100
    if "northing" in n or n.endswith("north"):
        return 80
    if t and t[0] == "y":
        return 70
    return 0


def _score_z(col: str) -> int:
    n = _norm(col)
    t = _tokens(col)
    if n in ("z", "elev", "elevation", "rl", "zcoord", "zcoordinate", "altitude"):
        return 100
    if "elev" in n or n in ("rl", "topo"):
        return 80
    # depth is common but ambiguous (down-hole vs elevation)
    if n in ("depth", "zdepth") or (t and t[0] == "z"):
        return 60
    return 0


def _score_hole(col: str) -> int:
    n = _norm(col)
    t = _tokens(col)
    if n in ("holeid", "hole", "borehole", "drillhole", "dhid", "bhid", "hole_id"):
        return 100
    if n.startswith("hole") or n.startswith("borehole") or n.startswith("drillhole"):
        return 90
    if t and t[0] in ("hole", "borehole", "drillhole", "bh", "dh"):
        return 85
    if "hole" in n and "id" in n:
        return 80
    return 0


def _score_value(col: str) -> int:
    n = _norm(col)
    t = _tokens(col)
    if n in ("value", "grade", "assay", "v", "zvalue"):
        return 100
    if n.startswith("grade") or n.startswith("value") or n.startswith("assay"):
        return 90
    # Au, Au ppm, Au_gpt, CU_PCT …
    if t and _ASSAY_TOKEN.match(t[0]):
        # prefer short assay names / with unit suffix
        base = t[0]
        prio = {name: 80 - i for i, name in enumerate(_ASSAY_PRIORITY)}
        return 70 + prio.get(base, 40)
    # "Au ppm" style already covered; "ppm Au" rare
    if any(_ASSAY_TOKEN.match(tok) for tok in t):
        return 55
    return 0


def _best(columns: list[str], scorer, used: set[str]) -> tuple[str | None, int]:
    best_col: str | None = None
    best_score = 0
    for c in columns:
        if c in used:
            continue
        s = scorer(c)
        if s > best_score:
            best_score = s
            best_col = c
    if best_score <= 0:
        return None, 0
    return best_col, best_score


def _value_candidates(columns: list[str], used: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in columns:
        if c in used:
            continue
        s = _score_value(c)
        if s > 0:
            out.append({"column": c, "score": s})
    out.sort(
        key=lambda x: (
            -x["score"],
            # stable preference Au > Cu > …
            next(
                (
                    i
                    for i, name in enumerate(_ASSAY_PRIORITY)
                    if name in _tokens(x["column"]) or _norm(x["column"]).startswith(name)
                ),
                99,
            ),
            x["column"].lower(),
        )
    )
    return out


def suggest_mapping(columns: list[str]) -> dict[str, Any]:
    cols = [c for c in columns if c and str(c).strip()]
    used: set[str] = set()
    mapping: dict[str, str | None] = {
        "X": None,
        "Y": None,
        "Z": None,
        "Value": None,
        "HoleID": None,
    }
    confidence: dict[str, str] = {}

    for role, scorer in (
        ("X", _score_x),
        ("Y", _score_y),
        ("Z", _score_z),
        ("HoleID", _score_hole),
    ):
        col, score = _best(cols, scorer, used)
        if col:
            mapping[role] = col
            used.add(col)
            confidence[role] = "high" if score >= 80 else "medium"

    value_cands = _value_candidates(cols, used)
    if value_cands:
        pick = value_cands[0]["column"]
        mapping["Value"] = pick
        confidence["Value"] = (
            "high" if value_cands[0]["score"] >= 80 else "medium"
        )
        if len(value_cands) > 1:
            confidence["Value"] = "choice"

    warnings: list[str] = []
    for role in ("X", "Y", "Z", "Value"):
        if not mapping[role]:
            warnings.append(f"Не удалось автоматически определить колонку {role}.")
    if mapping["HoleID"] is None:
        warnings.append(
            "HoleID не найден — будет spatial block CV вместо GroupKFold."
        )
    if len(value_cands) > 1:
        warnings.append(
            "Найдено несколько полей содержания — выбрано эвристически; "
            "проверьте и при необходимости смените."
        )

    return {
        "columns": cols,
        "mapping": mapping,
        "confidence": confidence,
        "value_candidates": [v["column"] for v in value_cands],
        "warnings": warnings,
        "ready": all(mapping[r] for r in ("X", "Y", "Z", "Value")),
    }


def decode_csv(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("invalid CSV encoding")


def read_header_and_rows(raw: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = decode_csv(raw)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("invalid CSV: empty header")
    columns = [c.strip() for c in reader.fieldnames if c and str(c).strip()]
    if not columns:
        raise ValueError("invalid CSV: empty header")
    # normalize keys on rows to stripped names
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append(
            {
                (k.strip() if k else k): (v if v is not None else "")
                for k, v in row.items()
                if k and str(k).strip()
            }
        )
    return columns, rows


def inspect_csv_bytes(raw: bytes) -> dict[str, Any]:
    columns, rows = read_header_and_rows(raw)
    suggestion = suggest_mapping(columns)
    suggestion["n_points"] = len(rows)
    suggestion["preview_rows"] = rows[:5]
    return suggestion


def apply_mapping(
    raw: bytes,
    *,
    col_x: str,
    col_y: str,
    col_z: str,
    col_value: str,
    col_holeid: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    columns, rows = read_header_and_rows(raw)
    colset = set(columns)
    for label, col in (
        ("X", col_x),
        ("Y", col_y),
        ("Z", col_z),
        ("Value", col_value),
    ):
        if not col or col not in colset:
            raise ValueError(f"missing mapped column for {label}: {col!r}")
    if col_holeid and col_holeid not in colset:
        raise ValueError(f"missing mapped column for HoleID: {col_holeid!r}")

    out = io.StringIO()
    fieldnames = ["X", "Y", "Z", "Value"] + (["HoleID"] if col_holeid else [])
    writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    n_written = 0
    for row in rows:
        rec = {
            "X": row.get(col_x, ""),
            "Y": row.get(col_y, ""),
            "Z": row.get(col_z, ""),
            "Value": row.get(col_value, ""),
        }
        if col_holeid:
            rec["HoleID"] = row.get(col_holeid, "")
        # skip completely empty value rows
        if str(rec["Value"]).strip() == "":
            continue
        writer.writerow(rec)
        n_written += 1

    if n_written == 0:
        raise ValueError("no data rows after mapping")

    meta = {
        "n_points": n_written,
        "has_hole_id": bool(col_holeid),
        "mapping": {
            "X": col_x,
            "Y": col_y,
            "Z": col_z,
            "Value": col_value,
            "HoleID": col_holeid,
        },
    }
    return out.getvalue().encode("utf-8"), meta
