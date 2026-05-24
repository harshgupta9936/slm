"""
Shared movie CSV loading — normalizes column names for this repo's dataset
(`databse.csv`: movie_title, release_year, etc.) and the README convention
(title, year, ...).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_LOADED = False


def load_project_env() -> None:
    """Load TMDB_API_KEY and other secrets from <repo>/.env (once per process)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


COLUMN_ALIASES = {
    "title": ["title", "movie_title", "name"],
    "overview": ["overview", "plot", "summary"],
    "director": ["director", "directors"],
    "genre": ["genre", "genres"],
    "year": ["year", "release_year", "release date"],
    "rating": ["rating", "vote_average", "score"],
    "movie_id": ["movie_id", "id", "tmdb_id"],
}


def _pick_column(df: pd.DataFrame, canonical: str) -> str | None:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        key = alias.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    return None


def load_movies_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    title_c = _pick_column(df, "title")
    overview_c = _pick_column(df, "overview")
    if not title_c or not overview_c:
        raise ValueError(
            f"CSV must include title and overview columns. Found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["title"] = df[title_c].astype(str).str.strip()
    out["overview"] = df[overview_c].astype(str).str.strip()

    d = _pick_column(df, "director")
    out["director"] = df[d].astype(str) if d else "Unknown"

    g = _pick_column(df, "genre")
    out["genre"] = df[g].astype(str) if g else "Unknown"

    y = _pick_column(df, "year")

    def _year_cell(x):
        if pd.isna(x) or str(x).strip() in ("", "nan"):
            return "Unknown"
        try:
            return str(int(float(x)))
        except (TypeError, ValueError):
            return str(x).strip()

    if y:
        out["year"] = df[y].map(_year_cell)
    else:
        out["year"] = "Unknown"

    r = _pick_column(df, "rating")

    def _rating_cell(x):
        if pd.isna(x):
            return "Unknown"
        try:
            return f"{float(x):.2f}"
        except (TypeError, ValueError):
            return str(x)

    if r:
        out["rating"] = df[r].map(_rating_cell)
    else:
        out["rating"] = "Unknown"

    mid = _pick_column(df, "movie_id")
    if mid:
        out["movie_id"] = df[mid].astype(str)
    else:
        out["movie_id"] = out.index.astype(str)

    out = out.fillna("Unknown")
    out = out[out["title"].str.len() > 0]
    return out.reset_index(drop=True)


def movies_to_csv_standard(df: pd.DataFrame, path: str) -> None:
    """Write normalized columns for tools that expect title,overview,director,..."""
    df[["title", "overview", "director", "genre", "year", "rating"]].to_csv(path, index=False)
