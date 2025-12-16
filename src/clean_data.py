#!/usr/bin/env python3
"""
Clean and feature-engineer TMDb raw data into a single CSV for analysis.

Inputs (expected in data/raw/):
- genres.json
- movie_details.jsonl
- movie_credits.jsonl

Output (written to data/processed/):
- movies_clean.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Paths:
    """Container for project data paths."""
    raw_dir: Path
    processed_dir: Path


def ensure_dirs(root: Path = Path(".")) -> Paths:
    """
    Ensure required data directories exist.

    Args:
        root: Project root directory.

    Returns:
        Paths: Paths object with raw and processed directories.
    """
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return Paths(raw_dir=raw_dir, processed_dir=processed_dir)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON dictionary.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL (one JSON object per line) file.

    Args:
        path: Path to a .jsonl file.

    Returns:
        A list of JSON dictionaries (one per line).
    """
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_genre_map(genres_data: Dict[str, Any]) -> Dict[int, str]:
    """
    Build a mapping from genre ID to genre name.

    Args:
        genres_data: JSON dictionary loaded from genres.json.

    Returns:
        Dictionary mapping genre_id -> genre_name.
    """
    return {g["id"]: g["name"] for g in genres_data.get("genres", [])}


def extract_genre_names(genres: Any) -> Optional[str]:
    """
    Convert TMDb 'genres' list-of-dicts into a '|' delimited string.

    Args:
        genres: Value from the 'genres' column (expected list of dicts).

    Returns:
        A '|' delimited genre string, or None if missing/invalid.
    """
    if not isinstance(genres, list):
        return None
    names = [g.get("name") for g in genres if isinstance(g, dict) and g.get("name")]
    return "|".join(names) if names else None


def first_genre(all_genres: Optional[str]) -> Optional[str]:
    """
    Extract the first genre from a '|' delimited genre string.

    Args:
        all_genres: Genre string such as "Action|Comedy".

    Returns:
        First genre name or None.
    """
    if isinstance(all_genres, str) and all_genres:
        return all_genres.split("|")[0]
    return None


def month_to_season(month: Any) -> Optional[str]:
    """
    Map month (1-12) to meteorological season.

    Args:
        month: Month value (int-like).

    Returns:
        Season string or None if missing.
    """
    if pd.isna(month):
        return None
    m = int(month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


def extract_credits_features(credits_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract director name and top 3 cast names from TMDb credits payload.

    Args:
        credits_rows: List of credits JSON dicts from movie_credits.jsonl.

    Returns:
        DataFrame with columns: movie_id, director_name, top_3_cast
    """
    records: List[Dict[str, Any]] = []
    for c in credits_rows:
        mid = c.get("id")
        crew = c.get("crew", []) or []
        cast = c.get("cast", []) or []

        director = next((p.get("name") for p in crew if p.get("job") == "Director"), None)
        top_cast = [p.get("name") for p in cast[:3] if p.get("name")]

        records.append(
            {
                "movie_id": mid,
                "director_name": director,
                "top_3_cast": "|".join(top_cast) if top_cast else None,
            }
        )
    return pd.DataFrame(records)


def build_clean_dataset(df_det: pd.DataFrame, df_cred: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Merge details + credits and build cleaned features for analysis.

    Args:
        df_det: Movie details DataFrame (from movie_details.jsonl).
        df_cred: Credits features DataFrame (director/cast).
        current_year: Used to compute movie_age.

    Returns:
        Cleaned DataFrame ready to save as CSV.
    """
    df = df_det.merge(df_cred, left_on="id", right_on="movie_id", how="left")

    # Genres
    df["all_genres"] = df["genres"].apply(extract_genre_names)
    df["main_genre"] = df["all_genres"].apply(first_genre)

    # Time features
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_season"] = df["release_month"].apply(month_to_season)
    df["movie_age"] = current_year - df["release_year"]

    # Numeric + ROI
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df.loc[df["budget"] <= 0, "budget"] = pd.NA
    df["roi"] = df["revenue"] / df["budget"]

    keep_cols = [
        "id", "title", "original_title",
        "release_date", "release_year", "release_month", "release_season", "movie_age",
        "budget", "revenue", "roi",
        "popularity", "vote_average", "vote_count",
        "all_genres", "main_genre",
        "director_name", "top_3_cast",
    ]

    df_final = df[keep_cols].copy()

    # Drop rows missing core financial values
    df_final = df_final.dropna(subset=["budget", "revenue"])

    return df_final


def main() -> None:
    """
    Main entry point: load raw files, clean data, and write movies_clean.csv.
    """
    paths = ensure_dirs(Path("."))

    # Load inputs
    genres_data = load_json(paths.raw_dir / "genres.json")
    _genre_map = build_genre_map(genres_data)  # not required downstream now, but kept for completeness

    details_rows = load_jsonl(paths.raw_dir / "movie_details.jsonl")
    credits_rows = load_jsonl(paths.raw_dir / "movie_credits.jsonl")

    df_det = pd.DataFrame(details_rows)
    df_cred = extract_credits_features(credits_rows)

    # Build final dataset
    df_final = build_clean_dataset(df_det, df_cred, current_year=2025)

    # Save
    out_path = paths.processed_dir / "movies_clean.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path}  (rows={len(df_final)}, cols={df_final.shape[1]})")


if __name__ == "__main__":
    main()
