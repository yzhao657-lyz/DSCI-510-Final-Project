#!/usr/bin/env python3
"""
Download movie data from the TMDb API and save raw artifacts to data/raw/.

Outputs:
- data/raw/genres.json
- data/raw/movie_ids.json
- data/raw/discover_page_<n>.json
- data/raw/movie_details.jsonl
- data/raw/movie_credits.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import os
import time

import requests


@dataclass(frozen=True)
class Paths:
    """Project paths for raw and processed data"""
    root: Path
    raw_dir: Path
    processed_dir: Path


def ensure_dirs(root: Path = Path(".")) -> Paths:
    """
    Create required project structure

    Args:
        root (Path): Project root directory.

    Returns:
        Paths: Object containing root/raw/processed paths.
    """
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, raw_dir=raw_dir, processed_dir=processed_dir)


def get_api_key() -> str:
    """
    Retrieve TMDb API key from environment

    Returns:
        str: TMDb API key.

    Raises:
        RuntimeError: If API key is not found
    """
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY not set. Set it in your environment, e.g.\n"
            "export TMDB_API_KEY='YOUR_KEY_HERE'"
        )
    return key


def tmdb_get(base_url: str, api_key: str, endpoint: str,
             params: Optional[Dict[str, Any]] = None,
             timeout: int = 15) -> Dict[str, Any]:
    """
    Call the TMDb API and return parsed JSON response.

    Args:
        base_url (str): Base TMDb URL, e.g. https://api.themoviedb.org/3
        api_key (str): TMDb API key.
        endpoint (str): Endpoint path starting with '/', e.g. '/movie/550'.
        params (Optional[Dict[str, Any]]): Query parameters (without api_key).
        timeout (int): Request timeout seconds.

    Returns:
        Dict[str, Any]: JSON response as a dictionary.

    Raises:
        requests.HTTPError: For non-200 HTTP status codes.
    """
    if params is None:
        params = {}
    params["api_key"] = api_key

    url = base_url + endpoint
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Save a JSON object

    Args:
        path (Path): Output file path.
        obj (Dict[str, Any]): JSON-serializable dictionary.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_lines(path: Path) -> int:
    """
    Count number of lines in a text file. Used to resume JSONL downloads.

    Args:
        path (Path): File path.

    Returns:
        int: Number of lines (0 if file doesn't exist)
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def fetch_genres(paths: Paths, base_url: str, api_key: str) -> Path:
    """
    Fetch movie genres list and save to data/raw/genres.json.

    Args:
        paths (Paths): Project paths.
        base_url (str): TMDb base URL.
        api_key (str): TMDb API key.

    Returns:
        Path: Path to saved genres.json.
    """
    data = tmdb_get(base_url, api_key, "/genre/movie/list", params={"language": "en-US"})
    out_path = paths.raw_dir / "genres.json"
    save_json(out_path, data)
    return out_path


def discover_movie_ids(paths: Paths, base_url: str, api_key: str,
                       max_pages: int = 50,
                       start_date: str = "1990-01-01",
                       sleep_s: float = 0.2) -> List[int]:
    """
    Discover popular movies and return unique movie IDs

    Saves each discover page JSON for traceability.

    Args:
        paths (Paths): Project paths.
        base_url (str): TMDb base URL.
        api_key (str): TMDb API key.
        max_pages (int): Max number of /discover pages to fetch.
        start_date (str): Earliest release date (YYYY-MM-DD).
        sleep_s (float): Sleep between requests.

    Returns:
        List[int]: Sorted list of unique movie IDs.
    """
    all_movies: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = {
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page,
            "primary_release_date.gte": start_date,
        }
        data = tmdb_get(base_url, api_key, "/discover/movie", params=params)

        page_path = paths.raw_dir / f"discover_page_{page}.json"
        save_json(page_path, data)

        results = data.get("results", [])
        all_movies.extend(results)

        if page >= data.get("total_pages", 0):
            break

        time.sleep(sleep_s)

    movie_ids = sorted({m["id"] for m in all_movies if "id" in m})
    ids_path = paths.raw_dir / "movie_ids.json"
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(movie_ids, f, indent=2)

    return movie_ids


def fetch_details_and_credits(paths: Paths, base_url: str, api_key: str,
                              movie_ids: List[int],
                              sleep_s: float = 0.35) -> None:
    """
    Fetch movie details and credits for each movie ID and save as JSONL.

    Files:
      - movie_details.jsonl: one JSON object per line
      - movie_credits.jsonl: one JSON object per line

    The function resumes automatically by counting existing lines.

    Args:
        paths (Paths): Project paths.
        base_url (str): TMDb base URL.
        api_key (str): TMDb API key.
        movie_ids (List[int]): Movie IDs to fetch.
        sleep_s (float): Sleep between requests.
    """
    details_path = paths.raw_dir / "movie_details.jsonl"
    credits_path = paths.raw_dir / "movie_credits.jsonl"

    done = min(count_lines(details_path), count_lines(credits_path))

    with details_path.open("a", encoding="utf-8") as f_det, \
         credits_path.open("a", encoding="utf-8") as f_cred:

        for idx, mid in enumerate(movie_ids[done:], start=done + 1):
            det = tmdb_get(base_url, api_key, f"/movie/{mid}", params={"language": "en-US"})
            f_det.write(json.dumps(det, ensure_ascii=False) + "\n")

            cred = tmdb_get(base_url, api_key, f"/movie/{mid}/credits")
            f_cred.write(json.dumps(cred, ensure_ascii=False) + "\n")

            if idx % 50 == 0 or idx == len(movie_ids):
                print(f"Fetched {idx}/{len(movie_ids)}")

            time.sleep(sleep_s)


def main() -> None:
    """Run the full data collection pipeline."""
    paths = ensure_dirs(Path("."))
    base_url = "https://api.themoviedb.org/3"
    api_key = get_api_key()

    genres_path = fetch_genres(paths, base_url, api_key)
    print(f"Saved genres: {genres_path}")

    movie_ids = discover_movie_ids(paths, base_url, api_key, max_pages=50)
    print(f"Collected {len(movie_ids)} movie IDs.")

    fetch_details_and_credits(paths, base_url, api_key, movie_ids)
    print("Done.")


if __name__ == "__main__":
    main()
