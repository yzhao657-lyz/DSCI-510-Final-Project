#!/usr/bin/env python3
"""
Run analysis on the cleaned TMDb dataset and write summary tables to disk.

Input:
- data/processed/movies_clean.csv

Outputs (data/processed/analysis_results/):
- correlation_matrix.csv
- age_correlation.csv
- genre_roi.csv
- genre_revenue.csv
- season_stats.csv
- top_directors.csv
- top_actors.csv
- budget_revenue_metrics.json
- multivariate_metrics.json
- feature_importance.csv
- top_roi_movies.csv
- bottom_roi_movies.csv
- summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class Paths:
    """Input/output paths for analysis artifacts."""
    input_csv: Path
    outdir: Path


def ensure_outdir(root: Path = Path(".")) -> Paths:
    """
    Ensure analysis output directory exists.

    Args:
        root: Project root directory.

    Returns:
        Paths: paths object containing input CSV and output directory.
    """
    input_csv = root / "data" / "processed" / "movies_clean.csv"
    outdir = root / "data" / "processed" / "analysis_results"
    outdir.mkdir(parents=True, exist_ok=True)
    return Paths(input_csv=input_csv, outdir=outdir)


def load_and_prepare(input_path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset and perform minimal numeric coercion.

    Args:
        input_path: Path to movies_clean.csv

    Returns:
        Prepared DataFrame with numeric columns coerced and ROI/movie_age ensured.
    """
    df = pd.read_csv(input_path)

    required = ["budget", "revenue", "popularity", "vote_average", "vote_count", "release_year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned CSV: {missing}")

    # numeric coercion
    num_cols = ["budget", "revenue", "popularity", "vote_average", "vote_count", "release_year"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic filtering
    df = df.dropna(subset=["budget", "revenue"]).copy()
    df = df[df["budget"] > 0].copy()
    df = df[df["revenue"] >= 0].copy()

    # movie age (if not already present)
    if "movie_age" not in df.columns:
        current_year = pd.Timestamp.now().year
        df["movie_age"] = current_year - df["release_year"]

    # ROI (if not already present)
    if "roi" not in df.columns:
        df["roi"] = df["revenue"] / df["budget"]

    # release_month numeric if exists
    if "release_month" in df.columns:
        df["release_month"] = pd.to_numeric(df["release_month"], errors="coerce")

    return df


def save_correlation_tables(df: pd.DataFrame, outdir: Path) -> None:
    """
    Compute and save correlation matrices.

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.
    """
    corr_cols = ["budget", "revenue", "roi", "popularity", "vote_average", "vote_count", "movie_age"]
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(outdir / "correlation_matrix.csv")

    age_corr = df[["movie_age", "vote_average", "popularity", "vote_count"]].corr(numeric_only=True)
    age_corr.to_csv(outdir / "age_correlation.csv")


def fit_budget_revenue_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit simple log-budget -> log-revenue linear regression.

    Args:
        df: Prepared DataFrame.

    Returns:
        Metrics dictionary.
    """
    X = np.log1p(df["budget"]).to_frame("log_budget")
    y = np.log1p(df["revenue"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "model": "LinearRegression(log_budget -> log_revenue)",
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n": int(len(df)),
    }


def genre_analysis(df: pd.DataFrame, outdir: Path) -> None:
    """
    Compute average ROI and revenue by genre (using exploded all_genres).

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.
    """
    if "all_genres" not in df.columns:
        return

    tmp = df.copy()
    tmp["genre_list"] = tmp["all_genres"].astype(str).str.split("|")
    exploded = tmp.explode("genre_list")

    genre_roi = (
        exploded.groupby("genre_list")["roi"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"genre_list": "genre", "roi": "avg_roi"})
    )
    genre_roi.to_csv(outdir / "genre_roi.csv", index=False)

    genre_revenue = (
        exploded.groupby("genre_list")["revenue"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"genre_list": "genre", "revenue": "avg_revenue"})
    )
    genre_revenue.to_csv(outdir / "genre_revenue.csv", index=False)


def season_analysis(df: pd.DataFrame, outdir: Path) -> None:
    """
    Compute summary statistics by release season.

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.
    """
    if "release_season" not in df.columns:
        return

    season_stats = (
        df.groupby("release_season")
        .agg(
            avg_revenue=("revenue", "mean"),
            avg_rating=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            n=("revenue", "size"),
        )
        .reset_index()
        .sort_values("avg_revenue", ascending=False)
    )
    season_stats.to_csv(outdir / "season_stats.csv", index=False)


def director_actor_tables(df: pd.DataFrame, outdir: Path) -> None:
    """
    Create top directors and top actors tables based on average revenue/ROI.

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.
    """
    if "director_name" in df.columns:
        director_stats = (
            df.groupby("director_name")
            .agg(
                avg_revenue=("revenue", "mean"),
                avg_roi=("roi", "mean"),
                avg_rating=("vote_average", "mean"),
                n=("revenue", "size"),
            )
            .query("n >= 3")
            .sort_values("avg_revenue", ascending=False)
            .head(30)
            .reset_index()
        )
        director_stats.to_csv(outdir / "top_directors.csv", index=False)

    if "top_3_cast" in df.columns:
        tmp = df.copy()
        tmp["actor_list"] = tmp["top_3_cast"].astype(str).str.split("|")
        exploded = tmp.explode("actor_list")

        actor_stats = (
            exploded.groupby("actor_list")
            .agg(
                avg_revenue=("revenue", "mean"),
                avg_roi=("roi", "mean"),
                avg_rating=("vote_average", "mean"),
                n=("revenue", "size"),
            )
            .query("n >= 5")
            .sort_values("avg_revenue", ascending=False)
            .head(50)
            .reset_index()
            .rename(columns={"actor_list": "actor"})
        )
        actor_stats.to_csv(outdir / "top_actors.csv", index=False)


def fit_multivariate_revenue_model(df: pd.DataFrame, outdir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Fit multivariate linear regression predicting log1p(revenue).

    Features:
    - budget, popularity, vote_average, vote_count, movie_age
    - optional: release_month
    - optional: director_name encoded

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.

    Returns:
        (metrics dict, feature_importance dataframe)
    """
    features: List[str] = ["budget", "popularity", "vote_average", "vote_count", "movie_age"]

    if "release_month" in df.columns:
        features.append("release_month")

    # Optional: director encoding (adds complexity, matches your scope)
    X_df = df[features].copy()

    if "director_name" in df.columns:
        le = LabelEncoder()
        X_df["director_encoded"] = le.fit_transform(df["director_name"].fillna("Unknown"))
        features = features + ["director_encoded"]

    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = np.log1p(df["revenue"]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "model": "LinearRegression(features -> log_revenue)",
        "features": features,
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n": int(len(df)),
    }

    feature_importance = pd.DataFrame(
        {"feature": features, "coefficient": model.coef_, "abs_coefficient": np.abs(model.coef_)}
    ).sort_values("abs_coefficient", ascending=False)

    feature_importance.to_csv(outdir / "feature_importance.csv", index=False)

    return metrics, feature_importance


def save_top_bottom_roi(df: pd.DataFrame, outdir: Path) -> None:
    """
    Save top/bottom ROI movie tables.

    Args:
        df: Prepared DataFrame.
        outdir: Output directory.
    """
    cols = [c for c in ["id", "title", "budget", "revenue", "roi", "vote_average", "popularity", "all_genres"] if c in df.columns]
    df.sort_values("roi", ascending=False).head(20)[cols].to_csv(outdir / "top_roi_movies.csv", index=False)
    df.sort_values("roi", ascending=True).head(20)[cols].to_csv(outdir / "bottom_roi_movies.csv", index=False)


def main() -> None:
    """Run full analysis pipeline and write outputs."""
    paths = ensure_outdir(Path("."))

    if not paths.input_csv.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {paths.input_csv}. Run clean_data.py first.")

    df = load_and_prepare(paths.input_csv)

    # Tables
    save_correlation_tables(df, paths.outdir)
    genre_analysis(df, paths.outdir)
    season_analysis(df, paths.outdir)
    director_actor_tables(df, paths.outdir)
    save_top_bottom_roi(df, paths.outdir)

    # Models
    budget_metrics = fit_budget_revenue_model(df)
    with (paths.outdir / "budget_revenue_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(budget_metrics, f, indent=2)

    multi_metrics, _fi = fit_multivariate_revenue_model(df, paths.outdir)
    with (paths.outdir / "multivariate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(multi_metrics, f, indent=2)

    summary = {
        "rows_used": int(len(df)),
        "budget_to_revenue": budget_metrics,
        "multivariate_revenue_model": multi_metrics,
        "outputs": sorted([p.name for p in paths.outdir.glob("*")]),
    }
    with (paths.outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Wrote files:")
    for p in sorted(paths.outdir.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()

