#!/usr/bin/env python3
"""
visualize_results.py

Creates and saves all project visualizations.

Inputs:
- data/processed/movies_clean.csv
- data/processed/analysis_results/*.csv  (produced by run_analysis.py)

Outputs:
- results/figures/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths

ROOT = Path(".")
INPUT_PATH = ROOT / "data" / "processed" / "movies_clean.csv"
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis_results"

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Helpers

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists; otherwise return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """In-place numeric coercion."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")



# Plots
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_cols = ["budget", "revenue", "roi", "popularity", "vote_average", "vote_count", "movie_age"]
    for c in corr_cols:
        if c not in df.columns:
            print(f"[skip] correlation heatmap: missing column {c}")
            return

    corr = df[corr_cols].corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Heatmap")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.colorbar(im)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def plot_budget_vs_revenue_outliers(df: pd.DataFrame, n_outliers: int = 6) -> None:
    if "budget" not in df.columns or "revenue" not in df.columns:
        print("[skip] budget vs revenue: missing budget/revenue")
        return

    x = np.log1p(pd.to_numeric(df["budget"], errors="coerce"))
    y = np.log1p(pd.to_numeric(df["revenue"], errors="coerce"))
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    df2 = df.loc[mask].copy()

    zx = (x - x.mean()) / x.std(ddof=0)
    zy = (y - y.mean()) / y.std(ddof=0)
    dist = np.sqrt(zx**2 + zy**2)

    out_idx = dist.nlargest(n_outliers).index
    offsets = [(6, 6), (6, -10), (-10, 6), (-10, -10), (10, 0), (-15, 0)]

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.25, label="Movies")
    plt.scatter(x.loc[out_idx], y.loc[out_idx], alpha=1.0, label="Statistical outliers")

    for k, i in enumerate(out_idx):
        label = df2.loc[i, "title"] if "title" in df2.columns else str(i)
        dx, dy = offsets[k % len(offsets)]
        plt.annotate(
            label,
            (x.loc[i], y.loc[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
        )

    plt.title("Budget vs Revenue (log1p) with Outliers")
    plt.xlabel("log1p(Budget)")
    plt.ylabel("log1p(Revenue)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "budget_vs_revenue_outliers.png", dpi=200)
    plt.close()


def plot_roi_distribution(df: pd.DataFrame) -> None:
    if "roi" not in df.columns:
        print("[skip] ROI distribution: missing roi")
        return

    roi = df["roi"].replace([np.inf, -np.inf], np.nan).dropna()
    if roi.empty:
        print("[skip] ROI distribution: no valid ROI")
        return

    plt.figure(figsize=(9, 6))
    plt.hist(roi, bins=60)
    plt.title("ROI Distribution (Revenue / Budget)")
    plt.xlabel("ROI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roi_distribution.png", dpi=200)
    plt.close()


def plot_popularity_vs_rating_with_trend(df: pd.DataFrame) -> None:
    if "popularity" not in df.columns or "vote_average" not in df.columns:
        print("[skip] popularity vs rating: missing columns")
        return

    pop = pd.to_numeric(df["popularity"], errors="coerce")
    rate = pd.to_numeric(df["vote_average"], errors="coerce")
    mask = pop.notna() & rate.notna()

    x = pop[mask].values
    y = rate[mask].values
    if len(x) < 2:
        print("[skip] popularity vs rating: not enough data")
        return

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.25)

    # trend line
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = coef[0] * xx + coef[1]
    plt.plot(xx, yy)

    plt.title("Popularity vs Rating (with Trend Line)")
    plt.xlabel("Popularity")
    plt.ylabel("Vote Average")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "popularity_vs_rating_trend.png", dpi=200)
    plt.close()


def plot_genre_proportion_pie(df: pd.DataFrame, top_k: int = 8) -> None:
    if "all_genres" not in df.columns:
        print("[skip] genre proportion: missing all_genres")
        return

    genres = df["all_genres"].dropna().astype(str).str.split("|")
    genre_counts = genres.explode().value_counts()
    if genre_counts.empty:
        print("[skip] genre proportion: no genre data")
        return

    top = genre_counts.head(top_k)
    other = pd.Series({"Other": genre_counts.iloc[top_k:].sum()}) if len(genre_counts) > top_k else pd.Series(dtype=float)
    plot_counts = pd.concat([top, other]) if not other.empty else top

    plt.figure(figsize=(8, 8))
    plt.pie(plot_counts.values, labels=plot_counts.index, autopct=lambda p: f"{p:.1f}%")
    plt.title(f"Genre Proportion (Top {top_k}" + (" + Other)" if not other.empty else ")"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "genre_proportion_pie.png", dpi=200)
    plt.close()


def plot_top_genres_by_roi_from_analysis() -> None:
    genre_roi_path = ANALYSIS_DIR / "genre_roi.csv"
    genre_roi = safe_read_csv(genre_roi_path)
    if genre_roi is None:
        print("[skip] top genres by ROI: genre_roi.csv not found (run run_analysis.py first)")
        return

    genre_roi = genre_roi.head(12)
    if "genre" not in genre_roi.columns or "avg_roi" not in genre_roi.columns:
        print("[skip] top genres by ROI: wrong columns in genre_roi.csv")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(genre_roi["genre"], genre_roi["avg_roi"])
    plt.title("Top Genres by Average ROI")
    plt.xlabel("Genre")
    plt.ylabel("Average ROI")
    plt.xticks(rotation=40, ha="right")
    plt.bar_label(bars, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_genres_by_roi_labeled.png", dpi=200)
    plt.close()


def plot_season_proportion_pie(df: pd.DataFrame) -> None:
    if "release_season" not in df.columns:
        print("[skip] season proportion: missing release_season")
        return

    season_counts = df["release_season"].value_counts(dropna=True)
    if season_counts.empty:
        print("[skip] season proportion: no season data")
        return

    plt.figure(figsize=(7, 7))
    plt.pie(
        season_counts.values,
        labels=season_counts.index,
        autopct=lambda p: f"{p:.1f}%\n(n={int(round(p/100*season_counts.sum()))})",
    )
    plt.title("Movie Proportion by Release Season")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "season_proportion_pie.png", dpi=200)
    plt.close()


def plot_avg_revenue_by_season_from_analysis() -> None:
    season_path = ANALYSIS_DIR / "season_stats.csv"
    season = safe_read_csv(season_path)
    if season is None:
        print("[skip] avg revenue by season: season_stats.csv not found (run run_analysis.py first)")
        return

    if "release_season" not in season.columns or "avg_revenue" not in season.columns:
        print("[skip] avg revenue by season: wrong columns in season_stats.csv")
        return

    plt.figure(figsize=(8, 6))
    bars = plt.bar(season["release_season"], season["avg_revenue"])
    plt.title("Average Revenue by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Revenue")
    plt.bar_label(bars, fmt="%.0f", padding=3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "avg_revenue_by_season_labeled.png", dpi=200)
    plt.close()


def plot_age_vs_rating_boxplot(df: pd.DataFrame) -> None:
    if "movie_age" not in df.columns or "vote_average" not in df.columns:
        print("[skip] age vs rating: missing columns")
        return

    bins = [0, 2, 5, 10, 20, 40]
    labels = ["0–2", "3–5", "6–10", "11–20", "20+"]

    df2 = df.copy()
    df2["age_group"] = pd.cut(df2["movie_age"], bins=bins, labels=labels, right=True)

    box_data = [df2.loc[df2["age_group"] == g, "vote_average"].dropna() for g in labels]

    plt.figure(figsize=(9, 6))
    plt.boxplot(box_data, tick_labels=labels, showfliers=True)  # tick_labels newer matplotlib
    plt.title("Movie Age vs Rating (Grouped Box Plot)")
    plt.xlabel("Movie Age Group (years)")
    plt.ylabel("Vote Average")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "Age_vs_rating_boxplot.png", dpi=200)
    plt.close()


def plot_top_directors_and_revenue_boxplot(df: pd.DataFrame) -> None:
    if "director_name" not in df.columns:
        print("[skip] directors plots: missing director_name")
        return

    director_stats = (
        df.groupby("director_name")
        .agg(avg_revenue=("revenue", "mean"), n_movies=("revenue", "size"))
        .query("n_movies >= 3")
        .sort_values("avg_revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    if director_stats.empty:
        print("[skip] directors plots: no directors meeting threshold")
        return

    # bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(director_stats["director_name"], director_stats["avg_revenue"])
    plt.title("Top 10 Directors by Average Revenue (min 3 movies)")
    plt.xlabel("Director")
    plt.ylabel("Average Revenue")
    plt.xticks(rotation=35, ha="right")
    plt.bar_label(bars, fmt="%.0f", padding=3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_directors_avg_revenue.png", dpi=200)
    plt.close()

    # boxplot of revenue distributions
    top_directors = director_stats["director_name"].tolist()
    data = [df.loc[df["director_name"] == d, "revenue"].dropna().values for d in top_directors]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, tick_labels=top_directors, showfliers=True)
    plt.title("Revenue Distribution of Top Directors")
    plt.xlabel("Director")
    plt.ylabel("Revenue")
    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "Director_revenue_boxplot.png", dpi=200)
    plt.close()


def plot_top_actors_roi_and_scatter(df: pd.DataFrame) -> None:
    if "top_3_cast" not in df.columns:
        print("[skip] actors plots: missing top_3_cast")
        return

    tmp = df.copy()
    tmp["actor_list"] = tmp["top_3_cast"].astype(str).str.split("|")
    exploded = tmp.explode("actor_list")

    actor_stats = (
        exploded.groupby("actor_list")
        .agg(avg_roi=("roi", "mean"), n_movies=("roi", "size"))
        .query("n_movies >= 5")
        .sort_values("avg_roi", ascending=False)
        .head(12)
        .reset_index()
        .rename(columns={"actor_list": "actor"})
    )

    if actor_stats.empty:
        print("[skip] actors plots: no actors meeting threshold")
        return

    # bar
    plt.figure(figsize=(10, 6))
    bars = plt.bar(actor_stats["actor"], actor_stats["avg_roi"])
    plt.title("Top Actors by Average ROI (min 5 movies)")
    plt.xlabel("Actor")
    plt.ylabel("Average ROI")
    plt.xticks(rotation=40, ha="right")
    plt.bar_label(bars, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_actors_avg_roi.png", dpi=200)
    plt.close()

    # scatter ROI vs count
    plt.figure(figsize=(9, 6))
    plt.scatter(actor_stats["n_movies"], actor_stats["avg_roi"], s=actor_stats["n_movies"] * 40, alpha=0.6)

    for _, r in actor_stats.iterrows():
        plt.text(r["n_movies"], r["avg_roi"], r["actor"], fontsize=9)

    plt.title("Actor Impact: ROI vs Number of Movies")
    plt.xlabel("Number of Movies")
    plt.ylabel("Average ROI")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "Actor_roi_vs_Count.png", dpi=200)
    plt.close()


def plot_feature_importance_from_analysis() -> None:
    fi_path = ANALYSIS_DIR / "feature_importance.csv"
    fi = safe_read_csv(fi_path)
    if fi is None:
        print("[skip] feature importance: feature_importance.csv not found (run run_analysis.py first)")
        return

    if "feature" not in fi.columns or "coefficient" not in fi.columns:
        print("[skip] feature importance: wrong columns in feature_importance.csv")
        return

    if "abs_coefficient" in fi.columns:
        fi = fi.sort_values("abs_coefficient", ascending=True)
    else:
        fi["abs_coefficient"] = np.abs(fi["coefficient"])
        fi = fi.sort_values("abs_coefficient", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(fi["feature"], fi["coefficient"])
    plt.title("Revenue Model: Feature Coefficients (log revenue)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run clean_data.py first.")

    df = pd.read_csv(INPUT_PATH)

    # numeric / basic cleaning for plotting consistency
    coerce_numeric(df, ["budget", "revenue", "popularity", "vote_average", "vote_count", "movie_age"])
    df = df.dropna(subset=["budget", "revenue"])
    df = df[df["budget"] > 0].copy()
    df["roi"] = df["revenue"] / df["budget"]

    print("Figures ->", FIG_DIR.resolve())
    print("Rows for visualization:", len(df))

    # core plots
    plot_correlation_heatmap(df)
    plot_budget_vs_revenue_outliers(df)
    plot_roi_distribution(df)
    plot_popularity_vs_rating_with_trend(df)
    plot_genre_proportion_pie(df)
    plot_season_proportion_pie(df)
    plot_age_vs_rating_boxplot(df)

    # analysis-driven plots (need run_analysis.py outputs)
    plot_top_genres_by_roi_from_analysis()
    plot_avg_revenue_by_season_from_analysis()
    plot_feature_importance_from_analysis()

    plot_top_directors_and_revenue_boxplot(df)
    plot_top_actors_roi_and_scatter(df)

    print("Saved figures in:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
