"""
Analytics service for CineIQ — Netflix Content Intelligence.

Computes aggregations and visual data structures for the executive Dashboard (8 charts)
and interactive Trend Analysis views with multi-criteria filtering.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.services import data_service

logger = logging.getLogger(__name__)


def get_dashboard_charts() -> Dict[str, Any]:
    """
    Generate data structures for the 8 Dashboard visualizations:
      1. Movies vs TV Shows (count and percentage)
      2. Titles by Release Year (time-series by type)
      3. Top Genres (individual exploded genres)
      4. Top Countries (individual exploded countries)
      5. Ratings Distribution
      6. Content Added Over Time (by addition year)
      7. Movie Duration Distribution (histogram bins in minutes)
      8. TV Show Seasons Distribution (1, 2, 3, 4, 5+ seasons)
    """
    df = data_service.get_dataframe()

    # 1. Movies vs TV Shows
    type_counts = df["type"].value_counts()
    chart_type = {
        "labels": type_counts.index.tolist(),
        "values": type_counts.values.tolist(),
        "percentages": [round((v / len(df)) * 100, 1) for v in type_counts.values],
    }

    # 2. Titles by Release Year (focus on modern era 1980-2021 for clean plotting, but full data available)
    by_year = (
        df.groupby(["release_year", "type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure both columns exist
    if "Movie" not in by_year.columns:
        by_year["Movie"] = 0
    if "TV Show" not in by_year.columns:
        by_year["TV Show"] = 0

    chart_release_years = {
        "years": by_year["release_year"].tolist(),
        "movies": by_year["Movie"].tolist(),
        "tv_shows": by_year["TV Show"].tolist(),
    }

    # 3. Top Genres (exploded)
    genres_series = df["genre_list"].explode().dropna()
    top_genres = (
        genres_series[genres_series != ""]
        .value_counts()
        .head(12)
    )
    chart_top_genres = {
        "genres": top_genres.index.tolist()[::-1],  # Reverse for bottom-to-top horizontal bar
        "counts": top_genres.values.tolist()[::-1],
    }

    # 4. Top Countries (exploded)
    country_series = df["country_list"].explode().dropna()
    top_countries = (
        country_series[country_series != ""]
        .value_counts()
        .head(12)
    )
    chart_top_countries = {
        "countries": top_countries.index.tolist()[::-1],
        "counts": top_countries.values.tolist()[::-1],
    }

    # 5. Ratings Distribution
    ratings_counts = df["rating"].value_counts().head(10)
    chart_ratings = {
        "ratings": ratings_counts.index.tolist(),
        "counts": ratings_counts.values.tolist(),
    }

    # 6. Content Added Over Time (use year_added)
    if "year_added" in df.columns:
        added_by_year = (
            df.dropna(subset=["year_added"])
            .groupby(["year_added", "type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if "Movie" not in added_by_year.columns:
            added_by_year["Movie"] = 0
        if "TV Show" not in added_by_year.columns:
            added_by_year["TV Show"] = 0

        chart_added_over_time = {
            "years": added_by_year["year_added"].astype(int).tolist(),
            "movies": added_by_year["Movie"].tolist(),
            "tv_shows": added_by_year["TV Show"].tolist(),
        }
    else:
        chart_added_over_time = {"years": [], "movies": [], "tv_shows": []}

    # 7. Movie Duration Distribution (binned in 20-min buckets)
    movies = df[df["type"] == "Movie"]
    durations = movies["duration_minutes"].dropna()
    bins = [0, 60, 80, 100, 120, 150, 999]
    labels = ["< 60 min", "60-80 min", "80-100 min", "100-120 min", "120-150 min", "> 150 min"]
    binned_durations = pd.cut(durations, bins=bins, labels=labels, right=False)
    duration_counts = binned_durations.value_counts()[labels]
    chart_movie_durations = {
        "bins": labels,
        "counts": duration_counts.values.tolist(),
    }

    # 8. TV Show Seasons Distribution
    tv_shows = df[df["type"] == "TV Show"]
    seasons = tv_shows["duration_seasons"].dropna()
    season_buckets = {
        "1 Season": int((seasons == 1).sum()),
        "2 Seasons": int((seasons == 2).sum()),
        "3 Seasons": int((seasons == 3).sum()),
        "4 Seasons": int((seasons == 4).sum()),
        "5+ Seasons": int((seasons >= 5).sum()),
    }
    chart_tv_seasons = {
        "labels": list(season_buckets.keys()),
        "counts": list(season_buckets.values()),
    }

    return {
        "chart_type": chart_type,
        "chart_release_years": chart_release_years,
        "chart_top_genres": chart_top_genres,
        "chart_top_countries": chart_top_countries,
        "chart_ratings": chart_ratings,
        "chart_added_over_time": chart_added_over_time,
        "chart_movie_durations": chart_movie_durations,
        "chart_tv_seasons": chart_tv_seasons,
    }


def get_trend_analytics(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute filtered catalog trend analytics for the dedicated Trends page.
    Supports filtering by: type, genre, country, rating, year_min, year_max.
    """
    df = data_service.get_dataframe().copy()

    filters = filters or {}
    type_filter = filters.get("type")
    genre_filter = filters.get("genre")
    country_filter = filters.get("country")
    rating_filter = filters.get("rating")
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")

    if type_filter and type_filter in ("Movie", "TV Show"):
        df = df[df["type"] == type_filter]

    if genre_filter:
        df = df[df["genre_list"].apply(lambda gl: genre_filter in gl if isinstance(gl, list) else False)]

    if country_filter:
        df = df[df["country_list"].apply(lambda cl: country_filter in cl if isinstance(cl, list) else False)]

    if rating_filter:
        df = df[df["rating"] == rating_filter]

    if year_min:
        try:
            df = df[df["release_year"] >= int(year_min)]
        except (ValueError, TypeError):
            pass

    if year_max:
        try:
            df = df[df["release_year"] <= int(year_max)]
        except (ValueError, TypeError):
            pass

    total_filtered = len(df)

    if total_filtered == 0:
        return {
            "total_matches": 0,
            "by_year": [],
            "top_genres": [],
            "top_countries": [],
            "ratings": [],
            "duration_trends": [],
        }

    # 1. Releases by year & type
    by_year = (
        df.groupby(["release_year", "type"])
        .size()
        .reset_index(name="count")
        .sort_values("release_year")
    )

    # 2. Top genres in filtered subset
    top_genres = (
        df["genre_list"].explode().dropna()
        .loc[lambda x: x != ""]
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_genres.columns = ["genre", "count"]

    # 3. Top countries in filtered subset
    top_countries = (
        df["country_list"].explode().dropna()
        .loc[lambda x: x != ""]
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_countries.columns = ["country", "count"]

    # 4. Rating distribution
    ratings = (
        df["rating"].value_counts()
        .head(10)
        .reset_index()
    )
    ratings.columns = ["rating", "count"]

    # 5. Average movie runtime evolution across years
    movie_subset = df[(df["type"] == "Movie") & (df["duration_minutes"] > 0)]
    if not movie_subset.empty:
        duration_trends = (
            movie_subset.groupby("release_year")["duration_minutes"]
            .mean()
            .round(1)
            .reset_index()
        )
        duration_trends.columns = ["year", "avg_duration"]
    else:
        duration_trends = pd.DataFrame(columns=["year", "avg_duration"])

    return {
        "total_matches": total_filtered,
        "by_year": by_year.to_dict(orient="records"),
        "top_genres": top_genres.to_dict(orient="records"),
        "top_countries": top_countries.to_dict(orient="records"),
        "ratings": ratings.to_dict(orient="records"),
        "duration_trends": duration_trends.to_dict(orient="records"),
    }
