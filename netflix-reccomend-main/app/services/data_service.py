"""
Data service for Netflix Content Intelligence.

Loads the cleaned DataFrame once and exposes helpers for KPIs, catalog metadata,
and data quality reporting.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "netflix_titles.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PICKLE_PATH = PROCESSED_DIR / "netflix_cleaned.pkl"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"

REQUIRED_COLUMNS = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration", "listed_in", "description"
]

# Module-level cache
_df: pd.DataFrame | None = None
_report: Dict[str, Any] | None = None


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load CSV dataset and validate presence of required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path, dtype=str)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    return df


def audit_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Compile preliminary audit statistics on the raw data."""
    return {
        "raw_rows": int(len(df)),
        "raw_cols": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_show_ids": int(df["show_id"].duplicated().sum()),
        "missing_values_by_column": {col: int(df[col].isnull().sum()) for col in df.columns},
        "type_counts": df["type"].value_counts().to_dict(),
    }


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, standardize, and parse all fields in the Netflix DataFrame."""
    df = df.copy()

    # 1. Strip whitespace on string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": None, "nan": None, "NaN": None, "None": None})

    # 2. Fix shifted duration anomaly (e.g., Louis C.K. standups)
    shifted_mask = df["rating"].str.contains(r"\d+\s*min", na=False)
    shifted_count = int(shifted_mask.sum())
    if shifted_count > 0:
        df.loc[shifted_mask & df["duration"].isnull(), "duration"] = df.loc[shifted_mask, "rating"]
        df.loc[shifted_mask, "rating"] = "Not Rated"

    df["rating"] = df["rating"].fillna("Not Rated")

    # 3. Standardize and validate Type
    df["type"] = df["type"].str.strip()
    valid_types = {"Movie", "TV Show"}
    invalid_types = set(df["type"].dropna()) - valid_types
    if invalid_types:
        raise ValueError(f"Unexpected content types found: {invalid_types}")

    # 4. Standardize Release Year
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    # 5. Parse date_added to ISO datetime string, year_added, month_added
    parsed_dates = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    df["date_added_iso"] = parsed_dates.dt.strftime("%Y-%m-%d")
    df["year_added"] = parsed_dates.dt.year.astype("Int64")
    df["month_added"] = parsed_dates.dt.month.astype("Int64")

    # 6. Parse Duration
    df["duration_raw"] = df["duration"].fillna("Unknown")

    def extract_minutes(val: str, content_type: str) -> float:
        if content_type != "Movie" or not val:
            return 0.0
        m = re.search(r"(\d+)\s*min", str(val), re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    def extract_seasons(val: str, content_type: str) -> int:
        if content_type != "TV Show" or not val:
            return 0
        m = re.search(r"(\d+)\s*Season", str(val), re.IGNORECASE)
        return int(m.group(1)) if m else 0

    df["duration_minutes"] = df.apply(
        lambda r: extract_minutes(r["duration_raw"], r["type"]), axis=1
    )
    df["duration_seasons"] = df.apply(
        lambda r: extract_seasons(r["duration_raw"], r["type"]), axis=1
    )
    df["duration_value"] = np.where(
        df["type"] == "Movie",
        df["duration_minutes"],
        df["duration_seasons"]
    )

    # 7. Tokenize multi-value fields into Python lists
    def split_tokens(val: Any) -> List[str]:
        if val is None or pd.isna(val):
            return []
        return [item.strip() for item in str(val).split(",") if item.strip()]

    df["genre_list"] = df["listed_in"].apply(split_tokens)
    df["country_list"] = df["country"].apply(split_tokens)
    df["director_list"] = df["director"].apply(split_tokens)
    df["cast_list"] = df["cast"].apply(split_tokens)

    df["primary_country"] = df["country_list"].apply(lambda lst: lst[0] if lst else "Unknown")
    df["primary_genre"] = df["genre_list"].apply(lambda lst: lst[0] if lst else "Unknown")
    df["description"] = df["description"].fillna("No description available.")

    report = {
        "total_titles": int(len(df)),
        "total_movies": int((df["type"] == "Movie").sum()),
        "total_tv_shows": int((df["type"] == "TV Show").sum()),
        "shifted_duration_anomalies_fixed": shifted_count,
        "missing_directors_count": int(df["director"].isnull().sum()),
        "missing_cast_count": int(df["cast"].isnull().sum()),
        "missing_country_count": int(df["country"].isnull().sum()),
        "missing_date_added_count": int(df["date_added"].isnull().sum()),
        "missing_rating_count_original": 4,
        "unique_genres": int(df["genre_list"].explode().dropna().loc[lambda x: x != ""].nunique()),
        "unique_countries": int(df["country_list"].explode().dropna().loc[lambda x: x != ""].nunique()),
        "unique_directors": int(df["director_list"].explode().dropna().loc[lambda x: x != ""].nunique()),
        "unique_actors": int(df["cast_list"].explode().dropna().loc[lambda x: x != ""].nunique()),
        "avg_movie_duration_min": round(float(df[df["type"] == "Movie"]["duration_minutes"].mean()), 2),
        "min_release_year": int(df["release_year"].min()),
        "max_release_year": int(df["release_year"].max()),
    }

    return df, report


def clear_cache():
    """Clear memory caches (useful for testing and reloading)."""
    global _df, _report
    _df = None
    _report = None


def _load_dataframe() -> pd.DataFrame:
    global _df, _report
    if _df is None:
        # Load from raw CSV or cleaned CSV directly to ensure version independence across pandas/numpy
        if RAW_CSV.exists():
            logger.info("Loading and cleaning raw dataset from %s", RAW_CSV)
            raw_df = load_raw_dataset(RAW_CSV)
            _df, _report = clean_dataset(raw_df)
        elif (PROCESSED_DIR / "netflix_cleaned.csv").exists():
            logger.info("Loading cleaned CSV from %s", PROCESSED_DIR / "netflix_cleaned.csv")
            raw_df = load_raw_dataset(PROCESSED_DIR / "netflix_cleaned.csv")
            _df, _report = clean_dataset(raw_df)
        elif PICKLE_PATH.exists():
            try:
                _df = pd.read_pickle(str(PICKLE_PATH))
            except Exception as exc:
                logger.error("Pickle loading failed: %s", exc)
                raise
        else:
            raise FileNotFoundError(f"Dataset missing at {RAW_CSV}")
    return _df


def get_dataframe() -> pd.DataFrame:
    """Return the cached cleaned DataFrame, loading it on first call."""
    return _load_dataframe()


def get_kpis() -> Dict[str, Any]:
    """
    Return a dictionary of high-level KPIs computed dynamically from the DataFrame.

    Keys:
      - total_titles: int
      - total_movies: int
      - total_tv_shows: int
      - unique_genres: int
      - unique_countries: int
      - avg_movie_duration_minutes: float
      - total_directors: int
      - movie_percentage: float
      - tv_percentage: float
    """
    df = _load_dataframe()

    total_titles = int(len(df))
    total_movies = int((df["type"] == "Movie").sum())
    total_tv_shows = int((df["type"] == "TV Show").sum())

    movie_percentage = round((total_movies / total_titles) * 100, 1) if total_titles > 0 else 0.0
    tv_percentage = round((total_tv_shows / total_titles) * 100, 1) if total_titles > 0 else 0.0

    # Unique individual genres
    if "genre_list" in df.columns:
        all_genres = df["genre_list"].dropna().explode()
        unique_genres = int(all_genres[all_genres != ""].nunique())
    else:
        unique_genres = 0

    # Unique countries
    if "country_list" in df.columns:
        all_countries = df["country_list"].dropna().explode()
        unique_countries = int(all_countries[all_countries != ""].nunique())
    else:
        unique_countries = 0

    # Average movie duration in minutes
    if "duration_minutes" in df.columns:
        movie_mask = (df["type"] == "Movie") & (df["duration_minutes"] > 0)
        avg_movie_duration_minutes = round(
            float(df.loc[movie_mask, "duration_minutes"].mean()), 1
        ) if movie_mask.any() else 0.0
    elif "duration_value" in df.columns:
        movie_mask = (df["type"] == "Movie") & (df["duration_value"] > 0)
        avg_movie_duration_minutes = round(
            float(df.loc[movie_mask, "duration_value"].mean()), 1
        ) if movie_mask.any() else 0.0
    else:
        avg_movie_duration_minutes = 0.0

    # Total unique directors
    if "director_list" in df.columns:
        all_directors = df["director_list"].dropna().explode()
        total_directors = int(all_directors[all_directors != ""].nunique())
    elif "director" in df.columns:
        directors_series = df["director"].dropna()
        try:
            exploded = directors_series.explode()
        except AttributeError:
            exploded = directors_series
        total_directors = int(exploded[exploded != ""].nunique())
    else:
        total_directors = 0

    return {
        "total_titles": total_titles,
        "total_movies": total_movies,
        "total_tv_shows": total_tv_shows,
        "movie_percentage": movie_percentage,
        "tv_percentage": tv_percentage,
        "unique_genres": unique_genres,
        "unique_countries": unique_countries,
        "avg_movie_duration_minutes": avg_movie_duration_minutes,
        "total_directors": total_directors,
    }


def get_filter_options() -> Dict[str, Any]:
    """Return distinct filter lists for Explorer and Trends pages."""
    df = _load_dataframe()

    genres = sorted({
        g for sub in df["genre_list"].dropna() for g in sub if g
    })
    countries = sorted({
        c for sub in df["country_list"].dropna() for c in sub if c
    })
    ratings = [
        r for r in df["rating"].dropna().unique().tolist()
        if r and r not in ("None", "nan")
    ]
    # Sort ratings by frequency
    rating_order = df["rating"].value_counts().index.tolist()
    ratings.sort(key=lambda r: rating_order.index(r) if r in rating_order else 999)

    min_year = int(df["release_year"].dropna().min()) if "release_year" in df.columns else 1925
    max_year = int(df["release_year"].dropna().max()) if "release_year" in df.columns else 2021

    return {
        "genres": genres,
        "countries": countries,
        "ratings": ratings,
        "min_year": min_year,
        "max_year": max_year,
        "types": ["Movie", "TV Show"],
    }


def get_data_quality_report() -> Dict[str, Any]:
    """Load or generate the comprehensive data quality audit report."""
    global _report
    if _report is None:
        if REPORT_PATH.exists():
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                _report = json.load(f)
        else:
            # Fallback basic stats
            df = _load_dataframe()
            _report = {
                "raw_audit": {"raw_rows": len(df), "raw_cols": len(df.columns)},
                "quality_report": get_kpis()
            }
    return _report
