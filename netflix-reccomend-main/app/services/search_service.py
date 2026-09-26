"""
Search & Explorer Service for CineIQ — Netflix Content Intelligence.

Provides multifaceted catalog filtering, case-insensitive partial title search,
sorting, pagination, and CSV export streaming over the 8,807 title catalog.
"""

import io
import csv
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from app.services import data_service

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 24


def _empty_catalog_result(page: int = 1, per_page: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
    return {
        "items": [],
        "total_count": 0,
        "page": max(1, int(page)),
        "per_page": max(1, int(per_page)),
        "total_pages": 1,
        "has_next": False,
        "has_prev": False,
    }


def query_catalog(
    q: Optional[str] = None,
    content_type: Optional[str] = None,
    genre: Optional[str] = None,
    country: Optional[str] = None,
    rating: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    sort_by: str = "year_desc",
    page: int = 1,
    per_page: int = DEFAULT_PAGE_SIZE,
) -> Dict[str, Any]:
    """
    Search and filter catalog records with sorting and pagination.

    Returns:
      - items: list[dict]
      - total_count: int
      - page: int
      - per_page: int
      - total_pages: int
      - has_next: bool
      - has_prev: bool
    """
    df = data_service.get_dataframe()

    # 1. Text Search (title, case-insensitive, whitespace tolerant)
    if q and str(q).strip():
        search_term = str(q).strip().lower()
        df = df[df["title"].str.lower().str.contains(search_term, na=False, regex=False)]

    # 2. Content Type filter
    if content_type and content_type in ("Movie", "TV Show"):
        df = df[df["type"] == content_type]
    if df.empty:
        return _empty_catalog_result(page, per_page)

    # 3. Genre filter (matching inside list)
    if genre and str(genre).strip():
        target_genre = str(genre).strip()
        df = df[df["genre_list"].apply(lambda gl: target_genre in gl if isinstance(gl, list) else False)]
    if df.empty:
        return _empty_catalog_result(page, per_page)

    # 4. Country filter (matching inside list)
    if country and str(country).strip():
        target_country = str(country).strip()
        df = df[df["country_list"].apply(lambda cl: target_country in cl if isinstance(cl, list) else False)]
    if df.empty:
        return _empty_catalog_result(page, per_page)

    # 5. Rating filter
    if rating and str(rating).strip():
        df = df[df["rating"] == str(rating).strip()]
    if df.empty:
        return _empty_catalog_result(page, per_page)

    # 6. Release Year range
    if year_min is not None:
        try:
            df = df[df["release_year"] >= int(year_min)]
        except (ValueError, TypeError):
            pass
    if df.empty:
        return _empty_catalog_result(page, per_page)

    if year_max is not None:
        try:
            df = df[df["release_year"] <= int(year_max)]
        except (ValueError, TypeError):
            pass
    if df.empty:
        return _empty_catalog_result(page, per_page)

    total_count = len(df)

    # 7. Sorting
    if sort_by == "year_desc":
        df = df.sort_values("release_year", ascending=False)
    elif sort_by == "year_asc":
        df = df.sort_values("release_year", ascending=True)
    elif sort_by == "title_asc":
        df = df.sort_values("title", ascending=True, key=lambda s: s.str.lower())
    elif sort_by == "title_desc":
        df = df.sort_values("title", ascending=False, key=lambda s: s.str.lower())
    elif sort_by == "date_added_desc":
        df = df.sort_values("date_added_iso", ascending=False, na_position="last")
    elif sort_by == "date_added_asc":
        df = df.sort_values("date_added_iso", ascending=True, na_position="last")
    else:
        df = df.sort_values("release_year", ascending=False)

    # 8. Pagination calculations
    page = max(1, int(page))
    per_page = max(6, min(int(per_page), 100))
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    offset = (page - 1) * per_page

    paged_df = df.iloc[offset: offset + per_page]

    # Convert records to dictionary format
    cols_to_extract = [
        "show_id", "title", "type", "release_year", "rating",
        "duration_raw", "genre_list", "country_list", "director", "cast", "description"
    ]
    existing_cols = [c for c in cols_to_extract if c in paged_df.columns]
    items = paged_df[existing_cols].to_dict(orient="records")

    # Format cleanly for presentation
    for item in items:
        item["release_year"] = int(item["release_year"]) if pd.notna(item.get("release_year")) else "Unknown"
        item["genre_list"] = list(item.get("genre_list", []))
        item["country_list"] = list(item.get("country_list", []))

    return {
        "items": items,
        "total_count": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def get_content_details(show_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch complete metadata for a title by show_id, with clean fallbacks.
    Returns None if show_id does not exist.
    """
    df = data_service.get_dataframe()
    matches = df[df["show_id"] == str(show_id).strip()]
    if matches.empty:
        return None

    row = matches.iloc[0].to_dict()

    # Clean missing values with human-friendly fallback
    fallback = "Not available in dataset"

    def clean_val(v):
        if v is None or pd.isna(v) or str(v).strip().lower() in ("nan", "none", "null", ""):
            return fallback
        return str(v).strip()

    def clean_list(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return []
        cleaned = [str(x).strip() for x in lst if x and str(x).strip().lower() not in ("nan", "none", "null")]
        return cleaned

    genres = clean_list(row.get("genre_list"))
    countries = clean_list(row.get("country_list"))
    directors = clean_list(row.get("director_list"))
    cast = clean_list(row.get("cast_list"))

    return {
        "show_id": str(row.get("show_id")),
        "title": clean_val(row.get("title")),
        "type": clean_val(row.get("type")),
        "release_year": int(row["release_year"]) if pd.notna(row.get("release_year")) else fallback,
        "date_added": clean_val(row.get("date_added")),
        "date_added_iso": clean_val(row.get("date_added_iso")),
        "rating": clean_val(row.get("rating")),
        "duration_raw": clean_val(row.get("duration_raw")),
        "duration_minutes": float(row.get("duration_minutes", 0)) if row.get("type") == "Movie" else None,
        "duration_seasons": int(row.get("duration_seasons", 0)) if row.get("type") == "TV Show" else None,
        "description": clean_val(row.get("description")),
        "director": ", ".join(directors) if directors else clean_val(row.get("director")),
        "director_list": directors,
        "cast": ", ".join(cast) if cast else clean_val(row.get("cast")),
        "cast_list": cast,
        "country": ", ".join(countries) if countries else clean_val(row.get("country")),
        "country_list": countries,
        "listed_in": ", ".join(genres) if genres else clean_val(row.get("listed_in")),
        "genre_list": genres,
    }


def export_catalog_csv(
    q: Optional[str] = None,
    content_type: Optional[str] = None,
    genre: Optional[str] = None,
    country: Optional[str] = None,
    rating: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    sort_by: str = "year_desc",
) -> io.StringIO:
    """Generate in-memory CSV file of filtered catalog records."""
    # Retrieve all matches without pagination (up to full dataset size)
    res = query_catalog(
        q=q,
        content_type=content_type,
        genre=genre,
        country=country,
        rating=rating,
        year_min=year_min,
        year_max=year_max,
        sort_by=sort_by,
        page=1,
        per_page=10000,
    )

    items = res["items"]
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "show_id", "type", "title", "director", "cast", "country",
        "release_year", "rating", "duration", "genres", "description"
    ])

    for it in items:
        writer.writerow([
            it.get("show_id", ""),
            it.get("type", ""),
            it.get("title", ""),
            it.get("director", ""),
            it.get("cast", ""),
            ", ".join(it.get("country_list", [])),
            it.get("release_year", ""),
            it.get("rating", ""),
            it.get("duration_raw", ""),
            ", ".join(it.get("genre_list", [])),
            it.get("description", ""),
        ])

    output.seek(0)
    return output
