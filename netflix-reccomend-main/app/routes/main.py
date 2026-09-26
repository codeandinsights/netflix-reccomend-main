"""
Main application routes & API endpoints for CineIQ — Netflix Content Intelligence.
"""

import io
import logging
from flask import (
    Blueprint, render_template, request, jsonify, abort, Response, current_app
)
from app.services import (
    data_service,
    analytics_service,
    recommendation_service,
    search_service,
)

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------

@main_bp.route("/")
def dashboard():
    """Executive Dashboard with dynamic KPIs and all 8 Plotly charts."""
    try:
        kpis = data_service.get_kpis()
    except Exception as exc:
        logger.error("Error loading KPIs: %s", exc)
        kpis = {}

    try:
        charts = analytics_service.get_dashboard_charts()
    except Exception as exc:
        logger.error("Error loading dashboard charts: %s", exc)
        charts = {}

    return render_template("pages/dashboard.html", kpis=kpis, charts=charts)


@main_bp.route("/explore")
def explore():
    """Full Netflix Catalog Explorer with multifaceted search, filters, and pagination."""
    q = request.args.get("q", "").strip()
    content_type = request.args.get("type", "").strip()
    genre = request.args.get("genre", "").strip()
    country = request.args.get("country", "").strip()
    rating = request.args.get("rating", "").strip()
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    sort_by = request.args.get("sort_by", "year_desc").strip()
    page = request.args.get("page", 1, type=int)

    filter_options = data_service.get_filter_options()

    catalog_result = search_service.query_catalog(
        q=q or None,
        content_type=content_type or None,
        genre=genre or None,
        country=country or None,
        rating=rating or None,
        year_min=year_min,
        year_max=year_max,
        sort_by=sort_by,
        page=page,
        per_page=24,
    )

    current_filters = {
        "q": q,
        "type": content_type,
        "genre": genre,
        "country": country,
        "rating": rating,
        "year_min": year_min,
        "year_max": year_max,
        "sort_by": sort_by,
    }

    return render_template(
        "pages/explore.html",
        items=catalog_result["items"],
        pagination=catalog_result,
        filters=filter_options,
        current_filters=current_filters,
    )


@main_bp.route("/content/<show_id>")
def content_details(show_id: str):
    """Detailed view for a single title with safe fallback and recommendation link."""
    content = search_service.get_content_details(show_id)
    if not content:
        abort(404, description=f"Content title with ID '{show_id}' was not found in catalog.")

    # Fetch top 4 similar titles as instant related items preview
    preview_recs = []
    try:
        if recommendation_service.is_ready():
            preview_recs = recommendation_service.get_recommendations(show_id, n=4)
    except Exception as exc:
        logger.warning("Could not compute preview recommendations for %s: %s", show_id, exc)

    return render_template(
        "pages/content_detail.html",
        item=content,
        preview_recs=preview_recs,
    )


@main_bp.route("/recommendations")
def recommendations():
    """Interactive Content-Based Recommendation Engine with explainability."""
    preselect_show_id = request.args.get("show_id", "").strip()
    n = request.args.get("n", 10, type=int)

    sample_titles = recommendation_service.get_sample_titles(count=150)
    initial_recs = []
    selected_item = None

    if preselect_show_id:
        selected_item = search_service.get_content_details(preselect_show_id)
        try:
            if recommendation_service.is_ready():
                initial_recs = recommendation_service.get_recommendations(preselect_show_id, n=n)
        except Exception as exc:
            logger.error("Error computing recommendations for %s: %s", preselect_show_id, exc)

    return render_template(
        "pages/recommendations.html",
        sample_titles=sample_titles,
        preselect_show_id=preselect_show_id,
        selected_item=selected_item,
        initial_recs=initial_recs,
    )


@main_bp.route("/trends")
def trends():
    """Dedicated Trend Analysis with interactive Plotly visualizations."""
    filter_options = data_service.get_filter_options()
    trend_data = analytics_service.get_trend_analytics()

    return render_template(
        "pages/trends.html",
        filters=filter_options,
        trend_data=trend_data,
    )


@main_bp.route("/about")
def about():
    """About & Methodology page featuring verified data quality audit and system specs."""
    quality_report = data_service.get_data_quality_report()
    kpis = data_service.get_kpis()
    return render_template(
        "pages/about.html",
        report=quality_report,
        kpis=kpis,
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@main_bp.route("/api/recommendations")
def api_recommendations():
    """JSON API: Returns top-N similar titles with explainability metadata."""
    show_id = request.args.get("show_id", "").strip()
    n = request.args.get("n", 10, type=int)

    if not show_id:
        return jsonify({"error": "Query parameter 'show_id' is required."}), 400

    try:
        if not recommendation_service.is_ready():
            # Graceful fallback to genre-based matching if artifacts are missing
            df = data_service.get_dataframe()
            row = df[df["show_id"] == show_id]
            if row.empty:
                return jsonify({"error": f"show_id '{show_id}' not found."}), 404
            row = row.iloc[0]
            genres = set(row.get("genre_list") or [])
            similar = df[df["show_id"] != show_id].copy()
            similar["shared"] = similar["genre_list"].apply(
                lambda gl: len(genres & set(gl or []))
            )
            top_similar = similar[similar["shared"] > 0].nlargest(n, "shared")
            results = []
            for _, r in top_similar.iterrows():
                results.append({
                    "show_id": str(r["show_id"]),
                    "title": str(r["title"]),
                    "type": str(r["type"]),
                    "release_year": int(r["release_year"]) if r["release_year"] else None,
                    "rating": str(r.get("rating", "")),
                    "duration_raw": str(r.get("duration_raw", "")),
                    "genre_list": list(r.get("genre_list", [])),
                    "similarity_score": 0.5,
                    "similarity_percentage": 50,
                    "explanations": [{"category": "genre", "label": "Shared Genres", "detail": ", ".join(list(genres & set(r.get("genre_list", []))))}]
                })
            return jsonify({"results": results, "source": "genre_fallback"})

        results = recommendation_service.get_recommendations(show_id, n=n)
        return jsonify({"results": results, "source": "tfidf_cosine"})

    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.error("API recommendations error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@main_bp.route("/api/search")
def api_search():
    """JSON API: Autocomplete title search for recommendation input."""
    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", 15, type=int)
    results = recommendation_service.search_recommender_titles(query, limit=limit)
    return jsonify({"results": results})


@main_bp.route("/api/explore")
def api_explore():
    """JSON API: Dynamic catalog search and filter results for async UI updates."""
    q = request.args.get("q", "").strip()
    content_type = request.args.get("type", "").strip()
    genre = request.args.get("genre", "").strip()
    country = request.args.get("country", "").strip()
    rating = request.args.get("rating", "").strip()
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    sort_by = request.args.get("sort_by", "year_desc").strip()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 24, type=int)

    result = search_service.query_catalog(
        q=q or None,
        content_type=content_type or None,
        genre=genre or None,
        country=country or None,
        rating=rating or None,
        year_min=year_min,
        year_max=year_max,
        sort_by=sort_by,
        page=page,
        per_page=per_page,
    )
    return jsonify(result)


@main_bp.route("/api/trends")
def api_trends():
    """JSON API: Filtered trend analytics payload for interactive charts."""
    filters = {
        "type": request.args.get("type", "").strip() or None,
        "genre": request.args.get("genre", "").strip() or None,
        "country": request.args.get("country", "").strip() or None,
        "rating": request.args.get("rating", "").strip() or None,
        "year_min": request.args.get("year_min", type=int),
        "year_max": request.args.get("year_max", type=int),
    }
    data = analytics_service.get_trend_analytics(filters)
    return jsonify(data)


@main_bp.route("/export")
@main_bp.route("/export/csv")
def export():
    """Download filtered catalog query as a CSV file."""
    q = request.args.get("q", "").strip()
    content_type = request.args.get("type", "").strip()
    genre = request.args.get("genre", "").strip()
    country = request.args.get("country", "").strip()
    rating = request.args.get("rating", "").strip()
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    sort_by = request.args.get("sort_by", "year_desc").strip()

    csv_buffer = search_service.export_catalog_csv(
        q=q or None,
        content_type=content_type or None,
        genre=genre or None,
        country=country or None,
        rating=rating or None,
        year_min=year_min,
        year_max=year_max,
        sort_by=sort_by,
    )

    filename = "cineiq_catalog_export.csv"
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@main_bp.app_errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found"}), 404
    return render_template("pages/404.html", error=error), 404


@main_bp.app_errorhandler(500)
def internal_error(error):
    logger.error("Internal Server Error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("pages/500.html", error=error), 500
