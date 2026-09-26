# Netflix Content Intelligence & Recommendation Platform (CineIQ)

An end-to-end data analytics, machine learning, and web platform built on the verified Netflix catalog dataset (8,807 titles). The project combines thorough data preprocessing, normalized relational storage, interactive visual analytics, and an explainable content-based recommendation engine.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Jupyter Notebooks (Data Science & Visualizations)](#jupyter-notebooks)
   - [Notebook 1: Data Cleaning, Preprocessing & Recommendation Engine](#notebook-1)
   - [Notebook 2: Exploratory Data Analysis & Visualizations](#notebook-2)
3. [Web Application Architecture & Capabilities](#web-application-architecture)
   - [Executive Dashboard (`/`)](#1-executive-dashboard)
   - [Catalog Explorer & Search Engine (`/explore`)](#2-catalog-explorer)
   - [Content Details View (`/content/<id>`)](#3-content-details-view)
   - [Explainable Recommendation Engine (`/recommendations`)](#4-explainable-recommendation-engine)
   - [Macro Trend Analysis (`/trends`)](#5-macro-trend-analysis)
   - [Data Quality Audit (`/about`)](#6-data-quality-audit)
4. [Technology Stack](#technology-stack)
5. [Project Directory Layout](#project-directory-layout)
6. [Quickstart & Execution Guide](#quickstart--execution-guide)
7. [Data Engineering & Algorithmic Methodology](#data-engineering--algorithmic-methodology)
   - [Data Cleaning & Anomaly Correction](#data-cleaning--anomaly-correction)
   - [Database Normalization (3NF)](#database-normalization-3nf)
   - [Weighted Content Soup Formulation](#weighted-content-soup-formulation)
   - [TF-IDF Vectorization & On-The-Fly Cosine Similarity](#tf-idf-vectorization--on-the-fly-cosine-similarity)
   - [Data-Grounded Explainability](#data-grounded-explainability)
8. [Comprehensive Viva / Evaluation Q&A](#comprehensive-viva--evaluation-qa)

---

## Project Overview

Streaming services manage massive catalogs where discovering relevant content is a primary challenge. While commercial streaming platforms rely on private user click-streams, academic and public analytics must operate on catalog metadata alone.

This project delivers:
- **Scientific Data Cleaning:** Repairs real-world catalog anomalies (such as misaligned duration fields), parses heterogeneous runtimes, and standardizes dates.
- **Normalized Relational Modeling:** Converts flat CSV records into an organized 3rd Normal Form (3NF) SQLite database.
- **In-Depth Exploratory Data Analysis:** 10 interactive Plotly visualizations extracting deep patterns in catalog composition, vintage, runtime distributions, and geographic production.
- **Explainable Content Recommendation:** A lightweight, sub-5ms content-based recommendation engine powered by TF-IDF and Cosine Similarity, featuring verified reasoning badges that tell users exactly *why* titles were suggested.
- **Interactive Web Interface:** A responsive Flask web application providing live search, filtering, and data export.

---

## Jupyter Notebooks

All data processing, model building, and visualization code is fully documented and independently executable across two dedicated Jupyter notebooks in the `notebooks/` directory:

### Notebook 1: `notebooks/01_data_cleaning_and_recommendation_engine.ipynb`
*Focused on data engineering, anomaly detection, database generation, and machine learning model training.*

1. **Environment Setup & Ingestion:** Loads `data/raw/netflix_titles.csv` (8,807 rows, 12 columns).
2. **Data Health Audit:** Analyzes column missingness (e.g., Director missing 30%, Country missing 9%, Cast missing 9%) and verifies zero duplicate `show_id` entries.
3. **Anomaly Correction:** 
   - Resolves the shifted duration anomaly in records `s5542`, `s5795`, and `s5814` where movie durations were erroneously entered in the `rating` column.
   - Extracts numeric `duration_minutes` for movies and integer `duration_seasons` for TV shows.
   - Parses dates into ISO-8601 (`YYYY-MM-DD`) format and extracts `year_added` and `month_added`.
   - Tokenizes multi-value fields (`listed_in`, `country`, `director`, `cast`) into trimmed Python lists.
4. **SQLite Relational Database Construction:** Creates `data/processed/netflix.db` under a 3NF schema (`content`, `genres`, `countries`, `directors`, `actors`, and 4 junction tables) and populates all 8,807 titles.
5. **Feature Engineering (Content Soup):** Builds weighted text profiles emphasizing genres (2x weight), synopsis, director, cast, country, format, and rating.
6. **TF-IDF Vectorization:** Fits a `TfidfVectorizer` (10,000 features, unigrams + bigrams, English stop-words removed) yielding a compact sparse matrix (~3.9 MB).
7. **On-the-Fly Similarity:** Computes cosine similarity for target titles in ~4 milliseconds using `linear_kernel`.
8. **Explainability Demonstration:** Evaluates set intersections for recommendations and outputs verified matching reasons.
9. **Artifact Serialization:** Exports `netflix_cleaned.pkl`, `netflix.db`, and `recommender_tfidf.pkl`.

### Notebook 2: `notebooks/02_eda_and_data_visualization.ipynb`
*A complete 14-section industry-standard Exploratory Data Analysis & Visualization portfolio notebook.*

Constructed with publication-quality dark-mode styling across both interactive Plotly figures and statistical Seaborn/Matplotlib charts:
1. **Setup & Imports:** Library configuration (`pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scipy`), custom cinematic Netflix palette (`#E50914`, `#3182CE`, `#D69E2E`, `#805AD5`), dark theme settings, and float precision formatting.
2. **Data Loading & Initial Inspection:** `.head()`, `.tail()`, `.sample()`, `.shape`, `.info()`, `.describe(include='all')`, and deep memory consumption profiling.
3. **Data Quality Assessment:** Missing value tabulation with percentage bar charts, duplication checks, data type validations, and shifted duration anomaly detection.
4. **Data Cleaning:** Shifted anomaly repair, missing value imputation strategies with explicit justifications, duration parsing (movie minutes & TV seasons), ISO date standardization, and a "Before vs. After" impact summary table.
5. **Feature Engineering:** Date part extraction (month, day of week), content age vintage (`content_age_years`), acquisition lag (`added_lag_years`), 5 runtime brackets (`<60m`, `60-90m`, `90-120m`, `120-150m`, `>150m`), and tokenized lists for multi-attribute analysis.
6. **Univariate Analysis:** Central tendency, spread, skewness, and kurtosis calculations; movie runtime distribution with KDE overlay and boxplot; interactive catalog composition donut chart.
7. **Bivariate Analysis:** Historical runtime evolution regression plot (1970–2021), genre-by-genre movie duration boxplots, and maturity rating vs. format crosstab heatmaps.
8. **Multivariate Analysis:** Pearson & Spearman correlation heatmaps across numeric features, and a multidimensional interactive Plotly bubble chart (Year vs. Runtime, sized by cast count, colored by rating).
9. **Time Series Analysis:** Monthly catalog additions timeline (2015–2021) with a 6-month moving average overlay; day-of-week and month-of-year seasonality analysis.
10. **Categorical Deep Dives:** Interactive hierarchical Treemap (Format $\rightarrow$ Top Genre $\rightarrow$ Rating), alongside top 10 most prolific directors and featured actors bar charts.
11. **Advanced / Interactive Visualizations:** 4-panel interactive Plotly dashboard grid (Countries, Runtime distribution, TV seasons longevity, and Maturity ratings).
12. **Statistical Testing:** Two-sample independent t-test with 95% confidence intervals comparing Indian vs. US movie durations ($p < 0.001$), and Chi-Square test of independence between content format and rating ($p < 0.001$).
13. **Key Insights & Strategic Recommendations:** 6 empirical takeaways, business recommendations for streaming portfolio expansion, and dataset limitations.
14. **Export & Save:** Serialization of clean assets (`netflix_cleaned.csv`, `netflix_cleaned.pkl`).

---

## Web Application Architecture

The platform includes a complete, responsive Flask web application that exposes the analytics, catalog search, and recommendation engine through an intuitive dark-mode user interface:

### 1. Executive Dashboard (`/`)
- **Dynamic KPI Cards:** Live catalog counts for Total Titles (8,807), Movies (6,131), TV Shows (2,676), Unique Genres (42), Countries (122), and Average Movie Runtime (99.6 min).
- **8 Responsive Plotly Visualizations:**
  1. *Catalog Composition* (Movie vs. TV Show Donut)
  2. *Release Year Trajectory* (Historical release curve)
  3. *Top 12 Genres* (Category rankings)
  4. *Top 12 Production Origins* (Country volume)
  5. *Audience Demographics* (Maturity rating split)
  6. *Acquisition Growth* (Content added by year)
  7. *Movie Duration Distribution* (Runtime frequency curve)
  8. *TV Show Longevity* (Seasons distribution)

### 2. Catalog Explorer (`/explore`)
- **Real-Time Search:** Instant title query matching with case-insensitive and whitespace tolerance.
- **Multi-Faceted Filtering:** Combine filters across Format (Movie/TV), 42 Genres, 122 Countries, Maturity Ratings, and Release Year Range (1925–2021).
- **Sorting & Pagination:** Sort by Release Year (Newest/Oldest) or Title (A–Z / Z–A) with clean 24-card pages.
- **CSV Data Export:** One-click download of the currently filtered catalog dataset.

### 3. Content Details View (`/content/<id>`)
- Detailed display of all metadata (Title, Format, Year, Added Date, Rating, Duration, Country, Genres, Director, Cast, and Synopsis).
- Null-safe formatting: Displays clean fallback notices (*"Not available in dataset"*); zero unhandled `NaN` or raw errors.
- **"Find Similar Content"** button directly launching the title in the recommendation engine.
- Inline preview of the top 4 related titles.

### 4. Explainable Recommendation Engine (`/recommendations`)
- Recommends titles based on the TF-IDF feature profile.
- Live autocomplete title search box.
- Configurable result count: Top 5, 10, 15, or 20 recommendations.
- **Data-Grounded Explainability Badges:** Explicitly highlights verified attributes shared with the query title:
  - `✓ Shared Genres` (e.g., *Sci-Fi, Drama*)
  - `✓ Same Director`
  - `✓ Shared Cast`
  - `✓ Shared Country`
  - `✓ Same Format` (Movie / TV Show)
  - `Match Score Percentage` (Cosine similarity score)
- Guaranteed self-exclusion: The query title is never shown in its own results.

### 5. Macro Trend Analysis (`/trends`)
- Analytical deep-dive with interactive client-side filter controls (Format, Genre, Country, Year Range).
- Live chart re-rendering via REST API (`/api/trends`).

### 6. Data Quality Audit (`/about`)
- Full transparency page detailing dataset boundaries (static catalog through 2021).
- Complete reconciliation matrix verifying 100% data parity between raw CSV and SQLite storage.

---

## Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.10+, Flask 3.0.3, Jinja2 | Routing, server-side template rendering, REST API endpoints. |
| **Database & ORM** | SQLite 3, SQLAlchemy 2.1.1 | Embedded relational 3NF database; zero external daemon requirements. |
| **Data Processing** | Pandas 2.2.2, NumPy 1.26.4 | Vectorized cleaning, duration parsing, tokenization, and metric math. |
| **Machine Learning** | Scikit-Learn 1.5.0 | Sublinear TF-IDF vectorization and cosine kernel similarity. |
| **Interactive Charts** | Plotly.js 2.35.2, Plotly Python | Interactive SVG/WebGL data visualizations with dark-mode styling. |
| **Frontend Styling** | Tailwind CSS (CDN) & Custom CSS | Responsive layout, dark theme (`#0a0a0f`, `#181824`, `#e50914`). |

---

## Project Directory Layout

```text
netflix-content-intelligence/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py               # SQLAlchemy ORM models (normalized SQLite schema)
│   ├── routes/
│   │   ├── __init__.py
│   │   └── main.py                 # Blueprint routes & REST API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analytics_service.py    # Plotly visualization logic & trend aggregations
│   │   ├── data_service.py         # Data cleaning, loading, and KPI calculation
│   │   ├── recommendation_service.py # TF-IDF engine, similarity, and explainability
│   │   └── search_service.py       # Catalog search, multi-filter, pagination, and CSV export
│   ├── static/
│   │   ├── css/
│   │   │   └── main.css            # Dark theme palette, responsive utilities, glowing badges
│   │   └── js/
│   │       └── main.js             # Mobile drawer toggle & UI interactions
│   └── templates/
│       ├── base.html               # Base layout template with header, nav, and footer
│       └── pages/
│           ├── dashboard.html      # KPI overview + 8 Plotly charts
│           ├── explore.html        # Catalog search & multifaceted filtering interface
│           ├── content_detail.html # Single title detail page with fallbacks
│           ├── recommendations.html# Interactive recommender & explainability badges
│           ├── trends.html         # Macro trend analysis interface
│           ├── about.html          # Methodology & verified data audit
│           ├── 404.html            # Custom not-found error page
│           └── 500.html            # Custom internal server error page
├── data/
│   ├── raw/
│   │   └── netflix_titles.csv      # Raw Netflix source dataset (8,807 rows)
│   └── processed/
│       ├── data_quality_report.json# Cleaned dataset audit and reconciliation report
│       ├── netflix.db              # Reconciled SQLite relational database (5.8 MB)
│       ├── netflix_cleaned.csv     # Cleaned catalog in exportable CSV format
│       ├── netflix_cleaned.pkl     # Processed Pandas DataFrame for fast loading
│       └── recommender_tfidf.pkl   # Compact sparse TF-IDF model bundle (~3.9 MB)
├── notebooks/
│   ├── 01_data_cleaning_and_recommendation_engine.ipynb  # Pipeline, DB & ML notebook
│   └── 02_eda_and_data_visualization.ipynb               # 10 publication-grade Plotly charts
├── .gitignore                      # Git ignore patterns
├── README.md                       # Complete project documentation
├── requirements.txt                # Pinned production dependencies
└── run.py                          # Application entrypoint
```

---

## Quickstart & Execution Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd netflix-content-intelligence

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python run.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
*Note: All processed database files (`netflix.db`) and ML models (`recommender_tfidf.pkl`) are pre-generated. If missing, `app/services/data_service.py` automatically cleans the raw dataset and re-creates them on startup.*

### 4. Running the Jupyter Notebooks
To interactively inspect the data cleaning, exploratory analysis, and recommendation modeling:
```bash
# Launch Jupyter Notebook
jupyter notebook
```
Open either notebook in the `notebooks/` directory:
- `01_data_cleaning_and_recommendation_engine.ipynb`
- `02_eda_and_data_visualization.ipynb`

---

## Data Engineering & Algorithmic Methodology

### Data Cleaning & Anomaly Correction
1. **Shifted Duration Anomaly:** In the raw dataset, three Louis C.K. standup comedy specials (`s5542`, `s5795`, `s5814`) contained runtimes (`"74 min"`, `"84 min"`, `"66 min"`) inside the `rating` column, leaving the `duration` column null. The cleaning routine identifies rows where `rating` matches `r'\d+\s*min'` and moves the value into `duration`, assigning the rating to `"Not Rated"`.
2. **Standardized Durations:** Movie runtimes are extracted into float minutes (`duration_minutes`), and TV runtimes into integer season counts (`duration_seasons`).
3. **Standardized Dates:** Date strings like `"September 25, 2021"` are converted into standard ISO-8601 strings (`2021-09-25`), populating `year_added` and `month_added`.
4. **Tokenization:** Multi-value comma-separated strings (`listed_in`, `country`, `director`, `cast`) are tokenized into clean Python lists with whitespace stripped.

### Database Normalization (3NF)
Rather than leaving catalog data in a flat table, the project establishes a 3rd Normal Form schema in SQLite:
- `content`: Primary entity table (`show_id`, `type`, `title`, `release_year`, `rating`, `duration_minutes`, `duration_seasons`, `description`).
- `genres`, `countries`, `directors`, `actors`: Normalized entity tables with unique string names.
- `content_genres`, `content_countries`, `content_directors`, `content_cast`: Junction tables managing many-to-many associations.

### Weighted Content Soup Formulation
To measure title similarity across multiple attributes, we construct a unified text string per title:
$$\text{Soup} = \text{Synopsis}_{(1\times)} + \text{Genres}_{(2\times)} + \text{Director}_{(1\times)} + \text{Cast}_{[:5](1\times)} + \text{Country}_{(1\times)} + \text{Format}_{(1\times)} + \text{Rating}_{(1\times)}$$
Genres receive **2x repetition** to ensure strong thematic alignment so that comedies recommend comedies and horror titles recommend horror titles.

### TF-IDF Vectorization & On-The-Fly Cosine Similarity
1. **Vectorization:** The content soup is transformed using `TfidfVectorizer` (sublinear term frequency scaling, English stop-words removed, 10,000 maximum features).
2. **On-the-Fly Cosine Similarity:** Instead of precomputing and storing an $8,807 \times 8,807$ dense matrix (which consumes over 310 MB of disk and RAM), the system stores only the compact sparse TF-IDF model (~3.9 MB). For any query title, the cosine similarity vector is computed on the fly using `linear_kernel`:
$$\text{Sim}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
This dot product across all 8,807 titles completes in **~4 milliseconds**.

### Data-Grounded Explainability
Commercial recommendation systems often behave as black boxes. CineIQ enforces explainability by comparing the query title's metadata directly against each candidate:
- $\text{Shared Genres} = \text{Genres}_q \cap \text{Genres}_c$
- $\text{Shared Country} = \text{Countries}_q \cap \text{Countries}_c$
- $\text{Shared Cast} = \text{Cast}_q \cap \text{Cast}_c$
- $\text{Same Director} = (\text{Director}_q == \text{Director}_c)$
- $\text{Same Format} = (\text{Type}_q == \text{Type}_c)$
Only intersecting attributes are rendered as explanation badges. No false reasons are displayed.

---

## Comprehensive Viva / Evaluation Q&A

### Q1: Why did you choose Content-Based Filtering over Collaborative Filtering?
> **Answer:** The Netflix catalog dataset provides item metadata (genres, synopsis, cast, director, country, release year) but **contains zero user accounts, ratings, or watch history**. Collaborative filtering requires a user-item rating matrix. Claiming collaborative filtering on this dataset would require fabricating fake user data. Using content-based filtering via TF-IDF and Cosine Similarity is scientifically honest and grounded entirely in real catalog attributes.

### Q2: Why did you compute Cosine Similarity on the fly instead of saving a precomputed matrix?
> **Answer:** Precomputing an $8,807 \times 8,807$ floating-point similarity matrix requires over 77 million floats, taking up 310 MB of disk space and RAM. In contrast, the sparse TF-IDF model occupies only ~3.9 MB (over 98% space reduction). Computing the single-row vector dot product using `linear_kernel` takes only **~4 milliseconds** on standard hardware, making on-the-fly computation both faster to load and far more memory efficient.

### Q3: How did you handle data quality issues in the raw dataset?
> **Answer:** 
> 1. In 3 records (`s5542`, `s5795`, `s5814`), movie duration was misplaced in the `rating` column while `duration` was null. We detected this pattern and shifted the values to `duration`, resetting `rating` to `"Not Rated"`.
> 2. Durations were parsed into numeric units: minutes for movies, season counts for TV shows.
> 3. Date strings were standardized into ISO-8601 (`YYYY-MM-DD`).
> 4. Multi-value strings were tokenized into lists to allow precise multi-attribute filtering.
> 5. Missing values in the UI are replaced with clean fallbacks (*"Not available in dataset"*); zero unhandled `NaN` or raw errors are exposed to users.

### Q4: What is the purpose of the 3NF SQLite database?
> **Answer:** The raw dataset is a denormalized CSV containing repeated text strings. Normalizing into a 3NF schema (`content`, `genres`, `countries`, `directors`, `actors`, and junction tables) prevents data redundancy, guarantees referential integrity, and allows efficient relational SQL queries across many-to-many relationships.

### Q5: How do the two Jupyter Notebooks fit into the project?
> **Answer:** 
> - `01_data_cleaning_and_recommendation_engine.ipynb` documents the complete data science pipeline: data loading, anomaly correction, database population, feature engineering, TF-IDF vectorization, on-the-fly similarity, and recommendation explainability.
> - `02_eda_and_data_visualization.ipynb` serves as the exploratory analysis and visual analytics portfolio, containing 10 publication-quality Plotly charts analyzing catalog composition, trends, runtimes, countries, and audience demographics.
