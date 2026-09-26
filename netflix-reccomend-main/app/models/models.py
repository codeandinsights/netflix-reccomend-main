"""
SQLAlchemy ORM models and database connection helpers for CineIQ.

Defines normalized schema:
  - Content (shows/movies)
  - Genre (distinct genres)
  - Country (distinct countries)
  - Director (distinct directors)
  - Many-to-many association tables
"""

from pathlib import Path
from typing import Dict, Any, List

from sqlalchemy import (
    Column, Integer, String, Text, Float, Table, ForeignKey, create_engine, Index
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, scoped_session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "netflix.db"
DB_URI = f"sqlite:///{DB_PATH}"

Base = declarative_base()

# ---------------------------------------------------------------------------
# Association tables
# ---------------------------------------------------------------------------

content_genres = Table(
    "content_genres",
    Base.metadata,
    Column("content_id", Integer, ForeignKey("content.content_id", ondelete="CASCADE"), primary_key=True),
    Column("genre_id", Integer, ForeignKey("genre.genre_id", ondelete="CASCADE"), primary_key=True),
    Index("idx_content_genres_genre", "genre_id"),
)

content_countries = Table(
    "content_countries",
    Base.metadata,
    Column("content_id", Integer, ForeignKey("content.content_id", ondelete="CASCADE"), primary_key=True),
    Column("country_id", Integer, ForeignKey("country.country_id", ondelete="CASCADE"), primary_key=True),
    Index("idx_content_countries_country", "country_id"),
)

content_directors = Table(
    "content_directors",
    Base.metadata,
    Column("content_id", Integer, ForeignKey("content.content_id", ondelete="CASCADE"), primary_key=True),
    Column("director_id", Integer, ForeignKey("director.director_id", ondelete="CASCADE"), primary_key=True),
    Index("idx_content_directors_director", "director_id"),
)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Content(Base):
    __tablename__ = "content"

    content_id = Column(Integer, primary_key=True, autoincrement=True)
    show_id = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)  # Movie or TV Show
    date_added = Column(String(50), nullable=True)
    date_added_iso = Column(String(20), nullable=True, index=True)
    release_year = Column(Integer, nullable=True, index=True)
    rating = Column(String(20), nullable=True, index=True)
    duration_raw = Column(String(50), nullable=True)
    duration_minutes = Column(Float, nullable=True)
    duration_seasons = Column(Integer, nullable=True)
    duration_value = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    cast = Column(Text, nullable=True)

    genres = relationship(
        "Genre",
        secondary=content_genres,
        back_populates="contents",
    )
    countries = relationship(
        "Country",
        secondary=content_countries,
        back_populates="contents",
    )
    directors = relationship(
        "Director",
        secondary=content_directors,
        back_populates="contents",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM object to clean JSON-serializable dictionary."""
        return {
            "content_id": self.content_id,
            "show_id": self.show_id,
            "title": self.title,
            "type": self.type,
            "date_added": self.date_added,
            "date_added_iso": self.date_added_iso,
            "release_year": self.release_year,
            "rating": self.rating,
            "duration_raw": self.duration_raw,
            "duration_minutes": self.duration_minutes,
            "duration_seasons": self.duration_seasons,
            "duration_value": self.duration_value,
            "description": self.description,
            "cast": self.cast,
            "genre_list": [g.genre_name for g in self.genres],
            "country_list": [c.country_name for c in self.countries],
            "director_list": [d.director_name for d in self.directors],
        }

    def __repr__(self):
        return f"<Content(id={self.content_id}, show_id={self.show_id!r}, title={self.title!r})>"


class Genre(Base):
    __tablename__ = "genre"

    genre_id = Column(Integer, primary_key=True, autoincrement=True)
    genre_name = Column(String(200), unique=True, nullable=False, index=True)

    contents = relationship(
        "Content",
        secondary=content_genres,
        back_populates="genres",
    )

    def __repr__(self):
        return f"<Genre(id={self.genre_id}, name={self.genre_name!r})>"


class Country(Base):
    __tablename__ = "country"

    country_id = Column(Integer, primary_key=True, autoincrement=True)
    country_name = Column(String(200), unique=True, nullable=False, index=True)

    contents = relationship(
        "Content",
        secondary=content_countries,
        back_populates="countries",
    )

    def __repr__(self):
        return f"<Country(id={self.country_id}, name={self.country_name!r})>"


class Director(Base):
    __tablename__ = "director"

    director_id = Column(Integer, primary_key=True, autoincrement=True)
    director_name = Column(String(500), unique=True, nullable=False, index=True)

    contents = relationship(
        "Content",
        secondary=content_directors,
        back_populates="directors",
    )

    def __repr__(self):
        return f"<Director(id={self.director_id}, name={self.director_name!r})>"


# ---------------------------------------------------------------------------
# Database Engine & Session Factory
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def get_engine(db_path: Path = DB_PATH):
    global _engine
    if _engine is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engine


def get_session(db_path: Path = DB_PATH):
    global _session_factory
    engine = get_engine(db_path)
    if _session_factory is None:
        _session_factory = scoped_session(sessionmaker(bind=engine))
    return _session_factory()


def init_db(db_path: Path = DB_PATH):
    """Create all tables in the database."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
