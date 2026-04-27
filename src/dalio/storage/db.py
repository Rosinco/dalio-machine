"""SQLite storage for macro time-series.

One row per (country, indicator, date, source) — long format. The unique constraint
prevents duplicate writes; updates go through upsert in the pipeline layer.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True)
    country = Column(String(8), nullable=False, index=True)
    indicator = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)
    source = Column(String(32), nullable=False)
    series_id = Column(String(64), nullable=False)
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "country", "indicator", "date", "source",
            name="uq_obs_country_ind_date_src",
        ),
        Index("ix_obs_lookup", "country", "indicator", "date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Observation {self.country}/{self.indicator} "
            f"{self.date}={self.value} from {self.source}>"
        )


def get_db_path() -> Path:
    raw = os.environ.get("DALIO_DB_PATH", "data/dalio.db")
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def make_engine(db_path: Path | None = None) -> Engine:
    path = db_path if db_path is not None else get_db_path()
    return create_engine(f"sqlite:///{path}", future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
