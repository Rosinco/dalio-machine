"""Asset-price-based regime signals (Slice 16).

Dalio's framework actually uses asset prices as cycle inputs, not just
outputs. Tight credit spreads, high CAPE, gold:bonds extreme — these are
bubble / distress detectors that complement the macro indicators.

This slice ships only the HY credit spread z-score. CAPE and gold:bonds
are documented gaps (see `data_sources/fred.py` ASSET_PRICE_SERIES
docstring) — the Yale spreadsheet integration and a replacement gold
series are follow-up work.

The z-score is computed against the country's own 20-year history of
the same series. None when there's <40 observations (≈10y at quarterly
cadence; daily series usually have decades).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.storage.db import Observation

# 20 years of daily data ≈ 5,000 obs; the floor below catches sparser
# monthly series too.
MIN_OBSERVATIONS = 40


@dataclass(frozen=True)
class AssetSignals:
    country: str
    hy_spread_latest: float | None = None
    hy_spread_z: float | None = None


def _zscore_latest(
    session: Session, country: str, indicator: str, window_days: int = 365 * 20,
) -> tuple[float | None, float | None]:
    """Return (latest, z) where z is the latest value's z-score against the
    country's last `window_days` of history. Both None when the indicator
    is absent or has too few observations.
    """
    cutoff = date.today() - timedelta(days=window_days)
    rows = session.execute(
        select(Observation.value, Observation.date)
        .where(
            Observation.country == country,
            Observation.indicator == indicator,
            Observation.date >= cutoff,
        )
        .order_by(Observation.date.asc())
    ).all()
    if len(rows) < MIN_OBSERVATIONS:
        return None, None
    values = np.asarray([r[0] for r in rows], dtype=float)
    latest = float(values[-1])
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        return latest, 0.0
    z = (latest - mean) / std
    return latest, float(z)


def compute_asset_signals(session: Session, country: str) -> AssetSignals:
    """Build the asset-signal snapshot for a country. Currently US-only —
    other countries simply return AssetSignals(country=country) with all
    values None.
    """
    if country != "US":
        return AssetSignals(country=country)
    hy_latest, hy_z = _zscore_latest(session, country, "hy_spread")
    return AssetSignals(country=country, hy_spread_latest=hy_latest, hy_spread_z=hy_z)
