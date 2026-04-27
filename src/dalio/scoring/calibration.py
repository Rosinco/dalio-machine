"""Per-country threshold calibration from historical observations.

Computes quantiles of a country's own series and maps them to classifier
thresholds. Falls back to default Thresholds when history is too short to
be meaningful. The default is conservative — 10 years of quarterly data
(40 observations) — because under-calibrated thresholds are worse than
honest fallback to US-derived defaults.

CPI thresholds get a floor: countries with chronically near-zero inflation
(Japan) shouldn't end up with a 0.8% "elevated" trigger, because that
would fire constantly without meaning what the framework says it means.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.scoring.thresholds import DEFAULT_THRESHOLDS, Thresholds
from dalio.storage.db import Observation

# 10 years of quarterly = 40, of monthly = 120. We pick 40 as the floor —
# enough to be meaningful, low enough that quarterly-only series qualify.
MIN_OBSERVATIONS = 40


def compute_country_quantiles(
    session: Session, country: str, indicator: str
) -> dict[str, float] | None:
    """Return q50/q75/q90/q95 of `indicator` for `country`. None if too few.

    Uses all observations in the DB regardless of source — duplicates are
    rare since the (country, indicator, date, source) constraint allows
    only one row per source-day, and most indicators only have one source.
    """
    rows = session.execute(
        select(Observation.value)
        .where(Observation.country == country, Observation.indicator == indicator)
    ).scalars().all()
    if len(rows) < MIN_OBSERVATIONS:
        return None
    values = np.asarray(rows, dtype=float)
    return {
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
    }


def compute_country_thresholds(session: Session, country: str) -> Thresholds:
    """Build per-country thresholds from historical quantiles.

    Each field falls back to its default when the source indicator has
    fewer than MIN_OBSERVATIONS rows. Floors are applied to CPI thresholds
    so chronically-low-inflation countries don't end up with absurdly low
    "elevated" / "peak" triggers.
    """
    debt = compute_country_quantiles(session, country, "total_credit_pct_gdp")
    dsr = compute_country_quantiles(session, country, "debt_service_ratio")
    cpi = compute_country_quantiles(session, country, "cpi_yoy")

    return Thresholds(
        debt_late_cycle_low=debt["q75"] if debt else DEFAULT_THRESHOLDS.debt_late_cycle_low,
        debt_extreme=debt["q95"] if debt else DEFAULT_THRESHOLDS.debt_extreme,
        dsr_stretched=dsr["q75"] if dsr else DEFAULT_THRESHOLDS.dsr_stretched,
        dsr_distress=dsr["q90"] if dsr else DEFAULT_THRESHOLDS.dsr_distress,
        dsr_extreme=dsr["q95"] if dsr else DEFAULT_THRESHOLDS.dsr_extreme,
        cpi_elevated=max(cpi["q75"], 2.5) if cpi else DEFAULT_THRESHOLDS.cpi_elevated,
        cpi_peak=max(cpi["q90"], 3.5) if cpi else DEFAULT_THRESHOLDS.cpi_peak,
    )


def threshold_deltas(country_t: Thresholds) -> dict[str, tuple[float, float, float]]:
    """For each per-country threshold, return (default, country, delta).

    Used by the dashboard to render the side-by-side table that makes
    calibration choices visible.
    """
    fields = (
        "debt_late_cycle_low", "debt_extreme",
        "dsr_stretched", "dsr_distress", "dsr_extreme",
        "cpi_elevated", "cpi_peak",
    )
    out: dict[str, tuple[float, float, float]] = {}
    for f in fields:
        d = getattr(DEFAULT_THRESHOLDS, f)
        c = getattr(country_t, f)
        out[f] = (d, c, c - d)
    return out
