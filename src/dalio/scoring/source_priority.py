"""Indicator-specific provider preference for same-economic-date ties.

Observation date remains the primary ordering key. These preferences only
decide between providers reporting the same indicator for the same country on
the same reference date; unmapped series retain the deterministic alphabetical
tie-break used before official Swedish sources were added.
"""

from __future__ import annotations

from sqlalchemy import case, literal

SOURCE_PREFERENCES: dict[tuple[str, str], tuple[str, ...]] = {
    ("SE", "cpi_yoy"): ("SCB_CPI", "IMF_CPI", "FRED"),
    ("SE", "policy_rate"): ("RIKSBANK_SWEA", "BIS_CBPOL", "FRED"),
    ("SE", "yield_10y"): ("RIKSBANK_SWEA", "FRED"),
    ("SE", "yield_2y"): ("RIKSBANK_SWEA", "FRED"),
}


def source_rank_expression(source_column, country: str, indicator: str):
    """Return a SQL rank expression for one concrete country/indicator query."""
    preferred = SOURCE_PREFERENCES.get((country, indicator))
    if not preferred:
        return literal(0)
    return case(
        {source: rank for rank, source in enumerate(preferred)},
        value=source_column,
        else_=len(preferred),
    )
