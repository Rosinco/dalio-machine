"""FRED (St. Louis Fed) data source adapter.

Fetches a series, optionally applies a year-over-year transform, and returns a
long-format DataFrame matching the storage schema.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from fredapi import Fred


@dataclass(frozen=True)
class FredSeriesSpec:
    indicator: str          # internal indicator name (snake_case)
    series_id: str          # FRED series ID
    country: str            # ISO2
    frequency: str = "M"    # M=monthly, Q=quarterly, A=annual, D=daily
    transform: str = "raw"  # "raw" or "yoy"


# Slice 1 minimum viable bundle — US short-term cycle indicators.
US_SHORT_TERM_SERIES: tuple[FredSeriesSpec, ...] = (
    FredSeriesSpec("policy_rate", "DFF", "US", frequency="D"),
    FredSeriesSpec("cpi_yoy", "CPIAUCSL", "US", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "UNRATE", "US", frequency="M"),
    FredSeriesSpec("yield_10y", "DGS10", "US", frequency="D"),
    FredSeriesSpec("yield_2y", "DGS2", "US", frequency="D"),
    FredSeriesSpec("real_gdp_yoy", "GDPC1", "US", frequency="Q", transform="yoy"),
)


class FredClient(Protocol):
    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series: ...


class FredSource:
    def __init__(self, client: FredClient | None = None, api_key: str | None = None):
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError(
                "FRED_API_KEY not set. Register at "
                "https://fred.stlouisfed.org/docs/api/api_key.html "
                "and add to .env"
            )
        self._client = Fred(api_key=key)

    def fetch(self, spec: FredSeriesSpec, start: str | None = None) -> pd.DataFrame:
        raw = self._client.get_series(spec.series_id, observation_start=start)
        series = pd.Series(raw).dropna()

        if spec.transform == "yoy":
            series = self._yoy(series, spec.frequency)
        elif spec.transform != "raw":
            raise ValueError(f"Unknown transform: {spec.transform!r}")

        df = series.reset_index()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["country"] = spec.country
        df["indicator"] = spec.indicator
        df["source"] = "FRED"
        df["series_id"] = spec.series_id
        return df[["country", "indicator", "date", "value", "source", "series_id"]]

    @staticmethod
    def _yoy(series: pd.Series, frequency: str) -> pd.Series:
        periods_map = {"M": 12, "Q": 4, "A": 1}
        if frequency not in periods_map:
            raise ValueError(
                f"YoY transform not supported for frequency {frequency!r} — "
                "resample to M/Q/A first"
            )
        return series.pct_change(periods=periods_map[frequency]).mul(100).dropna()
