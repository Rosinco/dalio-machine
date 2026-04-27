"""FRED (St. Louis Fed) data source adapter.

Fetches a series, optionally applies a year-over-year transform, and returns a
long-format DataFrame matching the storage schema.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FredSeriesSpec:
    indicator: str          # internal indicator name (snake_case)
    series_id: str          # FRED series ID
    country: str            # ISO2
    frequency: str = "M"    # M=monthly, Q=quarterly, A=annual, D=daily
    transform: str = "raw"  # "raw" or "yoy"


# Tier-1 short-term cycle bundle — 6 countries × 6 indicators.
#
# US uses native FRED series (DFF, CPIAUCSL, UNRATE, DGS10, DGS2, GDPC1) — daily
# frequency where available, BLS/BEA primary sources.
#
# International series use FRED's mirror of the OECD Main Economic Indicators
# (MEI) database, which uses standardized codes:
#   IRSTCI01<C>M156N  — short-term interbank rate (~3 months), proxy for policy
#   IRLTLT01<C>M156N  — long-term interest rate (10Y govt bond)
#   <C>CPIALLMINMEI   — headline CPI, monthly index level
#   LRHUTTTT<C>M156S  — harmonised unemployment rate (15-64)
#   NAEXKP01<C>Q652S  — real GDP, quarterly
#
# Where <C> is the OECD 3-letter country code (USA, GBR, DEU, JPN, CHN, SWE,
# IND, BRA). Eurozone uses EZ19 / EA19 aggregates with Eurostat-published series.
#
# 2-year yields are intentionally absent for non-US countries: FRED doesn't
# carry a consistent series for them, and the yield-curve slope (10y-2y) is a
# US-centric leading indicator. Other countries lose the curve modifier in
# the classifier — that's a documented coverage gap, not a bug.
TIER_1_SERIES: tuple[FredSeriesSpec, ...] = (
    # ─── US (native FRED, daily where possible) ───
    FredSeriesSpec("policy_rate", "DFF", "US", frequency="D"),
    FredSeriesSpec("cpi_yoy", "CPIAUCSL", "US", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "UNRATE", "US", frequency="M"),
    FredSeriesSpec("yield_10y", "DGS10", "US", frequency="D"),
    FredSeriesSpec("yield_2y", "DGS2", "US", frequency="D"),
    FredSeriesSpec("real_gdp_yoy", "GDPC1", "US", frequency="Q", transform="yoy"),

    # ─── China (gaps: no FRED 10Y yield; GDP annual already-YoY) ───
    FredSeriesSpec("policy_rate", "IRSTCI01CNM156N", "CN", frequency="M"),
    FredSeriesSpec("cpi_yoy", "CHNCPIALLMINMEI", "CN", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "LMUNRRTTCNQ156S", "CN", frequency="Q"),
    # NAEXKP01CNA657S has the OECD-MEI "657" suffix = already a YoY growth rate;
    # do NOT apply YoY transform on top.
    FredSeriesSpec("real_gdp_yoy", "NAEXKP01CNA657S", "CN", frequency="A", transform="raw"),

    # ─── Eurozone (EZ19 aggregate, ECB / Eurostat / OECD MEI) ───
    FredSeriesSpec("policy_rate", "ECBDFR", "EU", frequency="D"),
    FredSeriesSpec("cpi_yoy", "CP0000EZ19M086NEST", "EU", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "LRHUTTTTEZM156S", "EU", frequency="M"),
    FredSeriesSpec("yield_10y", "IRLTLT01EZM156N", "EU", frequency="M"),
    FredSeriesSpec("real_gdp_yoy", "CLVMNACSCAB1GQEA19", "EU", frequency="Q", transform="yoy"),

    # ─── United Kingdom ───
    FredSeriesSpec("policy_rate", "IRSTCI01GBM156N", "UK", frequency="M"),
    FredSeriesSpec("cpi_yoy", "GBRCPIALLMINMEI", "UK", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "LRHUTTTTGBM156S", "UK", frequency="M"),
    FredSeriesSpec("yield_10y", "IRLTLT01GBM156N", "UK", frequency="M"),
    FredSeriesSpec("real_gdp_yoy", "NGDPRSAXDCGBQ", "UK", frequency="Q", transform="yoy"),

    # ─── Japan (gap: FRED's OECD-MEI Japan CPI series ended 2021;
    #     no current monthly JP CPI on FRED. Slice 2 ships without it —
    #     a BoJ-direct adapter or IMF IFS could fill the gap later.) ───
    FredSeriesSpec("policy_rate", "IRSTCI01JPM156N", "JP", frequency="M"),
    FredSeriesSpec("unemployment_rate", "LRUN64TTJPM156S", "JP", frequency="M"),
    FredSeriesSpec("yield_10y", "IRLTLT01JPM156N", "JP", frequency="M"),
    FredSeriesSpec("real_gdp_yoy", "JPNRGDPEXP", "JP", frequency="Q", transform="yoy"),

    # ─── Sweden ───
    # Riksbank short-term rate. IRSTCI01SEM156N is discontinued (last 2020-10);
    # IR3TIB01SEM156N (3-month interbank rate) is the live OECD MEI series.
    FredSeriesSpec("policy_rate", "IR3TIB01SEM156N", "SE", frequency="M"),
    FredSeriesSpec("cpi_yoy", "SWECPIALLMINMEI", "SE", frequency="M", transform="yoy"),
    FredSeriesSpec("unemployment_rate", "LRHUTTTTSEM156S", "SE", frequency="M"),
    FredSeriesSpec("yield_10y", "IRLTLT01SEM156N", "SE", frequency="M"),
    FredSeriesSpec("real_gdp_yoy", "CLVMNACSCAB1GQSE", "SE", frequency="Q", transform="yoy"),
)


# Backwards-compat: slice-1 US-only subset.
US_SHORT_TERM_SERIES: tuple[FredSeriesSpec, ...] = tuple(
    s for s in TIER_1_SERIES if s.country == "US"
)


def specs_for_countries(countries: tuple[str, ...] | None = None) -> tuple[FredSeriesSpec, ...]:
    """Return the subset of TIER_1_SERIES for the given country codes (None = all)."""
    if countries is None:
        return TIER_1_SERIES
    wanted = {c.upper() for c in countries}
    return tuple(s for s in TIER_1_SERIES if s.country in wanted)


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
        raw = self._fetch_with_retry(spec.series_id, start)
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

    def _fetch_with_retry(
        self, series_id: str, start: str | None,
        attempts: int = 3, backoff_base: float = 1.0,
    ) -> pd.Series:
        """Retry on transient errors (5xx, network glitches). Fail fast on 4xx."""
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._client.get_series(series_id, observation_start=start)
            except ValueError as e:
                msg = str(e)
                # 4xx-style errors — bad series ID, no point retrying
                if "does not exist" in msg or "Bad Request" in msg:
                    raise
                last_error = e
            except Exception as e:  # noqa: BLE001 — network errors etc.
                last_error = e
            if attempt < attempts - 1:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    "FRED %s attempt %d/%d failed (%s) — retrying in %.1fs",
                    series_id, attempt + 1, attempts, last_error, wait,
                )
                time.sleep(wait)
        # All attempts exhausted
        assert last_error is not None
        raise last_error

    @staticmethod
    def _yoy(series: pd.Series, frequency: str) -> pd.Series:
        periods_map = {"M": 12, "Q": 4, "A": 1}
        if frequency not in periods_map:
            raise ValueError(
                f"YoY transform not supported for frequency {frequency!r} — "
                "resample to M/Q/A first"
            )
        return series.pct_change(periods=periods_map[frequency]).mul(100).dropna()
