"""IMF CPI adapter — monthly headline inflation, y/y % (slice 26).

Keyless SDMX 2.1 CSV endpoint, verified 2026-08-24 (7 of 8 cycle countries
current to 2026-M06/M07; the euro area has no code here and stays on FRED):

    https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI/{ISO3+…}.CPI._T.YOY_PCH_PA_PT.M
        ?startPeriod=YYYY            (Accept: application/vnd.sdmx.data+csv;version=1.0.0)

Dimensions ``COUNTRY.INDEX_TYPE.COICOP_1999.TYPE_OF_TRANSFORMATION.FREQUENCY``;
``_T`` = all items, ``YOY_PCH_PA_PT`` = year-over-year percent change — the
same quantity FRED's discontinued OECD-MEI mirrors gave us as ``cpi_yoy``.
The CSV carries a multi-line quoted ``FULL_DESCRIPTION`` column: parse with
pandas, never line by line. Stored as ``cpi_yoy`` under source ``IMF_CPI``,
date = first of the month (FRED's monthly convention), so the classifier's
"latest observation wins" picks it up without any change.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from dalio.countries import Country
from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    SDMX_CSV_ACCEPT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

IMF_CPI_BASE = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI"
IMF_CPI_KEY_SUFFIX = "CPI._T.YOY_PCH_PA_PT.M"
SOURCE_IMF_CPI = "IMF_CPI"
INDICATOR_CPI = "cpi_yoy"
SERIES_ID_CPI = "CPI/_T/YOY_PCH_PA_PT"
DEFAULT_TIMEOUT = 120
DEFAULT_START_YEAR = 2010


class ImfCpiSource:
    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, SDMX_CSV_ACCEPT)
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(cache_dir, "DALIO_IMF_CPI_CACHE", "data/cache/imf_cpi"),
            cache_ttl_hours, label="IMF CPI", timeout=DEFAULT_TIMEOUT,
            forbidden_hint="Akamai; try another network",
        )

    def fetch(
        self,
        countries: Sequence[Country],
        use_cache: bool = True,
        start_year: int = DEFAULT_START_YEAR,
    ) -> pd.DataFrame:
        """Long frame of monthly headline CPI y/y for every country with an
        ISO-3 code the IMF publishes (aggregates such as the euro area are
        skipped — the IMF CPI dataset has no euro-area entity)."""
        iso3_to_iso2 = {c.iso3: c.iso2 for c in countries if c.on_map}
        skipped = [c.iso2 for c in countries if not c.on_map]
        if skipped:
            logger.info("IMF CPI: no entity for %s — left to other sources", ", ".join(skipped))
        if not iso3_to_iso2:
            return _empty_long()
        url = self.url_for(list(iso3_to_iso2), start_year)
        raw = self._parse(self._fetcher.fetch(url, use_cache=use_cache))
        if raw.empty:
            return _empty_long()
        rows = []
        for r in raw.itertuples(index=False):
            iso2 = iso3_to_iso2.get(r.COUNTRY)
            when = _month_to_date(r.TIME_PERIOD)
            if iso2 is None or when is None:
                continue
            rows.append({
                "country": iso2, "indicator": INDICATOR_CPI, "date": when,
                "value": float(r.OBS_VALUE), "source": SOURCE_IMF_CPI, "series_id": SERIES_ID_CPI,
            })
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)

    @staticmethod
    def url_for(iso3s: Sequence[str], start_year: int) -> str:
        return f"{IMF_CPI_BASE}/{'+'.join(iso3s)}.{IMF_CPI_KEY_SUFFIX}?startPeriod={start_year}"

    @staticmethod
    def _parse(text: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return pd.DataFrame()
        need = ["COUNTRY", "COICOP_1999", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"]
        if df.empty or any(c not in df.columns for c in need):
            return pd.DataFrame()
        df = df[need].dropna(subset=["OBS_VALUE", "TIME_PERIOD"])
        df = df[(df["COICOP_1999"] == "_T") & (df["TYPE_OF_TRANSFORMATION"] == "YOY_PCH_PA_PT")]
        return df


def _month_to_date(period: str) -> date | None:
    """'2026-M06' → 2026-06-01."""
    if not isinstance(period, str) or "-M" not in period:
        return None
    year, month = period.split("-M")
    try:
        return date(int(year), int(month), 1)
    except ValueError:
        return None


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
