"""OECD SDMX adapter — quarterly real GDP growth and monthly unemployment (slice 26).

Keyless SDMX-CSV endpoints, verified 2026-08-24:

    https://sdmx.oecd.org/public/rest/data/{AGENCY},{FLOW},{VERSION}/{KEY}?startPeriod=…
        (Accept: application/vnd.sdmx.data+csv;version=1.0.0)

* Real GDP y/y %: ``OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1``
  key ``Q.Y.{AREAS}.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102`` — seasonally adjusted,
  chain-linked volumes, growth vs the same quarter a year earlier. Covers the
  OECD members plus BRA/CHN/IND ("key partners") and the euro area as ``EA``.
* Unemployment rate %: ``OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0``
  key ``{AREAS}.UNE_LF_M.PT_LF_SUB._Z.Y._T.Y_GE15._Z.M`` — SA, both sexes,
  15+. Members + ``EA`` only: CN/IN/BR have no monthly series here (documented gap).

``+`` joins areas so each flow is ONE call. Areas absent from a flow simply
yield no rows; a key with *no* data at all answers 404 (raised as ValueError).
Dates follow FRED's conventions (first month of the quarter / first of the month)
so the classifiers' "latest observation wins" picks the fresher source up.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from dalio.countries import ISO2_TO_OECD, Country
from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    SDMX_CSV_ACCEPT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

OECD_BASE = "https://sdmx.oecd.org/public/rest/data"
DEFAULT_TIMEOUT = 120
DEFAULT_START_YEAR = 2010


@dataclass(frozen=True)
class OecdFlow:
    name: str                 # "qna" | "lfs" — pipeline key
    indicator: str            # internal indicator name
    source: str               # storage source tag
    dataflow: str             # "AGENCY,FLOW,VERSION"
    key_template: str         # "{areas}" placeholder for the +-joined REF_AREA list
    frequency: str            # "Q" | "M" — how startPeriod and TIME_PERIOD are written
    series_id: str

    def url_for(self, areas: Sequence[str], start_year: int) -> str:
        key = self.key_template.format(areas="+".join(areas))
        start = f"{start_year}-Q1" if self.frequency == "Q" else f"{start_year}-01"
        return f"{OECD_BASE}/{self.dataflow}/{key}?startPeriod={start}"


QNA_GDP_GROWTH = OecdFlow(
    name="qna", indicator="real_gdp_yoy", source="OECD_QNA",
    dataflow="OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1",
    key_template="Q.Y.{areas}.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102",
    frequency="Q", series_id="QNA/B1GQ/GY",
)
LFS_UNEMPLOYMENT = OecdFlow(
    name="lfs", indicator="unemployment_rate", source="OECD_LFS",
    dataflow="OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0",
    key_template="{areas}.UNE_LF_M.PT_LF_SUB._Z.Y._T.Y_GE15._Z.M",
    frequency="M", series_id="LFS/UNE_LF_M/Y_GE15",
)
OECD_FLOWS: tuple[OecdFlow, ...] = (QNA_GDP_GROWTH, LFS_UNEMPLOYMENT)


class OecdSource:
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
            CachedTextFetcher.resolve_cache_dir(cache_dir, "DALIO_OECD_CACHE", "data/cache/oecd"),
            cache_ttl_hours, label="OECD", timeout=DEFAULT_TIMEOUT,
            forbidden_hint="try another network",
        )

    def fetch(
        self,
        flow: OecdFlow,
        countries: Sequence[Country],
        use_cache: bool = True,
        start_year: int = DEFAULT_START_YEAR,
    ) -> pd.DataFrame:
        """Long frame for one flow; one request for all requested areas."""
        area_to_iso2 = {ISO2_TO_OECD[c.iso2]: c.iso2 for c in countries}
        if not area_to_iso2:
            return _empty_long()
        url = flow.url_for(list(area_to_iso2), start_year)
        raw = self._parse(self._fetcher.fetch(url, use_cache=use_cache))
        if raw.empty:
            return _empty_long()
        rows = []
        for r in raw.itertuples(index=False):
            iso2 = area_to_iso2.get(r.REF_AREA)
            when = period_to_date(r.TIME_PERIOD)
            if iso2 is None or when is None:
                continue
            rows.append({
                "country": iso2, "indicator": flow.indicator, "date": when,
                "value": float(r.OBS_VALUE), "source": flow.source, "series_id": flow.series_id,
            })
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)

    def fetch_gdp_growth(self, countries: Sequence[Country], use_cache: bool = True,
                         start_year: int = DEFAULT_START_YEAR) -> pd.DataFrame:
        return self.fetch(QNA_GDP_GROWTH, countries, use_cache, start_year)

    def fetch_unemployment(self, countries: Sequence[Country], use_cache: bool = True,
                           start_year: int = DEFAULT_START_YEAR) -> pd.DataFrame:
        return self.fetch(LFS_UNEMPLOYMENT, countries, use_cache, start_year)

    @staticmethod
    def _parse(text: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return pd.DataFrame()
        need = ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]
        if df.empty or any(c not in df.columns for c in need):
            return pd.DataFrame()
        return df[need].dropna(subset=["OBS_VALUE", "TIME_PERIOD"])


def period_to_date(period: str) -> date | None:
    """'2026-Q2' → 2026-04-01 (first month of the quarter); '2026-06' → 2026-06-01."""
    if not isinstance(period, str):
        return None
    try:
        if "-Q" in period:
            year, q = period.split("-Q")
            return date(int(year), (int(q) - 1) * 3 + 1, 1)
        if len(period) == 7 and period[4] == "-":
            return date(int(period[:4]), int(period[5:]), 1)
    except ValueError:
        return None
    return None


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
