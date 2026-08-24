"""OEC Economic Complexity Index adapter (slice 23).

Keyless JSON endpoint of the Observatory of Economic Complexity, verified
2026-08-24 (all 21 individual players present 2022–2024):

    https://oec.world/api/olap-proxy/data.jsonrecords
        ?cube=complexity_eci_a_hs92_hs6&drilldowns=Country,Year&measures=ECI&parents=false
    → {"data": [{"Country ID": "asjpn", "Country": "Japan", "Year": 2023, "ECI": 2.19}, …]}

``Country ID`` is a continent prefix + lowercase ISO-3; the last three
characters map back to the registry. One call returns every country and year
(~4 k rows); filtering happens locally. Stored under source ``OEC_ECI``;
the euro area gets a flagged member mean in the pipeline (as WGI does).
Tier C: an ordinal index, revised with each trade vintage.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from dalio.countries import Country
from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

OEC_URL = ("https://oec.world/api/olap-proxy/data.jsonrecords"
           "?cube=complexity_eci_a_hs92_hs6&drilldowns=Country,Year&measures=ECI&parents=false")
OEC_SERIES_ID = "complexity_eci_a_hs92_hs6"
SOURCE_OEC = "OEC_ECI"
INDICATOR_ECI = "economic_complexity"
DEFAULT_TIMEOUT = 60


class OecSource:
    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent)
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(cache_dir, "DALIO_OEC_CACHE", "data/cache/oec"),
            cache_ttl_hours, label="OEC", timeout=DEFAULT_TIMEOUT, suffix=".json",
            forbidden_hint="endpoint refused or moved",
        )

    def fetch_eci(self, countries: Sequence[Country], use_cache: bool = True,
                  start_year: int = 1995) -> pd.DataFrame:
        """Long frame of ECI per basket country (by ISO-3) and year ≥ ``start_year``."""
        iso3_to_iso2 = {c.iso3.lower(): c.iso2 for c in countries if c.on_map}
        if not iso3_to_iso2:
            return _empty_long()
        text = self._fetcher.fetch(OEC_URL, use_cache=use_cache)
        rows = []
        for rec in self._parse(text):
            cid = str(rec.get("Country ID", ""))
            iso2 = iso3_to_iso2.get(cid[-3:].lower()) if len(cid) >= 3 else None
            if iso2 is None:
                continue
            try:
                year = int(rec["Year"])
                value = float(rec["ECI"])
            except (KeyError, TypeError, ValueError):
                continue
            if year < start_year:
                continue
            rows.append({"country": iso2, "indicator": INDICATOR_ECI, "date": date(year, 12, 31),
                         "value": value, "source": SOURCE_OEC, "series_id": OEC_SERIES_ID})
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)

    @staticmethod
    def _parse(text: str) -> list[dict]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        rows = data.get("data") if isinstance(data, dict) else None
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []



def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
