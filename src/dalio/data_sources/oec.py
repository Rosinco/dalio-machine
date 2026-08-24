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

import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from dalio.countries import Country

logger = logging.getLogger(__name__)

OEC_URL = ("https://oec.world/api/olap-proxy/data.jsonrecords"
           "?cube=complexity_eci_a_hs92_hs6&drilldowns=Country,Year&measures=ECI&parents=false")
OEC_SERIES_ID = "complexity_eci_a_hs92_hs6"
SOURCE_OEC = "OEC_ECI"
INDICATOR_ECI = "economic_complexity"
DEFAULT_TIMEOUT = 60
DEFAULT_USER_AGENT = "dalio-machine/0.1 (+https://github.com/Rosinco/dalio-machine)"


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


class OecSource:
    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self._client = client or requests.Session()
        headers = getattr(self._client, "headers", None)
        if isinstance(headers, dict | requests.structures.CaseInsensitiveDict):
            headers["User-Agent"] = user_agent
        self._cache_dir = cache_dir or Path(os.environ.get("DALIO_OEC_CACHE", "data/cache/oec"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600

    def fetch_eci(self, countries: Sequence[Country], use_cache: bool = True,
                  start_year: int = 1995) -> pd.DataFrame:
        """Long frame of ECI per basket country (by ISO-3) and year ≥ ``start_year``."""
        iso3_to_iso2 = {c.iso3.lower(): c.iso2 for c in countries if c.on_map}
        if not iso3_to_iso2:
            return _empty_long()
        text = self._fetch_text(OEC_URL, use_cache=use_cache)
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

    def _fetch_text(self, url: str, use_cache: bool, attempts: int = 3, backoff_base: float = 1.0) -> str:
        cache_path = self._cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                return cache_path.read_text()
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.get(url, timeout=DEFAULT_TIMEOUT)
                if resp.status_code in (403, 404):
                    raise ValueError(f"OEC refused or moved the endpoint ({resp.status_code}): {url}")
                if resp.status_code >= 500:
                    raise RuntimeError(f"Server error {resp.status_code}: {url}")
                resp.raise_for_status()
                cache_path.write_text(resp.text)
                return resp.text
            except (ValueError, FileNotFoundError):
                raise
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < attempts - 1:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning("oec %s attempt %d/%d failed (%s) — retrying in %.1fs",
                                   url, attempt + 1, attempts, e, wait)
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _cache_path_for(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.json"


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
