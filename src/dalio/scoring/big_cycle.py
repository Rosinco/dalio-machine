"""Big-cycle qualitative inputs (Slice 17).

Honors Dalio's complete framework — internal-order, external-order,
reserve-currency lifecycle — *without* making them automated classifiers.

This module loads two slow-moving series for visual context:
- World Bank Gini coefficient (per country) — internal-disorder proxy.
- IMF COFER USD share of allocated FX reserves (global) — reserve-
  currency lifecycle proxy.

Neither feeds `compute_tilts` or `classify_*`. They render only as
charts in the dashboard's deeply-collapsed "The bigger picture"
expander, framed as "for completeness, not for action."

Both fetchers cache to disk (24h TTL) to avoid hammering APIs on every
dashboard render.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WB_BASE_URL = "https://api.worldbank.org/v2"
IMF_COFER_URL = (
    "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,COFER/"
    "G001.AFXRA.CI_USD.SHRO_PT.Q?startPeriod=2000"
)
DEFAULT_TIMEOUT = 30


# ISO2 → World Bank country code (mostly ISO3, with "EMU" for Eurozone)
ISO2_TO_WB: dict[str, str] = {
    "US": "USA",
    "CN": "CHN",
    "EU": "EMU",
    "UK": "GBR",
    "JP": "JPN",
    "SE": "SWE",
    "IN": "IND",
    "BR": "BRA",
}


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


class BigCycleSource:
    """Fetch Gini (per country) + COFER USD share (global) with on-disk cache."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
    ):
        self._client = client or requests.Session()
        self._cache_dir = cache_dir or Path(os.environ.get(
            "DALIO_BIG_CYCLE_CACHE", "data/cache/big_cycle"
        ))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600

    # ─── Public API ──────────────────────────────────────────────────────

    def fetch_gini(self, country_iso2: str, use_cache: bool = True) -> pd.DataFrame:
        """Annual Gini index for one country, long-format. Sparse — typically
        one survey per 1–3 years. Returns empty frame if no data is published.
        """
        wb_code = self._iso2_to_wb(country_iso2)
        url = f"{WB_BASE_URL}/country/{wb_code}/indicator/SI.POV.GINI?format=json&per_page=200"
        text = self._fetch_text(url, use_cache=use_cache)
        return self._parse_worldbank_json(text, country_iso2)

    def fetch_cofer_usd_share(self, use_cache: bool = True) -> pd.DataFrame:
        """Quarterly USD share of allocated FX reserves, world aggregate.
        Returns long-format with country='WLD', indicator='cofer_usd_share'.
        """
        text = self._fetch_text(IMF_COFER_URL, use_cache=use_cache)
        return self._parse_cofer_xml(text)

    # ─── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _iso2_to_wb(iso2: str) -> str:
        try:
            return ISO2_TO_WB[iso2.upper()]
        except KeyError as e:
            raise KeyError(f"No World Bank code mapping for {iso2!r}") from e

    def _fetch_text(
        self, url: str, use_cache: bool, attempts: int = 3, backoff_base: float = 1.0,
    ) -> str:
        cache_path = self._cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                logger.debug("big_cycle cache hit (age=%.0fs): %s", age, url)
                return cache_path.read_text()

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.get(url, timeout=DEFAULT_TIMEOUT)
                if resp.status_code == 404:
                    raise ValueError(f"Resource not found (404): {url}")
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
                    logger.warning(
                        "big_cycle %s attempt %d/%d failed (%s) — retrying in %.1fs",
                        url, attempt + 1, attempts, e, wait,
                    )
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _cache_path_for(self, url: str) -> Path:
        import hashlib
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.txt"

    @staticmethod
    def _parse_worldbank_json(text: str, country_iso2: str) -> pd.DataFrame:
        """World Bank returns [meta, [observations]]. Some country/indicator
        combinations return a single message dict instead — treat as empty.
        """
        import json
        data = json.loads(text)
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
            return _empty_long()
        rows = []
        for obs in data[1]:
            v = obs.get("value")
            if v is None:
                continue
            year = int(obs["date"])
            rows.append({
                "country": country_iso2,
                "indicator": "gini",
                "date": pd.Timestamp(year=year, month=12, day=31).date(),
                "value": float(v),
                "source": "WORLD_BANK",
                "series_id": "SI.POV.GINI",
            })
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_cofer_xml(text: str) -> pd.DataFrame:
        """Extract TIME_PERIOD/OBS_VALUE pairs from the SDMX-ML payload.

        The IMF endpoint returns StructureSpecificData XML; observations are
        attributes on `<Obs ... />` elements. Quarterly periods come as
        "YYYY-Q1"/"YYYY-Q2"/etc.
        """
        pattern = re.compile(
            r'TIME_PERIOD="([^"]+)"\s+OBS_VALUE="([^"]+)"'
        )
        rows = []
        for tp, val in pattern.findall(text):
            d = _quarter_to_date(tp)
            if d is None:
                continue
            try:
                rows.append({
                    "country": "WLD",
                    "indicator": "cofer_usd_share",
                    "date": d,
                    "value": float(val),
                    "source": "IMF_COFER",
                    "series_id": "G001.AFXRA.CI_USD.SHRO_PT.Q",
                })
            except ValueError:
                continue
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "country", "indicator", "date", "value", "source", "series_id",
    ])


def _quarter_to_date(period: str):
    """Convert '2024-Q3' → date(2024, 9, 30); 'YYYY' → date(YYYY, 12, 31)."""
    m = re.match(r"^(\d{4})-Q([1-4])$", period)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        month = q * 3
        last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return last.date()
    m = re.match(r"^(\d{4})$", period)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=12, day=31).date()
    return None
