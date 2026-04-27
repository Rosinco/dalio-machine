"""BIS (Bank for International Settlements) data source adapter.

Pulls Total Credit and Debt Service Ratios via the BIS SDMX REST API in CSV
format. The endpoint accepts a dot-separated dimension key, e.g.
    Q.US.P.A.M.770.A
which means quarterly, United States, Private non-financial sector,
All lender sectors, Market valuation, % of GDP (770), Adjusted for breaks.

Documentation: https://stats.bis.org/statx/toc/LBS.html
"""
from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

logger = logging.getLogger(__name__)


BIS_BASE_URL = "https://stats.bis.org/api/v1/data"
DEFAULT_TIMEOUT = 30


# Sector codes used in WS_TC dimension TC_BORROWERS
class Sector:
    """BIS Total Credit sector codes (TC_BORROWERS)."""
    NON_FIN_TOTAL = "C"      # All non-financial = Government + Households + Corps
    GOVERNMENT = "G"         # General government
    PRIVATE_NON_FIN = "P"    # Private non-financial = Households + Non-fin corps
    HOUSEHOLDS = "H"
    NON_FIN_CORPS = "N"


# ISO2 → BIS country code (mostly identical, a few aggregates differ)
ISO2_TO_BIS: dict[str, str] = {
    "US": "US",
    "CN": "CN",
    "EU": "XM",   # Euro area aggregate
    "UK": "GB",
    "JP": "JP",
    "SE": "SE",
    "IN": "IN",
    "BR": "BR",
}


@dataclass(frozen=True)
class TotalCreditSpec:
    """One time-series request from the BIS WS_TC dataset."""
    indicator: str           # internal indicator name (snake_case)
    country: str             # ISO2 — adapter translates to BIS code
    sector: str              # Sector.* code (G/P/H/N/A)
    valuation: str = "M"     # M = market value, N = nominal
    unit_type: str = "770"   # 770 = % of GDP
    adjust: str = "A"        # A = adjusted for breaks


@dataclass(frozen=True)
class DsrSpec:
    """One time-series request from the BIS WS_DSR dataset."""
    indicator: str           # internal indicator name
    country: str             # ISO2
    borrower: str = "P"      # P = private non-financial sector


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


class BisSource:
    """Fetches BIS Total Credit + DSR series and returns long-format DataFrames.

    Uses an on-disk cache (default `data/cache/bis/`) to avoid re-downloading
    the same series within a window. Cache TTL is enforced loosely: we only
    consult the cache if `use_cache=True` is passed at fetch time.
    """

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
    ):
        self._client = client or requests.Session()
        self._cache_dir = cache_dir or Path(os.environ.get(
            "DALIO_BIS_CACHE", "data/cache/bis"
        ))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600

    # ─── Public API ──────────────────────────────────────────────────────

    def fetch_total_credit(self, spec: TotalCreditSpec, use_cache: bool = True) -> pd.DataFrame:
        bis_country = self._iso2_to_bis(spec.country)
        key = f"Q.{bis_country}.{spec.sector}.A.{spec.valuation}.{spec.unit_type}.{spec.adjust}"
        url = f"{BIS_BASE_URL}/WS_TC/{key}?format=csv"
        csv_text = self._fetch_csv(url, use_cache=use_cache)
        return self._parse_to_long(csv_text, spec.country, spec.indicator, "BIS_TC", url)

    def fetch_dsr(self, spec: DsrSpec, use_cache: bool = True) -> pd.DataFrame:
        bis_country = self._iso2_to_bis(spec.country)
        # DSR key: FREQ.BORROWERS_CTY.BORROWERS_SECTOR
        key = f"Q.{bis_country}.{spec.borrower}"
        url = f"{BIS_BASE_URL}/WS_DSR/{key}?format=csv"
        csv_text = self._fetch_csv(url, use_cache=use_cache)
        return self._parse_to_long(csv_text, spec.country, spec.indicator, "BIS_DSR", url)

    # ─── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _iso2_to_bis(iso2: str) -> str:
        try:
            return ISO2_TO_BIS[iso2.upper()]
        except KeyError as e:
            raise KeyError(f"No BIS code mapping for {iso2!r}") from e

    def _fetch_csv(
        self, url: str, use_cache: bool, attempts: int = 3, backoff_base: float = 1.0,
    ) -> str:
        cache_path = self._cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                logger.debug("BIS cache hit (age=%.0fs): %s", age, url)
                return cache_path.read_text()

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.get(url, timeout=DEFAULT_TIMEOUT)
                if resp.status_code == 404:
                    raise ValueError(f"BIS series not found (404): {url}")
                if resp.status_code >= 500:
                    raise RuntimeError(f"BIS server error {resp.status_code}: {url}")
                resp.raise_for_status()
                cache_path.write_text(resp.text)
                return resp.text
            except (ValueError, FileNotFoundError):
                raise  # don't retry permanent errors
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < attempts - 1:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning(
                        "BIS %s attempt %d/%d failed (%s) — retrying in %.1fs",
                        url, attempt + 1, attempts, e, wait,
                    )
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _cache_path_for(self, url: str) -> Path:
        # Hash URL into a stable filename
        import hashlib
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.csv"

    @staticmethod
    def _parse_to_long(
        csv_text: str, country: str, indicator: str, source: str, series_id: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return pd.DataFrame(columns=[
                "country", "indicator", "date", "value", "source", "series_id",
            ])

        # BIS time periods come as "2025-Q3", "2025-M01", "2025"; convert quarter to date
        df = df.dropna(subset=["OBS_VALUE"])
        df["date"] = df["TIME_PERIOD"].apply(BisSource._period_to_date)
        df = df.dropna(subset=["date"])
        df["country"] = country
        df["indicator"] = indicator
        df["source"] = source
        # Truncate the URL to keep series_id reasonable (the dimension key only)
        df["series_id"] = series_id.split("/")[-1].split("?")[0][:64]
        df = df.rename(columns={"OBS_VALUE": "value"})
        df["value"] = df["value"].astype(float)
        return df[["country", "indicator", "date", "value", "source", "series_id"]]

    @staticmethod
    def _period_to_date(period: str):
        """Convert BIS TIME_PERIOD ('2025-Q3', '2025-M01', '2025') to date."""
        from datetime import date as _date
        if not isinstance(period, str):
            return None
        if "-Q" in period:
            year, q = period.split("-Q")
            month = (int(q) - 1) * 3 + 1
            return _date(int(year), month, 1)
        if "-M" in period:
            year, m = period.split("-M")
            return _date(int(year), int(m), 1)
        if period.isdigit():
            return _date(int(period), 1, 1)
        return None


# ─── Tier-1 long-term cycle bundle ───────────────────────────────────────

# Per-country sector-debt indicators we want for the long-term cycle.
# All Tier-1 countries are covered by BIS Total Credit dataset.
TIER_1_TOTAL_CREDIT: tuple[TotalCreditSpec, ...] = tuple(
    spec
    for country in ("US", "CN", "EU", "UK", "JP", "SE")
    for spec in (
        TotalCreditSpec("total_credit_pct_gdp", country, Sector.NON_FIN_TOTAL),
        TotalCreditSpec("gov_debt_pct_gdp", country, Sector.GOVERNMENT),
        TotalCreditSpec("private_nonfin_pct_gdp", country, Sector.PRIVATE_NON_FIN),
        TotalCreditSpec("hh_debt_pct_gdp", country, Sector.HOUSEHOLDS),
        TotalCreditSpec("corp_debt_pct_gdp", country, Sector.NON_FIN_CORPS),
    )
)

TIER_1_DSR: tuple[DsrSpec, ...] = tuple(
    DsrSpec("debt_service_ratio", country)
    for country in ("US", "CN", "EU", "UK", "JP", "SE")
)


# ─── Tier-2 long-term cycle bundle ───────────────────────────────────────

# India + Brazil. BIS Total Credit covers both with the same sector taxonomy.
# Brazil's DSR series is incomplete in BIS (gov debt covered, private DSR
# patchy) — failing to fetch is documented as an expected gap, like JP CPI.
TIER_2_TOTAL_CREDIT: tuple[TotalCreditSpec, ...] = tuple(
    spec
    for country in ("IN", "BR")
    for spec in (
        TotalCreditSpec("total_credit_pct_gdp", country, Sector.NON_FIN_TOTAL),
        TotalCreditSpec("gov_debt_pct_gdp", country, Sector.GOVERNMENT),
        TotalCreditSpec("private_nonfin_pct_gdp", country, Sector.PRIVATE_NON_FIN),
        TotalCreditSpec("hh_debt_pct_gdp", country, Sector.HOUSEHOLDS),
        TotalCreditSpec("corp_debt_pct_gdp", country, Sector.NON_FIN_CORPS),
    )
)

TIER_2_DSR: tuple[DsrSpec, ...] = tuple(
    DsrSpec("debt_service_ratio", country)
    for country in ("IN", "BR")
)


# Aggregated convenience constants for callers that want everything.
ALL_TOTAL_CREDIT: tuple[TotalCreditSpec, ...] = TIER_1_TOTAL_CREDIT + TIER_2_TOTAL_CREDIT
ALL_DSR: tuple[DsrSpec, ...] = TIER_1_DSR + TIER_2_DSR
