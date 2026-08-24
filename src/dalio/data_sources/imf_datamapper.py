"""IMF DataMapper adapter (World Economic Outlook history + forecasts).

Keyless JSON endpoint, one call per indicator for ALL entities (the API
ignores country filters and returns ~230 entities — filter locally):

    https://www.imf.org/external/datamapper/api/v1/{INDICATOR}
    → {"values": {INDICATOR: {ENTITY: {"1980": 2.5, ..., "2031": 1.9}}}}

Entities are ISO-3 for countries plus group codes; the euro area is ``EURO``
(``Country.imf_id``). Years ≥ the current year are WEO projections and are
stored under the source tag ``IMF_WEO_FCST`` (history under ``IMF_WEO``) —
forecast rows are replaced wholesale each run because a vintage supersedes the
last one entirely.

Akamai may answer 403 from some networks; a User-Agent header is always sent.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
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

IMF_DM_BASE = "https://www.imf.org/external/datamapper/api/v1"
DEFAULT_TIMEOUT = 30
SOURCE_HISTORY = "IMF_WEO"
SOURCE_FORECAST = "IMF_WEO_FCST"


@dataclass(frozen=True)
class ImfSpec:
    indicator: str        # internal snake_case name
    imf_code: str         # DataMapper indicator id, e.g. "GGXWDG_NGDP"
    start_year: int = 1980


class ImfDataMapperSource:
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
            CachedTextFetcher.resolve_cache_dir(cache_dir, "DALIO_IMF_CACHE", "data/cache/imf"),
            cache_ttl_hours, label="IMF DataMapper", timeout=DEFAULT_TIMEOUT, suffix=".json",
            forbidden_hint="Akamai; try another network or the SDMX WEO fallback",
        )

    # ─── Public API ──────────────────────────────────────────────────────

    def fetch(
        self,
        spec: ImfSpec,
        countries: Sequence[Country],
        use_cache: bool = True,
        today: date | None = None,
    ) -> pd.DataFrame:
        """Long frame for every basket country with an ``imf_id``; years
        ≥ ``today.year`` tagged ``IMF_WEO_FCST``, earlier ``IMF_WEO``."""
        code_to_iso2 = {c.imf_id: c.iso2 for c in countries if c.imf_id}
        if not code_to_iso2:
            return _empty_long()
        url = f"{IMF_DM_BASE}/{spec.imf_code}"
        text = self._fetcher.fetch(url, use_cache=use_cache)
        values = self._parse(text, spec.imf_code)
        cutoff = (today or date.today()).year
        rows = []
        for code, series in values.items():
            iso2 = code_to_iso2.get(code)
            if iso2 is None:
                continue
            for year_s, v in series.items():
                try:
                    year = int(year_s)
                    value = float(v)
                except (TypeError, ValueError):
                    continue
                if year < spec.start_year or v is None:
                    continue
                rows.append({
                    "country": iso2,
                    "indicator": spec.indicator,
                    "date": date(year, 12, 31),
                    "value": value,
                    "source": SOURCE_FORECAST if year >= cutoff else SOURCE_HISTORY,
                    "series_id": spec.imf_code,
                })
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)

    # ─── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _parse(text: str, code: str) -> dict[str, dict[str, float]]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        values = data.get("values") if isinstance(data, dict) else None
        if not isinstance(values, dict):
            return {}
        block = values.get(code)
        return block if isinstance(block, dict) else {}



def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])


# ─── Derived ────────────────────────────────────────────────────────────────


def derive_interest_burden(primary: pd.DataFrame, overall: pd.DataFrame,
                           indicator_out: str = "interest_burden_pct_gdp") -> pd.DataFrame:
    """Net interest paid ≈ primary balance − overall balance (both % of GDP),
    aligned on (country, date, source). Rows without both inputs are dropped."""
    if primary.empty or overall.empty:
        return _empty_long()
    key = ["country", "date", "source"]
    m = primary[[*key, "value"]].merge(overall[[*key, "value"]], on=key, suffixes=("_p", "_o"))
    if m.empty:
        return _empty_long()
    out = m.assign(
        indicator=indicator_out,
        value=m["value_p"] - m["value_o"],
        series_id="primary-overall",
    )
    return out[["country", "indicator", "date", "value", "source", "series_id"]].reset_index(drop=True)


# ─── Fundamentals bundle ────────────────────────────────────────────────────

# WEO does not expose a primary balance in DataMapper; the Fiscal Monitor block
# does ("Primary net lending/borrowing", % of GDP, with projections). Verified
# against /indicators on 2026-08-24.
IMF_PRIMARY_BALANCE_CODE = "GGXONLB_G01_GDP_PT"

IMF_FUNDAMENTALS: tuple[ImfSpec, ...] = (
    ImfSpec("real_gdp_growth", "NGDP_RPCH"),                 # base for gdp_growth_fwd5
    ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP"),
    ImfSpec("fiscal_balance_pct_gdp", "GGXCNL_NGDP"),
    ImfSpec("primary_balance_pct_gdp", IMF_PRIMARY_BALANCE_CODE),
    ImfSpec("current_account_pct_gdp", "BCA_NGDPD"),
)
