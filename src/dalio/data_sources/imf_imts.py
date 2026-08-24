"""IMF IMTS adapter — bilateral goods trade by partner (the successor of DOTS).

Keyless SDMX 2.1 CSV endpoint, verified 2026-08-24:

    https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS/{REPORTERS}.{INDICATOR}.{COUNTERPARTS}.A
        ?startPeriod=YYYY            (Accept: application/vnd.sdmx.data+csv;version=1.0.0)

Dimensions ``COUNTRY.INDICATOR.COUNTERPART_COUNTRY.FREQUENCY``; ``+`` joins
codes, so the whole basket comes back in ONE call per flow (≈ 3 k rows, 2 MB).
Only four indicators exist — exports FOB, imports CIF/FOB, balance — there are
no energy-specific flows. ``OBS_VALUE`` is plain US dollars. Counterpart
``G001`` is the world total; ``G163`` is the euro area (as reporter and as
counterpart).

Storage needs no schema change: the partner is encoded in the indicator name
(``exports_to_CN`` / ``imports_from_WLD``), source ``IMF_IMTS``. For the
euro-area reporter, rows against its own member players (intra-union trade)
are dropped; its world total still includes intra-area trade — the UI says so.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from dalio.countries import ISO2_TO_IMTS, Country
from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    SDMX_CSV_ACCEPT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

IMTS_BASE = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS"
IMTS_ACCEPT = SDMX_CSV_ACCEPT
IMTS_WORLD = "G001"
WORLD_PARTNER = "WLD"
IMTS_YEARS = 7                  # latest + a 5-year lag with slack
DEFAULT_TIMEOUT = 120
SOURCE_IMTS = "IMF_IMTS"


@dataclass(frozen=True)
class ImtsSpec:
    flow: Literal["exports", "imports"]
    imts_code: str                # "XG_FOB_USD" | "MG_CIF_USD"
    indicator_prefix: str         # "exports_to" | "imports_from"

    def indicator(self, partner_iso2: str) -> str:
        return f"{self.indicator_prefix}_{partner_iso2}"


IMTS_FLOWS: tuple[ImtsSpec, ...] = (
    ImtsSpec("exports", "XG_FOB_USD", "exports_to"),
    ImtsSpec("imports", "MG_CIF_USD", "imports_from"),
)


class ImtsSource:
    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, IMTS_ACCEPT)
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(cache_dir, "DALIO_IMTS_CACHE", "data/cache/imts"),
            cache_ttl_hours, label="IMF IMTS", timeout=DEFAULT_TIMEOUT,
            forbidden_hint="Akamai; try another network",
        )

    # ─── Public API ──────────────────────────────────────────────────────

    def fetch(
        self,
        spec: ImtsSpec,
        countries: Sequence[Country],
        use_cache: bool = True,
        today: date | None = None,
        years: int = IMTS_YEARS,
    ) -> pd.DataFrame:
        """Long frame: one row per (reporter, partner, year) with the partner in
        the indicator name; the world total is partner ``WLD``. Counterparts
        outside the basket, reporter == partner rows, and the euro-area
        reporter's rows against its own member players are dropped."""
        code_to_iso2 = {ISO2_TO_IMTS[c.iso2]: c.iso2 for c in countries}
        if not code_to_iso2:
            return _empty_long()
        url = self.url_for(spec, list(code_to_iso2), (today or date.today()).year - years + 1)
        text = self._fetcher.fetch(url, use_cache=use_cache)
        raw = self._parse(text)
        if raw.empty:
            return _empty_long()
        by_iso2 = {c.iso2: c for c in countries}
        member_iso2 = {c.iso2 for c in countries if c.eu_member}
        rows = []
        for r in raw.itertuples(index=False):
            reporter = code_to_iso2.get(r.COUNTRY)
            partner = WORLD_PARTNER if r.COUNTERPART_COUNTRY == IMTS_WORLD else code_to_iso2.get(r.COUNTERPART_COUNTRY)
            if reporter is None or partner is None or reporter == partner:
                continue
            if by_iso2[reporter].members and partner in member_iso2:
                continue                                  # intra-union trade of the aggregate
            rows.append({
                "country": reporter,
                "indicator": spec.indicator(partner),
                "date": date(int(r.TIME_PERIOD), 12, 31),
                "value": float(r.OBS_VALUE),
                "source": SOURCE_IMTS,
                "series_id": f"{spec.imts_code}/{r.COUNTERPART_COUNTRY}",
            })
        if not rows:
            return _empty_long()
        return pd.DataFrame(rows).sort_values(["country", "indicator", "date"]).reset_index(drop=True)

    @staticmethod
    def url_for(spec: ImtsSpec, codes: Sequence[str], start_year: int) -> str:
        reporters = "+".join(codes)
        counterparts = "+".join([*codes, IMTS_WORLD])
        return f"{IMTS_BASE}/{reporters}.{spec.imts_code}.{counterparts}.A?startPeriod={start_year}"

    # ─── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return pd.DataFrame()
        need = ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "OBS_VALUE"]
        if df.empty or any(c not in df.columns for c in need):
            return pd.DataFrame()
        df = df[need].dropna(subset=["OBS_VALUE", "TIME_PERIOD"])
        df = df[df["TIME_PERIOD"].str.fullmatch(r"\d{4}")]
        return df



def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
