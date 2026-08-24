"""World Bank API v2 adapter (WDI source 2, WGI source 3).

Keyless JSON REST API. One request per (indicator, page) for the whole
country basket:

    {WB_BASE_URL}/country/USA;CHN;...;WLD/indicator/{code}
        ?format=json&per_page=1000&date=1960:2026&source=2&page=1

Rules learned the hard way (slice 18):

* Never use ``mrv=1`` — it returns the latest *row* even when its value is
  null. Fetch a date range and keep every non-null year; consumers pick the
  latest non-null per country.
* WGI series live in source 3 with ids like ``GOV_WGI_RL.EST`` /
  ``GOV_WGI_RL.SE`` (standard error). Passing ``source=3`` is mandatory.
* Aggregates (``EMU`` euro area, ``WLD`` world) come back with
  ``countryiso3code`` set to the aggregate code.
* Some indicator/country combinations return a single message dict instead
  of ``[meta, rows]`` — treated as empty.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from dalio.countries import Country

logger = logging.getLogger(__name__)

WB_BASE_URL = "https://api.worldbank.org/v2"
DEFAULT_TIMEOUT = 30
WORLD_CODE = "WLD"
PER_PAGE = 1000

SOURCE_LABELS = {2: "WORLD_BANK", 3: "WORLD_BANK_WGI"}


@dataclass(frozen=True)
class WbIndicatorSpec:
    """One World Bank indicator to pull for the whole basket."""
    indicator: str            # internal snake_case name
    wb_code: str              # e.g. "NY.GDP.PCAP.PP.KD"
    source_id: int = 2        # 2 = WDI, 3 = WGI
    start_year: int = 1960
    end_year: int | None = None      # None → current year
    include_world: bool = False      # also fetch the WLD aggregate (for world shares)

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source_id, f"WORLD_BANK_{self.source_id}")


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


class WorldBankSource:
    """Fetch World Bank indicators for many countries in one paginated call."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        page_pause_seconds: float = 0.5,
    ):
        self._client = client or requests.Session()
        self._cache_dir = cache_dir or Path(os.environ.get(
            "DALIO_WB_CACHE", "data/cache/worldbank"
        ))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._page_pause = page_pause_seconds

    # ─── Public API ──────────────────────────────────────────────────────

    def fetch(
        self,
        spec: WbIndicatorSpec,
        countries: Sequence[Country],
        use_cache: bool = True,
        today: date | None = None,
    ) -> pd.DataFrame:
        """Long-format frame ``country, indicator, date, value, source, series_id``
        for every non-null observation of ``spec`` across ``countries``
        (plus ``WLD`` if ``spec.include_world``). ISO2 codes are the registry's;
        the world aggregate is reported as country ``"WLD"``.
        """
        wb_to_iso2 = {c.wb_id: c.iso2 for c in countries if c.wb_id}
        if not wb_to_iso2:
            return _empty_long()
        codes = list(wb_to_iso2)
        if spec.include_world:
            codes.append(WORLD_CODE)
            wb_to_iso2[WORLD_CODE] = WORLD_CODE

        end_year = spec.end_year or (today or date.today()).year
        rows: list[dict] = []
        page = 1
        while True:
            url = self._url(spec, codes, page, end_year)
            text = self._fetch_text(url, use_cache=use_cache)
            meta, obs = self._parse_page(text)
            rows.extend(self._rows_from(obs, spec, wb_to_iso2))
            pages = int(meta.get("pages", 1) or 1)
            if page >= pages:
                break
            page += 1
            if self._page_pause:
                time.sleep(self._page_pause)

        if not rows:
            return _empty_long()
        return (
            pd.DataFrame(rows)
            .sort_values(["country", "date"])
            .reset_index(drop=True)
        )

    # ─── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _url(spec: WbIndicatorSpec, codes: Sequence[str], page: int, end_year: int) -> str:
        return (
            f"{WB_BASE_URL}/country/{';'.join(codes)}/indicator/{spec.wb_code}"
            f"?format=json&per_page={PER_PAGE}&date={spec.start_year}:{end_year}"
            f"&source={spec.source_id}&page={page}"
        )

    @staticmethod
    def _parse_page(text: str) -> tuple[dict, list[dict]]:
        """Return ``(meta, observations)``; the single-message error shape and
        anything malformed yield ``({}, [])``."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}, []
        if (
            not isinstance(data, list) or len(data) < 2
            or not isinstance(data[0], dict) or not isinstance(data[1], list)
        ):
            return {}, []
        return data[0], data[1]

    @staticmethod
    def _rows_from(obs: list[dict], spec: WbIndicatorSpec, wb_to_iso2: dict[str, str]) -> list[dict]:
        rows = []
        unknown: set[str] = set()
        for o in obs:
            v = o.get("value")
            if v is None:
                continue
            # World Bank populates ``countryiso3code`` for countries AND aggregates
            # (EMU, WLD — verified on live pages); ``country.id`` is a different,
            # 2-letter namespace (US, GB, XC, 1W) and is deliberately not used.
            code = o.get("countryiso3code") or ""
            iso2 = wb_to_iso2.get(code)
            if iso2 is None:
                unknown.add(code or "<blank>")
                continue
            try:
                year = int(o["date"])
                value = float(v)
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "country": iso2,
                "indicator": spec.indicator,
                "date": date(year, 12, 31),
                "value": value,
                "source": spec.source_label,
                "series_id": spec.wb_code,
            })
        if unknown:
            logger.warning("worldbank %s: dropped rows for unrequested/unknown codes %s",
                           spec.wb_code, sorted(unknown))
        return rows

    def _fetch_text(
        self, url: str, use_cache: bool, attempts: int = 3, backoff_base: float = 1.0,
    ) -> str:
        cache_path = self._cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                logger.debug("worldbank cache hit (age=%.0fs): %s", age, url)
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
                        "worldbank %s attempt %d/%d failed (%s) — retrying in %.1fs",
                        url, attempt + 1, attempts, e, wait,
                    )
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _cache_path_for(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.json"


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "country", "indicator", "date", "value", "source", "series_id",
    ])


# ─── Derived series ──────────────────────────────────────────────────────────

WORLD_SHARE_SUFFIX = "÷WLD"
MEMBER_MEAN_SUFFIX = ":member-mean"


def derive_world_share(df: pd.DataFrame, indicator_out: str) -> pd.DataFrame:
    """``country ÷ WLD × 100`` per year → a new long frame for ``indicator_out``.

    Input is one indicator's long frame that includes the ``WLD`` pseudo-country
    (fetch with ``include_world=True``). Rows without a matching world value
    are dropped; the world row itself is not emitted.
    """
    if df.empty:
        return _empty_long()
    world = df[df["country"] == WORLD_CODE].set_index("date")["value"]
    rest = df[df["country"] != WORLD_CODE].copy()
    rest["_w"] = rest["date"].map(world)
    rest = rest[rest["_w"].notna() & (rest["_w"] != 0)]
    if rest.empty:
        return _empty_long()
    out = rest.assign(
        indicator=indicator_out,
        value=rest["value"] / rest["_w"] * 100.0,
        series_id=rest["series_id"] + WORLD_SHARE_SUFFIX,
    ).drop(columns="_w")
    return out[["country", "indicator", "date", "value", "source", "series_id"]].reset_index(drop=True)


def derive_member_mean(
    df: pd.DataFrame,
    members: Sequence[str],
    out_country: str,
    min_members: int = 3,
) -> pd.DataFrame:
    """Unweighted mean over ``members`` per (indicator, date) → rows for
    ``out_country`` (the euro-area aggregate, which WGI does not publish).

    Flagged in ``series_id`` (``:member-mean``) so the UI can say so. For
    standard-error siblings this is an approximation, not a pooled SE.
    """
    sub = df[df["country"].isin(members)]
    if sub.empty:
        return _empty_long()
    g = sub.groupby(["indicator", "date", "source"], as_index=False).agg(
        value=("value", "mean"), n=("value", "size"), series_id=("series_id", "first"),
    )
    g = g[g["n"] >= min_members].drop(columns="n")
    if g.empty:
        return _empty_long()
    g["country"] = out_country
    g["series_id"] = g["series_id"] + MEMBER_MEAN_SUFFIX
    return g[["country", "indicator", "date", "value", "source", "series_id"]].reset_index(drop=True)


# ─── Fundamentals bundle ─────────────────────────────────────────────────────

# Raw pulls. Derived indicators (world shares, EU member-means) are produced by
# the pipeline from these via the helpers above.
WB_FUNDAMENTALS: tuple[WbIndicatorSpec, ...] = (
    # real stuff
    WbIndicatorSpec("energy_net_imports_pct", "EG.IMP.CONS.ZS", start_year=1960),
    WbIndicatorSpec("old_age_dependency", "SP.POP.DPND.OL", start_year=1960),
    # production
    WbIndicatorSpec("gdp_pc_ppp", "NY.GDP.PCAP.PP.KD", start_year=1990),
    WbIndicatorSpec("gdp_usd", "NY.GDP.MKTP.CD", start_year=1960),          # bubble size only
    WbIndicatorSpec("rd_pct_gdp", "GB.XPD.RSDV.GD.ZS", start_year=1996),
    # exchange
    WbIndicatorSpec("exports_usd", "NE.EXP.GNFS.CD", start_year=1960, include_world=True),
    WbIndicatorSpec("current_account_pct_gdp", "BN.CAB.XOKA.GD.ZS", start_year=1960),
    WbIndicatorSpec("reserves_months_imports", "FI.RES.TOTL.MO", start_year=1960),
    # enforcer
    WbIndicatorSpec("military_pct_gdp", "MS.MIL.XPND.GD.ZS", start_year=1960),
    WbIndicatorSpec("military_usd", "MS.MIL.XPND.CD", start_year=1960, include_world=True),
    WbIndicatorSpec("rule_of_law", "GOV_WGI_RL.EST", source_id=3, start_year=1996),
    WbIndicatorSpec("rule_of_law_se", "GOV_WGI_RL.SE", source_id=3, start_year=1996),
    WbIndicatorSpec("political_stability", "GOV_WGI_PV.EST", source_id=3, start_year=1996),
    WbIndicatorSpec("political_stability_se", "GOV_WGI_PV.SE", source_id=3, start_year=1996),
)

# (raw indicator → derived indicator) world shares computed after fetch
WB_WORLD_SHARES: tuple[tuple[str, str], ...] = (
    ("exports_usd", "exports_share_world"),
    ("military_usd", "military_share_world"),
)

# WGI has no euro-area aggregate: mean of the basket's members, flagged.
WB_MEMBER_MEAN_INDICATORS: tuple[str, ...] = (
    "rule_of_law", "rule_of_law_se", "political_stability", "political_stability_se",
)
