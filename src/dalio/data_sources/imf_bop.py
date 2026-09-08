"""IMF Balance of Payments transactions: where cross-border money moved.

Official, keyless SDMX 2.1 CSV endpoint, verified 2026-09-08::

    https://api.imf.org/external/sdmx/2.1/data/IMF.STA,BOP/
        {COUNTRY}.{BOP_ACCOUNTING_ENTRY}.{INDICATOR}.{UNIT}.{FREQUENCY}

The accounting entry is part of every internal indicator name and native
``series_id``.  This is not cosmetic: under BPM6 a positive resident net
acquisition of foreign assets (``A_NFA_T``) is an outward placement, while a
positive net incurrence of liabilities (``L_NIL_T``) is inward financing.
``NNAFANIL_T`` is assets less liabilities, so positive means net lending to the
rest of the world.  These transaction flows must never be reconstructed from
changes in IIP/CPIS/CDIS positions, which also contain valuation effects.

``OBS_VALUE`` is already returned in the requested base unit (USD here); the
IMF ``SCALE`` column is display metadata and is deliberately not applied a
second time.  The query requests SDMX ``detail=dataonly`` because the default
response repeats several kilobytes of series documentation for every
observation.  Native observation-status codes are retained when supplied;
blank status means simply ``observed``.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dalio.countries import COUNTRIES, Country
from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    SDMX_CSV_ACCEPT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

BOP_BASE_URL = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,BOP"
BOP_SOURCE = "IMF_BOP"
BOP_HISTORY_START_YEAR = 2000
DEFAULT_TIMEOUT = 180


@dataclass(frozen=True)
class BopSeriesSpec:
    """One explicit IMF BOP transaction series.

    ``indicator`` is the stable internal name.  ``accounting_entry`` and
    ``imf_indicator`` are the first-hand IMF codes and remain embedded in the
    native series id for auditability.
    """

    indicator: str
    accounting_entry: str
    imf_indicator: str
    unit: str = "USD"
    frequency: str = "Q"

    @property
    def series_id(self) -> str:
        return (
            f"BOP/{self.accounting_entry}/{self.imf_indicator}/"
            f"{self.unit}/{self.frequency}"
        )

    @property
    def native_key_suffix(self) -> tuple[str, str, str, str]:
        return self.accounting_entry, self.imf_indicator, self.unit, self.frequency


def _asset_liability_net(prefix: str, imf_indicator: str) -> tuple[BopSeriesSpec, ...]:
    return (
        BopSeriesSpec(f"{prefix}_assets_flow_usd", "A_NFA_T", imf_indicator),
        BopSeriesSpec(f"{prefix}_liabilities_flow_usd", "L_NIL_T", imf_indicator),
        BopSeriesSpec(f"{prefix}_net_flow_usd", "NNAFANIL_T", imf_indicator),
    )


def _asset_liability(prefix: str, imf_indicator: str) -> tuple[BopSeriesSpec, ...]:
    return (
        BopSeriesSpec(f"{prefix}_assets_flow_usd", "A_NFA_T", imf_indicator),
        BopSeriesSpec(f"{prefix}_liabilities_flow_usd", "L_NIL_T", imf_indicator),
    )


# Minimal useful BPM6 financial-account bundle.  Totals retain all three
# published entries; instrument splits retain their separately published asset
# and liability legs.  Derivatives and the financial-account balance are
# published as net series in the current IMF flow.
BOP_SERIES: tuple[BopSeriesSpec, ...] = (
    *_asset_liability_net("direct_investment", "D_F"),
    *_asset_liability("direct_equity", "D_F5"),
    *_asset_liability("direct_debt", "D_FL"),
    *_asset_liability_net("portfolio_investment", "P_F"),
    *_asset_liability("portfolio_equity", "P_F5"),
    *_asset_liability("portfolio_debt", "P_F3"),
    *_asset_liability_net("other_investment", "O_F"),
    *_asset_liability_net("other_deposits", "O_F2"),
    *_asset_liability_net("other_loans", "O_F4"),
    BopSeriesSpec("financial_derivatives_net_flow_usd", "NNAFANIL_T", "F_F7"),
    BopSeriesSpec("financial_account_net_flow_usd", "NNAFANIL_T", "FAB"),
)

# The IMF BOP flow has country reporters but no euro-area aggregate series for
# this selection.  Member countries remain available; excluding the aggregate
# also prevents double counting in global flow summaries.
BOP_COUNTRIES: tuple[Country, ...] = tuple(
    country for country in COUNTRIES if country.imf_id and country.on_map
)

_OUTPUT_COLUMNS = [
    "country",
    "indicator",
    "date",
    "value",
    "source",
    "series_id",
    "status",
]
_REQUIRED_COLUMNS = {
    "DATAFLOW",
    "COUNTRY",
    "BOP_ACCOUNTING_ENTRY",
    "INDICATOR",
    "UNIT",
    "FREQUENCY",
    "TIME_PERIOD",
    "OBS_VALUE",
    "STATUS",
}


class ImfBopSource:
    """Fetch a complete multi-country bundle and return long observations."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, SDMX_CSV_ACCEPT)
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(
                cache_dir,
                "DALIO_IMF_BOP_CACHE",
                "data/cache/imf_bop",
            ),
            cache_ttl_hours,
            label="IMF BOP",
            timeout=DEFAULT_TIMEOUT,
            forbidden_hint="Akamai; try another network",
        )

    def fetch(
        self,
        countries: Sequence[Country] = BOP_COUNTRIES,
        *,
        specs: Sequence[BopSeriesSpec] = BOP_SERIES,
        start_year: int = BOP_HISTORY_START_YEAR,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return quarterly USD transactions for the requested explicit series."""
        mapped = tuple(country for country in countries if country.imf_id)
        selected_specs = tuple(specs)
        if not mapped or not selected_specs:
            return _empty_long()
        if isinstance(start_year, bool) or not isinstance(start_year, int) or start_year < 1:
            raise ValueError("start_year must be a positive integer")

        url = self.url_for(mapped, specs=selected_specs, start_year=start_year)
        text = self._fetcher.fetch(url, use_cache=use_cache)
        return self._parse(text, mapped, selected_specs)

    @staticmethod
    def url_for(
        countries: Sequence[Country],
        *,
        specs: Sequence[BopSeriesSpec] = BOP_SERIES,
        start_year: int = BOP_HISTORY_START_YEAR,
    ) -> str:
        mapped_ids = _ordered_unique(
            country.imf_id for country in countries if country.imf_id
        )
        selected_specs = tuple(specs)
        if not mapped_ids:
            raise ValueError("IMF BOP query needs at least one mapped country")
        if not selected_specs:
            raise ValueError("IMF BOP query needs at least one series")
        if isinstance(start_year, bool) or not isinstance(start_year, int) or start_year < 1:
            raise ValueError("start_year must be a positive integer")

        entries = _ordered_unique(spec.accounting_entry for spec in selected_specs)
        indicators = _ordered_unique(spec.imf_indicator for spec in selected_specs)
        units = _ordered_unique(spec.unit for spec in selected_specs)
        frequencies = _ordered_unique(spec.frequency for spec in selected_specs)
        key = ".".join(
            "+".join(values)
            for values in (mapped_ids, entries, indicators, units, frequencies)
        )
        return f"{BOP_BASE_URL}/{key}?startPeriod={start_year}&detail=dataonly"

    @staticmethod
    def _parse(
        text: str,
        countries: Sequence[Country],
        specs: Sequence[BopSeriesSpec],
    ) -> pd.DataFrame:
        try:
            raw = pd.read_csv(io.StringIO(text), dtype=str)
        except pd.errors.EmptyDataError:
            return _empty_long()
        except pd.errors.ParserError as exc:
            raise ValueError("IMF BOP returned malformed CSV") from exc

        missing = _REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"IMF BOP snapshot missing columns: {sorted(missing)}")
        if raw.empty:
            return _empty_long()
        dataflows = raw["DATAFLOW"].dropna().astype(str)
        if dataflows.empty or not dataflows.str.startswith("IMF.STA:BOP(").all():
            raise ValueError("IMF BOP snapshot contains an unexpected dataflow")

        id_to_iso2 = {
            str(country.imf_id): country.iso2 for country in countries if country.imf_id
        }
        spec_by_native = {spec.native_key_suffix: spec for spec in specs}
        if len(spec_by_native) != len(tuple(specs)):
            raise ValueError("IMF BOP specs contain a duplicate native series")

        work = raw[
            raw["COUNTRY"].isin(id_to_iso2)
            & raw.apply(
                lambda row: (
                    row["BOP_ACCOUNTING_ENTRY"],
                    row["INDICATOR"],
                    row["UNIT"],
                    row["FREQUENCY"],
                )
                in spec_by_native,
                axis=1,
            )
        ].copy()
        if work.empty:
            return _empty_long()

        values = pd.to_numeric(work["OBS_VALUE"], errors="coerce")
        supplied = work["OBS_VALUE"].notna() & work["OBS_VALUE"].astype(str).str.strip().ne("")
        if values[supplied].isna().any():
            raise ValueError("IMF BOP snapshot contains a non-numeric observation")
        work["value"] = values
        work = work.dropna(subset=["value"])
        if work.empty:
            return _empty_long()
        if not np.isfinite(work["value"].astype(float)).all():
            raise ValueError("IMF BOP snapshot contains a non-finite observation")

        rows: list[dict] = []
        for row in work.itertuples(index=False):
            observed_on = _quarter_to_date(row.TIME_PERIOD)
            spec = spec_by_native[
                (row.BOP_ACCOUNTING_ENTRY, row.INDICATOR, row.UNIT, row.FREQUENCY)
            ]
            native_status = "" if pd.isna(row.STATUS) else str(row.STATUS).strip()
            if len(native_status) > 24:
                raise ValueError("IMF BOP native observation status is too long")
            rows.append(
                {
                    "country": id_to_iso2[row.COUNTRY],
                    "indicator": spec.indicator,
                    "date": observed_on,
                    "value": float(row.value),
                    "source": BOP_SOURCE,
                    "series_id": spec.series_id,
                    "status": native_status or "observed",
                }
            )

        frame = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
        duplicate_key = ["country", "indicator", "date", "source"]
        if frame.duplicated(duplicate_key, keep=False).any():
            raise ValueError("IMF BOP snapshot contains duplicate observations")
        return frame.sort_values(
            ["country", "indicator", "date"], kind="stable"
        ).reset_index(drop=True)


def _quarter_to_date(period: str) -> date:
    if not isinstance(period, str) or len(period) != 7 or period[4:6] != "-Q":
        raise ValueError(f"IMF BOP invalid quarterly period: {period!r}")
    try:
        year = int(period[:4])
        quarter = int(period[6])
        if quarter not in (1, 2, 3, 4):
            raise ValueError
        return date(year, (quarter - 1) * 3 + 1, 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"IMF BOP invalid quarterly period: {period!r}") from exc


def _ordered_unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)
