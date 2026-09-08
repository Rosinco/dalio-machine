"""IMF bilateral portfolio and direct-investment position adapters.

First-hand SDMX 2.1 datasets:

* ``PIP`` (formerly CPIS) reports portfolio assets held by residents of the
  reporter against an immediate counterpart economy. Only accounting entry
  ``A`` and total-economy sectors ``S1``/``S1`` are selected. Annual and
  semiannual histories remain distinct.
* ``DIP`` (formerly CDIS) reports direct-investment positions against an
  immediate counterpart economy. Only ``DV_TYPE=O`` (the reporter's own
  observation) is selected; ``SCC`` counterparty-derived mirror estimates are
  deliberately excluded. ``NETLA`` means liabilities less assets for inward
  positions and ``NETAL`` means assets less liabilities for outward positions.

The API's ``OBS_VALUE`` is already an unscaled USD amount. ``SCALE`` is display
metadata and is never multiplied into the value. Positions are stocks and must
not be described as transactions or money flows during a period.
"""

from __future__ import annotations

import io
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

IMF_POSITION_BASE_URL = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA"
PIP_SOURCE = "IMF_PIP"
DIP_SOURCE = "IMF_DIP"
PIP_HISTORY_START_YEAR = 1997
DIP_HISTORY_START_YEAR = 2009
PIP_FREQUENCIES = ("A", "S")
DIP_FREQUENCIES = ("A",)
DEFAULT_TIMEOUT = 180


@dataclass(frozen=True)
class PositionCounterpart:
    """One selected counterpart with internal and native IMF identifiers."""

    internal_code: str
    imf_code: str
    name: str


@dataclass(frozen=True)
class PositionSeriesSpec:
    """Explicit native IMF series and its non-lossy economic interpretation."""

    dataset: str
    indicator: str
    native_indicator: str
    direction: str
    accounting_basis: str
    instrument_code: str
    instrument_label: str
    unit: str = "USD"

    def series_id(
        self,
        reporter_code: str,
        counterpart_code: str,
        frequency: str,
    ) -> str:
        if self.dataset == "PIP":
            key = f"{reporter_code}.A.{self.native_indicator}.S1.S1.{counterpart_code}.{frequency}"
        elif self.dataset == "DIP":
            key = f"{reporter_code}.O.{self.native_indicator}.{counterpart_code}.{frequency}"
        else:  # pragma: no cover - catalogue construction prevents this
            raise ValueError(f"unsupported IMF position dataset: {self.dataset}")
        return f"{self.dataset}/{key}"


PIP_SERIES: tuple[PositionSeriesSpec, ...] = (
    PositionSeriesSpec(
        "PIP",
        "portfolio_total",
        "P_TOTINV_P_USD",
        "outward_assets",
        "assets",
        "portfolio_total",
        "Total portfolio investment",
    ),
    PositionSeriesSpec(
        "PIP",
        "portfolio_equity",
        "P_F51_P_USD",
        "outward_assets",
        "assets",
        "portfolio_equity",
        "Equity and investment fund shares",
    ),
    PositionSeriesSpec(
        "PIP",
        "portfolio_long_term_debt",
        "P_F3_L_P_USD",
        "outward_assets",
        "assets",
        "portfolio_long_term_debt",
        "Long-term debt securities",
    ),
    PositionSeriesSpec(
        "PIP",
        "portfolio_short_term_debt",
        "P_F3_S_P_USD",
        "outward_assets",
        "assets",
        "portfolio_short_term_debt",
        "Short-term debt securities",
    ),
)

DIP_SERIES: tuple[PositionSeriesSpec, ...] = (
    PositionSeriesSpec(
        "DIP",
        "direct_total_inward",
        "INWD_D_NETLA_FALL_ALL",
        "inward",
        "net_liabilities_less_assets",
        "direct_total",
        "Direct investment, all financial instruments",
    ),
    PositionSeriesSpec(
        "DIP",
        "direct_equity_inward",
        "INWD_D_NETLA_F51_ALL",
        "inward",
        "net_liabilities_less_assets",
        "direct_equity",
        "Direct investment equity",
    ),
    PositionSeriesSpec(
        "DIP",
        "direct_debt_inward",
        "INWD_D_NETLA_FL_ALL",
        "inward",
        "net_liabilities_less_assets",
        "direct_debt",
        "Direct investment debt",
    ),
    PositionSeriesSpec(
        "DIP",
        "direct_total_outward",
        "OTWD_D_NETAL_FALL_ALL",
        "outward",
        "net_assets_less_liabilities",
        "direct_total",
        "Direct investment, all financial instruments",
    ),
    PositionSeriesSpec(
        "DIP",
        "direct_equity_outward",
        "OTWD_D_NETAL_F51_ALL",
        "outward",
        "net_assets_less_liabilities",
        "direct_equity",
        "Direct investment equity",
    ),
    PositionSeriesSpec(
        "DIP",
        "direct_debt_outward",
        "OTWD_D_NETAL_FL_ALL",
        "outward",
        "net_assets_less_liabilities",
        "direct_debt",
        "Direct investment debt",
    ),
)

# Both reporter datasets returned no euro-area aggregate for the tested native
# selection. Its 21 constituent/other individual economies remain reporters;
# excluding the aggregate prevents double counting.
IMF_POSITION_COUNTRIES: tuple[Country, ...] = tuple(
    country for country in COUNTRIES if country.imf_id and country.on_map
)
IMF_POSITION_COUNTERPARTS: tuple[PositionCounterpart, ...] = (
    *(
        PositionCounterpart(country.iso2, str(country.imf_id), country.name)
        for country in IMF_POSITION_COUNTRIES
    ),
    PositionCounterpart("WLD", "G001", "World"),
)

_OUTPUT_COLUMNS = [
    "dataset",
    "reporter_country",
    "reporter_code",
    "counterpart_country",
    "counterpart_code",
    "date",
    "direction",
    "accounting_basis",
    "instrument_code",
    "instrument_label",
    "frequency",
    "value",
    "unit",
    "source",
    "native_indicator",
    "reporter_sector_code",
    "counterpart_sector_code",
    "derivation_type",
    "series_id",
    "status",
]
_PIP_REQUIRED_COLUMNS = {
    "DATAFLOW",
    "COUNTRY",
    "ACCOUNTING_ENTRY",
    "INDICATOR",
    "SECTOR",
    "COUNTERPART_SECTOR",
    "COUNTERPART_COUNTRY",
    "FREQUENCY",
    "TIME_PERIOD",
    "OBS_VALUE",
    "STATUS",
}
_DIP_REQUIRED_COLUMNS = {
    "DATAFLOW",
    "COUNTRY",
    "DV_TYPE",
    "INDICATOR",
    "COUNTERPART_COUNTRY",
    "FREQUENCY",
    "TIME_PERIOD",
    "OBS_VALUE",
    "STATUS",
}


class ImfPipSource:
    """Fetch one complete reporter/indicator/frequency PIP partition."""

    dataset = "PIP"

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
                cache_dir, "DALIO_IMF_PIP_CACHE", "data/cache/imf_pip"
            ),
            cache_ttl_hours,
            label="IMF PIP",
            timeout=DEFAULT_TIMEOUT,
            forbidden_hint="Akamai; try another network",
        )

    def fetch(
        self,
        reporter: Country,
        spec: PositionSeriesSpec,
        frequency: str,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = PIP_HISTORY_START_YEAR,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        return self.fetch_bundle(
            reporter,
            (spec,),
            (frequency,),
            counterparts=counterparts,
            start_year=start_year,
            use_cache=use_cache,
        )

    def fetch_bundle(
        self,
        reporter: Country,
        specs: Sequence[PositionSeriesSpec] = PIP_SERIES,
        frequencies: Sequence[str] = PIP_FREQUENCIES,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = PIP_HISTORY_START_YEAR,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch several native partitions in one bounded reporter request."""
        selected_specs = tuple(specs)
        selected_frequencies = tuple(str(value).upper() for value in frequencies)
        selected_counterparts = tuple(counterparts)
        url = self.bundle_url_for(
            reporter,
            selected_specs,
            selected_frequencies,
            counterparts=selected_counterparts,
            start_year=start_year,
        )
        text = self._fetcher.fetch(url, use_cache=use_cache)
        return _parse_pip_bundle(
            text,
            reporter,
            selected_specs,
            selected_frequencies,
            selected_counterparts,
        )

    @staticmethod
    def url_for(
        reporter: Country,
        spec: PositionSeriesSpec,
        frequency: str,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = PIP_HISTORY_START_YEAR,
    ) -> str:
        return ImfPipSource.bundle_url_for(
            reporter,
            (spec,),
            (frequency,),
            counterparts=counterparts,
            start_year=start_year,
        )

    @staticmethod
    def bundle_url_for(
        reporter: Country,
        specs: Sequence[PositionSeriesSpec] = PIP_SERIES,
        frequencies: Sequence[str] = PIP_FREQUENCIES,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = PIP_HISTORY_START_YEAR,
    ) -> str:
        reporter_code, selected_specs, selected_frequencies, selected = _validate_bundle_request(
            "PIP", reporter, specs, frequencies, counterparts, start_year
        )
        counterpart_key = "+".join(item.imf_code for item in selected)
        indicator_key = "+".join(spec.native_indicator for spec in selected_specs)
        frequency_key = "+".join(selected_frequencies)
        key = f"{reporter_code}.A.{indicator_key}.S1.S1.{counterpart_key}.{frequency_key}"
        return f"{IMF_POSITION_BASE_URL},PIP/{key}?startPeriod={start_year}&detail=dataonly"


class ImfDipSource:
    """Fetch one complete reporter/indicator annual DIP partition."""

    dataset = "DIP"

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
                cache_dir, "DALIO_IMF_DIP_CACHE", "data/cache/imf_dip"
            ),
            cache_ttl_hours,
            label="IMF DIP",
            timeout=DEFAULT_TIMEOUT,
            forbidden_hint="Akamai; try another network",
        )

    def fetch(
        self,
        reporter: Country,
        spec: PositionSeriesSpec,
        frequency: str = "A",
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = DIP_HISTORY_START_YEAR,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        return self.fetch_bundle(
            reporter,
            (spec,),
            (frequency,),
            counterparts=counterparts,
            start_year=start_year,
            use_cache=use_cache,
        )

    def fetch_bundle(
        self,
        reporter: Country,
        specs: Sequence[PositionSeriesSpec] = DIP_SERIES,
        frequencies: Sequence[str] = DIP_FREQUENCIES,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = DIP_HISTORY_START_YEAR,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch several native partitions in one bounded reporter request."""
        selected_specs = tuple(specs)
        selected_frequencies = tuple(str(value).upper() for value in frequencies)
        selected_counterparts = tuple(counterparts)
        url = self.bundle_url_for(
            reporter,
            selected_specs,
            selected_frequencies,
            counterparts=selected_counterparts,
            start_year=start_year,
        )
        text = self._fetcher.fetch(url, use_cache=use_cache)
        return _parse_dip_bundle(
            text,
            reporter,
            selected_specs,
            selected_frequencies,
            selected_counterparts,
        )

    @staticmethod
    def url_for(
        reporter: Country,
        spec: PositionSeriesSpec,
        frequency: str = "A",
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = DIP_HISTORY_START_YEAR,
    ) -> str:
        return ImfDipSource.bundle_url_for(
            reporter,
            (spec,),
            (frequency,),
            counterparts=counterparts,
            start_year=start_year,
        )

    @staticmethod
    def bundle_url_for(
        reporter: Country,
        specs: Sequence[PositionSeriesSpec] = DIP_SERIES,
        frequencies: Sequence[str] = DIP_FREQUENCIES,
        *,
        counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
        start_year: int = DIP_HISTORY_START_YEAR,
    ) -> str:
        reporter_code, selected_specs, selected_frequencies, selected = _validate_bundle_request(
            "DIP", reporter, specs, frequencies, counterparts, start_year
        )
        counterpart_key = "+".join(item.imf_code for item in selected)
        indicator_key = "+".join(spec.native_indicator for spec in selected_specs)
        frequency_key = "+".join(selected_frequencies)
        key = f"{reporter_code}.O.{indicator_key}.{counterpart_key}.{frequency_key}"
        return f"{IMF_POSITION_BASE_URL},DIP/{key}?startPeriod={start_year}&detail=dataonly"


def _validate_bundle_request(
    dataset: str,
    reporter: Country,
    specs: Sequence[PositionSeriesSpec],
    frequencies: Sequence[str],
    counterparts: Sequence[PositionCounterpart],
    start_year: int,
) -> tuple[
    str,
    tuple[PositionSeriesSpec, ...],
    tuple[str, ...],
    tuple[PositionCounterpart, ...],
]:
    reporters = {country.iso2 for country in IMF_POSITION_COUNTRIES}
    if reporter.iso2 not in reporters or not reporter.imf_id:
        raise ValueError(f"IMF {dataset} reporter must be one of the 21 individual countries")
    selected_specs = tuple(specs)
    if not selected_specs:
        raise ValueError(f"IMF {dataset} query needs at least one series")
    if any(spec.dataset != dataset for spec in selected_specs):
        raise ValueError(f"IMF {dataset} source received a different-dataset series")
    native_indicators = [spec.native_indicator for spec in selected_specs]
    if len(set(native_indicators)) != len(native_indicators):
        raise ValueError(f"IMF {dataset} series selection contains duplicates")

    selected_frequencies = tuple(str(value).upper() for value in frequencies)
    if not selected_frequencies:
        raise ValueError(f"IMF {dataset} query needs at least one frequency")
    allowed_frequencies = PIP_FREQUENCIES if dataset == "PIP" else DIP_FREQUENCIES
    if any(value not in allowed_frequencies for value in selected_frequencies):
        raise ValueError(f"IMF {dataset} frequencies must be within {allowed_frequencies}")
    if len(set(selected_frequencies)) != len(selected_frequencies):
        raise ValueError(f"IMF {dataset} frequency selection contains duplicates")
    if isinstance(start_year, bool) or not isinstance(start_year, int) or start_year < 1:
        raise ValueError("start_year must be a positive integer")

    selected = tuple(counterparts)
    if not selected:
        raise ValueError(f"IMF {dataset} query needs at least one counterpart")
    allowed = {(item.internal_code, item.imf_code) for item in IMF_POSITION_COUNTERPARTS}
    requested = [(item.internal_code, item.imf_code) for item in selected]
    if any(item not in allowed for item in requested):
        raise ValueError(f"IMF {dataset} query contains an unsupported counterpart")
    if len(set(requested)) != len(requested):
        raise ValueError(f"IMF {dataset} counterpart selection contains duplicates")
    return str(reporter.imf_id), selected_specs, selected_frequencies, selected


def _read_csv(text: str, required: set[str], dataset: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=sorted(required))
    except pd.errors.ParserError as exc:
        raise ValueError(f"IMF {dataset} returned malformed CSV") from exc
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"IMF {dataset} snapshot missing columns: {sorted(missing)}")
    if raw.empty:
        return raw
    dataflows = raw["DATAFLOW"].fillna("").astype(str).str.strip()
    present = dataflows.ne("")
    if not present.any() or not dataflows[present].str.startswith(f"IMF.STA:{dataset}(").all():
        raise ValueError(f"IMF {dataset} snapshot contains an unexpected dataflow")
    payload_columns = required - {"DATAFLOW", "STATUS"}
    payload = raw.loc[:, sorted(payload_columns)].notna().any(axis=1)
    if (payload & ~present).any():
        raise ValueError(f"IMF {dataset} snapshot contains an unexpected dataflow")
    return raw


def _parse_pip_bundle(
    text: str,
    reporter: Country,
    specs: tuple[PositionSeriesSpec, ...],
    frequencies: tuple[str, ...],
    counterparts: Sequence[PositionCounterpart],
) -> pd.DataFrame:
    raw = _read_csv(text, _PIP_REQUIRED_COLUMNS, "PIP")
    if raw.empty:
        return _empty_long()
    reporter_code = str(reporter.imf_id)
    counterpart_by_native = {item.imf_code: item.internal_code for item in counterparts}
    spec_by_indicator = {spec.native_indicator: spec for spec in specs}
    payload = _payload_rows(raw, _PIP_REQUIRED_COLUMNS - {"DATAFLOW", "STATUS"})
    if payload.empty:
        return _empty_long()
    expected = (
        (payload["COUNTRY"] == reporter_code)
        & (payload["ACCOUNTING_ENTRY"] == "A")
        & payload["INDICATOR"].isin(spec_by_indicator)
        & (payload["SECTOR"] == "S1")
        & (payload["COUNTERPART_SECTOR"] == "S1")
        & payload["COUNTERPART_COUNTRY"].isin(counterpart_by_native)
        & payload["FREQUENCY"].isin(frequencies)
    )
    if not expected.all():
        raise ValueError("IMF PIP snapshot contains a row outside the requested native key")
    frames = []
    for spec in specs:
        for frequency in frequencies:
            partition = payload[
                (payload["INDICATOR"] == spec.native_indicator)
                & (payload["FREQUENCY"] == frequency)
            ]
            if not partition.empty:
                frames.append(
                    _position_rows(
                        partition,
                        reporter,
                        spec,
                        frequency,
                        counterpart_by_native,
                        source=PIP_SOURCE,
                        reporter_sector_code="S1",
                        counterpart_sector_code="S1",
                        derivation_type=None,
                    )
                )
    return _concat_positions(frames)


def _parse_dip_bundle(
    text: str,
    reporter: Country,
    specs: tuple[PositionSeriesSpec, ...],
    frequencies: tuple[str, ...],
    counterparts: Sequence[PositionCounterpart],
) -> pd.DataFrame:
    raw = _read_csv(text, _DIP_REQUIRED_COLUMNS, "DIP")
    if raw.empty:
        return _empty_long()
    reporter_code = str(reporter.imf_id)
    counterpart_by_native = {item.imf_code: item.internal_code for item in counterparts}
    spec_by_indicator = {spec.native_indicator: spec for spec in specs}
    payload = _payload_rows(raw, _DIP_REQUIRED_COLUMNS - {"DATAFLOW", "STATUS"})
    if payload.empty:
        return _empty_long()
    expected = (
        (payload["COUNTRY"] == reporter_code)
        & (payload["DV_TYPE"] == "O")
        & payload["INDICATOR"].isin(spec_by_indicator)
        & payload["COUNTERPART_COUNTRY"].isin(counterpart_by_native)
        & payload["FREQUENCY"].isin(frequencies)
    )
    if not expected.all():
        raise ValueError("IMF DIP snapshot contains a row outside the requested native key")
    frames = []
    for spec in specs:
        for frequency in frequencies:
            partition = payload[
                (payload["INDICATOR"] == spec.native_indicator)
                & (payload["FREQUENCY"] == frequency)
            ]
            if not partition.empty:
                frames.append(
                    _position_rows(
                        partition,
                        reporter,
                        spec,
                        frequency,
                        counterpart_by_native,
                        source=DIP_SOURCE,
                        reporter_sector_code=None,
                        counterpart_sector_code=None,
                        derivation_type="O",
                    )
                )
    return _concat_positions(frames)


def _payload_rows(raw: pd.DataFrame, columns: set[str]) -> pd.DataFrame:
    supplied = raw.loc[:, sorted(columns)].notna().any(axis=1)
    return raw.loc[supplied].copy()


def _concat_positions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_long()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            [
                "dataset",
                "reporter_country",
                "native_indicator",
                "frequency",
                "counterpart_country",
                "date",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _position_rows(
    payload: pd.DataFrame,
    reporter: Country,
    spec: PositionSeriesSpec,
    frequency: str,
    counterpart_by_native: dict[str, str],
    *,
    source: str,
    reporter_sector_code: str | None,
    counterpart_sector_code: str | None,
    derivation_type: str | None,
) -> pd.DataFrame:
    values = pd.to_numeric(payload["OBS_VALUE"], errors="coerce")
    supplied = payload["OBS_VALUE"].notna() & payload["OBS_VALUE"].astype(str).str.strip().ne("")
    if values[supplied].isna().any():
        raise ValueError(f"IMF {spec.dataset} snapshot contains a non-numeric observation")
    payload = payload.loc[values.notna()].copy()
    if payload.empty:
        return _empty_long()
    payload["value"] = values.loc[payload.index].astype(float)
    if not np.isfinite(payload["value"]).all():
        raise ValueError(f"IMF {spec.dataset} snapshot contains a non-finite observation")

    reporter_code = str(reporter.imf_id)
    rows: list[dict] = []
    for row in payload.itertuples(index=False):
        native_status = "" if pd.isna(row.STATUS) else str(row.STATUS).strip()
        if len(native_status) > 24:
            raise ValueError(f"IMF {spec.dataset} native observation status is too long")
        rows.append(
            {
                "dataset": spec.dataset,
                "reporter_country": reporter.iso2,
                "reporter_code": reporter_code,
                "counterpart_country": counterpart_by_native[row.COUNTERPART_COUNTRY],
                "counterpart_code": row.COUNTERPART_COUNTRY,
                "date": _period_to_date(row.TIME_PERIOD, frequency, spec.dataset),
                "direction": spec.direction,
                "accounting_basis": spec.accounting_basis,
                "instrument_code": spec.instrument_code,
                "instrument_label": spec.instrument_label,
                "frequency": frequency,
                "value": float(row.value),
                "unit": spec.unit,
                "source": source,
                "native_indicator": spec.native_indicator,
                "reporter_sector_code": reporter_sector_code,
                "counterpart_sector_code": counterpart_sector_code,
                "derivation_type": derivation_type,
                "series_id": spec.series_id(reporter_code, row.COUNTERPART_COUNTRY, frequency),
                "status": native_status or "observed",
            }
        )
    frame = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    duplicate_key = [
        "dataset",
        "reporter_code",
        "counterpart_code",
        "date",
        "direction",
        "instrument_code",
        "frequency",
        "source",
        "series_id",
    ]
    if frame.duplicated(duplicate_key, keep=False).any():
        raise ValueError(f"IMF {spec.dataset} snapshot contains duplicate position observations")
    return frame.sort_values(
        ["reporter_country", "counterpart_country", "date"], kind="stable"
    ).reset_index(drop=True)


def _period_to_date(period: str, frequency: str, dataset: str) -> date:
    if frequency == "A":
        if not isinstance(period, str) or len(period) != 4 or not period.isdigit():
            raise ValueError(f"IMF {dataset} invalid annual period: {period!r}")
        return date(int(period), 1, 1)
    if frequency == "S":
        if (
            not isinstance(period, str)
            or len(period) != 7
            or period[4:6] != "-S"
            or not period[:4].isdigit()
            or period[6] not in {"1", "2"}
        ):
            raise ValueError(f"IMF {dataset} invalid semiannual period: {period!r}")
        return date(int(period[:4]), 1 if period[6] == "1" else 7, 1)
    raise ValueError(f"IMF {dataset} unsupported frequency: {frequency!r}")


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)
