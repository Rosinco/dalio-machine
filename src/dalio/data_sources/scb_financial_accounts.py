"""SCB Financial Accounts adapter for holders of Swedish government debt.

Statistics Sweden table TAB1203 is the official quarterly Financial Accounts
table.  This adapter deliberately owns one complete, multidimensional
selection: central-government debt-security liabilities, split into total,
short-term and long-term instruments and every reported counterpart sector.

The response is JSON-stat 2.0.  Its declared dimension order and category
positions are authoritative; JSON object insertion order is not.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

SCB_FINANCIAL_ACCOUNTS_BASE_URL = "https://api.scb.se/OV0104/v2beta/api/v2"
SCB_FINANCIAL_ACCOUNTS_TABLE_ID = "TAB1203"
SOURCE_SCB_FINANCIAL_ACCOUNTS = "SCB_FINANCIAL_ACCOUNTS"
DEFAULT_TIMEOUT = 30

ISSUER_SECTOR_CODE = "S1311"
INSTRUMENT_CODES = ("FL3000", "FL3100", "FL3200")
MEASURE_CODE = "FM0103AS"

SCB_GOVERNMENT_DEBT_HOLDERS_URL = (
    f"{SCB_FINANCIAL_ACCOUNTS_BASE_URL}/tables/{SCB_FINANCIAL_ACCOUNTS_TABLE_ID}/data"
    "?valueCodes%5BSektor%5D=S1311"
    "&valueCodes%5BKontopost%5D=FL3000%2CFL3100%2CFL3200"
    "&valueCodes%5BMotsektor%5D=%2A"
    "&valueCodes%5BContentsCode%5D=FM0103AS"
    "&valueCodes%5BTid%5D=%2A"
    "&outputFormat=json-stat2&lang=en"
)

LONG_COLUMNS = [
    "country",
    "date",
    "issuer_sector_code",
    "issuer_sector_label",
    "instrument_code",
    "instrument_label",
    "holder_sector_code",
    "holder_sector_label",
    "measure_code",
    "measure_label",
    "unit",
    "value",
    "source",
    "series_id",
    "status",
]

_EXPECTED_DIMENSIONS = {
    "Sektor",
    "Kontopost",
    "Motsektor",
    "ContentsCode",
    "Tid",
}


@dataclass(frozen=True)
class ScbFinancialAccountsSpec:
    """Fixed native selection for Swedish central-government debt holders."""

    country: str = "SE"
    table_id: str = SCB_FINANCIAL_ACCOUNTS_TABLE_ID
    issuer_sector_code: str = ISSUER_SECTOR_CODE
    instrument_codes: tuple[str, ...] = INSTRUMENT_CODES
    measure_code: str = MEASURE_CODE

    @property
    def url(self) -> str:
        """Exact, stable URL for the complete SCB selection."""
        return SCB_GOVERNMENT_DEBT_HOLDERS_URL

    def series_id(self, instrument_code: str, holder_sector_code: str) -> str:
        """Native identity of one holder/instrument time series."""
        return "/".join(
            (
                self.table_id,
                self.issuer_sector_code,
                instrument_code,
                holder_sector_code,
                self.measure_code,
            )
        )


SCB_GOVERNMENT_DEBT_HOLDERS = ScbFinancialAccountsSpec()


class ScbFinancialAccountsSource:
    """Fetch the complete official SCB government-debt-holder selection."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, "application/json")
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(
                cache_dir,
                "DALIO_SCB_FINANCIAL_ACCOUNTS_CACHE",
                "data/cache/scb_financial_accounts",
            ),
            cache_ttl_hours,
            label="SCB Financial Accounts",
            timeout=DEFAULT_TIMEOUT,
            suffix=".json",
            forbidden_hint="PxWeb endpoint refused the request",
        )

    def fetch(
        self,
        spec: ScbFinancialAccountsSpec = SCB_GOVERNMENT_DEBT_HOLDERS,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return typed long rows for every non-null published observation."""
        text = self._fetcher.fetch(spec.url, use_cache=use_cache)
        return parse_jsonstat(text, spec)


def parse_jsonstat(
    text: str,
    spec: ScbFinancialAccountsSpec = SCB_GOVERNMENT_DEBT_HOLDERS,
) -> pd.DataFrame:
    """Parse and validate a TAB1203 JSON-stat 2.0 response.

    Publisher nulls represent unavailable cells and are omitted.  Malformed,
    non-finite, duplicated or selection-leaking data fail closed.
    """
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("SCB Financial Accounts returned invalid JSON") from exc
    if not isinstance(raw, dict) or raw.get("class") != "dataset":
        raise ValueError("SCB Financial Accounts response is not a JSON-stat dataset")

    dimension_ids, sizes = _validate_dimensions(raw)
    categories = {
        dimension_id: _ordered_categories(raw, dimension_id, size)
        for dimension_id, size in zip(dimension_ids, sizes, strict=True)
    }
    _validate_selection(categories, spec)

    cell_count = math.prod(sizes)
    values = _dense_values(raw.get("value"), cell_count)
    measure_label = _category_label(raw, "ContentsCode", spec.measure_code)
    unit = _measure_unit(raw, spec.measure_code)

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, date]] = set()
    for flat_position, coordinates in enumerate(
        itertools.product(*(range(size) for size in sizes))
    ):
        raw_value = values.get(flat_position)
        if raw_value is None:
            continue
        value = _finite_number(raw_value, flat_position)
        codes = {
            dimension_id: categories[dimension_id][coordinate]
            for dimension_id, coordinate in zip(dimension_ids, coordinates, strict=True)
        }
        observed_on = _parse_quarter(codes["Tid"])
        key = (
            codes["Sektor"],
            codes["Kontopost"],
            codes["Motsektor"],
            codes["ContentsCode"],
            observed_on,
        )
        if key in seen_keys:
            raise ValueError(f"SCB Financial Accounts duplicate dimension key: {key!r}")
        seen_keys.add(key)

        instrument_code = codes["Kontopost"]
        holder_sector_code = codes["Motsektor"]
        rows.append(
            {
                "country": spec.country,
                "date": observed_on,
                "issuer_sector_code": codes["Sektor"],
                "issuer_sector_label": _category_label(raw, "Sektor", codes["Sektor"]),
                "instrument_code": instrument_code,
                "instrument_label": _category_label(raw, "Kontopost", instrument_code),
                "holder_sector_code": holder_sector_code,
                "holder_sector_label": _category_label(raw, "Motsektor", holder_sector_code),
                "measure_code": codes["ContentsCode"],
                "measure_label": measure_label,
                "unit": unit,
                "value": value,
                "source": SOURCE_SCB_FINANCIAL_ACCOUNTS,
                "series_id": spec.series_id(instrument_code, holder_sector_code),
                "status": "observed",
            }
        )

    if not rows:
        raise ValueError("SCB Financial Accounts response has no usable observations")
    return (
        pd.DataFrame(rows, columns=LONG_COLUMNS)
        .sort_values(["date", "instrument_code", "holder_sector_code"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_dimensions(raw: dict[str, Any]) -> tuple[list[str], list[int]]:
    dimension_ids = raw.get("id")
    raw_sizes = raw.get("size")
    if not isinstance(dimension_ids, list) or not all(
        isinstance(item, str) for item in dimension_ids
    ):
        raise ValueError("SCB Financial Accounts response has invalid dimension ids")
    if len(dimension_ids) != len(set(dimension_ids)):
        raise ValueError("SCB Financial Accounts response has duplicate dimension ids")
    if set(dimension_ids) != _EXPECTED_DIMENSIONS:
        raise ValueError(
            f"SCB Financial Accounts response has unexpected dimensions: {dimension_ids!r}"
        )
    if not isinstance(raw_sizes, list) or len(raw_sizes) != len(dimension_ids):
        raise ValueError("SCB Financial Accounts response has inconsistent dimension sizes")
    try:
        sizes = [int(size) for size in raw_sizes]
    except (TypeError, ValueError) as exc:
        raise ValueError("SCB Financial Accounts response has non-integer dimension sizes") from exc
    if any(size <= 0 for size in sizes):
        raise ValueError("SCB Financial Accounts response has an empty dimension")
    return dimension_ids, sizes


def _ordered_categories(raw: dict[str, Any], dimension_id: str, size: int) -> list[str]:
    try:
        category_index = raw["dimension"][dimension_id]["category"]["index"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"SCB Financial Accounts response is missing {dimension_id!r} categories"
        ) from exc

    if isinstance(category_index, list):
        codes = category_index
        if not all(isinstance(code, str) for code in codes):
            raise ValueError(f"SCB dimension {dimension_id!r} has invalid category codes")
        if len(codes) != len(set(codes)):
            raise ValueError(f"SCB dimension {dimension_id!r} has duplicate category codes")
    elif isinstance(category_index, dict):
        if not all(isinstance(code, str) for code in category_index):
            raise ValueError(f"SCB dimension {dimension_id!r} has invalid category codes")
        try:
            positions = [int(position) for position in category_index.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SCB dimension {dimension_id!r} has invalid category positions"
            ) from exc
        if len(positions) != len(set(positions)):
            raise ValueError(f"SCB dimension {dimension_id!r} has duplicate category positions")
        if set(positions) != set(range(size)):
            raise ValueError(
                f"SCB dimension {dimension_id!r} category positions do not match its size"
            )
        codes = [
            code
            for code, _position in sorted(category_index.items(), key=lambda item: int(item[1]))
        ]
    else:
        raise ValueError(f"SCB dimension {dimension_id!r} has unsupported category index")

    if len(codes) != size:
        raise ValueError(
            f"SCB dimension {dimension_id!r} has {len(codes)} categories, expected {size}"
        )
    return codes


def _validate_selection(
    categories: dict[str, list[str]],
    spec: ScbFinancialAccountsSpec,
) -> None:
    if categories["Sektor"] != [spec.issuer_sector_code]:
        raise ValueError("SCB Financial Accounts response leaked outside issuer S1311")
    if set(categories["Kontopost"]) != set(spec.instrument_codes) or len(
        categories["Kontopost"]
    ) != len(spec.instrument_codes):
        raise ValueError("SCB Financial Accounts response has unexpected instruments")
    if categories["ContentsCode"] != [spec.measure_code]:
        raise ValueError("SCB Financial Accounts response has an unexpected measure")
    if not categories["Motsektor"]:
        raise ValueError("SCB Financial Accounts response has no holder sectors")


def _dense_values(raw_values: Any, cell_count: int) -> dict[int, Any]:
    if isinstance(raw_values, list):
        if len(raw_values) != cell_count:
            raise ValueError(
                "SCB Financial Accounts response size mismatch: "
                f"{len(raw_values)} values vs {cell_count} dimension cells"
            )
        return dict(enumerate(raw_values))
    if isinstance(raw_values, dict):
        values: dict[int, Any] = {}
        for raw_position, value in raw_values.items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SCB Financial Accounts sparse values contain an invalid position"
                ) from exc
            if str(position) != str(raw_position) or not 0 <= position < cell_count:
                raise ValueError(
                    "SCB Financial Accounts sparse values contain an out-of-range position"
                )
            values[position] = value
        return values
    raise ValueError("SCB Financial Accounts response has unsupported values")


def _category_label(raw: dict[str, Any], dimension_id: str, code: str) -> str:
    try:
        label = raw["dimension"][dimension_id]["category"]["label"][code]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"SCB dimension {dimension_id!r} is missing label for {code!r}") from exc
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"SCB dimension {dimension_id!r} has an invalid label for {code!r}")
    return label.strip()


def _measure_unit(raw: dict[str, Any], measure_code: str) -> str:
    try:
        unit = raw["dimension"]["ContentsCode"]["category"]["unit"][measure_code]["base"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"SCB Financial Accounts response is missing unit for {measure_code!r}"
        ) from exc
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            f"SCB Financial Accounts response has an invalid unit for {measure_code!r}"
        )
    return unit.strip()


def _finite_number(raw_value: Any, flat_position: int) -> float:
    if isinstance(raw_value, bool):
        raise ValueError(f"SCB Financial Accounts cell {flat_position} has a non-numeric value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SCB Financial Accounts cell {flat_position} has a non-numeric value"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"SCB Financial Accounts cell {flat_position} is non-finite")
    return value


def _parse_quarter(label: str) -> date:
    match = re.fullmatch(r"(\d{4})K([1-4])", label)
    if not match:
        raise ValueError(f"Unsupported SCB Financial Accounts quarter: {label!r}")
    return date(int(match.group(1)), 1 + (int(match.group(2)) - 1) * 3, 1)
