"""World Bank Commodity Price Data (Pink Sheet) monthly-workbook adapter.

The publisher exposes the complete history as one rolling XLSX workbook rather
than an API.  This adapter therefore treats the validated workbook bytes as the
source artifact: it archives them by SHA-256, discovers every native benchmark
column, preserves the workbook unit in a catalogue, and emits canonical long
observations.  The small XLSX reader uses only the standard library; it reads
the Open XML parts needed by this particular tabular workbook and does not
require an implicit ``openpyxl`` dependency.

Missing cells remain missing. Nonpositive cells are quarantined because neither
a commodity price nor a price index can validly be zero or negative; the exact
publisher cell remains available in the archived workbook. No interpolation,
zero-filling, currency conversion, return calculation, or scoring happens in
the source adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import posixpath
import re
import tempfile
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import DEFAULT_USER_AGENT, set_default_headers

logger = logging.getLogger(__name__)

PINK_SHEET_LANDING_URL = "https://www.worldbank.org/en/research/commodity-markets"
# Last verified URL, also useful as an explicit reproducible override. Normal
# runs discover the current link from PINK_SHEET_LANDING_URL because the World
# Bank changes the document path when it republishes the rolling workbook.
PINK_SHEET_MONTHLY_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)
SOURCE_WORLD_BANK_COMMODITIES = "WORLD_BANK_PINK_SHEET"
WORLD_CODE = "WLD"
MONTHLY_SHEET_NAME = "Monthly Prices"
MONTHLY_INDICES_SHEET_NAME = "Monthly Indices"
DEFAULT_TIMEOUT = 60.0
MAX_WORKBOOK_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_XML_BYTES = 128 * 1024 * 1024
CATALOGUE_SCHEMA_VERSION = 2
CATALOGUE_GENERATOR_VERSION = "canonical-series-v2"
PINK_SHEET_VINTAGE_PREFIX = "Pink Sheet XLSX sha256:"
PINK_SHEET_CATALOGUE_VINTAGE_TAG = (
    f"catalogue:v{CATALOGUE_SCHEMA_VERSION}/{CATALOGUE_GENERATOR_VERSION}"
)

# Accepted catalogue identities and conservative rolling-workbook guards live
# beside the adapter semantics so ingestion and read-only inventory use exactly
# the same contract. Additions are permitted; silent substitution/removal is not.
EXPECTED_PINK_SHEET_MIN_PRICE_COLUMNS = 71
EXPECTED_PINK_SHEET_MIN_INDEX_COLUMNS = 16
EXPECTED_PINK_SHEET_PRICE_SERIES_COUNT = 70
EXPECTED_PINK_SHEET_INDEX_SERIES_COUNT = 17
EXPECTED_PINK_SHEET_HISTORY_START = date(1960, 1, 1)
MIN_PINK_SHEET_HISTORY_MONTHS = 720
MIN_PINK_SHEET_OBSERVATIONS_PER_SERIES = 12
MAX_PINK_SHEET_LATEST_LAG_MONTHS = 3
_EXPECTED_MONTHLY_PRICE_SLUGS = frozenset(
    {
        "aluminum",
        "banana_europe",
        "banana_us",
        "barley",
        "beef",
        "chicken",
        "coal_australian",
        "coal_south_african",
        "cocoa",
        "coconut_oil",
        "coffee_arabica",
        "coffee_robusta",
        "copper",
        "cotton_a_index",
        "crude_oil_average",
        "crude_oil_brent",
        "crude_oil_dubai",
        "crude_oil_wti",
        "dap",
        "fish_meal",
        "gold",
        "groundnut_oil",
        "groundnuts",
        "iron_ore_cfr_spot",
        "lamb",
        "lead",
        "liquefied_natural_gas_japan",
        "logs_cameroon",
        "logs_malaysian",
        "maize",
        "natural_gas_europe",
        "natural_gas_index",
        "natural_gas_us",
        "nickel",
        "orange",
        "palm_kernel_oil",
        "palm_oil",
        "phosphate_rock",
        "platinum",
        "plywood",
        "potassium_chloride",
        "rapeseed_oil",
        "rice_thai_25",
        "rice_thai_5",
        "rice_thai_a_1",
        "rice_viet_namese_5",
        "rubber_rss3",
        "rubber_tsr20",
        "sawnwood_cameroon",
        "sawnwood_malaysian",
        "shrimps_mexican",
        "silver",
        "sorghum",
        "soybean_meal",
        "soybean_oil",
        "soybeans",
        "sugar_eu",
        "sugar_us",
        "sugar_world",
        "sunflower_oil",
        "tea_avg_3_auctions",
        "tea_colombo",
        "tea_kolkata",
        "tea_mombasa",
        "tin",
        "tobacco_us_import_u_v",
        "tsp",
        "urea",
        "wheat_us_hrw",
        "wheat_us_srw",
        "zinc",
    }
)
_EXPECTED_MONTHLY_INDEX_SLUGS = frozenset(
    {
        "agriculture",
        "base_metals_ex_iron_ore",
        "beverages",
        "energy",
        "fertilizers",
        "food",
        "grains",
        "metals_minerals",
        "non_energy",
        "oils_meals",
        "other_food",
        "other_raw_mat",
        "precious_metals",
        "raw_materials",
        "timber",
        "total_index",
    }
)
EXPECTED_PINK_SHEET_SERIES_IDS_BY_WORKSHEET = {
    MONTHLY_SHEET_NAME: frozenset(
        f"monthly_prices:{slug}" for slug in _EXPECTED_MONTHLY_PRICE_SLUGS
    ),
    MONTHLY_INDICES_SHEET_NAME: frozenset(
        f"monthly_indices:{slug}" for slug in _EXPECTED_MONTHLY_INDEX_SLUGS
    ),
}

_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CELL_REFERENCE_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
_MONTH_RE = re.compile(r"^([12][0-9]{3})\s*(?:M|-)(0?[1-9]|1[0-2])$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_INDEX_BASE_RE = re.compile(r"(?<![0-9])([12][0-9]{3})\s*=\s*100(?![0-9])")

# The Pink Sheet has no native machine ID and its display labels do drift. Keep
# known publisher wording changes explicit; punctuation and footnote stars are
# handled by _normal_name. The canonical value is identity, never presentation.
_SERIES_NAME_ALIASES = {
    "coal australia": "coal australian",
    "fishmeal": "fish meal",
    "natural gas u s": "natural gas us",
    "palmkernel oil": "palm kernel oil",
    "rice thailand 5": "rice thai 5",
    "rice thailand 25": "rice thai 25",
    "rice thailand a1": "rice thai a 1",
    "tea average": "tea avg 3 auctions",
}


class BinaryHttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


@dataclass(frozen=True)
class CommoditySeries:
    """Metadata for one native Pink Sheet workbook column."""

    indicator: str
    series_id: str
    benchmark: str
    unit: str
    currency: str | None
    frequency: str
    price_basis: str
    category: str
    curated: bool
    worksheet: str = MONTHLY_SHEET_NAME
    series_kind: str = "price"


@dataclass(frozen=True)
class CommodityQualityIssue:
    """One source cell deliberately omitted from canonical observations."""

    worksheet: str
    cell: str
    date: date
    benchmark: str
    raw_value: float
    reason: str


@dataclass(frozen=True)
class CommodityDataset:
    """A validated workbook, its complete catalogue, and long observations."""

    observations: pd.DataFrame
    catalogue: tuple[CommoditySeries, ...]
    workbook_sha256: str
    source_url: str = PINK_SHEET_MONTHLY_URL
    artifact_path: Path | None = None
    catalogue_path: Path | None = None
    quality_issues: tuple[CommodityQualityIssue, ...] = ()


# A deliberately compact analytical panel.  All workbook columns are retained;
# this flag only identifies benchmarks useful for a first macro dashboard.
_CURATED_BENCHMARKS = frozenset(
    {
        "aluminum",
        "coal australian",
        "copper",
        "crude oil average",
        "crude oil brent",
        "crude oil wti",
        "gold",
        "iron ore cfr spot",
        "liquefied natural gas japan",
        "maize",
        "natural gas europe",
        "natural gas us",
        "nickel",
        "potassium chloride",
        "rice thai 5",
        "silver",
        "soybeans",
        "urea",
        "wheat us hrw",
        "woodpulp",
        "zinc",
    }
)
_CURATED_INDICES = frozenset(
    {
        "total index",
        "energy",
        "non energy",
        "agriculture",
        "food",
        "fertilizers",
        "metals minerals",
        "base metals ex iron ore",
        "precious metals",
    }
)

_ENERGY_TERMS = (
    "coal",
    "crude oil",
    "diesel",
    "gasoline",
    "liquefied natural gas",
    "natural gas",
    "petroleum",
    "propane",
)
_FERTILIZER_TERMS = (
    "dap",
    "fertilizer",
    "phosphate rock",
    "potassium chloride",
    "tsp",
    "urea",
)
_BASE_METALS = (
    "aluminum",
    "cobalt",
    "copper",
    "iron ore",
    "lead",
    "molybdenum",
    "nickel",
    "tin",
    "uranium",
    "zinc",
)
_PRECIOUS_METALS = ("gold", "palladium", "platinum", "silver")
_RAW_MATERIAL_TERMS = (
    "cotton",
    "logs",
    "plywood",
    "rubber",
    "sawnwood",
    "tobacco",
    "woodpulp",
)
_FOOD_TERMS = (
    "banana",
    "barley",
    "beef",
    "chicken",
    "cocoa",
    "coconut",
    "coffee",
    "fish meal",
    "groundnut",
    "lamb",
    "maize",
    "meat",
    "oil",
    "orange",
    "palm",
    "rapeseed",
    "rice",
    "shrimp",
    "sorghum",
    "soy",
    "sugar",
    "sunflower",
    "tea",
    "wheat",
)


def _normal_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(_NON_ALNUM_RE.sub(" ", ascii_value.lower()).split())


def _canonical_name(value: str) -> str:
    normalized = _normal_name(value)
    return _SERIES_NAME_ALIASES.get(normalized, normalized)


def canonical_series_id(benchmark: str, worksheet: str) -> str:
    """Return a stable native-column ID while leaving ``benchmark`` untouched."""

    scope = {
        MONTHLY_SHEET_NAME: "monthly_prices",
        MONTHLY_INDICES_SHEET_NAME: "monthly_indices",
    }.get(worksheet)
    if scope is None:
        raise ValueError(f"Unsupported Pink Sheet worksheet {worksheet!r}")
    canonical_name = _canonical_name(benchmark)
    if not canonical_name:
        raise ValueError("Pink Sheet benchmark cannot produce a canonical series id")
    candidate = f"{scope}:{canonical_name.replace(' ', '_')}"
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55].rstrip('_')}_{digest}"


def pink_sheet_vintage_label(workbook_sha256: str) -> str:
    """Bind an immutable release to workbook bytes and catalogue semantics."""

    if not _SHA256_RE.fullmatch(workbook_sha256):
        raise ValueError("Pink Sheet vintage label needs a lowercase SHA-256")
    label = (
        f"{PINK_SHEET_VINTAGE_PREFIX}{workbook_sha256};"
        f"{PINK_SHEET_CATALOGUE_VINTAGE_TAG}"
    )
    if len(label) > 128:
        raise ValueError("Pink Sheet catalogue version makes the vintage label too long")
    return label


def canonical_indicator(benchmark: str, *, series_kind: str = "price") -> str:
    """Return the stable project indicator for a publisher benchmark label."""

    canonical_name = _canonical_name(benchmark)
    slug = canonical_name.replace(" ", "_")
    if not slug:
        raise ValueError("Pink Sheet benchmark cannot produce an indicator name")
    if series_kind not in {"price", "index"}:
        raise ValueError(f"Unsupported Pink Sheet series kind {series_kind!r}")
    indicator = f"commodity_{series_kind}_{slug}"
    if len(indicator) <= 64:
        return indicator
    digest = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:8]
    return f"{indicator[:55].rstrip('_')}_{digest}"


def _category_for(benchmark: str) -> str:
    normalized = _canonical_name(benchmark)
    if any(term in normalized for term in _ENERGY_TERMS):
        return "energy"
    if any(term in normalized for term in _FERTILIZER_TERMS):
        return "fertilizers"
    if any(term in normalized for term in _PRECIOUS_METALS):
        return "precious_metals"
    if any(term in normalized for term in _BASE_METALS):
        return "base_metals"
    if any(term in normalized for term in _RAW_MATERIAL_TERMS):
        return "agricultural_raw_materials"
    if any(term in normalized for term in _FOOD_TERMS):
        return "food_and_beverages"
    return "other"


def _index_category_for(benchmark: str) -> str:
    normalized = _canonical_name(benchmark)
    if normalized == "total index":
        return "all_commodities"
    if normalized == "energy":
        return "energy"
    if normalized == "non energy":
        return "non_energy"
    if normalized == "agriculture":
        return "agriculture"
    if normalized in {"raw materials", "other raw mat", "timber"}:
        return "agricultural_raw_materials"
    category = _category_for(benchmark)
    if category != "other":
        return category
    if normalized in {"beverages", "food", "oils meals", "grains", "other food"}:
        return "food_and_beverages"
    if normalized in {"metals minerals", "base metals ex iron ore"}:
        return "base_metals"
    if normalized == "precious metals":
        return "precious_metals"
    return "other"


def _currency_for(unit: str) -> str | None:
    normalized = unit.lower().replace(" ", "")
    if "index" in normalized or ("=" in normalized and "$" not in normalized):
        return None
    if "$" in unit or "dollar" in unit.lower() or "cent" in unit.lower():
        return "USD"
    # The Pink Sheet's Monthly Prices worksheet is denominated in nominal US
    # dollars. Keep an unknown unit explicit rather than inventing a currency.
    return None


def _index_unit(unit: str) -> tuple[str, int] | None:
    """Return canonical index unit/base, rejecting ambiguous index-like units."""

    text = unit.strip()
    bases = {int(match.group(1)) for match in _INDEX_BASE_RE.finditer(text)}
    if len(bases) == 1:
        base_year = bases.pop()
        return f"{base_year}=100", base_year
    if bases or "=" in text or "index" in text.casefold():
        raise ValueError(f"Pink Sheet has an unsupported index unit {unit!r}")
    return None


def _index_unit_from_metadata(
    rows: list[tuple[int, dict[int, object]]],
    first_data_position: int,
) -> tuple[str, int]:
    bases = {
        int(match.group(1))
        for _row_number, values in rows[:first_data_position]
        for value in values.values()
        for match in _INDEX_BASE_RE.finditer(str(value))
    }
    if len(bases) != 1:
        raise ValueError(
            "Pink Sheet Monthly Indices must declare exactly one index base as YYYY=100"
        )
    base_year = bases.pop()
    return f"{base_year}=100", base_year


def _column_number(reference: str) -> int:
    match = _CELL_REFERENCE_RE.fullmatch(reference.upper())
    if not match:
        raise ValueError(f"Pink Sheet contains invalid XLSX cell reference {reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - 64
    return result


def _column_name(column: int) -> str:
    if column < 1:
        raise ValueError("XLSX column number must be positive")
    result = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_root(archive: zipfile.ZipFile, member: str) -> ElementTree.Element:
    try:
        info = archive.getinfo(member)
    except KeyError as exc:
        raise ValueError(f"Pink Sheet XLSX is missing {member}") from exc
    if info.file_size > MAX_UNCOMPRESSED_XML_BYTES:
        raise ValueError(f"Pink Sheet XLSX member {member} is unreasonably large")
    try:
        return ElementTree.fromstring(archive.read(info))
    except ElementTree.ParseError as exc:
        raise ValueError(f"Pink Sheet XLSX has malformed XML in {member}") from exc


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _xml_root(archive, "xl/sharedStrings.xml")
    return [
        "".join(node.text or "" for node in item.iter() if node.tag.rsplit("}", 1)[-1] == "t")
        for item in root.iter()
        if item.tag.rsplit("}", 1)[-1] == "si"
    ]


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = _xml_root(archive, "xl/workbook.xml")
    relationship_id: str | None = None
    for element in workbook.iter():
        if element.tag.rsplit("}", 1)[-1] != "sheet":
            continue
        if element.attrib.get("name", "").strip().casefold() == sheet_name.casefold():
            relationship_id = next(
                (
                    value
                    for key, value in element.attrib.items()
                    if key.rsplit("}", 1)[-1] == "id"
                ),
                None,
            )
            break
    if not relationship_id:
        raise ValueError(f"Pink Sheet XLSX has no {sheet_name!r} worksheet")

    relationships = _xml_root(archive, "xl/_rels/workbook.xml.rels")
    target: str | None = None
    for element in relationships.iter():
        if (
            element.tag.rsplit("}", 1)[-1] == "Relationship"
            and element.attrib.get("Id") == relationship_id
        ):
            target = element.attrib.get("Target")
            break
    if not target:
        raise ValueError("Pink Sheet XLSX worksheet relationship is missing")
    path = (
        posixpath.normpath(target.lstrip("/"))
        if target.startswith("/")
        else posixpath.normpath(posixpath.join("xl", target))
    )
    if not path.startswith("xl/") or path == "xl" or ".." in path.split("/"):
        raise ValueError("Pink Sheet XLSX has an unsafe worksheet path")
    if path not in archive.namelist():
        raise ValueError("Pink Sheet XLSX points to a missing worksheet")
    return path


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> object | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter() if node.tag.rsplit("}", 1)[-1] == "t"
        )
    value_node = next(
        (node for node in cell if node.tag.rsplit("}", 1)[-1] == "v"), None
    )
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "e":
        # Formula error cells in the publisher workbook (for example #VALUE!)
        # are unavailable observations, not commodity prices.
        return None
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError) as exc:
            raise ValueError("Pink Sheet XLSX has an invalid shared-string reference") from exc
    if cell_type == "str":
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        return float(raw)
    except ValueError:
        return raw


def _worksheet_rows(
    archive: zipfile.ZipFile,
    worksheet_path: str,
    shared: list[str],
) -> list[tuple[int, dict[int, object]]]:
    root = _xml_root(archive, worksheet_path)
    rows: list[tuple[int, dict[int, object]]] = []
    last_row = 0
    for row in root.iter():
        if row.tag.rsplit("}", 1)[-1] != "row":
            continue
        try:
            row_number = int(row.attrib.get("r", last_row + 1))
        except ValueError as exc:
            raise ValueError("Pink Sheet XLSX contains an invalid row number") from exc
        if row_number <= last_row:
            raise ValueError("Pink Sheet XLSX worksheet rows are not strictly ordered")
        last_row = row_number
        values: dict[int, object] = {}
        for cell in row:
            if cell.tag.rsplit("}", 1)[-1] != "c":
                continue
            reference = cell.attrib.get("r")
            if not reference:
                raise ValueError("Pink Sheet XLSX contains a cell without a reference")
            column = _column_number(reference)
            if column in values:
                raise ValueError("Pink Sheet XLSX contains duplicate cells")
            value = _cell_value(cell, shared)
            if value is not None:
                values[column] = value
        rows.append((row_number, values))
    return rows


def _parse_month(value: object) -> date | None:
    if isinstance(value, datetime):
        return date(value.year, value.month, 1)
    if isinstance(value, date):
        return date(value.year, value.month, 1)
    match = _MONTH_RE.fullmatch(str(value).strip())
    if not match:
        return None
    return date(int(match.group(1)), int(match.group(2)), 1)


def _next_month(period: date) -> date:
    if period.month == 12:
        return date(period.year + 1, 1, 1)
    return date(period.year, period.month + 1, 1)


def _monthly_periods(
    rows: list[tuple[int, dict[int, object]]],
    *,
    worksheet: str,
) -> list[date]:
    periods = [
        period
        for _row_number, values in rows
        if (period := _parse_month(values.get(1))) is not None
    ]
    if not periods:
        raise ValueError(f"Pink Sheet {worksheet} contains no valid monthly observation rows")
    for previous, current in zip(periods, periods[1:], strict=False):
        if current != _next_month(previous):
            raise ValueError(
                f"Pink Sheet {worksheet} monthly rows are not contiguous: "
                f"{previous.isoformat()} then {current.isoformat()}"
            )
    return periods


def _looks_like_unit(value: object) -> bool:
    text = str(value).strip().lower()
    compact = text.replace(" ", "")
    return (
        text in {"unit", "units"}
        or "$" in text
        or "cent" in text
        or "=" in compact
        or any(token in compact for token in ("/kg", "/mt", "/bbl", "/mmbtu", "/dmtu"))
    )


def _as_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Pink Sheet {field} must not be empty")
    return text


def _numeric_value(
    value: object,
    *,
    row: int,
    column: int,
    period: date,
    benchmark: str,
    worksheet: str,
    quality_issues: list[CommodityQualityIssue],
) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "..",
        "...",
        "…",
        "#value!",
        "#n/a",
        "na",
        "n/a",
        "-",
    }:
        return None
    try:
        number = float(str(value).replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"Pink Sheet has a non-numeric value for {benchmark!r} in worksheet row {row}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Pink Sheet has a non-finite value for {benchmark!r}")
    if number <= 0:
        quality_issues.append(
            CommodityQualityIssue(
                worksheet=worksheet,
                cell=f"{_column_name(column)}{row}",
                date=period,
                benchmark=benchmark,
                raw_value=number,
                reason="nonpositive_value",
            )
        )
        logger.warning(
            "Quarantining nonpositive Pink Sheet value %s for %s in worksheet row %d",
            number,
            benchmark,
            row,
        )
        return None
    return number


def _locate_header(
    rows: list[tuple[int, dict[int, object]]],
    first_data_position: int,
) -> tuple[int, int, dict[int, object]]:
    candidates: list[tuple[int, int, int, dict[int, object]]] = []
    for position, (_row_number, values) in enumerate(rows[:first_data_position]):
        series_values = [value for column, value in values.items() if column > 1]
        if not series_values:
            continue
        non_units = sum(not _looks_like_unit(value) for value in series_values)
        candidates.append((non_units, len(series_values), position, values))
    if not candidates:
        raise ValueError("Pink Sheet cannot locate the benchmark header row")
    non_units, _filled, position, values = max(candidates)
    if non_units < 1:
        raise ValueError("Pink Sheet cannot locate the benchmark header row")
    return position, rows[position][0], values


def _parse_index_rows(
    rows: list[tuple[int, dict[int, object]]],
    quality_issues: list[CommodityQualityIssue],
) -> tuple[list[CommoditySeries], list[dict[str, object]]]:
    data_positions = [
        position
        for position, (_row_number, values) in enumerate(rows)
        if _parse_month(values.get(1)) is not None
    ]
    if not data_positions:
        raise ValueError("Pink Sheet Monthly Indices contains no valid monthly observation rows")
    first_data_position = min(data_positions)
    unit, base_year = _index_unit_from_metadata(rows, first_data_position)
    series_columns = sorted(
        {
            column
            for _row_number, values in rows[:first_data_position]
            for column, value in values.items()
            if column > 1 and str(value).strip()
        }
    )
    if not series_columns:
        raise ValueError("Pink Sheet Monthly Indices has no index columns")

    catalogue: list[CommoditySeries] = []
    benchmarks: set[str] = set()
    indicators: set[str] = set()
    for column in series_columns:
        header_values = [
            str(values[column]).strip()
            for _row_number, values in rows[:first_data_position]
            if column in values and str(values[column]).strip()
        ]
        if not header_values:
            raise ValueError(f"Pink Sheet Monthly Indices column {column} has no benchmark")
        # The tab uses a stepped hierarchy. The lowest non-blank label in each
        # column is the native series heading (for example Food or Grains).
        benchmark = header_values[-1]
        if benchmark in benchmarks:
            raise ValueError("Pink Sheet Monthly Indices contains a duplicate benchmark")
        benchmarks.add(benchmark)
        indicator = canonical_indicator(benchmark, series_kind="index")
        if indicator in indicators:
            raise ValueError("Pink Sheet index names collide after indicator normalization")
        indicators.add(indicator)
        catalogue.append(
            CommoditySeries(
                indicator=indicator,
                series_id=canonical_series_id(benchmark, MONTHLY_INDICES_SHEET_NAME),
                benchmark=benchmark,
                unit=unit,
                currency=None,
                frequency="monthly",
                price_basis=f"nominal_usd_index_{base_year}_100",
                category=_index_category_for(benchmark),
                curated=_canonical_name(benchmark) in _CURATED_INDICES,
                worksheet=MONTHLY_INDICES_SHEET_NAME,
                series_kind="index",
            )
        )

    records: list[dict[str, object]] = []
    seen_periods: set[date] = set()
    observations_per_column = dict.fromkeys(series_columns, 0)
    by_column = dict(zip(series_columns, catalogue, strict=True))
    for row_number, values in rows[first_data_position:]:
        period = _parse_month(values.get(1))
        has_series_value = any(column in values for column in series_columns)
        if period is None:
            if has_series_value:
                raise ValueError(
                    "Pink Sheet Monthly Indices has values without a valid month "
                    f"in worksheet row {row_number}"
                )
            continue
        if period in seen_periods:
            raise ValueError(
                f"Pink Sheet Monthly Indices contains duplicate month {period.isoformat()}"
            )
        seen_periods.add(period)
        for column, series in by_column.items():
            number = _numeric_value(
                values.get(column),
                row=row_number,
                column=column,
                period=period,
                benchmark=series.benchmark,
                worksheet=MONTHLY_INDICES_SHEET_NAME,
                quality_issues=quality_issues,
            )
            if number is None:
                continue
            observations_per_column[column] += 1
            records.append(
                {
                    "country": WORLD_CODE,
                    "indicator": series.indicator,
                    "date": period,
                    "value": number,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series.series_id,
                }
            )
    empty_series = [
        by_column[column].benchmark
        for column, count in observations_per_column.items()
        if count == 0
    ]
    if empty_series:
        raise ValueError(f"Pink Sheet index columns have no observations: {empty_series}")
    return catalogue, records


def parse_pink_sheet_workbook(content: bytes) -> CommodityDataset:
    """Parse every series on the official monthly price and index worksheets."""

    if not isinstance(content, bytes) or not content.startswith(b"PK"):
        raise ValueError("Pink Sheet response is not a valid XLSX workbook")
    if len(content) > MAX_WORKBOOK_BYTES:
        raise ValueError("Pink Sheet XLSX is unreasonably large")
    digest = hashlib.sha256(content).hexdigest()
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            total_size = sum(item.file_size for item in archive.infolist())
            if total_size > MAX_UNCOMPRESSED_XML_BYTES:
                raise ValueError("Pink Sheet XLSX expands to an unreasonable size")
            shared = _shared_strings(archive)
            sheet_path = _worksheet_path(archive, MONTHLY_SHEET_NAME)
            rows = _worksheet_rows(archive, sheet_path, shared)
            index_sheet_path = _worksheet_path(archive, MONTHLY_INDICES_SHEET_NAME)
            index_rows = _worksheet_rows(archive, index_sheet_path, shared)
    except zipfile.BadZipFile as exc:
        raise ValueError("Pink Sheet response is not a valid XLSX workbook") from exc

    price_periods = _monthly_periods(rows, worksheet=MONTHLY_SHEET_NAME)
    index_periods = _monthly_periods(index_rows, worksheet=MONTHLY_INDICES_SHEET_NAME)
    if price_periods != index_periods:
        raise ValueError("Pink Sheet price and index worksheets cover different monthly periods")

    data_positions = [
        position for position, (_row_number, values) in enumerate(rows)
        if _parse_month(values.get(1)) is not None
    ]
    if not data_positions:
        raise ValueError("Pink Sheet contains no valid monthly observation rows")
    first_data_position = min(data_positions)
    header_position, _header_row_number, header = _locate_header(rows, first_data_position)

    series_columns = sorted(column for column, value in header.items() if column > 1 and value)
    if not series_columns:
        raise ValueError("Pink Sheet benchmark header row is empty")
    benchmarks = [_as_text(header[column], field="benchmark") for column in series_columns]
    if len(benchmarks) != len(set(benchmarks)):
        raise ValueError("Pink Sheet contains a duplicate benchmark column")
    data_columns = {
        column
        for position in data_positions
        for column in rows[position][1]
        if column > 1
    }
    unknown_data_columns = data_columns - set(series_columns)
    if unknown_data_columns:
        raise ValueError(
            "Pink Sheet contains data columns absent from its benchmark header: "
            f"{sorted(unknown_data_columns)}"
        )

    units: dict[int, str] = {}
    for column in series_columns:
        candidates = [
            values[column]
            for _row_number, values in rows[header_position + 1:first_data_position]
            if column in values and _looks_like_unit(values[column])
        ]
        if not candidates:
            raise ValueError(f"Pink Sheet benchmark {header[column]!r} has no unit metadata")
        units[column] = _as_text(candidates[-1], field="unit")

    catalogue: list[CommoditySeries] = []
    indicators: set[str] = set()
    for column, benchmark in zip(series_columns, benchmarks, strict=True):
        unit = units[column]
        index_metadata = _index_unit(unit)
        series_kind = "index" if index_metadata is not None else "price"
        indicator = canonical_indicator(benchmark, series_kind=series_kind)
        if indicator in indicators:
            raise ValueError("Pink Sheet benchmark names collide after indicator normalization")
        indicators.add(indicator)
        if index_metadata is None:
            canonical_unit = unit
            price_basis = "nominal_monthly_average"
            category = _category_for(benchmark)
            curated_set = _CURATED_BENCHMARKS
        else:
            canonical_unit, base_year = index_metadata
            price_basis = f"nominal_usd_index_{base_year}_100"
            category = _index_category_for(benchmark)
            curated_set = _CURATED_INDICES
        catalogue.append(
            CommoditySeries(
                indicator=indicator,
                series_id=canonical_series_id(benchmark, MONTHLY_SHEET_NAME),
                benchmark=benchmark,
                unit=canonical_unit,
                currency=_currency_for(canonical_unit),
                frequency="monthly",
                price_basis=price_basis,
                category=category,
                curated=_canonical_name(benchmark) in curated_set,
                worksheet=MONTHLY_SHEET_NAME,
                series_kind=series_kind,
            )
        )

    records: list[dict[str, object]] = []
    quality_issues: list[CommodityQualityIssue] = []
    seen_periods: set[date] = set()
    observations_per_column = dict.fromkeys(series_columns, 0)
    by_column = dict(zip(series_columns, catalogue, strict=True))
    for row_number, values in rows[first_data_position:]:
        period = _parse_month(values.get(1))
        has_series_value = any(column in values for column in series_columns)
        if period is None:
            if has_series_value:
                raise ValueError(
                    f"Pink Sheet has commodity values without a valid month in worksheet row {row_number}"
                )
            continue
        if period in seen_periods:
            raise ValueError(f"Pink Sheet contains duplicate monthly row {period.isoformat()}")
        seen_periods.add(period)
        for column, series in by_column.items():
            number = _numeric_value(
                values.get(column),
                row=row_number,
                column=column,
                period=period,
                benchmark=series.benchmark,
                worksheet=MONTHLY_SHEET_NAME,
                quality_issues=quality_issues,
            )
            if number is None:
                continue
            observations_per_column[column] += 1
            records.append(
                {
                    "country": WORLD_CODE,
                    "indicator": series.indicator,
                    "date": period,
                    "value": number,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series.series_id,
                }
            )
    empty_series = [
        by_column[column].benchmark
        for column, count in observations_per_column.items()
        if count == 0
    ]
    if empty_series:
        raise ValueError(f"Pink Sheet benchmark columns have no observations: {empty_series}")
    index_catalogue, index_records = _parse_index_rows(index_rows, quality_issues)
    all_catalogue = [*catalogue, *index_catalogue]
    all_indicators = [series.indicator for series in all_catalogue]
    if len(all_indicators) != len(set(all_indicators)):
        raise ValueError("Pink Sheet price and index indicators collide")
    records.extend(index_records)
    frame = pd.DataFrame(records, columns=_LONG_COLUMNS).sort_values(
        ["indicator", "date"], kind="stable"
    ).reset_index(drop=True)
    return CommodityDataset(
        observations=frame,
        catalogue=tuple(all_catalogue),
        workbook_sha256=digest,
        quality_issues=tuple(quality_issues),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _WorkbookLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = next((value for key, value in attrs if key.casefold() == "href"), None)
        if href:
            self.hrefs.append(href.strip())


def _validated_official_workbook_url(value: str) -> str:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().strip(".")
    filename = posixpath.basename(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not (host == "worldbank.org" or host.endswith(".worldbank.org"))
        or filename != "CMO-Historical-Data-Monthly.xlsx"
    ):
        raise ValueError("Pink Sheet workbook URL is not an official HTTPS monthly-workbook link")
    return value


class PinkSheetSource:
    """Download, validate, cache, and permanently archive the monthly workbook."""

    def __init__(
        self,
        client: BinaryHttpClient | None = None,
        cache_dir: Path | None = None,
        artifact_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        *,
        attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        timeout: float = DEFAULT_TIMEOUT,
        source_url: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        for name, value in (
            ("cache_ttl_hours", cache_ttl_hours),
            ("retry_backoff_seconds", retry_backoff_seconds),
            ("timeout", timeout),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        self._client = client or requests.Session()
        set_default_headers(
            self._client,
            user_agent,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self._cache_dir = cache_dir or Path(
            os.environ.get("DALIO_PINK_SHEET_CACHE", "data/cache/worldbank_commodities")
        )
        self._artifact_dir = artifact_dir or Path(
            os.environ.get(
                "DALIO_PINK_SHEET_ARTIFACTS",
                "data/artifacts/worldbank_commodities",
            )
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._attempts = attempts
        self._retry_backoff = retry_backoff_seconds
        self._timeout = timeout
        self._source_url_override = (
            _validated_official_workbook_url(source_url) if source_url is not None else None
        )

    def _cache_path_for(self, source_url: str) -> Path:
        digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{digest}.xlsx"

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.get(url, timeout=self._timeout)
                if response.status_code in {403, 404}:
                    raise ValueError(
                        f"World Bank Pink Sheet returned permanent HTTP {response.status_code}: {url}"
                    )
                if response.status_code >= 500:
                    raise RuntimeError(
                        f"World Bank Pink Sheet server error {response.status_code}: {url}"
                    )
                response.raise_for_status()
                return bytes(response.content)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient HTTP/transport errors
                last_error = exc
                if attempt < self._attempts - 1:
                    wait = self._retry_backoff * (2**attempt)
                    logger.warning(
                        "Pink Sheet attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        self._attempts,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _discover_source_url(self) -> str:
        content = self._download(PINK_SHEET_LANDING_URL)
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("Pink Sheet landing page is unreasonably large")
        try:
            html = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Cannot discover Pink Sheet workbook from non-UTF-8 landing page") from exc
        parser = _WorkbookLinkParser()
        parser.feed(html)
        candidates: set[str] = set()
        for href in parser.hrefs:
            candidate = urljoin(PINK_SHEET_LANDING_URL, href)
            try:
                candidates.add(_validated_official_workbook_url(candidate))
            except ValueError:
                continue
        if len(candidates) != 1:
            raise ValueError(
                "Cannot safely discover exactly one official Pink Sheet monthly workbook "
                f"(found {len(candidates)})"
            )
        return candidates.pop()

    @staticmethod
    def _catalogue_bytes(parsed: CommodityDataset) -> bytes:
        payload = {
            "schema_version": CATALOGUE_SCHEMA_VERSION,
            "generator_version": CATALOGUE_GENERATOR_VERSION,
            "source": SOURCE_WORLD_BANK_COMMODITIES,
            "workbook_sha256": parsed.workbook_sha256,
            "quality": {
                "quarantined_cell_count": len(parsed.quality_issues),
                "quarantined_cells": [
                    {
                        "worksheet": issue.worksheet,
                        "cell": issue.cell,
                        "date": issue.date.isoformat(),
                        "benchmark": issue.benchmark,
                        "raw_value": issue.raw_value,
                        "reason": issue.reason,
                    }
                    for issue in parsed.quality_issues
                ],
            },
            "series": [
                {
                    "worksheet": series.worksheet,
                    "series_kind": series.series_kind,
                    "series_id": series.series_id,
                    "benchmark": series.benchmark,
                    "indicator": series.indicator,
                    "unit": series.unit,
                    "currency": series.currency,
                    "frequency": series.frequency,
                    "price_basis": series.price_basis,
                    "category": series.category,
                    "curated": series.curated,
                }
                for series in parsed.catalogue
            ],
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _archive(self, content: bytes, parsed: CommodityDataset) -> tuple[Path, Path]:
        digest = parsed.workbook_sha256
        artifact_path = self._artifact_dir / digest[:2] / f"{digest}.xlsx"
        catalogue_path = (
            self._artifact_dir
            / digest[:2]
            / (
                f"{digest}.catalogue-v{CATALOGUE_SCHEMA_VERSION}-"
                f"{CATALOGUE_GENERATOR_VERSION}.json"
            )
        )
        if artifact_path.exists():
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pink Sheet artifact archive is corrupt: {artifact_path}")
        else:
            _atomic_write(artifact_path, content)

        catalogue_content = self._catalogue_bytes(parsed)
        if catalogue_path.exists():
            if catalogue_path.read_bytes() != catalogue_content:
                raise ValueError(f"Pink Sheet catalogue archive is corrupt: {catalogue_path}")
        else:
            _atomic_write(catalogue_path, catalogue_content)
        return artifact_path, catalogue_path

    def _validated_dataset(self, content: bytes, source_url: str) -> CommodityDataset:
        parsed = parse_pink_sheet_workbook(content)
        artifact_path, catalogue_path = self._archive(content, parsed)
        return CommodityDataset(
            observations=parsed.observations,
            catalogue=parsed.catalogue,
            workbook_sha256=parsed.workbook_sha256,
            source_url=source_url,
            artifact_path=artifact_path,
            catalogue_path=catalogue_path,
            quality_issues=parsed.quality_issues,
        )

    def fetch(self, *, use_cache: bool = True) -> CommodityDataset:
        """Return the complete validated monthly universe and metadata catalogue."""

        source_url = self._source_url_override or self._discover_source_url()
        cache_path = self._cache_path_for(source_url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                try:
                    return self._validated_dataset(cache_path.read_bytes(), source_url)
                except ValueError as exc:
                    logger.warning("Ignoring invalid Pink Sheet rolling cache: %s", exc)

        content = self._download(source_url)
        dataset = self._validated_dataset(content, source_url)
        _atomic_write(cache_path, content)
        return dataset
