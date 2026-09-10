"""Three original SCB selections for the Sweden monitoring pilot.

The acquisition layer owns HTTP and archives the exact metadata and response
bytes. This module has no I/O: the fixed request is checked again during replay,
including native dimensions, units, adjustment and the publisher release clock.
``observed`` means a published statistical observation, not a claim of finality;
the original revision notes and missing-value flags remain attached.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlencode

SCB_BASE_URL = "https://api.scb.se/OV0104/v2beta/api/v2"
SOURCE_SCB_MONITORING = "SCB_MONITORING"
MIR_INSTRUCTIONS_URL = (
    "https://www.scb.se/contentassets/1339ecb0b32a47eab3d9ef773e1e6738/"
    "2025-02-13/instruktioner_for_rapportering_av_rantestatistik_2025-02-10.pdf"
)


@dataclass(frozen=True)
class ScbMonitoringSpec:
    """A pinned scalar series; only the complete native time axis varies."""

    key: str
    table_id: str
    selection: tuple[tuple[str, str], ...]
    native_source: str
    native_unit: str
    native_adjustment: str
    price_type: str
    base_period: str | None
    definition: str
    documentation_url: str

    @property
    def metadata_url(self) -> str:
        return f"{SCB_BASE_URL}/tables/{self.table_id}/metadata?lang=en"

    @property
    def source_url(self) -> str:
        parameters = [(f"valueCodes[{name}]", code) for name, code in self.selection]
        parameters.extend(
            [("valueCodes[Tid]", "*"), ("outputFormat", "json-stat2"), ("lang", "en")]
        )
        return f"{SCB_BASE_URL}/tables/{self.table_id}/data?{urlencode(parameters)}"

    @property
    def request(self) -> dict:
        return {
            "method": "GET",
            "metadata_url": self.metadata_url,
            "source_url": self.source_url,
            "table_id": self.table_id,
            "selection": {**dict(self.selection), "Tid": "*"},
            "output_format": "json-stat2",
            "language": "en",
        }

    @property
    def series_id(self) -> str:
        return "/".join([self.table_id, *(code for _, code in self.selection)])


SCB_MONITORING_SPECS = (
    ScbMonitoringSpec(
        key="industrial_production",
        table_id="TAB1872",
        selection=(("SNI2007", "B+C"), ("ContentsCode", "NV0402AL")),
        native_source="Statistics Sweden",
        native_unit="index",
        native_adjustment="WorkAndSes",
        price_type="Fixed",
        base_period="2021",
        definition=(
            "Industrial production volume chain index, 2021=100; mining, quarrying "
            "and manufacturing (NACE Rev.2 B+C), excluding energy; calendar and "
            "seasonally adjusted. Monthly statistical observations remain revisable."
        ),
        documentation_url="https://www.scb.se/NV0402",
    ),
    ScbMonitoringSpec(
        key="industrial_orders",
        table_id="TAB1710",
        selection=(
            ("Marknad", "TOTALA"), ("SNI2007", "B+C"), ("ContentsCode", "NV0501BD")
        ),
        native_source="Statistics Sweden",
        native_unit="index",
        native_adjustment="WorkAndSes",
        price_type="Fixed",
        base_period="2021",
        definition=(
            "Industrial orders at constant prices, index 2021=100; total domestic "
            "and export market, mining, quarrying and manufacturing (NACE Rev.2 "
            "B+C); calendar and seasonally adjusted. Large individual orders and "
            "revisions can affect monthly and three-month comparisons."
        ),
        documentation_url="https://www.scb.se/NV0501",
    ),
    ScbMonitoringSpec(
        key="corporate_new_lending_rate",
        table_id="TAB5780",
        selection=(
            ("Referenssektor", "1.1"),
            ("Motpartssektor", "1"),
            ("Avtal", "0100"),
            ("Rantebindningstid", "1.1"),
            ("ContentsCode", "000004ZT"),
        ),
        native_source="The Riksbank",
        native_unit="percent",
        native_adjustment="None",
        price_type="NotApplicable",
        base_period=None,
        definition=(
            "MFI lending to Swedish non-financial corporations in SEK: amount-weighted "
            "annualised agreed interest rate on new and renegotiated agreements "
            "during the month, across all original fixation periods including "
            "floating rates. Transaction-account balances counted as new business "
            "are excluded by the loans-with-rate-fixation selection. The rate "
            "generally excludes fees. This measures borrowing terms and loan mix, "
            "not credit growth or a company's actual borrowing cost."
        ),
        documentation_url=MIR_INSTRUCTIONS_URL,
    ),
)


def parse_scb_input(
    key: str, metadata_bytes: bytes, response_bytes: bytes, request: dict
) -> dict:
    """Validate archived originals and return JSON-safe observations and context.

    The caller must compare ``publisher_updated_at`` against the actual response
    receipt clock. No retrieval time is invented or inferred here. All native
    periods are retained, including null cells; no growth rates are calculated.
    """
    spec = next((item for item in SCB_MONITORING_SPECS if item.key == key), None)
    if spec is None:
        raise ValueError(f"Unknown SCB monitoring key: {key!r}")
    if request != spec.request:
        raise ValueError("SCB request differs from the pinned native selection")
    metadata = _load_json(metadata_bytes)
    response = _load_json(response_bytes)
    meta_categories = _validate_dataset(metadata, spec, selected=False)
    data_categories = _validate_dataset(response, spec, selected=True)
    publication = _clock(metadata.get("updated"))
    if _clock(response.get("updated")) != publication:
        raise ValueError("SCB metadata and response have different release clocks")
    if data_categories["Tid"] != meta_categories["Tid"]:
        raise ValueError("SCB response does not cover the complete requested time periods")

    for name, code in spec.selection:
        native_label = _label(metadata, name, code)
        if _label(response, name, code) != native_label:
            raise ValueError(f"SCB response label differs from metadata for {name}")

    periods = data_categories["Tid"]
    values = _cells(response.get("value"), len(periods), field="value")
    statuses = _cells(response.get("status"), len(periods), field="status")
    observations = []
    for position, label in enumerate(periods):
        start, end = _month(label)
        if end > publication.date():
            raise ValueError("SCB observation period ends after publication")
        value = values[position]
        native_status = statuses[position]
        if native_status not in (None, "", ".."):
            raise ValueError(f"Unsupported SCB native status: {native_status!r}")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("SCB value must be a JSON number or null")
            if not math.isfinite(value):
                raise ValueError("SCB returned a non-finite numeric value")
            if native_status == "..":
                raise ValueError("SCB missing status conflicts with a numeric value")
            value = float(value)
        observations.append(
            {
                "date": end.isoformat(),
                "period": start.strftime("%Y-%m"),
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                "value": value,
                "status": "not_reported" if value is None else "observed",
                "native_status": native_status,
                "source_locator": f"/value/{position}; {spec.series_id}/Tid={label}",
            }
        )
    observations.sort(key=lambda item: item["date"])
    metric = dict(spec.selection)["ContentsCode"]
    measure = metadata["dimension"]["ContentsCode"]
    publisher_metadata = {
        "source": metadata["source"],
        "producer": "Statistics Sweden",
        "table_id": spec.table_id,
        "title": metadata["label"],
        "updated": metadata["updated"],
        "native_adjustment": spec.native_adjustment,
        "calendar_adjusted": spec.native_adjustment == "WorkAndSes",
        "seasonally_adjusted": spec.native_adjustment == "WorkAndSes",
        "price_type": spec.price_type,
        "base_period": spec.base_period,
        "native_unit": measure["category"]["unit"][metric],
        "native_reference_period": measure.get("extension", {}).get("refperiod", {}).get(metric),
        "dimensions": {
            name: {"code": code, "label": _label(metadata, name, code)}
            for name, code in spec.selection
        },
        "notes": metadata.get("note", []),
        "response_notes": response.get("note", []),
        "native_dimension_metadata": metadata["dimension"],
        "native_extension": metadata.get("extension", {}),
        "documentation_url": spec.documentation_url,
        "status_convention": (
            "observed denotes a published statistical observation, not finality; "
            "SCB's .. flag means unavailable or confidential, never zero. "
            "Original release notes describe revisions; no per-point finality is inferred."
        ),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }
    if key == "corporate_new_lending_rate":
        publisher_metadata.update(
            {
                "denomination_currency": "SEK",
                "rate_basis": "annualised agreed rate, amount weighted",
                "reference_period_convention": "agreements made during the month",
                "documentation_locator": (
                    "MIR instructions 2025-02-10 version6, sections3/3.1 (page4), "
                    "4 (page11), 5/5.1 (page15), 5.3 (page18)"
                ),
                "reference_period_note": (
                    "The table's generic month-end reference text covers both stocks "
                    "and new business; MIR instructions section3.1 specifies within-month "
                    "agreements for this new-business selection."
                ),
            }
        )
    missing = [row for row in observations if row["value"] is None]
    return {
        "indicator": key,
        "country": "SE",
        "source": SOURCE_SCB_MONITORING,
        "source_url": spec.source_url,
        "metadata_url": spec.metadata_url,
        "series_id": spec.series_id,
        "unit": spec.native_unit,
        "frequency": "monthly",
        "adjustment": (
            "seasonally_adjusted" if spec.native_adjustment == "WorkAndSes" else "not_adjusted"
        ),
        "definition": spec.definition,
        "publisher_updated_at": publication.isoformat(),
        "publisher_metadata": publisher_metadata,
        "observations": observations,
        "missingness": {
            "total": len(observations),
            "observed": len(observations) - len(missing),
            "not_reported": len(missing),
            "periods": [row["period"] for row in missing],
        },
    }


def _load_json(content: bytes) -> dict:
    if not isinstance(content, bytes) or not content:
        raise ValueError("SCB original content must be non-empty bytes")

    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"SCB JSON has duplicate key: {name}")
            result[name] = value
        return result

    try:
        result = json.loads(content, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("SCB original content is not valid unambiguous JSON") from exc
    if not isinstance(result, dict) or result.get("class") != "dataset":
        raise ValueError("SCB original content is not a JSON-stat dataset")
    return result


def _validate_dataset(raw: dict, spec: ScbMonitoringSpec, *, selected: bool) -> dict:
    try:
        if raw["version"] != "2.0" or raw["extension"]["px"]["tableid"] != spec.table_id:
            raise ValueError("SCB response has the wrong native table or JSON-stat version")
        if raw["source"] != spec.native_source:
            raise ValueError("SCB response has the wrong native source")
        ids, sizes = raw["id"], raw["size"]
        required = {name for name, _ in spec.selection} | {"Tid"}
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != required:
            raise ValueError("SCB response dimensions differ from the pinned selection")
        if not isinstance(sizes, list) or len(sizes) != len(ids):
            raise ValueError("SCB response dimension size mismatch")
        if set(raw["dimension"]) != required:
            raise ValueError("SCB response has unexpected dimension metadata")
        categories = {}
        for name, size in zip(ids, sizes, strict=True):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("SCB dimension size must be a positive integer")
            index = raw["dimension"][name]["category"]["index"]
            if not isinstance(index, dict) or len(index) != size:
                raise ValueError("SCB dimension category count mismatch")
            if any(type(position) is not int for position in index.values()):
                raise ValueError("SCB category positions must be integers")
            if set(index.values()) != set(range(size)):
                raise ValueError("SCB category positions must be unique and contiguous")
            categories[name] = sorted(index, key=index.get)
        for name, code in spec.selection:
            if code not in categories[name] or (selected and categories[name] != [code]):
                raise ValueError(f"SCB native selection differs for dimension {name}")
        metric = dict(spec.selection)["ContentsCode"]
        measure = raw["dimension"]["ContentsCode"]
        if measure["category"]["unit"][metric]["base"] != spec.native_unit:
            raise ValueError("SCB native unit changed")
        extension = measure["extension"]
        for name, expected in (
            ("adjustment", spec.native_adjustment), ("priceType", spec.price_type),
            ("basePeriod", spec.base_period),
        ):
            if extension.get(name, {}).get(metric) != expected:
                raise ValueError(f"SCB native measure {name} changed")
        return categories
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("SCB original has incomplete native metadata") from exc


def _label(raw: dict, name: str, code: str) -> str:
    value = raw["dimension"][name]["category"].get("label", {}).get(code)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SCB native label missing for {name}/{code}")
    return value


def _clock(value) -> datetime:
    if not isinstance(value, str):
        raise ValueError("SCB publisher clock missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("SCB publisher clock malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("SCB publisher clock must have an explicit timezone")
    return parsed.astimezone(UTC)


def _month(value: str) -> tuple[date, date]:
    match = re.fullmatch(r"(\d{4})M(\d{2})", value)
    if not match:
        raise ValueError(f"SCB period is not monthly: {value!r}")
    year, month = int(match[1]), int(match[2])
    start = date(year, month, 1)
    return start, date(year, month, calendar.monthrange(year, month)[1])


def _cells(raw, size: int, *, field: str) -> list:
    if raw is None and field == "status":
        return [None] * size
    if isinstance(raw, list):
        if len(raw) != size:
            raise ValueError(f"SCB {field} cell count mismatch")
        return raw
    if isinstance(raw, dict):
        if any(not re.fullmatch(r"0|[1-9]\d*", key) or int(key) >= size for key in raw):
            raise ValueError(f"SCB {field} contains an invalid cell position")
        return [raw.get(str(index)) for index in range(size)]
    raise ValueError(f"SCB {field} has an unsupported JSON-stat shape")
