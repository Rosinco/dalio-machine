"""Strict loader for page-located facts transcribed from official AP-fund reports.

This module does not download or interpret PDFs.  It validates the checked
reference transcription that connects every numeric fact to an immutable
official artifact and a physical PDF page.  Artifact bytes are independently
verified by the ingestion pipeline and storage layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

SOURCE_AP_FUNDS = "AP_FUNDS"
AP_FUND_REFERENCE_SCHEMA_VERSION = 1
AP_FUND_PARSER_NAME = "direct-table-transcription"
AP_FUND_PARSER_VERSION = "1"
AP_FUNDS_H1_2026_REFERENCE = (
    Path(__file__).resolve().parents[3] / "data" / "reference" / "ap_funds_h1_2026.json"
)

ALLOCATOR_FACT_COLUMNS = (
    "fund",
    "as_of_date",
    "period_start",
    "period_end",
    "record_type",
    "item_code",
    "reported_amount",
    "reported_unit",
    "amount_sek_mn",
    "exposure_pct",
    "basis",
    "row_role",
    "physical_page",
    "table_heading",
    "extraction_status",
    "quality_flag",
    "notes",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {"schema_version", "method_note", "releases"}
_RELEASE_FIELDS = {
    "fund",
    "report_date",
    "title",
    "source_url",
    "official_domains",
    "sha256",
    "page_count",
    "rows",
}
_REQUIRED_ROW_FIELDS = {
    "record_type",
    "item_code",
    "reported_amount",
    "reported_unit",
    "amount_sek_mn",
    "exposure_pct",
    "basis",
    "row_role",
    "physical_page",
    "table_heading",
}
_OPTIONAL_ROW_FIELDS = {
    "as_of_date",
    "period_start",
    "period_end",
    "extraction_status",
    "quality_flag",
    "notes",
}
_SEMANTIC_FACT_FIELDS = (
    "fund",
    "as_of_date",
    "period_start",
    "period_end",
    "record_type",
    "item_code",
    "basis",
    "row_role",
    "physical_page",
    "table_heading",
)


@dataclass(frozen=True)
class ApFundRelease:
    """One checked AP-fund report and its complete page-located fact table."""

    fund: str
    report_date: date
    title: str
    source_url: str
    official_domains: tuple[str, ...]
    sha256: str
    page_count: int
    facts: pd.DataFrame

    @property
    def artifact_filename(self) -> str:
        """Deterministic local input filename for an artifact directory."""
        if (self.report_date.month, self.report_date.day) == (6, 30):
            period = "h1"
        elif (self.report_date.month, self.report_date.day) == (12, 31):
            period = "fy"
        else:
            period = self.report_date.isoformat()
        return f"{self.fund.lower()}-{period}-{self.report_date.year}.pdf"


@dataclass(frozen=True)
class ApFundReference:
    """Versioned set of checked AP-fund report releases."""

    schema_version: int
    method_note: str
    releases: tuple[ApFundRelease, ...]


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"AP-fund reference contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"AP-fund reference contains non-finite JSON number {value}")


def _validate_fields(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    unexpected = set(value) - required - set(optional)
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string") from exc
    if value != parsed.isoformat():
        raise ValueError(f"{field} must use canonical YYYY-MM-DD format")
    return parsed


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _official_domains(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("official_domains must be a non-empty list")
    domains = tuple(_required_string(item, "official domain").lower().strip(".") for item in value)
    if any("/" in domain or ":" in domain for domain in domains):
        raise ValueError("official domains must be host names")
    if len(domains) != len(set(domains)):
        raise ValueError("official_domains contains duplicates")
    return domains


def _source_url(value: Any, domains: tuple[str, ...]) -> str:
    source_url = _required_string(value, "source_url")
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").lower().strip(".")
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise ValueError("source_url must be an HTTPS URL on an official domain")
    if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
        raise ValueError("source_url is outside the official domain allowlist")
    return source_url


def _parse_fact(
    raw: Any,
    *,
    fund: str,
    report_date: date,
    page_count: int,
    position: int,
) -> dict[str, Any]:
    context = f"AP-fund fact {fund}[{position}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _validate_fields(
        raw,
        required=_REQUIRED_ROW_FIELDS,
        optional=_OPTIONAL_ROW_FIELDS,
        context=context,
    )

    as_of_date = (
        _iso_date(raw["as_of_date"], f"{context}.as_of_date")
        if ("as_of_date" in raw)
        else report_date
    )
    if as_of_date > report_date:
        raise ValueError(f"{context}.as_of_date cannot be later than report_date")

    raw_period_start = raw.get("period_start")
    raw_period_end = raw.get("period_end")
    if (raw_period_start is None) != (raw_period_end is None):
        raise ValueError("allocator period_start and period_end must both be set or both be null")
    period_start = (
        _iso_date(raw_period_start, f"{context}.period_start")
        if raw_period_start is not None
        else None
    )
    period_end = (
        _iso_date(raw_period_end, f"{context}.period_end") if raw_period_end is not None else None
    )
    if period_start is not None and period_end < period_start:
        raise ValueError(f"{context}.period_end cannot be earlier than period_start")
    if period_end is not None and period_end > report_date:
        raise ValueError(f"{context}.period_end cannot be later than report_date")

    reported_amount = _optional_number(raw["reported_amount"], f"{context}.reported_amount")
    amount_sek_mn = _optional_number(raw["amount_sek_mn"], f"{context}.amount_sek_mn")
    exposure_pct = _optional_number(raw["exposure_pct"], f"{context}.exposure_pct")
    reported_unit = raw["reported_unit"]
    if reported_amount is None:
        if reported_unit is not None or amount_sek_mn is not None:
            raise ValueError(f"{context} amount normalization requires reported_amount")
    else:
        if reported_unit not in {"SEK_mn", "SEK_bn"}:
            raise ValueError(f"{context}.reported_unit must be SEK_mn or SEK_bn")
        if amount_sek_mn is None:
            raise ValueError(f"{context} reported amount requires amount_sek_mn normalization")
        expected = reported_amount if reported_unit == "SEK_mn" else reported_amount * 1_000
        if not math.isclose(amount_sek_mn, expected, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{context} has inconsistent SEK normalization")
    if reported_amount is None and exposure_pct is None:
        raise ValueError(f"{context} needs a reported amount or exposure percentage")

    physical_page = _positive_integer(raw["physical_page"], f"{context}.physical_page")
    if physical_page > page_count:
        raise ValueError(
            f"{context} physical page {physical_page} exceeds PDF page_count {page_count}"
        )

    return {
        "fund": fund,
        "as_of_date": as_of_date,
        "period_start": period_start,
        "period_end": period_end,
        "record_type": _required_string(raw["record_type"], f"{context}.record_type"),
        "item_code": _required_string(raw["item_code"], f"{context}.item_code"),
        "reported_amount": reported_amount,
        "reported_unit": reported_unit,
        "amount_sek_mn": amount_sek_mn,
        "exposure_pct": exposure_pct,
        "basis": _required_string(raw["basis"], f"{context}.basis"),
        "row_role": _required_string(raw["row_role"], f"{context}.row_role"),
        "physical_page": physical_page,
        "table_heading": _required_string(raw["table_heading"], f"{context}.table_heading"),
        "extraction_status": _required_string(
            raw.get("extraction_status", "model_visual_check"),
            f"{context}.extraction_status",
        ),
        "quality_flag": _required_string(
            raw.get("quality_flag", "none"), f"{context}.quality_flag"
        ),
        "notes": _optional_string(raw.get("notes"), f"{context}.notes"),
    }


def _parse_release(raw: Any, position: int) -> ApFundRelease:
    context = f"AP-fund release[{position}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    _validate_fields(raw, required=_RELEASE_FIELDS, context=context)

    fund = _required_string(raw["fund"], f"{context}.fund")
    report_date = _iso_date(raw["report_date"], f"{context}.report_date")
    title = _required_string(raw["title"], f"{context}.title")
    domains = _official_domains(raw["official_domains"])
    source_url = _source_url(raw["source_url"], domains)
    sha256 = _required_string(raw["sha256"], f"{context}.sha256")
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"{context}.sha256 must be 64 lowercase hexadecimal characters")
    page_count = _positive_integer(raw["page_count"], f"{context}.page_count")

    raw_rows = raw["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{context}.rows must be a non-empty list")
    rows = [
        _parse_fact(
            row,
            fund=fund,
            report_date=report_date,
            page_count=page_count,
            position=row_position,
        )
        for row_position, row in enumerate(raw_rows)
    ]
    semantic_keys = [tuple(row[field] for field in _SEMANTIC_FACT_FIELDS) for row in rows]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError(f"{context} contains a duplicate semantic fact")

    return ApFundRelease(
        fund=fund,
        report_date=report_date,
        title=title,
        source_url=source_url,
        official_domains=domains,
        sha256=sha256,
        page_count=page_count,
        facts=pd.DataFrame(rows, columns=ALLOCATOR_FACT_COLUMNS),
    )


def load_ap_fund_reference(path: Path = AP_FUNDS_H1_2026_REFERENCE) -> ApFundReference:
    """Load and fail-closed validate a versioned AP-fund fact reference file."""
    reference_path = Path(path)
    try:
        raw = json.loads(
            reference_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"AP-fund reference is invalid JSON: {reference_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("AP-fund reference root must be an object")
    _validate_fields(raw, required=_TOP_LEVEL_FIELDS, context="AP-fund reference")

    schema_version = raw["schema_version"]
    if type(schema_version) is not int or schema_version != AP_FUND_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"AP-fund reference schema_version must be {AP_FUND_REFERENCE_SCHEMA_VERSION}"
        )
    method_note = _required_string(raw["method_note"], "AP-fund reference method_note")
    raw_releases = raw["releases"]
    if not isinstance(raw_releases, list) or not raw_releases:
        raise ValueError("AP-fund reference releases must be a non-empty list")
    releases = tuple(
        _parse_release(release, position) for position, release in enumerate(raw_releases)
    )
    funds = [release.fund for release in releases]
    if len(funds) != len(set(funds)):
        raise ValueError("AP-fund reference contains a duplicate fund release")
    return ApFundReference(schema_version, method_note, releases)
