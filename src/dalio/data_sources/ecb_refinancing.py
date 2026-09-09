"""Pinned ECB euro-area government refinancing histories.

The ECB Government Finance Statistics (GFS) dataset publishes two compact,
decision-useful histories for the fixed-composition euro area 21 (``I10``):

* average residual maturity of outstanding general-government debt securities;
* scheduled gross redemptions during the coming 1--12 months, as percent of
  the annual moving sum of GDP.

These are publisher-defined, non-consolidated general-government measures.
They are not an issuance calendar for a single sovereign and must not be mixed
with a national debt-management-office denominator.  This adapter requests
the complete native monthly history, validates the full SDMX-CSV identity, and
does no resampling, interpolation, currency conversion, or imputation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    HttpClient,
    set_default_headers,
)

logger = logging.getLogger(__name__)

SOURCE_ECB_GFS = "ECB_GFS"
ECB_GFS_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data/GFS"
DEFAULT_TIMEOUT = 60.0

ECB_REFINANCING_NATIVE_PAYLOAD_SCHEMA_VERSION = "ecb-gfs-native-series-v1"
ECB_REFINANCING_MISSINGNESS_SCHEMA_VERSION = "ecb-gfs-missingness-v1"
ECB_REFINANCING_MISSING_VALUE_POLICY = (
    "ECB blank, dot, or double-dot OBS_VALUE cells remain missing; never zero-fill or impute them"
)

_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_STATUS_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,23}$")
_MISSING_TOKENS = frozenset({"", ".", ".."})


# ECB's csvdata representation includes the complete dimension identity plus
# observation and time-series attributes.  Pinning the ordered header makes a
# provider schema change explicit rather than silently dropping a new or
# renamed dimension.
ECB_GFS_CSV_COLUMNS: tuple[str, ...] = (
    "KEY",
    "FREQ",
    "ADJUSTMENT",
    "REF_AREA",
    "COUNTERPART_AREA",
    "REF_SECTOR",
    "COUNTERPART_SECTOR",
    "CONSOLIDATION",
    "ACCOUNTING_ENTRY",
    "STO",
    "INSTR_ASSET",
    "MATURITY",
    "EXPENDITURE",
    "UNIT_MEASURE",
    "CURRENCY_DENOM",
    "VALUATION",
    "PRICES",
    "TRANSFORMATION",
    "CUST_BREAKDOWN",
    "TIME_PERIOD",
    "OBS_VALUE",
    "OBS_STATUS",
    "CONF_STATUS",
    "PRE_BREAK_VALUE",
    "COMMENT_OBS",
    "EMBARGO_DATE",
    "OBS_EDP_WBB",
    "TIME_FORMAT",
    "COLL_PERIOD",
    "COMMENT_TS",
    "COMPILING_ORG",
    "CURRENCY",
    "CUST_BREAKDOWN_LB",
    "DATA_COMP",
    "DECIMALS",
    "DISS_ORG",
    "GFS_ECOFUNC",
    "GFS_TAXCAT",
    "LAST_UPDATE",
    "REF_PERIOD_DETAIL",
    "REF_YEAR_PRICE",
    "REPYEAREND",
    "REPYEARSTART",
    "TABLE_IDENTIFIER",
    "TIME_PER_COLLECT",
    "TITLE",
    "TITLE_COMPL",
    "UNIT_MULT",
    "COMMENT_DSET",
)


@dataclass(frozen=True)
class EcbRefinancingSeries:
    """Exact GFS identity and interpretation for one refinancing measure."""

    indicator: str
    native_series_id: str
    title: str
    unit: str
    native_unit_measure: str
    accounting_entry: str
    stock_flow: str
    maturity: str
    transformation: str
    collection_period: str
    definition: str
    country: str = "EA21"
    frequency: str = "monthly"
    source_family: str = SOURCE_ECB_GFS
    publisher: str = "European Central Bank"
    delivery_service: str = "ECB Data Portal SDMX API"
    reference_area: str = "I10"
    reference_area_label: str = "Euro area 21 (fixed composition) as of 1 January 2026"
    reference_sector: str = "S13"
    counterpart_sector: str = "S1"
    consolidation: str = "N"
    valuation: str = "F"
    expected_start: date = date(2009, 12, 1)
    verified_through: date = date(2026, 7, 1)

    @property
    def url(self) -> str:
        return build_ecb_gfs_csv_url(self)


ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY = EcbRefinancingSeries(
    indicator="euro_area_gov_debt_avg_residual_maturity_years",
    native_series_id="GFS.M.N.I10.W0.S13.S1.N.L.LE.F3.TT._Z.YR._T.F.V.A1._T",
    title=(
        "Average residual maturity for total government debt securities "
        "(non-consolidated - outstanding amounts) in years"
    ),
    unit="years",
    native_unit_measure="YR",
    accounting_entry="L",
    stock_flow="LE",
    maturity="TT",
    transformation="A1",
    collection_period="A",
    definition=(
        "Average residual maturity of outstanding non-consolidated general-government "
        "debt securities at face value"
    ),
)

ECB_GOV_REDEMPTIONS_1_12M = EcbRefinancingSeries(
    indicator="euro_area_gov_redemptions_1_12m_pct_gdp",
    native_series_id=("GFS.M.N.I10.W0.S13.S1.N.LD.F.F3.TS._Z.XDC_R_B1GQ_CY._T.F.V.C12._T"),
    title=(
        "Scheduled debt repayment of total government debt securities "
        "(non-consolidated), sum of coming months (between the 1st and 12th), "
        "percent of GDP"
    ),
    unit="percent_of_gdp",
    native_unit_measure="XDC_R_B1GQ_CY",
    accounting_entry="LD",
    stock_flow="F",
    maturity="TS",
    transformation="C12",
    collection_period="S",
    definition=(
        "Scheduled gross redemption of outstanding non-consolidated general-government "
        "debt securities during the coming 1--12 months, divided by the annual moving "
        "sum of GDP"
    ),
)

ECB_REFINANCING_SERIES: tuple[EcbRefinancingSeries, ...] = (
    ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY,
    ECB_GOV_REDEMPTIONS_1_12M,
)

_EXPECTED_SERIES_BY_ID = {spec.native_series_id: spec for spec in ECB_REFINANCING_SERIES}


def validate_ecb_refinancing_catalogue(
    specs: Iterable[EcbRefinancingSeries] = ECB_REFINANCING_SERIES,
) -> None:
    """Fail if the deliberately two-series denominator drifts."""
    selected = tuple(specs)
    expected_ids = {
        "GFS.M.N.I10.W0.S13.S1.N.L.LE.F3.TT._Z.YR._T.F.V.A1._T",
        ("GFS.M.N.I10.W0.S13.S1.N.LD.F.F3.TS._Z.XDC_R_B1GQ_CY._T.F.V.C12._T"),
    }
    expected_indicators = {
        "euro_area_gov_debt_avg_residual_maturity_years",
        "euro_area_gov_redemptions_1_12m_pct_gdp",
    }
    if len(selected) != 2:
        raise ValueError("ECB refinancing catalogue must contain exactly two native series")
    ids = [spec.native_series_id for spec in selected]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("ECB refinancing catalogue native-series denominator changed")
    if {spec.indicator for spec in selected} != expected_indicators:
        raise ValueError("ECB refinancing catalogue indicator denominator changed")
    for spec in selected:
        expected = _EXPECTED_SERIES_BY_ID.get(spec.native_series_id)
        if expected is None or spec != expected:
            raise ValueError(f"ECB refinancing series metadata changed: {spec.native_series_id!r}")
        if (
            spec.country != "EA21"
            or spec.frequency != "monthly"
            or spec.source_family != SOURCE_ECB_GFS
            or spec.reference_area != "I10"
            or spec.reference_sector != "S13"
            or spec.counterpart_sector != "S1"
            or spec.consolidation != "N"
            or spec.valuation != "F"
            or spec.expected_start != date(2009, 12, 1)
            or spec.verified_through != date(2026, 7, 1)
        ):
            raise ValueError(f"ECB refinancing series scope changed: {spec.native_series_id!r}")


def build_ecb_gfs_csv_url(spec_or_series_id: EcbRefinancingSeries | str) -> str:
    """Build the exact first-party full-history CSV endpoint.

    No ``startPeriod``, ``endPeriod``, transformation, or frequency query is
    allowed.  The leading ``GFS.`` remains part of stored series identity but
    is represented by the endpoint's dataset path rather than repeated in its
    key segment.
    """
    native_series_id = (
        spec_or_series_id.native_series_id
        if isinstance(spec_or_series_id, EcbRefinancingSeries)
        else str(spec_or_series_id)
    )
    if native_series_id not in _EXPECTED_SERIES_BY_ID:
        raise ValueError(f"unsupported ECB GFS refinancing series: {native_series_id!r}")
    key = native_series_id.removeprefix("GFS.")
    if not key or key == native_series_id:
        raise ValueError("ECB GFS native series ID must start with 'GFS.'")
    return f"{ECB_GFS_API_BASE_URL}/{key}?format=csvdata"


def ecb_refinancing_catalogue_sha256(
    specs: Iterable[EcbRefinancingSeries] = ECB_REFINANCING_SERIES,
) -> str:
    """Hash the complete pinned two-series semantics."""
    selected = tuple(specs)
    validate_ecb_refinancing_catalogue(selected)
    records = [
        {
            "accounting_entry": spec.accounting_entry,
            "collection_period": spec.collection_period,
            "consolidation": spec.consolidation,
            "country": spec.country,
            "definition": spec.definition,
            "delivery_service": spec.delivery_service,
            "expected_start": spec.expected_start.isoformat(),
            "frequency": spec.frequency,
            "indicator": spec.indicator,
            "maturity": spec.maturity,
            "native_series_id": spec.native_series_id,
            "native_unit_measure": spec.native_unit_measure,
            "publisher": spec.publisher,
            "reference_area": spec.reference_area,
            "reference_area_label": spec.reference_area_label,
            "reference_sector": spec.reference_sector,
            "source_family": spec.source_family,
            "stock_flow": spec.stock_flow,
            "title": spec.title,
            "transformation": spec.transformation,
            "unit": spec.unit,
            "valuation": spec.valuation,
            "verified_through": spec.verified_through.isoformat(),
        }
        for spec in sorted(selected, key=lambda item: item.native_series_id)
    ]
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


class EcbRefinancingSource:
    """Fetch and archive complete histories for the pinned ECB GFS series."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
        artifact_dir: Path | None = None,
    ) -> None:
        if not math.isfinite(cache_ttl_hours) or cache_ttl_hours < 0:
            raise ValueError("cache_ttl_hours must be a finite non-negative number")
        self._client = client or requests.Session()
        set_default_headers(
            self._client,
            user_agent,
            "text/csv, application/vnd.sdmx.data+csv;version=1.0.0;q=0.9",
        )
        resolved_cache = cache_dir or Path(
            os.environ.get(
                "DALIO_ECB_REFINANCING_CACHE",
                "data/cache/ecb_refinancing",
            )
        )
        self._cache_dir = resolved_cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._artifact_dir = artifact_dir or Path(
            os.environ.get(
                "DALIO_ECB_REFINANCING_ARTIFACTS",
                "data/artifacts/debt_refinancing/ecb_gfs",
            )
        )

    def fetch(
        self,
        spec: EcbRefinancingSeries,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch, validate, then atomically admit one complete native history."""
        _require_pinned_spec(spec)
        url = build_ecb_gfs_csv_url(spec)
        cached = self._read_valid_cache(url, spec) if use_cache else None
        if cached is None:
            source_bytes = self._request_source_bytes(url)
            frame = _parse_source_bytes(source_bytes, spec)
            self._write_cache(url, source_bytes)
        else:
            source_bytes, frame = cached

        # Parsing and every derived-document construction happen before any
        # durable artifact write.  Invalid publisher bytes therefore cannot be
        # admitted to the evidence archive.
        native_bytes = frame.attrs.pop("_native_payload_bytes")
        missing_bytes = frame.attrs.pop("_missing_provenance_bytes")
        assert isinstance(native_bytes, bytes)
        assert isinstance(missing_bytes, bytes)

        documents = (
            (
                source_bytes,
                self._artifact_dir / "source-responses",
                ".csv",
                "source response",
            ),
            (
                native_bytes,
                self._artifact_dir / "native-series",
                ".json",
                "canonical native payload",
            ),
            (
                missing_bytes,
                self._artifact_dir / "missingness-ledgers",
                ".json",
                "missingness ledger",
            ),
        )
        archived = _archive_document_batch(documents)
        (source_path, source_sha), (native_path, native_sha), (missing_path, missing_sha) = archived
        if native_sha != frame.attrs["native_payload_sha256"]:
            raise ValueError(
                f"ECB canonical native payload hash mismatch for {spec.native_series_id}"
            )
        if missing_sha != frame.attrs["missing_provenance_sha256"]:
            raise ValueError(f"ECB missingness ledger hash mismatch for {spec.native_series_id}")

        frame.attrs.update(
            {
                "source_url": url,
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_sha,
                "native_payload_artifact_path": str(native_path),
                "missing_provenance_artifact_path": str(missing_path),
            }
        )
        return frame

    def fetch_many(
        self,
        specs: Iterable[EcbRefinancingSeries] = ECB_REFINANCING_SERIES,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch a unique subset, retaining one exact response per native key."""
        selected = tuple(specs)
        if not selected:
            raise ValueError("at least one ECB refinancing series is required")
        ids = [spec.native_series_id for spec in selected]
        if len(ids) != len(set(ids)):
            raise ValueError("ECB refinancing request contains duplicate native series IDs")
        return {spec.native_series_id: self.fetch(spec, use_cache=use_cache) for spec in selected}

    def _read_valid_cache(
        self,
        url: str,
        spec: EcbRefinancingSeries,
    ) -> tuple[bytes, pd.DataFrame] | None:
        cache_path = self._cache_path_for(url)
        try:
            if not cache_path.exists():
                return None
            age = time.time() - cache_path.stat().st_mtime
            if age >= self._cache_ttl_seconds:
                return None
            source_bytes = cache_path.read_bytes()
            return source_bytes, _parse_source_bytes(source_bytes, spec)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring invalid ECB GFS refinancing cache %s: %s", cache_path, exc)
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                logger.warning(
                    "Could not discard invalid ECB GFS cache %s: %s",
                    cache_path,
                    unlink_exc,
                )
            return None

    def _request_source_bytes(self, url: str, attempts: int = 3) -> bytes:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.get(url, timeout=DEFAULT_TIMEOUT)
                status_code = response.status_code
                if status_code in {403, 404} or (400 <= status_code < 500 and status_code != 429):
                    raise ValueError(f"ECB GFS refinancing request failed ({status_code}): {url}")
                if status_code == 429 or status_code < 200 or status_code >= 300:
                    raise RuntimeError(f"ECB GFS refinancing server response {status_code}: {url}")
                source_bytes = response.content
                if not isinstance(source_bytes, bytes) or not source_bytes:
                    raise ValueError("ECB GFS returned an empty or non-byte response body")
                return source_bytes
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - injected HTTP clients vary
                last_error = exc
                if attempt < attempts - 1:
                    wait = 2**attempt
                    logger.warning(
                        "ECB GFS refinancing %s attempt %d/%d failed (%s) — retrying in %.1fs",
                        url,
                        attempt + 1,
                        attempts,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _cache_path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{digest}.csv"

    def _write_cache(self, url: str, payload: bytes) -> None:
        cache_path = self._cache_path_for(url)
        try:
            _atomic_write(cache_path, payload, hashlib.sha256(payload).hexdigest())
        except OSError as exc:
            logger.warning("Could not write ECB GFS refinancing cache %s: %s", cache_path, exc)


def parse_ecb_refinancing_csv(
    text: str,
    spec: EcbRefinancingSeries,
) -> pd.DataFrame:
    """Validate and convert one exact ECB GFS CSV history to release-long form."""
    _require_pinned_spec(spec)
    rows = _strict_csv_rows(text)
    if not rows:
        raise ValueError(f"ECB GFS series {spec.native_series_id} has no observation rows")

    expected_dimensions = _expected_dimensions(spec)
    native_periods: list[str] = []
    native_statuses: list[str] = []
    parsed_dates: list[date] = []
    parsed_values: list[float | None] = []
    missing_records: list[dict[str, object]] = []

    for position, row in enumerate(rows):
        _validate_dimension_row(row, expected_dimensions, spec)
        native_period = row["TIME_PERIOD"].strip()
        observed_on = _parse_month(native_period)
        native_status = _parse_native_status(row["OBS_STATUS"], spec)
        native_value = row["OBS_VALUE"].strip()
        value = _parse_native_value(native_value, spec)

        native_periods.append(native_period)
        native_statuses.append(native_status)
        parsed_dates.append(observed_on)
        parsed_values.append(value)
        if value is None:
            missing_records.append(
                {
                    "native_period": native_period,
                    "native_position": position,
                    "missing_kind": "blank" if native_value == "" else "null_token",
                    "native_token": native_value,
                    "native_obs_status": native_status,
                    "reason": "publisher_missing_not_zero",
                    "evidence": "ecb_csv_obs_value_cell",
                }
            )

    _validate_strict_period_order(parsed_dates, spec)
    source_updated_at = _source_updated_at(rows, spec)

    valid_positions = [index for index, value in enumerate(parsed_values) if value is not None]
    if not valid_positions:
        raise ValueError(f"ECB GFS series {spec.native_series_id} has no finite observations")
    frame = pd.DataFrame(
        {
            "country": [spec.country] * len(valid_positions),
            "indicator": [spec.indicator] * len(valid_positions),
            "date": [parsed_dates[index] for index in valid_positions],
            "value": [float(parsed_values[index]) for index in valid_positions],
            "source": [spec.source_family] * len(valid_positions),
            "series_id": [spec.native_series_id] * len(valid_positions),
            # Deliberately retain the publisher's native OBS_STATUS code.  The
            # generic release layer may normalize case, but the adapter does
            # not replace ECB status semantics with invented labels.
            "status": [native_statuses[index] for index in valid_positions],
        },
        columns=[*_LONG_COLUMNS, "status"],
    )

    native_payload = {
        "schema_version": ECB_REFINANCING_NATIVE_PAYLOAD_SCHEMA_VERSION,
        "native_series_id": spec.native_series_id,
        "columns": list(ECB_GFS_CSV_COLUMNS),
        "rows": rows,
    }
    missing_payload = {
        "schema_version": ECB_REFINANCING_MISSINGNESS_SCHEMA_VERSION,
        "native_series_id": spec.native_series_id,
        "missing_value_policy": ECB_REFINANCING_MISSING_VALUE_POLICY,
        "records": missing_records,
    }
    native_bytes = _canonical_json_bytes(native_payload)
    missing_bytes = _canonical_json_bytes(missing_payload)
    frame.attrs.update(
        {
            "native_periods": tuple(native_periods),
            "native_period_format": "P1M",
            "native_observation_statuses": tuple(native_statuses),
            "reference_area": spec.reference_area,
            "reference_area_label": spec.reference_area_label,
            "reference_sector": spec.reference_sector,
            "consolidation": spec.consolidation,
            "valuation": spec.valuation,
            "native_unit_measure": spec.native_unit_measure,
            "maturity_code": spec.maturity,
            "transformation_code": spec.transformation,
            "source_updated_at": source_updated_at,
            "missing_period_records": tuple(missing_records),
            "missing_value_policy": ECB_REFINANCING_MISSING_VALUE_POLICY,
            "native_payload_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "missing_provenance_sha256": hashlib.sha256(missing_bytes).hexdigest(),
            "native_payload_json": native_bytes.decode("utf-8"),
            "missing_provenance_json": missing_bytes.decode("utf-8"),
            "_native_payload_bytes": native_bytes,
            "_missing_provenance_bytes": missing_bytes,
        }
    )
    return frame


def _parse_source_bytes(source_bytes: bytes, spec: EcbRefinancingSeries) -> pd.DataFrame:
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("ECB GFS response is not valid UTF-8") from exc
    return parse_ecb_refinancing_csv(text, spec)


def _require_pinned_spec(spec: EcbRefinancingSeries) -> None:
    expected = _EXPECTED_SERIES_BY_ID.get(spec.native_series_id)
    if expected is None or spec != expected:
        raise ValueError(f"ECB refinancing series is not in the pinned catalogue: {spec!r}")


def _strict_csv_rows(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ECB GFS returned an empty response body")
    try:
        raw_rows = list(csv.reader(StringIO(text)))
    except csv.Error as exc:
        raise ValueError("ECB GFS returned malformed CSV") from exc
    if not raw_rows:
        raise ValueError("ECB GFS returned an empty response body")
    header = tuple(raw_rows[0])
    if header != ECB_GFS_CSV_COLUMNS:
        raise ValueError("ECB GFS CSV schema does not match the pinned ordered header")
    output: list[dict[str, str]] = []
    for position, cells in enumerate(raw_rows[1:], start=2):
        if not cells:
            continue
        if len(cells) != len(ECB_GFS_CSV_COLUMNS):
            raise ValueError(f"ECB GFS CSV row {position} has an unexpected field count")
        output.append(dict(zip(ECB_GFS_CSV_COLUMNS, cells, strict=True)))
    return output


def _expected_dimensions(spec: EcbRefinancingSeries) -> dict[str, str]:
    return {
        "KEY": spec.native_series_id,
        "FREQ": "M",
        "ADJUSTMENT": "N",
        "REF_AREA": spec.reference_area,
        "COUNTERPART_AREA": "W0",
        "REF_SECTOR": spec.reference_sector,
        "COUNTERPART_SECTOR": spec.counterpart_sector,
        "CONSOLIDATION": spec.consolidation,
        "ACCOUNTING_ENTRY": spec.accounting_entry,
        "STO": spec.stock_flow,
        "INSTR_ASSET": "F3",
        "MATURITY": spec.maturity,
        "EXPENDITURE": "_Z",
        "UNIT_MEASURE": spec.native_unit_measure,
        "CURRENCY_DENOM": "_T",
        "VALUATION": spec.valuation,
        "PRICES": "V",
        "TRANSFORMATION": spec.transformation,
        "CUST_BREAKDOWN": "_T",
        "TIME_FORMAT": "P1M",
        "CONF_STATUS": "F",
        "COMPILING_ORG": "4F0",
        "DECIMALS": "4",
        "TIME_PER_COLLECT": spec.collection_period,
        "UNIT_MULT": "0",
    }


def _validate_dimension_row(
    row: Mapping[str, str],
    expected_dimensions: Mapping[str, str],
    spec: EcbRefinancingSeries,
) -> None:
    for column, expected in expected_dimensions.items():
        actual = row[column].strip()
        if actual != expected:
            raise ValueError(
                f"ECB GFS {column} mismatch for {spec.native_series_id}: "
                f"expected {expected!r}, got {actual!r}"
            )
    composition_prefix = f"{spec.reference_area_label} - "
    if not row["COMMENT_TS"].strip().startswith(composition_prefix):
        raise ValueError(
            f"ECB GFS reference-area composition label mismatch for {spec.native_series_id}"
        )


def _parse_month(value: object) -> date:
    match = _MONTH_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"ECB GFS returned an invalid monthly period: {value!r}")
    return date(int(match.group(1)), int(match.group(2)), 1)


def _parse_native_status(value: object, spec: EcbRefinancingSeries) -> str:
    native = str(value).strip()
    if not _STATUS_RE.fullmatch(native):
        raise ValueError(
            f"ECB GFS returned an invalid OBS_STATUS for {spec.native_series_id}: {value!r}"
        )
    return native


def _parse_native_value(value: str, spec: EcbRefinancingSeries) -> float | None:
    if value in _MISSING_TOKENS:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"ECB GFS returned a non-numeric observation for {spec.native_series_id}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"ECB GFS returned a non-finite observation for {spec.native_series_id}")
    if parsed < 0:
        raise ValueError(f"ECB GFS returned a negative refinancing measure: {value!r}")
    return parsed


def _source_updated_at(
    rows: Sequence[Mapping[str, str]],
    spec: EcbRefinancingSeries,
) -> datetime | None:
    """Parse an optional uniform ECB LAST_UPDATE time-series attribute."""
    raw_values = {row["LAST_UPDATE"].strip() for row in rows}
    if raw_values == {""}:
        return None
    if "" in raw_values or len(raw_values) != 1:
        raise ValueError(
            f"ECB GFS LAST_UPDATE is inconsistent for {spec.native_series_id}: "
            f"{sorted(raw_values)!r}"
        )
    raw = next(iter(raw_values))
    try:
        parsed = pd.Timestamp(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ECB GFS LAST_UPDATE is invalid for {spec.native_series_id}: {raw!r}"
        ) from exc
    if pd.isna(parsed):
        raise ValueError(f"ECB GFS LAST_UPDATE is invalid for {spec.native_series_id}: {raw!r}")
    if parsed.tzinfo is None:
        raise ValueError(
            f"ECB GFS LAST_UPDATE lacks a timezone for {spec.native_series_id}: {raw!r}"
        )
    return parsed.tz_convert(UTC).to_pydatetime()


def _validate_strict_period_order(
    periods: Sequence[date],
    spec: EcbRefinancingSeries,
) -> None:
    if len(periods) != len(set(periods)):
        raise ValueError(
            f"ECB GFS returned duplicate observation periods for {spec.native_series_id}"
        )
    if list(periods) != sorted(periods):
        raise ValueError(
            f"ECB GFS observation periods are not strictly increasing for {spec.native_series_id}"
        )
    if periods[0] != spec.expected_start:
        raise ValueError(
            f"ECB GFS history start mismatch for {spec.native_series_id}: expected "
            f"{spec.expected_start.isoformat()}, got {periods[0].isoformat()}"
        )
    month_ordinals = [period.year * 12 + period.month for period in periods]
    if any(
        right - left != 1 for left, right in zip(month_ordinals, month_ordinals[1:], strict=False)
    ):
        raise ValueError(f"ECB GFS monthly history has a cadence gap for {spec.native_series_id}")
    if periods[-1] < spec.verified_through:
        raise ValueError(
            f"ECB GFS history does not reach the verified-through floor for "
            f"{spec.native_series_id}: expected at least "
            f"{spec.verified_through.isoformat()}, got {periods[-1].isoformat()}"
        )


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("ECB GFS payload cannot be represented as canonical JSON") from exc


def _content_addressed_target(payload: bytes, directory: Path, suffix: str) -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    return directory / digest[:2] / f"{digest}{suffix}", digest


def _archive_document_batch(
    documents: Sequence[tuple[bytes, Path, str, str]],
) -> tuple[tuple[Path, str], ...]:
    """Preflight all hashes, then atomically write each validated document."""
    targets = [
        (*_content_addressed_target(payload, directory, suffix), payload, label)
        for payload, directory, suffix, label in documents
    ]
    for destination, _digest, payload, label in targets:
        if not destination.exists():
            continue
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise ValueError(f"ECB GFS {label} artifact cannot be read: {destination}") from exc
        if existing != payload:
            raise ValueError(f"ECB GFS {label} artifact hash collision or corrupt archive")

    archived: list[tuple[Path, str]] = []
    for destination, digest, payload, _label in targets:
        if not destination.exists():
            _atomic_write(destination, payload, digest)
        archived.append((destination, digest))
    return tuple(archived)


def _atomic_write(destination: Path, payload: bytes, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


validate_ecb_refinancing_catalogue()
