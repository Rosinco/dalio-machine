"""Eurostat general-government refinancing-risk scalar adapter.

The catalogue pins 29 annual partitions from four Eurostat government-debt
tables.  Each API request selects exactly one country and one native measure,
and deliberately omits a time filter so the response is a complete published
history.  Eurostat's JSON-stat payload is sparse: an absent or explicit-null
cell is evidence of missingness, never a zero.

Coverage is enforced against a deliberately dated floor in each catalogue
specification.  The native annual axis must be contiguous, start no later than
the verified first year, and extend at least through the verified-through year.
Publisher backfills and later annual additions are therefore accepted.  A
null or sparse-absent cell at either endpoint still proves that the native year
was returned, but does not become an observation; at least one finite value is
required for the partition as a whole.

The scope is general government (ESA 2010 sector ``S13``), which is not
interchangeable with the central-government scope in QPSD.  Ratios between
these measures are consequently a downstream concern and must retain their
Eurostat scope denominator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import DEFAULT_USER_AGENT, HttpClient, set_default_headers

logger = logging.getLogger(__name__)

EUROSTAT_REFINANCING_BASE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
)
EUROSTAT_REFINANCING_SOURCE = "EUROSTAT_GOV_DEBT"
EUROSTAT_REFINANCING_MISSINGNESS_SCHEMA_VERSION = "eurostat-refinancing-missingness-v1"
DEFAULT_TIMEOUT = 30.0

INDICATOR_AVG_RESIDUAL_MATURITY = "general_gov_debt_avg_residual_maturity_years"
INDICATOR_RMD_SCOPE = "general_gov_debt_rmd_scope_pct_gdp"
INDICATOR_DUE_LE1Y = "general_gov_debt_due_le1y_pct_gdp"
INDICATOR_FOREIGN_CURRENCY = "general_gov_debt_foreign_currency_pct_gdp"
INDICATOR_APPARENT_COST = "general_gov_debt_apparent_cost_pct"
INDICATOR_LT_VARIABLE_RATE = "general_gov_debt_lt_variable_rate_pct_gdp"

_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]
_YEAR_RE = re.compile(r"^[0-9]{4}$")
_SAFE_DATASET_RE = re.compile(r"^[a-z0-9_]+$")
_MISSING_VALUE_POLICY = (
    "Publisher JSON nulls and sparse absent native values remain missing; "
    "never zero-fill or impute them"
)

_DATASET_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "gov_10dd_rmd": ("freq", "sector", "maturity", "na_item", "unit", "geo", "time"),
    "gov_10dd_dcur": ("freq", "sector", "currency", "na_item", "unit", "geo", "time"),
    "gov_10dd_acd": ("freq", "sector", "unit", "geo", "time"),
    "gov_10dd_ggd": (
        "freq",
        "na_item",
        "sector2",
        "sector",
        "maturity",
        "unit",
        "geo",
        "time",
    ),
}


class EurostatRefinancingResponseError(ValueError):
    """An upstream body cannot safely be interpreted as the pinned partition."""


@dataclass(frozen=True)
class EurostatRefinancingSeries:
    """Exact metadata and singleton filters for one Eurostat native partition."""

    indicator: str
    country: str
    dataset: str
    dimension_codes: tuple[tuple[str, str], ...]
    title: str
    unit: str
    native_unit: str
    verified_start_year: int
    verified_through_year: int = 2025

    @property
    def series_id(self) -> str:
        """Stable official key assembled in the dataset's dimension order."""

        codes = ".".join(code for _dimension, code in self.dimension_codes)
        return f"{self.dataset.upper()}.{codes}"

    @property
    def dimensions(self) -> dict[str, str]:
        """Return a fresh mapping of the pinned non-time dimension codes."""

        return dict(self.dimension_codes)


_COUNTRIES: tuple[str, ...] = ("DE", "FR", "IT", "ES", "SE")


def _series(
    *,
    indicator: str,
    country: str,
    dataset: str,
    codes: tuple[tuple[str, str], ...],
    title: str,
    unit: str,
    native_unit: str,
    start: int,
) -> EurostatRefinancingSeries:
    return EurostatRefinancingSeries(
        indicator=indicator,
        country=country,
        dataset=dataset,
        dimension_codes=codes,
        title=title,
        unit=unit,
        native_unit=native_unit,
        verified_start_year=start,
    )


AVG_RESIDUAL_MATURITY_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_AVG_RESIDUAL_MATURITY,
        country=country,
        dataset="gov_10dd_rmd",
        codes=(
            ("freq", "A"),
            ("sector", "S13"),
            ("maturity", "TOTAL"),
            ("na_item", "GD"),
            ("unit", "YR"),
            ("geo", country),
        ),
        title="Average remaining maturity of general government gross debt",
        unit="years",
        native_unit="YR",
        start={"DE": 1995, "FR": 2020, "IT": 2020, "ES": 2020, "SE": 2021}[country],
    )
    for country in _COUNTRIES
)

RMD_SCOPE_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_RMD_SCOPE,
        country=country,
        dataset="gov_10dd_rmd",
        codes=(
            ("freq", "A"),
            ("sector", "S13"),
            ("maturity", "TOTAL"),
            ("na_item", "GD"),
            ("unit", "PC_GDP"),
            ("geo", country),
        ),
        title="General government gross debt in the remaining-maturity table",
        unit="percent_of_gdp",
        native_unit="PC_GDP",
        start={"DE": 1995, "FR": 2020, "IT": 2020, "ES": 2020, "SE": 2022}[country],
    )
    for country in _COUNTRIES
)

DUE_LE1Y_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_DUE_LE1Y,
        country=country,
        dataset="gov_10dd_rmd",
        codes=(
            ("freq", "A"),
            ("sector", "S13"),
            ("maturity", "Y_LE1"),
            ("na_item", "GD"),
            ("unit", "PC_GDP"),
            ("geo", country),
        ),
        title="General government gross debt with residual maturity of one year or less",
        unit="percent_of_gdp",
        native_unit="PC_GDP",
        start={"DE": 1995, "FR": 2020, "IT": 2020, "ES": 2020, "SE": 2022}[country],
    )
    for country in _COUNTRIES
)

FOREIGN_CURRENCY_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_FOREIGN_CURRENCY,
        country=country,
        dataset="gov_10dd_dcur",
        codes=(
            ("freq", "A"),
            ("sector", "S13"),
            ("currency", "FOR"),
            ("na_item", "GD"),
            ("unit", "PC_GDP"),
            ("geo", country),
        ),
        title="General government gross debt denominated in foreign currency",
        unit="percent_of_gdp",
        native_unit="PC_GDP",
        start={"DE": 2012, "FR": 2020, "IT": 2020, "ES": 2020, "SE": 2022}[country],
    )
    for country in _COUNTRIES
)

APPARENT_COST_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_APPARENT_COST,
        country=country,
        dataset="gov_10dd_acd",
        codes=(
            ("freq", "A"),
            ("sector", "S13"),
            ("unit", "RT"),
            ("geo", country),
        ),
        title="Apparent cost of general government gross debt",
        unit="percent",
        native_unit="RT",
        start={"DE": 1997, "FR": 2020, "IT": 2020, "ES": 2020, "SE": 2021}[country],
    )
    for country in _COUNTRIES
)

LT_VARIABLE_RATE_SERIES: tuple[EurostatRefinancingSeries, ...] = tuple(
    _series(
        indicator=INDICATOR_LT_VARIABLE_RATE,
        country=country,
        dataset="gov_10dd_ggd",
        codes=(
            ("freq", "A"),
            ("na_item", "GD_VAR"),
            ("sector2", "S1_S2"),
            ("sector", "S13"),
            ("maturity", "Y_GT1"),
            ("unit", "PC_GDP"),
            ("geo", country),
        ),
        title="Long-term variable-rate general government gross debt",
        unit="percent_of_gdp",
        native_unit="PC_GDP",
        start={"DE": 1995, "FR": 2020, "IT": 2020, "ES": 2020}[country],
    )
    # Sweden has no observations for this exact combination and is therefore
    # an explicit catalogue gap, not an empty thirtieth partition.
    for country in ("DE", "FR", "IT", "ES")
)

EUROSTAT_REFINANCING_SERIES: tuple[EurostatRefinancingSeries, ...] = (
    *AVG_RESIDUAL_MATURITY_SERIES,
    *RMD_SCOPE_SERIES,
    *DUE_LE1Y_SERIES,
    *FOREIGN_CURRENCY_SERIES,
    *APPARENT_COST_SERIES,
    *LT_VARIABLE_RATE_SERIES,
)


def _validate_catalogue(specs: Sequence[EurostatRefinancingSeries]) -> None:
    if len(specs) != 29:
        raise RuntimeError(
            f"Eurostat refinancing catalogue must contain 29 partitions, got {len(specs)}"
        )
    identities: set[tuple[str, str]] = set()
    for spec in specs:
        _validate_spec(spec)
        identity = (spec.country, spec.indicator)
        if identity in identities:
            raise RuntimeError(f"duplicate Eurostat refinancing catalogue partition: {identity}")
        identities.add(identity)


def _validate_spec(spec: EurostatRefinancingSeries) -> tuple[str, ...]:
    expected_dimensions = _DATASET_DIMENSIONS.get(spec.dataset)
    if expected_dimensions is None:
        raise ValueError(f"unsupported Eurostat refinancing dataset: {spec.dataset!r}")
    if not _SAFE_DATASET_RE.fullmatch(spec.dataset):
        raise ValueError(f"unsafe Eurostat refinancing dataset: {spec.dataset!r}")
    actual_dimensions = tuple(dimension for dimension, _code in spec.dimension_codes)
    expected_query_dimensions = tuple(
        dimension for dimension in expected_dimensions if dimension != "time"
    )
    if actual_dimensions != expected_query_dimensions:
        raise ValueError(
            f"Eurostat {spec.dataset} filter dimensions must be {expected_query_dimensions!r}, "
            f"got {actual_dimensions!r}"
        )
    codes = dict(spec.dimension_codes)
    if any(not isinstance(code, str) or not code for code in codes.values()):
        raise ValueError(f"Eurostat {spec.dataset} filter codes must be non-empty strings")
    if codes.get("freq") != "A":
        raise ValueError("Eurostat refinancing partitions must use annual frequency code 'A'")
    if codes.get("sector") != "S13":
        raise ValueError("Eurostat refinancing partitions must use general-government sector S13")
    if codes.get("geo") != spec.country:
        raise ValueError("Eurostat refinancing country must equal its pinned geo code")
    for field, year in (
        ("verified_start_year", spec.verified_start_year),
        ("verified_through_year", spec.verified_through_year),
    ):
        if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999:
            raise ValueError(f"Eurostat {field} must be a valid calendar year")
    if spec.verified_start_year > spec.verified_through_year:
        raise ValueError("Eurostat verified coverage start cannot follow its end")
    if not spec.indicator or not spec.title or not spec.unit or not spec.native_unit:
        raise ValueError("Eurostat refinancing catalogue text fields must be non-empty")
    return expected_dimensions


_validate_catalogue(EUROSTAT_REFINANCING_SERIES)


def build_eurostat_refinancing_url(spec: EurostatRefinancingSeries) -> str:
    """Build a full-history official JSON-stat API URL for one partition."""

    _validate_spec(spec)
    # No time, sinceTimePeriod, untilTimePeriod, or lastTimePeriod parameter is
    # permitted here: the immutable release must represent full native history.
    query = urlencode((("lang", "en"), *spec.dimension_codes))
    return f"{EUROSTAT_REFINANCING_BASE_URL}/{spec.dataset}?{query}"


class EurostatRefinancingSource:
    """Fetch and archive validated, full-history Eurostat scalar partitions."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        *,
        artifact_dir: Path | None = None,
        attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        for name, value in (
            ("cache_ttl_hours", cache_ttl_hours),
            ("retry_backoff_seconds", retry_backoff_seconds),
            ("timeout", timeout),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0 or name == "timeout" and value == 0:
                raise ValueError(f"{name} must be a finite non-negative number")

        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, "application/json")
        self._cache_dir = cache_dir or Path(
            os.environ.get(
                "DALIO_EUROSTAT_REFINANCING_CACHE",
                "data/cache/eurostat_refinancing",
            )
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir = artifact_dir or Path(
            os.environ.get(
                "DALIO_EUROSTAT_REFINANCING_ARTIFACTS",
                "data/artifacts/eurostat_refinancing",
            )
        )
        self._cache_ttl_seconds = float(cache_ttl_hours) * 3600
        self._attempts = attempts
        self._retry_backoff = float(retry_backoff_seconds)
        self._timeout = float(timeout)

    def fetch(
        self,
        spec: EurostatRefinancingSeries,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return all finite observations from one exact native partition."""

        url = build_eurostat_refinancing_url(spec)
        cached = self._read_valid_cache(url, spec) if use_cache else None
        if cached is None:
            source_bytes, frame, native_bytes, missing_bytes = self._fetch_validated(url, spec)
        else:
            source_bytes, frame, native_bytes, missing_bytes = cached

        artifacts = _archive_validated_bundle(
            source_bytes=source_bytes,
            native_bytes=native_bytes,
            missing_bytes=missing_bytes,
            artifact_dir=self._artifact_dir,
        )
        if cached is None:
            self._write_cache(url, source_bytes)

        frame.attrs.update(
            {
                "source_url": url,
                "source_artifact_path": str(artifacts["source"][0]),
                "source_artifact_sha256": artifacts["source"][1],
                "native_payload_artifact_path": str(artifacts["native"][0]),
                "native_payload_sha256": artifacts["native"][1],
                "missing_provenance_artifact_path": str(artifacts["missing"][0]),
                "missing_provenance_sha256": artifacts["missing"][1],
                "missing_provenance_json": missing_bytes.decode("utf-8"),
            }
        )
        return frame

    def fetch_all(
        self,
        specs: Sequence[EurostatRefinancingSeries] = EUROSTAT_REFINANCING_SERIES,
        *,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch each requested full-history partition without combining scopes."""

        selected = tuple(specs)
        series_ids = [spec.series_id for spec in selected]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("Eurostat refinancing fetch contains duplicate series IDs")
        return {spec.series_id: self.fetch(spec, use_cache=use_cache) for spec in selected}

    def _read_valid_cache(
        self,
        url: str,
        spec: EurostatRefinancingSeries,
    ) -> tuple[bytes, pd.DataFrame, bytes, bytes] | None:
        cache_path = self._cache_path_for(url)
        try:
            if not cache_path.exists():
                return None
            age = time.time() - cache_path.stat().st_mtime
            if age >= self._cache_ttl_seconds:
                return None
            source_bytes = cache_path.read_bytes()
            frame, native_bytes, missing_bytes = _parse_validated_payload(source_bytes, spec)
            return source_bytes, frame, native_bytes, missing_bytes
        except (OSError, EurostatRefinancingResponseError) as exc:
            logger.warning("Ignoring invalid Eurostat refinancing cache %s: %s", cache_path, exc)
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                logger.warning(
                    "Could not discard invalid Eurostat cache %s: %s", cache_path, unlink_exc
                )
            return None

    def _fetch_validated(
        self,
        url: str,
        spec: EurostatRefinancingSeries,
    ) -> tuple[bytes, pd.DataFrame, bytes, bytes]:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.get(url, timeout=self._timeout)
                status_code = response.status_code
                if status_code in {403, 404} or 400 <= status_code < 500 and status_code != 429:
                    raise ValueError(f"Eurostat refinancing request failed ({status_code}): {url}")
                if status_code == 429 or status_code < 200 or status_code >= 300:
                    raise RuntimeError(f"Eurostat refinancing server response {status_code}: {url}")
                source_bytes = _response_content(response)
                frame, native_bytes, missing_bytes = _parse_validated_payload(
                    source_bytes,
                    spec,
                )
                return source_bytes, frame, native_bytes, missing_bytes
            except ValueError as exc:
                if not isinstance(exc, EurostatRefinancingResponseError):
                    raise
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - injected HTTP clients vary
                last_error = exc

            if attempt == self._attempts - 1:
                assert last_error is not None
                raise last_error
            wait = self._retry_backoff * 2**attempt
            logger.warning(
                "Eurostat refinancing %s attempt %d/%d failed (%s) — retrying in %.1fs",
                url,
                attempt + 1,
                self._attempts,
                last_error,
                wait,
            )
            if wait:
                time.sleep(wait)
        raise AssertionError("unreachable Eurostat retry state")

    def _cache_path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{digest}.json"

    def _write_cache(self, url: str, payload: bytes) -> None:
        cache_path = self._cache_path_for(url)
        try:
            _atomic_write(cache_path, payload)
        except OSError as exc:
            logger.warning("Could not write Eurostat refinancing cache %s: %s", cache_path, exc)


def parse_eurostat_refinancing_json(
    body: bytes | str,
    spec: EurostatRefinancingSeries,
) -> pd.DataFrame:
    """Validate one JSON-stat response and return its canonical long frame."""

    frame, _native_bytes, _missing_bytes = _parse_validated_payload(body, spec)
    return frame


def _parse_validated_payload(
    body: bytes | str,
    spec: EurostatRefinancingSeries,
) -> tuple[pd.DataFrame, bytes, bytes]:
    expected_dimensions = _validate_spec(spec)
    source_bytes = _coerce_utf8_bytes(body)
    payload = _load_json(source_bytes)
    if not isinstance(payload, dict):
        raise EurostatRefinancingResponseError("Eurostat JSON-stat response must be an object")

    _validate_dataset_identity(payload, spec)
    dimension_order, sizes, categories = _validate_dimensions(
        payload,
        spec,
        expected_dimensions,
    )
    source_updated_at = _parse_source_updated_at(payload)
    cell_count = math.prod(sizes)
    value_cells, value_presence = _cell_vector(
        payload.get("value", _ABSENT),
        cell_count,
        field="value",
        required=True,
    )
    status_cells, status_presence = _cell_vector(
        payload.get("status", _ABSENT),
        cell_count,
        field="status",
        required=False,
    )

    time_codes = categories["time"]
    time_by_position = {position: code for code, position in time_codes.items()}
    ordered_years = [
        _parse_year(time_by_position[position])
        for position in range(sizes[dimension_order.index("time")])
    ]
    _validate_history_axis(ordered_years, spec)

    rows: list[dict[str, object]] = []
    missing_records: list[dict[str, object]] = []
    status_records: list[dict[str, object]] = []
    seen_years: set[int] = set()
    for flat_position in range(cell_count):
        coordinates = _coordinates(flat_position, dimension_order, sizes)
        time_position = coordinates["time"]
        native_period = time_by_position[time_position]
        observed_on = _parse_year(native_period)
        if observed_on.year in seen_years:
            raise EurostatRefinancingResponseError(
                f"Eurostat response contains duplicate annual period {native_period!r}"
            )
        seen_years.add(observed_on.year)

        raw_status = status_cells[flat_position]
        canonical_status = _canonical_status(raw_status, native_period=native_period)
        status_records.append(
            {
                "native_period": native_period,
                "native_position": flat_position,
                "native_status": raw_status,
                "status_presence": status_presence[flat_position],
                "canonical_status": canonical_status,
            }
        )

        raw_value = value_cells[flat_position]
        if raw_value is None:
            presence = value_presence[flat_position]
            missing_kind = "json_null" if presence == "explicit" else "sparse_absent"
            missing_records.append(
                {
                    "native_period": native_period,
                    "native_position": flat_position,
                    "missing_kind": missing_kind,
                    "native_token": None,
                    "reason": (
                        "publisher_null_not_zero"
                        if missing_kind == "json_null"
                        else "publisher_sparse_absence_not_zero"
                    ),
                    "evidence": "eurostat_jsonstat_value_cell",
                    "native_status": raw_status,
                    "status_presence": status_presence[flat_position],
                }
            )
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise EurostatRefinancingResponseError(
                f"Eurostat period {native_period!r} has a non-numeric value"
            )
        try:
            value = float(raw_value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise EurostatRefinancingResponseError(
                f"Eurostat period {native_period!r} has an invalid numeric value"
            ) from exc
        if not math.isfinite(value):
            raise EurostatRefinancingResponseError(
                f"Eurostat period {native_period!r} has a non-finite value"
            )
        if value == 0.0:
            value = 0.0
        rows.append(
            {
                "country": spec.country,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": EUROSTAT_REFINANCING_SOURCE,
                "series_id": spec.series_id,
                "status": canonical_status,
            }
        )

    frame = pd.DataFrame(rows, columns=_LONG_COLUMNS)
    if frame.empty:
        raise EurostatRefinancingResponseError(
            f"Eurostat partition {spec.series_id} has no finite observations"
        )
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    native_bytes = _canonical_json_bytes(payload, label="Eurostat native payload")
    missing_payload = {
        "schema_version": EUROSTAT_REFINANCING_MISSINGNESS_SCHEMA_VERSION,
        "dataset": spec.dataset,
        "series_id": spec.series_id,
        "missing_value_policy": _MISSING_VALUE_POLICY,
        "records": missing_records,
        "status_records": status_records,
    }
    missing_bytes = _canonical_json_bytes(
        missing_payload,
        label="Eurostat missingness ledger",
    )
    frame.attrs.update(
        {
            "source_url": build_eurostat_refinancing_url(spec),
            "native_payload_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "missing_provenance_sha256": hashlib.sha256(missing_bytes).hexdigest(),
            "missing_provenance_json": missing_bytes.decode("utf-8"),
            "missing_period_records": tuple(missing_records),
            "source_updated_at": source_updated_at,
        }
    )
    return frame, native_bytes, missing_bytes


_ABSENT = object()


def _coerce_utf8_bytes(body: bytes | str) -> bytes:
    if isinstance(body, str):
        body = body.encode("utf-8")
    if not isinstance(body, bytes) or not body:
        raise EurostatRefinancingResponseError("Eurostat returned an empty response body")
    try:
        body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EurostatRefinancingResponseError("Eurostat response is not valid UTF-8") from exc
    return body


def _load_json(body: bytes) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EurostatRefinancingResponseError(
                    f"Eurostat JSON contains duplicate object key {key!r}"
                )
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise EurostatRefinancingResponseError(
            f"Eurostat JSON contains non-standard numeric token {token!r}"
        )

    try:
        return json.loads(
            body.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise EurostatRefinancingResponseError("Eurostat returned malformed JSON") from exc


def _validate_dataset_identity(
    payload: Mapping[str, object],
    spec: EurostatRefinancingSeries,
) -> None:
    if payload.get("version") != "2.0" or payload.get("class") != "dataset":
        raise EurostatRefinancingResponseError("Eurostat response is not a JSON-stat 2.0 dataset")
    if payload.get("source") != "ESTAT":
        raise EurostatRefinancingResponseError(
            f"Eurostat response source must be 'ESTAT', got {payload.get('source')!r}"
        )
    extension = payload.get("extension")
    if not isinstance(extension, dict):
        raise EurostatRefinancingResponseError("Eurostat response is missing dataset extension")
    expected_dataset = spec.dataset.upper()
    if extension.get("id") != expected_dataset or extension.get("agencyId") != "ESTAT":
        raise EurostatRefinancingResponseError(
            f"Eurostat dataset identity mismatch: expected {expected_dataset!r} from ESTAT"
        )
    if extension.get("lang") != "EN":
        raise EurostatRefinancingResponseError("Eurostat response language must be EN")
    datastructure = extension.get("datastructure")
    if (
        not isinstance(datastructure, dict)
        or datastructure.get("id") != expected_dataset
        or datastructure.get("agencyId") != "ESTAT"
    ):
        raise EurostatRefinancingResponseError(
            f"Eurostat data-structure identity mismatch for {expected_dataset}"
        )


def _validate_dimensions(
    payload: Mapping[str, object],
    spec: EurostatRefinancingSeries,
    expected_dimensions: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...], dict[str, dict[str, int]]]:
    raw_order = payload.get("id")
    raw_sizes = payload.get("size")
    raw_dimensions = payload.get("dimension")
    if (
        not isinstance(raw_order, list)
        or not all(isinstance(item, str) and item for item in raw_order)
        or len(raw_order) != len(set(raw_order))
    ):
        raise EurostatRefinancingResponseError("Eurostat dimension order is malformed")
    dimension_order = tuple(raw_order)
    if set(dimension_order) != set(expected_dimensions):
        raise EurostatRefinancingResponseError(
            f"Eurostat {spec.dataset} dimension schema mismatch: expected "
            f"{sorted(expected_dimensions)!r}, got {sorted(dimension_order)!r}"
        )
    if (
        not isinstance(raw_sizes, list)
        or len(raw_sizes) != len(dimension_order)
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in raw_sizes
        )
    ):
        raise EurostatRefinancingResponseError("Eurostat dimension sizes are malformed")
    sizes = tuple(raw_sizes)
    if not isinstance(raw_dimensions, dict) or set(raw_dimensions) != set(dimension_order):
        raise EurostatRefinancingResponseError(
            "Eurostat dimension metadata does not exactly match the declared dimensions"
        )

    categories: dict[str, dict[str, int]] = {}
    for dimension, declared_size in zip(dimension_order, sizes, strict=True):
        metadata = raw_dimensions[dimension]
        if not isinstance(metadata, dict):
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} metadata is malformed"
            )
        category = metadata.get("category")
        if not isinstance(category, dict) or "index" not in category:
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} has no category index"
            )
        index = _category_index(category["index"], dimension)
        if len(index) != declared_size:
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} size does not match its category index"
            )
        categories[dimension] = index

    expected_codes = spec.dimensions
    for dimension, expected_code in expected_codes.items():
        actual_codes = set(categories[dimension])
        if actual_codes != {expected_code} or sizes[dimension_order.index(dimension)] != 1:
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} code mismatch: expected only "
                f"{expected_code!r}, got {sorted(actual_codes)!r}"
            )
    return dimension_order, sizes, categories


def _category_index(raw_index: object, dimension: str) -> dict[str, int]:
    if isinstance(raw_index, list):
        if not all(isinstance(code, str) and code for code in raw_index):
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} category list is malformed"
            )
        if len(raw_index) != len(set(raw_index)):
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} contains duplicate category codes"
            )
        return {code: position for position, code in enumerate(raw_index)}
    if not isinstance(raw_index, dict):
        raise EurostatRefinancingResponseError(
            f"Eurostat dimension {dimension!r} category index is malformed"
        )
    normalized: dict[str, int] = {}
    for code, position in raw_index.items():
        if (
            not isinstance(code, str)
            or not code
            or isinstance(position, bool)
            or not isinstance(position, int)
        ):
            raise EurostatRefinancingResponseError(
                f"Eurostat dimension {dimension!r} category index is malformed"
            )
        normalized[code] = position
    if set(normalized.values()) != set(range(len(normalized))):
        raise EurostatRefinancingResponseError(
            f"Eurostat dimension {dimension!r} category positions are not contiguous"
        )
    return normalized


def _cell_vector(
    raw: object,
    expected_count: int,
    *,
    field: str,
    required: bool,
) -> tuple[list[object | None], list[str]]:
    if raw is _ABSENT:
        if required:
            raise EurostatRefinancingResponseError(f"Eurostat response is missing {field!r}")
        return [None] * expected_count, ["not_supplied"] * expected_count
    if isinstance(raw, list):
        if len(raw) != expected_count:
            raise EurostatRefinancingResponseError(
                f"Eurostat {field} length {len(raw)} does not match cell count {expected_count}"
            )
        return list(raw), ["explicit"] * expected_count
    if isinstance(raw, dict):
        values: list[object | None] = [None] * expected_count
        presence = ["sparse_absent"] * expected_count
        for raw_position, value in raw.items():
            if not isinstance(raw_position, str) or not re.fullmatch(
                r"0|[1-9][0-9]*", raw_position
            ):
                raise EurostatRefinancingResponseError(
                    f"Eurostat sparse {field} has an invalid cell position"
                )
            position = int(raw_position)
            if position >= expected_count:
                raise EurostatRefinancingResponseError(
                    f"Eurostat sparse {field} contains an out-of-range cell position"
                )
            values[position] = value
            presence[position] = "explicit"
        return values, presence
    raise EurostatRefinancingResponseError(f"Eurostat response uses an unsupported {field} shape")


def _coordinates(
    flat_position: int,
    dimension_order: tuple[str, ...],
    sizes: tuple[int, ...],
) -> dict[str, int]:
    remainder = flat_position
    coordinates: dict[str, int] = {}
    for dimension, size in reversed(tuple(zip(dimension_order, sizes, strict=True))):
        coordinates[dimension] = remainder % size
        remainder //= size
    if remainder:
        raise EurostatRefinancingResponseError("Eurostat flat cell position exceeds data cube")
    return coordinates


def _parse_year(native_period: str) -> date:
    if not isinstance(native_period, str) or not _YEAR_RE.fullmatch(native_period):
        raise EurostatRefinancingResponseError(
            f"Eurostat annual period is invalid: {native_period!r}"
        )
    try:
        return date(int(native_period), 1, 1)
    except ValueError as exc:
        raise EurostatRefinancingResponseError(
            f"Eurostat annual period is invalid: {native_period!r}"
        ) from exc


def _validate_history_axis(
    ordered_years: Sequence[date],
    spec: EurostatRefinancingSeries,
) -> None:
    """Enforce the dated history floor using native axis evidence only.

    A JSON-stat time category proves that the publisher returned that year,
    even when its value is null. Publisher extensions outside the verified
    window are accepted, but every returned annual category must be contiguous.
    """

    if not ordered_years:
        raise EurostatRefinancingResponseError(
            f"Eurostat partition {spec.series_id} has no native annual periods"
        )
    if ordered_years != sorted(ordered_years):
        raise EurostatRefinancingResponseError(
            "Eurostat time categories are not in ascending native position order"
        )
    year_numbers = [observed_on.year for observed_on in ordered_years]
    gaps = [
        (left, right)
        for left, right in zip(year_numbers, year_numbers[1:], strict=False)
        if right != left + 1
    ]
    if gaps:
        left, right = gaps[0]
        raise EurostatRefinancingResponseError(
            f"Eurostat annual time axis has a gap between {left} and {right}"
        )
    if year_numbers[0] > spec.verified_start_year:
        raise EurostatRefinancingResponseError(
            f"Eurostat partition {spec.series_id} starts in {year_numbers[0]}; "
            f"verified history starts by {spec.verified_start_year}"
        )
    if year_numbers[-1] < spec.verified_through_year:
        raise EurostatRefinancingResponseError(
            f"Eurostat partition {spec.series_id} ends in {year_numbers[-1]}; "
            f"verified history extends through {spec.verified_through_year}"
        )


def _canonical_status(raw_status: object, *, native_period: str) -> str:
    if raw_status is None:
        return "observed"
    if not isinstance(raw_status, str):
        raise EurostatRefinancingResponseError(
            f"Eurostat period {native_period!r} has a non-string status flag"
        )
    status = raw_status.strip().lower()
    if not status:
        return "observed"
    if len(status) > 24:
        raise EurostatRefinancingResponseError(
            f"Eurostat period {native_period!r} status exceeds release storage width"
        )
    return status


def _parse_source_updated_at(payload: Mapping[str, object]) -> datetime:
    raw = payload.get("updated")
    if not isinstance(raw, str) or not raw.strip():
        raise EurostatRefinancingResponseError("Eurostat response has no source update timestamp")
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise EurostatRefinancingResponseError(
            f"Eurostat source update timestamp is invalid: {raw!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EurostatRefinancingResponseError(
            "Eurostat source update timestamp must include a UTC offset"
        )
    return parsed.astimezone(UTC)


def _canonical_json_bytes(payload: object, *, label: str) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EurostatRefinancingResponseError(
            f"{label} cannot be represented as canonical JSON"
        ) from exc


def _response_content(response: object) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise EurostatRefinancingResponseError(
            "Eurostat HTTP response exposes neither byte content nor decoded text"
        )
    return text.encode("utf-8")


def _archive_validated_bundle(
    *,
    source_bytes: bytes,
    native_bytes: bytes,
    missing_bytes: bytes,
    artifact_dir: Path,
) -> dict[str, tuple[Path, str]]:
    payloads = {
        "source": (source_bytes, artifact_dir / "source-responses"),
        "native": (native_bytes, artifact_dir / "native-payloads"),
        "missing": (missing_bytes, artifact_dir / "missingness-ledgers"),
    }
    planned: dict[str, tuple[Path, str, bytes]] = {}
    # Preflight every existing destination before writing any member.  A
    # corrupt derived ledger therefore cannot leave a newly partial bundle.
    for role, (payload, directory) in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        destination = directory / digest[:2] / f"{digest}.json"
        if destination.exists():
            try:
                existing = destination.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"Eurostat content-addressed artifact cannot be read: {destination}"
                ) from exc
            if existing != payload:
                raise ValueError(
                    f"Eurostat artifact hash collision or corrupt archive: {destination}"
                )
        planned[role] = (destination, digest, payload)

    for destination, _digest, payload in planned.values():
        if not destination.exists():
            _atomic_write(destination, payload)
    return {
        role: (destination, digest) for role, (destination, digest, _payload) in planned.items()
    }


def _atomic_write(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
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
