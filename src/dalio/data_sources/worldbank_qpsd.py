"""World Bank/IMF Quarterly Public Sector Debt (QPSD) adapter.

QPSD is exposed through the World Bank Indicators API as source 20 (``PSD``).
The database is based on voluntary country submissions, so a missing country /
series is a coverage fact rather than a zero.  This adapter consequently emits
only observations actually returned by the publisher.  The companion pipeline
reports absent partitions and never manufactures or forward-fills them.

Each request covers one native QPSD indicator for the complete country basket.
The returned frame uses Dalio's canonical long shape and maps the registry's
World Bank/ISO-3 identifiers back to internal ISO-2 country codes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from dalio.countries import COUNTRIES, Country
from dalio.data_sources.sdmx_csv import DEFAULT_USER_AGENT, HttpClient, set_default_headers

logger = logging.getLogger(__name__)

QPSD_BASE_URL = "https://api.worldbank.org/v2"
QPSD_SOURCE_ID = 20
QPSD_SOURCE = "WORLD_BANK_QPSD"
DEFAULT_TIMEOUT = 30.0
# A full 22-country history can exceed 1,000 rows.  The Indicators API accepts
# 20,000 and returning a complete native series in one response materially
# reduces timeout and cross-page consistency risk.
PER_PAGE = 20_000

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_QUARTER_RE = re.compile(r"^(\d{4})-?Q([1-4])$")
_MISSING = object()


class QpsdResponseError(ValueError):
    """The upstream response cannot safely be interpreted as a QPSD page."""


class QpsdApiError(QpsdResponseError):
    """The World Bank returned its JSON error-message envelope."""


@dataclass(frozen=True)
class QpsdSeriesSpec:
    """One native central-government QPSD series, expressed as percent of GDP."""

    indicator: str
    wb_code: str
    description: str
    unit: str = "percent_of_gdp"

    @property
    def series_id(self) -> str:
        return self.wb_code


# Twelve raw components needed to assess refinancing, instrument, currency and
# creditor-residence risk.  These are kept raw; ratios between components are a
# downstream concern and must only be calculated within a coherent QPSD scope.
QPSD_SERIES: tuple[QpsdSeriesSpec, ...] = (
    QpsdSeriesSpec(
        "central_gov_debt_total_pct_gdp",
        "DP.DOD.DECT.CR.CG.Z1",
        "Gross central-government debt, all reported instruments",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_short_term_pct_gdp",
        "DP.DOD.DSTC.CR.CG.Z1",
        "Short-term central-government debt by original maturity",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_lt_due_1y_pct_gdp",
        "DP.DOD.DLTC.CR.L1.CG.Z1",
        "Long-term central-government debt due within one year",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_lt_due_over_1y_pct_gdp",
        "DP.DOD.DLTC.CR.M1.CG.Z1",
        "Long-term central-government debt due after one year",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_securities_pct_gdp",
        "DP.DOD.DLDS.CR.CG.Z1",
        "Central-government debt securities",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_loans_pct_gdp",
        "DP.DOD.DLLO.CR.CG.Z1",
        "Central-government loans",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_domestic_currency_pct_gdp",
        "DP.DOD.DECN.CR.CG.Z1",
        "Central-government debt denominated in domestic currency",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_foreign_currency_pct_gdp",
        "DP.DOD.DECF.CR.CG.Z1",
        "Central-government debt denominated in foreign currency",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_domestic_creditors_pct_gdp",
        "DP.DOD.DECD.CR.CG.Z1",
        "Central-government debt held by domestic creditors",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_external_creditors_pct_gdp",
        "DP.DOD.DECX.CR.CG.Z1",
        "Central-government debt held by external creditors",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_d1_pct_gdp",
        "DP.DOD.DLD1.CR.CG.Z1",
        "Central-government D1 debt (debt securities plus loans)",
    ),
    QpsdSeriesSpec(
        "central_gov_debt_d2a_pct_gdp",
        "DP.DOD.DLD2A.CR.CG.Z1",
        "Central-government D2A debt (Maastricht-like definition)",
    ),
)

# Descriptive aliases make call sites explicit while retaining a short canonical
# catalogue name.
CENTRAL_GOVERNMENT_PCT_GDP_SERIES = QPSD_SERIES
QPSD_COUNTRIES: tuple[Country, ...] = tuple(country for country in COUNTRIES if country.wb_id)


def parse_qpsd_period(value: object) -> date:
    """Convert ``2026Q1`` / ``2026-Q1`` to the first day of that quarter."""

    match = _QUARTER_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Unsupported QPSD quarterly period: {value!r}")
    year = int(match.group(1))
    quarter = int(match.group(2))
    return date(year, 1 + (quarter - 1) * 3, 1)


class WorldBankQpsdSource:
    """Fetch complete, paginated QPSD native-series histories for a basket."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        *,
        attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        page_pause_seconds: float = 0.25,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        for name, value in (
            ("cache_ttl_hours", cache_ttl_hours),
            ("retry_backoff_seconds", retry_backoff_seconds),
            ("page_pause_seconds", page_pause_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, "application/json")
        self._cache_dir = cache_dir or Path(
            os.environ.get("DALIO_QPSD_CACHE", "data/cache/worldbank_qpsd")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._attempts = attempts
        self._retry_backoff = retry_backoff_seconds
        self._page_pause = page_pause_seconds

    @staticmethod
    def url_for(
        spec: QpsdSeriesSpec,
        country_codes: Sequence[str],
        *,
        page: int = 1,
    ) -> str:
        """Build the official endpoint, always pinning QPSD source 20."""

        codes = [str(code).strip().upper() for code in country_codes]
        if not codes or any(not code for code in codes):
            raise ValueError("QPSD request needs at least one non-empty country code")
        if len(codes) != len(set(codes)):
            raise ValueError("QPSD request country codes must be unique")
        if page < 1:
            raise ValueError("QPSD page must be positive")
        return (
            f"{QPSD_BASE_URL}/country/{';'.join(codes)}/indicator/{spec.wb_code}"
            f"?format=json&per_page={PER_PAGE}&source={QPSD_SOURCE_ID}&page={page}"
        )

    def fetch(
        self,
        spec: QpsdSeriesSpec,
        countries: Sequence[Country] = QPSD_COUNTRIES,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return every non-null published observation for one native series.

        No date or MRV filter is sent: each successful result is a complete
        history suitable for an immutable per-country release snapshot.
        """

        country_codes, wb_to_iso2 = _country_mapping(countries)
        if not country_codes:
            return _empty_long()

        records: list[dict[str, object]] = []
        seen: set[tuple[str, date]] = set()
        expected_pages: int | None = None
        expected_total: int | None = None
        raw_count = 0
        page = 1
        requested_urls: list[str] = []
        pending_cache: list[tuple[str, str]] = []

        try:
            while True:
                url = self.url_for(spec, country_codes, page=page)
                requested_urls.append(url)
                meta, observations, fresh_text = self._fetch_page(url, use_cache=use_cache)
                _validate_source_id(meta)
                actual_page = _meta_int(meta, "page")
                if actual_page != page:
                    raise QpsdResponseError(
                        f"QPSD pagination mismatch: requested page {page}, received {actual_page}"
                    )

                reported_per_page = _meta_int(meta, "per_page")
                if reported_per_page != PER_PAGE:
                    raise QpsdResponseError(
                        "QPSD response per_page does not match the requested page size: "
                        f"received {reported_per_page}, expected {PER_PAGE}"
                    )

                reported_total = _meta_int(meta, "total", allow_zero=True, default=None)
                reported_pages = _meta_int(meta, "pages", allow_zero=True)
                # The Indicators API may describe a valid empty result as zero pages.
                normalized_pages = (
                    1 if reported_pages == 0 and reported_total == 0 else reported_pages
                )
                if normalized_pages < 1:
                    raise QpsdResponseError("QPSD response reported an invalid page count")

                if expected_pages is None:
                    expected_pages = normalized_pages
                    expected_total = reported_total
                elif normalized_pages != expected_pages or reported_total != expected_total:
                    raise QpsdResponseError("QPSD pagination metadata changed during retrieval")

                raw_count += len(observations)
                records.extend(
                    _observations_to_rows(
                        observations,
                        spec=spec,
                        wb_to_iso2=wb_to_iso2,
                        seen=seen,
                    )
                )
                if fresh_text is not None:
                    pending_cache.append((url, fresh_text))
                if page >= expected_pages:
                    break
                page += 1
                if self._page_pause:
                    time.sleep(self._page_pause)

            if expected_total is not None and raw_count != expected_total:
                raise QpsdResponseError(
                    f"QPSD pagination incomplete: received {raw_count} of {expected_total} rows"
                )
        except QpsdResponseError:
            # A cached page that is structurally valid JSON can still be wrong
            # for this partition (page/source/series/country/value).  Never let
            # such a body poison subsequent runs for the full TTL.
            self._discard_cached(requested_urls)
            raise

        for url, text in pending_cache:
            self._write_cache(url, text)
        if not records:
            return _empty_long()
        return (
            pd.DataFrame(records, columns=_COLUMNS)
            .sort_values(["country", "date"], kind="stable")
            .reset_index(drop=True)
        )

    def _fetch_page(self, url: str, *, use_cache: bool) -> tuple[dict, list[dict], str | None]:
        cache_path = self._cache_path_for(url)
        if use_cache:
            try:
                if cache_path.exists():
                    age = time.time() - cache_path.stat().st_mtime
                    if age < self._cache_ttl_seconds:
                        meta, rows = _parse_page(cache_path.read_text(encoding="utf-8"), url)
                        return meta, rows, None
            except (OSError, QpsdResponseError) as exc:
                logger.warning("Ignoring invalid QPSD cache %s: %s", cache_path, exc)

        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.get(url, timeout=DEFAULT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - injected HTTP clients vary
                last_error = exc
                if attempt == self._attempts - 1:
                    raise
                self._sleep_before_retry(url, attempt, exc)
                continue

            status = response.status_code
            if status in {403, 404} or 400 <= status < 500 and status != 429:
                raise ValueError(f"World Bank QPSD request failed ({status}): {url}")
            if status == 429 or status >= 500 or status < 200 or status >= 300:
                last_error = RuntimeError(f"World Bank QPSD server response {status}: {url}")
                if attempt == self._attempts - 1:
                    raise last_error
                self._sleep_before_retry(url, attempt, last_error)
                continue

            try:
                meta, rows = _parse_page(response.text, url)
            except QpsdApiError:
                raise
            except QpsdResponseError as exc:
                last_error = exc
                if attempt == self._attempts - 1:
                    raise
                self._sleep_before_retry(url, attempt, exc)
                continue

            # Cache only after every page and observation in the complete
            # response has passed semantic validation in ``fetch``.
            return meta, rows, response.text

        assert last_error is not None
        raise last_error

    def _sleep_before_retry(self, url: str, attempt: int, error: Exception) -> None:
        wait = self._retry_backoff * 2**attempt
        logger.warning(
            "QPSD %s attempt %d/%d failed (%s) — retrying in %.1fs",
            url,
            attempt + 1,
            self._attempts,
            error,
            wait,
        )
        time.sleep(wait)

    def _cache_path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{digest}.json"

    def _write_cache(self, url: str, text: str) -> None:
        cache_path = self._cache_path_for(url)
        try:
            cache_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write QPSD cache %s: %s", cache_path, exc)

    def _discard_cached(self, urls: Sequence[str]) -> None:
        for url in urls:
            cache_path = self._cache_path_for(url)
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not discard invalid QPSD cache %s: %s", cache_path, exc)


def _country_mapping(countries: Sequence[Country]) -> tuple[list[str], dict[str, str]]:
    codes: list[str] = []
    mapping: dict[str, str] = {}
    for country in countries:
        # wb_id is ISO-3 for countries and the API-native EMU id for the euro area.
        code = str(country.wb_id or country.iso3).strip().upper()
        if not code:
            raise ValueError(f"Country {country.iso2!r} has no QPSD/ISO-3 identifier")
        previous = mapping.get(code)
        if previous is not None and previous != country.iso2:
            raise ValueError(f"QPSD country code {code!r} maps to multiple basket rows")
        if previous is None:
            codes.append(code)
            mapping[code] = country.iso2
    return codes, mapping


def _parse_page(text: str, url: str) -> tuple[dict, list[dict]]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise QpsdResponseError(f"World Bank QPSD returned invalid JSON: {url}") from exc

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        messages = payload[0].get("message")
        if messages is not None:
            detail = _message_detail(messages)
            raise QpsdApiError(f"World Bank QPSD API error: {detail}")

    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or (payload[1] is not None and not isinstance(payload[1], list))
        or (isinstance(payload[1], list) and not all(isinstance(row, dict) for row in payload[1]))
    ):
        raise QpsdResponseError(f"World Bank QPSD returned an unsupported payload: {url}")
    return payload[0], payload[1] or []


def _message_detail(messages: object) -> str:
    if not isinstance(messages, list):
        return str(messages)
    parts = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("value") or message.get("key") or message))
        else:
            parts.append(str(message))
    return "; ".join(parts) or "unspecified upstream error"


def _meta_int(
    meta: dict,
    field: str,
    *,
    allow_zero: bool = False,
    default: int | None | object = _MISSING,
) -> int | None:
    raw = meta.get(field, default)
    if raw is _MISSING:
        raise QpsdResponseError(f"QPSD response is missing pagination field {field!r}")
    if raw is None and default is None:
        return None
    if isinstance(raw, bool) or isinstance(raw, float) and (
        not math.isfinite(raw) or not raw.is_integer()
    ):
        raise QpsdResponseError(f"QPSD pagination field {field!r} is invalid")
    try:
        value = int(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise QpsdResponseError(f"QPSD pagination field {field!r} is invalid") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise QpsdResponseError(f"QPSD pagination field {field!r} is invalid")
    return value


def _validate_source_id(meta: dict) -> None:
    raw = meta.get("sourceid")
    if raw is None:
        return
    if isinstance(raw, bool) or isinstance(raw, float) and (
        not math.isfinite(raw) or not raw.is_integer()
    ):
        raise QpsdResponseError("QPSD response sourceid is invalid")
    try:
        source_id = int(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise QpsdResponseError("QPSD response sourceid is invalid") from exc
    if source_id != QPSD_SOURCE_ID:
        raise QpsdResponseError(
            f"QPSD response came from source {source_id}, expected {QPSD_SOURCE_ID}"
        )


def _observations_to_rows(
    observations: list[dict],
    *,
    spec: QpsdSeriesSpec,
    wb_to_iso2: dict[str, str],
    seen: set[tuple[str, date]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, observation in enumerate(observations):
        wb_code = str(observation.get("countryiso3code") or "").strip().upper()
        iso2 = wb_to_iso2.get(wb_code)
        if iso2 is None:
            raise QpsdResponseError(
                f"QPSD observation {index} belongs to unrequested country {wb_code or '<blank>'!r}"
            )

        native_indicator = observation.get("indicator")
        if not isinstance(native_indicator, dict):
            raise QpsdResponseError(f"QPSD observation {index} has invalid indicator metadata")
        returned_code = native_indicator.get("id")
        if returned_code != spec.wb_code:
            raise QpsdResponseError(
                f"QPSD returned native series {returned_code!r}, expected {spec.wb_code!r}"
            )

        if "date" not in observation:
            raise QpsdResponseError(f"QPSD observation {index} is missing its quarter")
        try:
            observed_on = parse_qpsd_period(observation["date"])
        except ValueError as exc:
            raise QpsdResponseError(f"QPSD observation {index} has an invalid quarter") from exc

        key = (iso2, observed_on)
        if key in seen:
            raise QpsdResponseError(
                f"QPSD response contains duplicate {iso2} observation for {observed_on}"
            )
        seen.add(key)

        raw_value = observation.get("value")
        if raw_value is None:
            continue

        try:
            if isinstance(raw_value, bool):
                raise TypeError
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise QpsdResponseError(f"QPSD observation {index} has a non-numeric value") from exc
        if not math.isfinite(value):
            raise QpsdResponseError(f"QPSD observation {index} has a non-finite value")

        rows.append(
            {
                "country": iso2,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": QPSD_SOURCE,
                "series_id": spec.series_id,
            }
        )
    return rows


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)
