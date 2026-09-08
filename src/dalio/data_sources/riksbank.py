"""Riksbank SWEA adapter for official Swedish rates and exchange rates.

The SWEA observations endpoint is public and can be used without an API key.
Unauthenticated clients are limited to five calls per minute, so the companion
pipeline advertises a deliberately conservative default interval between calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RIKSBANK_API_BASE = "https://api.riksbank.se/swea/v1"
RIKSBANK_SOURCE = "RIKSBANK_SWEA"
RIKSBANK_API_KEY_HEADER = "Ocp-Apim-Subscription-Key"
DEFAULT_FROM_DATE = date(2005, 1, 1)
NOK_COMPARABLE_FROM = date(2023, 11, 27)
DEFAULT_TIMEOUT = 30.0

# SWEA's published keyless quota is five requests/minute. Twelve seconds is the
# exact boundary, so use a small safety margin rather than running on the edge.
UNAUTHENTICATED_CALLS_PER_MINUTE = 5
DEFAULT_UNAUTHENTICATED_PACING_SECONDS = 13.0
AUTHENTICATED_CALLS_PER_MINUTE = 200
DEFAULT_AUTHENTICATED_PACING_SECONDS = 0.35

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


@dataclass(frozen=True)
class RiksbankSeriesSpec:
    """One native SWEA series mapped to Dalio's indicator vocabulary."""

    indicator: str
    series_id: str
    unit: str
    description: str
    country: str = "SE"
    frequency: str = "D"
    history_start: date = DEFAULT_FROM_DATE


# These eight identifiers were verified against the current observations API.
# SEKKIXPMI is intentionally absent: that legacy alias returned HTTP 204 during
# verification and must not be presented as a working canonical KIX series.
RIKSBANK_SERIES: tuple[RiksbankSeriesSpec, ...] = (
    RiksbankSeriesSpec(
        "policy_rate",
        "SECBREPOEFF",
        "percent",
        "Effective Riksbank policy/repo rate",
    ),
    RiksbankSeriesSpec(
        "yield_2y",
        "SEGVB2YC",
        "percent",
        "Swedish government 2-year benchmark yield",
    ),
    RiksbankSeriesSpec(
        "yield_5y",
        "SEGVB5YC",
        "percent",
        "Swedish government 5-year benchmark yield",
    ),
    RiksbankSeriesSpec(
        "yield_10y",
        "SEGVB10YC",
        "percent",
        "Swedish government 10-year benchmark yield",
    ),
    RiksbankSeriesSpec(
        "sek_per_usd",
        "SEKUSDPMI",
        "SEK per USD",
        "Indicative SEK per US dollar mid-rate",
    ),
    RiksbankSeriesSpec(
        "sek_per_eur",
        "SEKEURPMI",
        "SEK per EUR",
        "Indicative SEK per euro mid-rate",
    ),
    RiksbankSeriesSpec(
        "sek_per_nok",
        "SEKNOKPMI",
        "SEK per NOK",
        "Indicative SEK per Norwegian krone mid-rate",
        history_start=NOK_COMPARABLE_FROM,
    ),
    RiksbankSeriesSpec(
        "sek_per_gbp",
        "SEKGBPPMI",
        "SEK per GBP",
        "Indicative SEK per pound sterling mid-rate",
    ),
)


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> requests.Response: ...


class RiksbankSource:
    """Fetch complete SWEA observation windows in Dalio's long format."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        api_key: str | None = None,
    ) -> None:
        self._client = client or requests.Session()
        self._cache_dir = cache_dir or Path(
            os.environ.get("DALIO_RIKSBANK_CACHE", "data/cache/riksbank")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        configured_key = api_key if api_key is not None else os.environ.get("RIKSBANK_API_KEY")
        self._api_key = configured_key.strip() if configured_key else None

    @property
    def recommended_pacing_seconds(self) -> float:
        """Safe request interval for the active SWEA quota class."""
        if self._api_key:
            return DEFAULT_AUTHENTICATED_PACING_SECONDS
        return DEFAULT_UNAUTHENTICATED_PACING_SECONDS

    @staticmethod
    def url_for(
        spec: RiksbankSeriesSpec,
        *,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
    ) -> str:
        start = max(_as_date(from_date, default=DEFAULT_FROM_DATE), spec.history_start)
        end = _as_date(to_date, default=date.today())
        if start > end:
            raise ValueError("Riksbank from_date cannot be later than to_date")
        return (
            f"{RIKSBANK_API_BASE}/Observations/{spec.series_id}/"
            f"{start.isoformat()}/{end.isoformat()}"
        )

    def fetch(
        self,
        spec: RiksbankSeriesSpec,
        *,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch one complete requested date window for one native series."""
        url = self.url_for(spec, from_date=from_date, to_date=to_date)
        payload = self._fetch_json(url, use_cache=use_cache)
        return self._to_long(payload, spec)

    def _fetch_json(
        self,
        url: str,
        *,
        use_cache: bool,
        attempts: int = 3,
        backoff_base: float = 1.0,
    ) -> list[dict[str, object]]:
        cache_path = self._cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                logger.debug("Riksbank cache hit (age=%.0fs): %s", age, url)
                return self._parse_payload(cache_path.read_text(encoding="utf-8"), url)

        headers = {"Accept": "application/json"}
        if self._api_key:
            headers[RIKSBANK_API_KEY_HEADER] = self._api_key

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - HTTP client is injected
                last_error = exc
                if attempt == attempts - 1:
                    raise
                self._sleep_before_retry(url, attempt, attempts, exc, backoff_base * 2**attempt)
                continue

            status = response.status_code
            if status == 204:
                return []
            if status == 429:
                last_error = RuntimeError(f"Riksbank rate limited (429): {url}")
                if attempt == attempts - 1:
                    raise last_error
                wait = self._retry_after_seconds(response.headers)
                self._sleep_before_retry(url, attempt, attempts, last_error, wait)
                continue
            if 400 <= status < 500:
                raise ValueError(f"Riksbank request failed ({status}): {url}")
            if status >= 500:
                last_error = RuntimeError(f"Riksbank server error {status}: {url}")
                if attempt == attempts - 1:
                    raise last_error
                self._sleep_before_retry(
                    url,
                    attempt,
                    attempts,
                    last_error,
                    backoff_base * 2**attempt,
                )
                continue
            if status < 200 or status >= 300:
                raise RuntimeError(f"Unexpected Riksbank response {status}: {url}")

            payload = self._parse_payload(response.text, url)
            try:
                cache_path.write_text(response.text, encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not write Riksbank cache %s: %s", cache_path, exc)
            return payload

        assert last_error is not None
        raise last_error

    @staticmethod
    def _sleep_before_retry(
        url: str,
        attempt: int,
        attempts: int,
        error: Exception,
        wait: float,
    ) -> None:
        logger.warning(
            "Riksbank %s attempt %d/%d failed (%s) — retrying in %.1fs",
            url,
            attempt + 1,
            attempts,
            error,
            wait,
        )
        time.sleep(wait)

    @staticmethod
    def _retry_after_seconds(headers: object) -> float:
        try:
            raw = headers.get("Retry-After")  # type: ignore[union-attr]
            wait = float(raw)
        except (AttributeError, TypeError, ValueError):
            return 60.0
        return max(wait, 0.0)

    @staticmethod
    def _parse_payload(raw: str, url: str) -> list[dict[str, object]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Riksbank returned invalid JSON: {url}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Riksbank response must be a JSON list: {url}")
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Riksbank observations must be JSON objects: {url}")
        return payload

    @staticmethod
    def _to_long(
        payload: list[dict[str, object]],
        spec: RiksbankSeriesSpec,
    ) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(columns=_COLUMNS)

        records: list[dict[str, object]] = []
        seen_dates: set[date] = set()
        for index, row in enumerate(payload):
            if "date" not in row or "value" not in row:
                raise ValueError(f"Riksbank observation {index} is missing date or value")
            try:
                observed_on = date.fromisoformat(str(row["date"]))
            except ValueError as exc:
                raise ValueError(f"Riksbank observation {index} has an invalid date") from exc
            try:
                if isinstance(row["value"], bool):
                    raise TypeError
                value = float(row["value"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Riksbank observation {index} has a non-numeric value") from exc
            if not math.isfinite(value):
                raise ValueError(f"Riksbank observation {index} has a non-finite value")
            if observed_on in seen_dates:
                raise ValueError(f"Riksbank response contains duplicate date {observed_on}")
            seen_dates.add(observed_on)
            records.append(
                {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "date": observed_on,
                    "value": value,
                    "source": RIKSBANK_SOURCE,
                    "series_id": spec.series_id,
                }
            )

        return pd.DataFrame(records, columns=_COLUMNS).sort_values("date").reset_index(drop=True)

    def _cache_path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{digest}.json"


def _as_date(value: date | str | None, *, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an ISO date, got {value!r}") from exc
