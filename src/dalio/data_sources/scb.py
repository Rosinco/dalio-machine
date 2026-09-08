"""Statistics Sweden (SCB) headline CPI adapter.

SCB's keyless PxWebApi 2.0 endpoint publishes the official Swedish monthly
12-month CPI change directly as JSON-stat 2.0. This module intentionally owns
one series only: the exact ``cpi_yoy`` input consumed by the cycle classifier.
Index levels and annual HICP are different quantities and stay out of this
catalogue until a concrete consumer needs them.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

SCB_BASE_URL = "https://api.scb.se/ov0104/v2beta/api/v2"
SOURCE_SCB_CPI = "SCB_CPI"
INDICATOR_CPI = "cpi_yoy"
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class ScbSeriesSpec:
    """One single-measure SCB PxWeb table selection."""

    indicator: str
    table_id: str
    contents_code: str
    country: str = "SE"

    @property
    def series_id(self) -> str:
        """Stable native identity: SCB table plus selected contents measure."""
        return f"{self.table_id}/{self.contents_code}"

    @property
    def url(self) -> str:
        return (
            f"{SCB_BASE_URL}/tables/{self.table_id}/data"
            f"?valueCodes[ContentsCode]={self.contents_code}"
            "&valueCodes[Tid]=*"
            "&outputFormat=json-stat2&lang=en"
        )


SCB_CPI_YOY = ScbSeriesSpec(
    indicator=INDICATOR_CPI,
    table_id="TAB6596",
    contents_code="00000804",
)
SCB_SERIES: tuple[ScbSeriesSpec, ...] = (SCB_CPI_YOY,)


class ScbSource:
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
                "DALIO_SCB_CACHE",
                "data/cache/scb",
            ),
            cache_ttl_hours,
            label="SCB",
            timeout=DEFAULT_TIMEOUT,
            suffix=".json",
            forbidden_hint="PxWeb endpoint refused the request",
        )

    def fetch(
        self,
        spec: ScbSeriesSpec = SCB_CPI_YOY,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch one complete native SCB series as Dalio's long format."""
        text = self._fetcher.fetch(spec.url, use_cache=use_cache)
        return parse_jsonstat(text, spec)


def parse_jsonstat(text: str, spec: ScbSeriesSpec = SCB_CPI_YOY) -> pd.DataFrame:
    """Parse a single-measure JSON-stat 2.0 response, dropping publisher nulls."""
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _empty_long()
    if not isinstance(raw, dict) or not raw.get("value"):
        return _empty_long()

    values = raw["value"]
    try:
        time_index = raw["dimension"]["Tid"]["category"]["index"]
    except (KeyError, TypeError) as exc:
        raise ValueError("SCB response is missing the Tid dimension") from exc
    if not isinstance(values, list) or not isinstance(time_index, dict):
        raise ValueError("SCB response uses an unsupported JSON-stat shape")

    time_labels = [
        label for label, _position in sorted(time_index.items(), key=lambda item: item[1])
    ]
    _validate_single_measure_shape(raw, len(time_labels), len(values))

    rows = []
    for label, raw_value in zip(time_labels, values, strict=True):
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SCB returned a non-numeric value for {label!r}") from exc
        if not math.isfinite(value):
            continue
        rows.append(
            {
                "country": spec.country,
                "indicator": spec.indicator,
                "date": _parse_time_label(label),
                "value": value,
                "source": SOURCE_SCB_CPI,
                "series_id": spec.series_id,
            }
        )
    if not rows:
        return _empty_long()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _validate_single_measure_shape(
    raw: dict,
    time_count: int,
    value_count: int,
) -> None:
    dimensions = raw.get("id")
    sizes = raw.get("size")
    if isinstance(dimensions, list) and isinstance(sizes, list):
        if len(dimensions) != len(sizes) or "Tid" not in dimensions:
            raise ValueError("SCB response has inconsistent JSON-stat dimensions")
        for dimension, size in zip(dimensions, sizes, strict=True):
            if dimension != "Tid" and int(size) != 1:
                raise ValueError("SCB response has multiple measures; add explicit filters")
        if int(sizes[dimensions.index("Tid")]) != time_count:
            raise ValueError("SCB response time dimension size mismatch")
    if value_count != time_count:
        raise ValueError(
            f"SCB response size mismatch: {value_count} values vs {time_count} time labels"
        )


def _parse_time_label(label: str) -> date:
    """Map SCB monthly, quarterly (K), or annual period labels to period start."""
    if match := re.fullmatch(r"(\d{4})M(\d{2})", str(label)):
        return date(int(match.group(1)), int(match.group(2)), 1)
    if match := re.fullmatch(r"(\d{4})K([1-4])", str(label)):
        return date(int(match.group(1)), 1 + (int(match.group(2)) - 1) * 3, 1)
    if match := re.fullmatch(r"\d{4}", str(label)):
        return date(int(match.group()), 1, 1)
    raise ValueError(f"Unsupported SCB time label: {label!r}")


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
