"""BIS Global Liquidity Indicators for offshore reserve-currency credit.

The three pinned series measure credit to non-bank borrowers outside the
issuing currency area.  They are *credit stocks*, not monetary aggregates and
not additive across currencies without an explicit FX methodology.  Exact BIS
dimension identities and native units are validated before observations leave
this adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

SOURCE_BIS_GLI = "BIS_GLI"
BIS_GLI_CATALOGUE_VINTAGE_PREFIX = "bis-gli-catalogue-sha256:"
DEFAULT_TIMEOUT = 60.0
_QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
_LONG_COLUMNS = [
    "country",
    "indicator",
    "date",
    "value",
    "source",
    "series_id",
    "status",
]
_STATUS = {
    "A": "observed",
    "B": "break",
    "E": "estimated",
    "P": "provisional",
}


@dataclass(frozen=True)
class BisGlobalLiquiditySeries:
    """Exact BIS series identity plus its economic/non-additivity semantics."""

    indicator: str
    country: str
    currency: str
    unit: str
    native_unit: str
    unit_multiplier: int
    frequency: str
    adjustment: str
    observation_basis: str
    measure_kind: str
    claim_side: str
    from_sector: str
    to_sector: str
    instrument: str
    aggregation_role: str
    non_additive_group: str
    publisher: str
    delivery_service: str
    source_family: str
    native_series_id: str
    title: str
    url: str
    expected_start: date
    minimum_observations: int
    max_latest_lag_days: int


def _offshore_credit(currency: str, area: str) -> BisGlobalLiquiditySeries:
    currency_name = {"USD": "US dollar", "EUR": "Euro", "JPY": "Yen"}[currency]
    area_slug = {"USD": "us", "EUR": "euro_area", "JPY": "japan"}[currency]
    native_series_id = f"Q.{currency}.3P.N.A.I.B.{currency}"
    return BisGlobalLiquiditySeries(
        indicator=f"offshore_{currency.lower()}_credit_nonbanks_stock",
        country="GLOBAL",
        currency=currency,
        unit=f"{currency} million",
        native_unit=currency_name,
        unit_multiplier=1_000_000,
        frequency="quarterly",
        adjustment="not seasonally adjusted",
        observation_basis="quarter-end stock; date stores quarter start",
        measure_kind="credit_stock",
        claim_side="borrower_liability",
        from_sector="banks_and_global_bond_investors",
        to_sector=f"nonbank_borrowers_outside_{area_slug}",
        instrument="bank_loans_and_debt_securities",
        aggregation_role="currency_total",
        non_additive_group="bis_gli_offshore_credit_by_currency",
        publisher="Bank for International Settlements",
        delivery_service="BIS SDMX REST API",
        source_family=SOURCE_BIS_GLI,
        native_series_id=native_series_id,
        title=(
            f"{currency} denominated credit (bank loans & debt securities) to "
            f"non-bank borrowers located outside {area}"
        ),
        url=(f"https://stats.bis.org/api/v1/data/WS_GLI/{native_series_id}?format=csv"),
        expected_start=date(2000, 1, 1),
        minimum_observations=100,
        # Stored dates use quarter start, so a still-current Q1 observation can
        # be roughly 250 days old by the early-September publication window.
        # 300 days permits one completed-quarter publication lag but rejects a
        # history that has fallen a further quarter behind.
        max_latest_lag_days=300,
    )


BIS_GLI_USD = _offshore_credit("USD", "the US")
BIS_GLI_EUR = _offshore_credit("EUR", "the euro area")
BIS_GLI_JPY = _offshore_credit("JPY", "Japan")

BIS_GLOBAL_LIQUIDITY_SERIES: tuple[BisGlobalLiquiditySeries, ...] = (
    BIS_GLI_USD,
    BIS_GLI_EUR,
    BIS_GLI_JPY,
)


def bis_global_liquidity_catalogue_sha256(
    specs: tuple[BisGlobalLiquiditySeries, ...] = BIS_GLOBAL_LIQUIDITY_SERIES,
) -> str:
    """Hash every field that controls interpretation or source identity."""

    records = []
    for spec in sorted(specs, key=lambda item: item.native_series_id):
        record = asdict(spec)
        record["expected_start"] = spec.expected_start.isoformat()
        records.append(record)
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BisGlobalLiquiditySource:
    """Fetch and validate the pinned BIS GLI scalar histories."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        artifact_dir: Path | None = None,
    ) -> None:
        self._client = client or requests.Session()
        set_default_headers(self._client, DEFAULT_USER_AGENT, "text/csv")
        resolved_cache = CachedTextFetcher.resolve_cache_dir(
            cache_dir,
            "DALIO_SHADOW_LIQUIDITY_CACHE",
            "data/cache/shadow_liquidity",
        )
        self._fetcher = CachedTextFetcher(
            self._client,
            resolved_cache,
            cache_ttl_hours,
            label="BIS GLI",
            timeout=DEFAULT_TIMEOUT,
            suffix=".csv",
        )
        self._artifact_dir = (
            artifact_dir
            if artifact_dir is not None
            else Path(
                os.environ.get(
                    "DALIO_SHADOW_LIQUIDITY_ARTIFACTS",
                    "data/artifacts/liquidity_frontier",
                )
            )
            / "bis"
        )

    def fetch(
        self,
        spec: BisGlobalLiquiditySeries,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        text = self._fetcher.fetch(spec.url, use_cache=use_cache)
        frame = parse_bis_global_liquidity_csv(text, spec)
        artifact_path, artifact_sha256 = _archive_validated_response(
            text,
            self._artifact_dir,
        )
        frame.attrs["source_url"] = spec.url
        frame.attrs["source_artifact_path"] = str(artifact_path)
        frame.attrs["source_artifact_sha256"] = artifact_sha256
        frame.attrs["native_payload_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return frame


def _archive_validated_response(text: str, artifact_dir: Path) -> tuple[Path, str]:
    """Atomically retain an exact validated BIS CSV response by SHA-256."""
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_dir / digest[:2] / f"{digest}.csv"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"BIS GLI artifact hash collision or corrupt archive: {destination}")
        return destination, digest

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
    return destination, digest


def parse_bis_global_liquidity_csv(
    text: str,
    spec: BisGlobalLiquiditySeries,
) -> pd.DataFrame:
    """Parse one exact WS_GLI response and fail closed on semantic drift."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("BIS GLI returned an empty response body")
    try:
        raw = pd.read_csv(StringIO(text), dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError("BIS GLI returned malformed CSV") from exc

    required = {
        "FREQ",
        "CURR_DENOM",
        "BORROWERS_CTY",
        "BORROWERS_SECTOR",
        "LENDERS_SECTOR",
        "L_POS_TYPE",
        "L_INSTR",
        "UNIT_MEASURE",
        "TITLE",
        "UNIT_MULT",
        "TIME_PERIOD",
        "OBS_VALUE",
        "OBS_STATUS",
        "OBS_PRE_BREAK",
        "OBS_CONF",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"BIS GLI response missing columns: {sorted(missing)}")
    if raw.empty:
        return pd.DataFrame(columns=_LONG_COLUMNS)

    expected_identity = {
        "FREQ": "Q",
        "CURR_DENOM": spec.currency,
        "BORROWERS_CTY": "3P",
        "BORROWERS_SECTOR": "N",
        "LENDERS_SECTOR": "A",
        "L_POS_TYPE": "I",
        "L_INSTR": "B",
        "UNIT_MEASURE": spec.currency,
        "TITLE": spec.title,
        "UNIT_MULT": "6",
        "OBS_CONF": "F",
    }
    for column, expected in expected_identity.items():
        normalized = raw[column].astype("string").str.strip()
        values = normalized.drop_duplicates().tolist()
        if (
            normalized.isna().any()
            or (normalized == "").any()
            or not (normalized == expected).all()
        ):
            raise ValueError(
                f"BIS GLI {column} identity mismatch for {spec.native_series_id}: "
                f"expected {expected!r}, got {values!r}"
            )

    if raw["OBS_PRE_BREAK"].notna().any():
        raise ValueError(
            f"BIS GLI {spec.native_series_id} introduced pre-break values that "
            "require an explicit storage decision"
        )

    periods = raw["TIME_PERIOD"].astype("string").str.strip()
    dates = periods.map(_quarter_start)
    normalized_values = (
        raw["OBS_VALUE"].astype("string").str.strip().replace({"": pd.NA, ".": pd.NA, "..": pd.NA})
    )
    values = pd.to_numeric(normalized_values, errors="coerce")
    invalid = normalized_values.notna() & values.isna()
    if invalid.any():
        raise ValueError(
            f"BIS GLI returned a non-numeric observation: "
            f"{normalized_values.loc[invalid].iloc[0]!r}"
        )
    valid = values.notna()
    values = values.loc[valid]
    dates = dates.loc[valid]
    statuses = raw.loc[valid, "OBS_STATUS"].map(_parse_status)
    if not values.map(math.isfinite).all():
        raise ValueError("BIS GLI returned a non-finite observation")
    if (values <= 0).any():
        raise ValueError("BIS GLI credit stocks must be positive")

    frame = pd.DataFrame(
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": dates.tolist(),
            "value": values.astype(float).tolist(),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
            "status": statuses.tolist(),
        }
    )
    frame.attrs["native_periods"] = tuple(periods.loc[valid])
    frame.attrs["native_period_format"] = "quarterly"
    return frame


def _quarter_start(value: object) -> date:
    match = _QUARTER_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"BIS GLI returned an invalid quarterly period: {value!r}")
    return date(int(match.group(1)), (int(match.group(2)) - 1) * 3 + 1, 1)


def _parse_status(value: object) -> str:
    if pd.isna(value):
        raise ValueError("BIS GLI response has a missing observation status")
    code = str(value).strip().upper()
    if not code:
        raise ValueError("BIS GLI response has a missing observation status")
    return _STATUS.get(code, f"native_{code.lower()}"[:24])
