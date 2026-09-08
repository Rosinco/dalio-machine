"""Official money-stock and central-bank-balance-sheet source adapter.

The catalogue is intentionally small and exact. Values stay in their
publisher-native denomination and frequency. In particular, weekly
Fed/Eurosystem assets are not silently resampled to monthly observations, and
no currency conversion or scoring occurs here. Nested measures such as UK M4
and M4ex or Japan M3 and L retain explicit research roles rather than becoming
duplicate votes.

Delivery endpoints are all first-party public infrastructure: Federal Reserve
Economic Data (FRED), the ECB Data Portal SDMX service, SCB's PxWeb API for a
table sourced from Sveriges Riksbank, and the Bank of England Interactive
Statistical Database (IADB), and the Bank of Japan Time-Series Data Search API.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, date
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

SOURCE_FED_FRED = "FED_FRED"
SOURCE_ECB_DATA = "ECB_DATA"
SOURCE_SCB_RIKSBANK_MONEY = "SCB_RIKSBANK_MONEY"
SOURCE_BOE_IADB = "BOE_IADB"
SOURCE_BOJ_DATA = "BOJ_DATA"
MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX = "money-liquidity-catalogue-sha256:"
MONEY_LIQUIDITY_MISSINGNESS_SCHEMA_VERSION = "money-liquidity-missingness-v1"

INDICATOR_BROAD_MONEY_M2 = "broad_money_m2_stock"
INDICATOR_BROAD_MONEY_M3 = "broad_money_m3_stock"
INDICATOR_BROAD_MONEY_M4 = "broad_money_m4_stock"
INDICATOR_BROAD_MONEY_M4EX = "broad_money_m4ex_stock"
INDICATOR_BROAD_MONEY_M4EX_QUARTERLY = "broad_money_m4ex_quarterly_stock"
INDICATOR_BROADLY_DEFINED_LIQUIDITY = "broadly_defined_liquidity_stock"
INDICATOR_CENTRAL_BANK_ASSETS = "central_bank_total_assets"

DEFAULT_TIMEOUT = 30.0
_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_ECB_STATUS = {
    "A": "observed",
    "B": "break",
    "E": "estimated",
    "F": "forecast",
    "P": "provisional",
}
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_BOJ_MONTH_RE = re.compile(r"^(\d{4})(0[1-9]|1[0-2])$")
_BOJ_LAST_UPDATE_RE = re.compile(r"^(\d{4})(0[1-9]|1[0-2])([0-3]\d)$")
_SCB_MONTH_RE = re.compile(r"^(\d{4})M(0[1-9]|1[0-2])$")
_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")
_NULL_TEXT_TOKENS = frozenset(
    {
        ".",
        "..",
        "#n/a",
        "#na",
        "<na>",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
    }
)
_MISSING_VALUE_POLICY = (
    "Publisher null, blank, or explicitly absent native values remain missing; "
    "never zero-fill or impute them"
)
_SPARSE_ABSENT = object()


@dataclass(frozen=True)
class MoneyLiquiditySeries:
    """Exact native-series metadata needed to interpret one stored scalar."""

    indicator: str
    country: str
    currency: str
    unit: str
    native_unit: str
    unit_multiplier: int
    frequency: str
    adjustment: str
    observation_basis: str
    publisher: str
    delivery_service: str
    source_family: str
    native_series_id: str
    title: str
    url: str
    response_format: str
    expected_start: date
    minimum_observations: int
    max_latest_lag_days: int
    research_role: str = "primary"
    perimeter: str = ""
    definition_notes: str = ""
    parent_native_series_id: str | None = None
    non_additive_groups: tuple[str, ...] = ()


FED_M2 = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M2,
    country="US",
    currency="USD",
    unit="USD billion",
    native_unit="Billions of Dollars",
    unit_multiplier=1_000_000_000,
    frequency="monthly",
    adjustment="seasonally adjusted",
    observation_basis="publisher monthly period",
    publisher="Board of Governors of the Federal Reserve System (US)",
    delivery_service="Federal Reserve Bank of St. Louis FRED",
    source_family=SOURCE_FED_FRED,
    native_series_id="M2SL",
    title="M2",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL",
    response_format="fred_csv",
    expected_start=date(1959, 1, 1),
    minimum_observations=800,
    max_latest_lag_days=120,
    perimeter=(
        "Federal Reserve M2 liabilities and retail liquid claims held by the U.S. nonbank public"
    ),
    definition_notes=(
        "Publisher-defined M2 level; no discontinued U.S. M4 or M5 history is spliced in"
    ),
    non_additive_groups=("us_money_and_central_bank_assets_unlike_measures",),
)

FED_TOTAL_ASSETS = MoneyLiquiditySeries(
    indicator=INDICATOR_CENTRAL_BANK_ASSETS,
    country="US",
    currency="USD",
    unit="USD million",
    native_unit="Millions of U.S. Dollars",
    unit_multiplier=1_000_000,
    frequency="weekly",
    adjustment="not seasonally adjusted",
    observation_basis="Wednesday level",
    publisher="Board of Governors of the Federal Reserve System (US)",
    delivery_service="Federal Reserve Bank of St. Louis FRED",
    source_family=SOURCE_FED_FRED,
    native_series_id="WALCL",
    title=(
        "Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level"
    ),
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL",
    response_format="fred_csv",
    expected_start=date(2002, 12, 18),
    minimum_observations=1_200,
    max_latest_lag_days=21,
    perimeter="Consolidated total assets of the Federal Reserve Banks",
    definition_notes="Weekly H.4.1 Wednesday level; not a component of M2",
    non_additive_groups=("us_money_and_central_bank_assets_unlike_measures",),
)

ECB_M3 = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M3,
    country="EU",
    currency="EUR",
    unit="EUR million",
    native_unit="EUR with unit multiplier 10^6",
    unit_multiplier=1_000_000,
    frequency="monthly",
    adjustment="working-day and seasonally adjusted",
    observation_basis="end-of-period stock; date stores month start",
    publisher="European Central Bank",
    delivery_service="ECB Data Portal SDMX API",
    source_family=SOURCE_ECB_DATA,
    native_series_id="BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E",
    title="Euro area M3 monetary aggregate, stock",
    url=(
        "https://data-api.ecb.europa.eu/service/data/BSI/"
        "M.U2.Y.V.M30.X.1.U2.2300.Z01.E?format=csvdata"
    ),
    response_format="ecb_csv",
    expected_start=date(1980, 1, 1),
    minimum_observations=550,
    max_latest_lag_days=120,
    perimeter=(
        "Euro-area MFI and central-government monetary liabilities included by the ECB "
        "in M3 and held by the euro-area money-holding sector"
    ),
    definition_notes=(
        "Publisher-defined M3 stock; already includes certain repo, MMF-share and short "
        "MFI-debt claims, so frontier instruments are not added to it"
    ),
    non_additive_groups=("euro_money_and_central_bank_assets_unlike_measures",),
)

ECB_EUROSYSTEM_ASSETS = MoneyLiquiditySeries(
    indicator=INDICATOR_CENTRAL_BANK_ASSETS,
    country="EU",
    currency="EUR",
    unit="EUR million",
    native_unit="EUR with unit multiplier 10^6",
    unit_multiplier=1_000_000,
    frequency="weekly",
    adjustment="not seasonally adjusted",
    observation_basis=(
        "end-of-period stock; date stores ISO-week Friday representative; "
        "official reporting dates may differ around holidays and quarter ends"
    ),
    publisher="European Central Bank",
    delivery_service="ECB Data Portal SDMX API",
    source_family=SOURCE_ECB_DATA,
    native_series_id="ILM.W.U2.C.T000000.Z5.Z01",
    title="Eurosystem consolidated financial statement, total assets",
    url=("https://data-api.ecb.europa.eu/service/data/ILM/W.U2.C.T000000.Z5.Z01?format=csvdata"),
    response_format="ecb_csv",
    expected_start=date(1999, 1, 1),
    minimum_observations=1_400,
    max_latest_lag_days=21,
    perimeter="Consolidated total assets of the Eurosystem",
    definition_notes="Weekly financial-statement stock; not a component of euro-area M3",
    non_additive_groups=("euro_money_and_central_bank_assets_unlike_measures",),
)

SCB_M3 = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M3,
    country="SE",
    currency="SEK",
    unit="SEK million",
    native_unit="SEK millions (mnkr)",
    unit_multiplier=1_000_000,
    frequency="monthly",
    adjustment="not adjusted; current-price stock",
    observation_basis="publisher monthly period",
    publisher="Sveriges Riksbank",
    delivery_service="Statistics Sweden (SCB) PxWeb API",
    source_family=SOURCE_SCB_RIKSBANK_MONEY,
    native_series_id="TAB6541/5LLM3a.1E.NEP.V.A/000007WQ",
    title="Outstanding M3 money supply",
    url=(
        "https://api.scb.se/ov0104/v2beta/api/v2/tables/TAB6541/data?"
        "valueCodes%5BPenningm%5D=5LLM3a.1E.NEP.V.A&"
        "valueCodes%5BContentsCode%5D=000007WQ&valueCodes%5BTid%5D=%2A&"
        "outputFormat=json-stat2&lang=en"
    ),
    response_format="scb_jsonstat2",
    expected_start=date(1999, 1, 1),
    minimum_observations=325,
    max_latest_lag_days=120,
    perimeter="Publisher-defined Swedish M3 outstanding money supply",
    definition_notes=(
        "Current SCB/Riksbank table definition from 1999; older definitions are not spliced"
    ),
    non_additive_groups=("sweden_broad_money_native_definition",),
)

BOE_M4EX = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M4EX,
    country="UK",
    currency="GBP",
    unit="GBP million",
    native_unit="Sterling millions",
    unit_multiplier=1_000_000,
    frequency="monthly",
    adjustment="seasonally adjusted",
    observation_basis=("end-of-period stock; publisher month-end date normalized to month start"),
    publisher="Bank of England",
    delivery_service="Bank of England Interactive Statistical Database (IADB)",
    source_family=SOURCE_BOE_IADB,
    native_series_id="RPMB53Q",
    title=(
        "Monthly amounts outstanding of UK resident monetary financial institutions' "
        "sterling M4 liabilities to Private sector excluding intermediate OFCs "
        "(in sterling millions) seasonally adjusted"
    ),
    url=(
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jul/2009&Dateto=now&"
        "SeriesCodes=RPMB53Q&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    ),
    response_format="boe_csv",
    expected_start=date(2009, 7, 1),
    minimum_observations=200,
    max_latest_lag_days=120,
    research_role="primary",
    perimeter=(
        "UK monetary financial institutions' sterling M4 liabilities to the private "
        "sector excluding intermediate other financial corporations"
    ),
    definition_notes=(
        "Preferred monthly M4ex perimeter begins July 2009; the separate quarterly "
        "RPQB53Q bridge is not a second vote"
    ),
    parent_native_series_id="LPMAUYN",
    non_additive_groups=("uk_m4_nested_and_frequency_overlap",),
)

BOE_M4EX_QUARTERLY = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M4EX_QUARTERLY,
    country="UK",
    currency="GBP",
    unit="GBP million",
    native_unit="Sterling millions",
    unit_multiplier=1_000_000,
    frequency="quarterly",
    adjustment="seasonally adjusted",
    observation_basis=(
        "end-of-quarter stock; publisher quarter-end date normalized to quarter start"
    ),
    publisher="Bank of England",
    delivery_service="Bank of England Interactive Statistical Database (IADB)",
    source_family=SOURCE_BOE_IADB,
    native_series_id="RPQB53Q",
    title=(
        "Quarterly amounts outstanding of UK resident monetary financial institutions' "
        "sterling M4 liabilities to Private sector excluding intermediate OFCs "
        "(in sterling millions) seasonally adjusted"
    ),
    url=(
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Oct/1997&Dateto=now&"
        "SeriesCodes=RPQB53Q&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    ),
    response_format="boe_csv",
    expected_start=date(1997, 10, 1),
    minimum_observations=110,
    max_latest_lag_days=240,
    research_role="diagnostic_bridge",
    perimeter=("Same M4ex perimeter as RPMB53Q, observed at native quarterly frequency"),
    definition_notes=(
        "Independent pre-2009 historical bridge; overlapping quarter ends equal the "
        "monthly M4ex series and must not be counted twice"
    ),
    parent_native_series_id="RPMB53Q",
    non_additive_groups=("uk_m4_nested_and_frequency_overlap",),
)

BOE_M4 = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M4,
    country="UK",
    currency="GBP",
    unit="GBP million",
    native_unit="Sterling millions",
    unit_multiplier=1_000_000,
    frequency="monthly",
    adjustment="seasonally adjusted",
    observation_basis=("end-of-period stock; publisher month-end date normalized to month start"),
    publisher="Bank of England",
    delivery_service="Bank of England Interactive Statistical Database (IADB)",
    source_family=SOURCE_BOE_IADB,
    native_series_id="LPMAUYN",
    title=(
        "Monthly amounts outstanding of M4 (monetary financial institutions' "
        "sterling M4 liabilities to private sector) (in sterling millions) "
        "seasonally adjusted"
    ),
    url=(
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jun/1982&Dateto=now&"
        "SeriesCodes=LPMAUYN&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    ),
    response_format="boe_csv",
    expected_start=date(1982, 6, 1),
    minimum_observations=525,
    max_latest_lag_days=120,
    research_role="diagnostic",
    perimeter=("UK monetary financial institutions' sterling M4 liabilities to the private sector"),
    definition_notes=(
        "Headline M4 has a wider intermediate-OFC perimeter than M4ex and is retained "
        "for diagnosis, not as an additive component"
    ),
    non_additive_groups=("uk_m4_nested_and_frequency_overlap",),
)

BOJ_M3 = MoneyLiquiditySeries(
    indicator=INDICATOR_BROAD_MONEY_M3,
    country="JP",
    currency="JPY",
    unit="JPY 100 million",
    native_unit="100 million yen",
    unit_multiplier=100_000_000,
    frequency="monthly",
    adjustment="not seasonally adjusted",
    observation_basis=(
        "average amounts outstanding during publisher monthly period; date stores month start"
    ),
    publisher="Bank of Japan",
    delivery_service="Bank of Japan Time-Series Data Search API",
    source_family=SOURCE_BOJ_DATA,
    native_series_id="MAM1NAM3M3MO",
    title="M3/Average Amounts Outstanding/Money Stock",
    url=(
        "https://www.stat-search.boj.or.jp/api/v1/getDataCode?"
        "format=csv&lang=en&db=MD02&code=MAM1NAM3M3MO"
    ),
    response_format="boj_csv",
    expected_start=date(2003, 4, 1),
    minimum_observations=275,
    max_latest_lag_days=120,
    research_role="diagnostic",
    perimeter="Bank of Japan publisher-defined M3 money stock holders and issuers",
    definition_notes=(
        "Current exact definition begins April 2003; older definitionally different "
        "series are not spliced"
    ),
    parent_native_series_id="MAM1NABLBLMO",
    non_additive_groups=("japan_m3_and_broad_liquidity_nested",),
)

BOJ_BROADLY_DEFINED_LIQUIDITY = MoneyLiquiditySeries(
    indicator=INDICATOR_BROADLY_DEFINED_LIQUIDITY,
    country="JP",
    currency="JPY",
    unit="JPY 100 million",
    native_unit="100 million yen",
    unit_multiplier=100_000_000,
    frequency="monthly",
    adjustment="not seasonally adjusted",
    observation_basis=(
        "average amounts outstanding during publisher monthly period; date stores month start"
    ),
    publisher="Bank of Japan",
    delivery_service="Bank of Japan Time-Series Data Search API",
    source_family=SOURCE_BOJ_DATA,
    native_series_id="MAM1NABLBLMO",
    title="L/Average Amounts Outstanding/Money Stock",
    url=(
        "https://www.stat-search.boj.or.jp/api/v1/getDataCode?"
        "format=csv&lang=en&db=MD02&code=MAM1NABLBLMO"
    ),
    response_format="boj_csv",
    expected_start=date(2003, 4, 1),
    minimum_observations=275,
    max_latest_lag_days=120,
    research_role="primary",
    perimeter=(
        "Bank of Japan broadly-defined liquidity L: M3 plus publisher-defined wider "
        "liquid claims issued by additional sectors"
    ),
    definition_notes=(
        "Current exact definition begins April 2003; this is Japan-specific and is not "
        "labelled as a universal M5"
    ),
    non_additive_groups=("japan_m3_and_broad_liquidity_nested",),
)

MONEY_LIQUIDITY_SERIES: tuple[MoneyLiquiditySeries, ...] = (
    FED_M2,
    FED_TOTAL_ASSETS,
    ECB_M3,
    ECB_EUROSYSTEM_ASSETS,
    SCB_M3,
    BOE_M4EX,
    BOE_M4EX_QUARTERLY,
    BOE_M4,
    BOJ_M3,
    BOJ_BROADLY_DEFINED_LIQUIDITY,
)


def money_liquidity_catalogue_sha256(
    specs: tuple[MoneyLiquiditySeries, ...] = MONEY_LIQUIDITY_SERIES,
) -> str:
    """Return a deterministic hash of the pinned catalogue and its semantics."""
    records = [
        {
            "indicator": spec.indicator,
            "country": spec.country,
            "currency": spec.currency,
            "unit": spec.unit,
            "native_unit": spec.native_unit,
            "unit_multiplier": spec.unit_multiplier,
            "frequency": spec.frequency,
            "adjustment": spec.adjustment,
            "observation_basis": spec.observation_basis,
            "publisher": spec.publisher,
            "delivery_service": spec.delivery_service,
            "source_family": spec.source_family,
            "native_series_id": spec.native_series_id,
            "title": spec.title,
            "url": spec.url,
            "response_format": spec.response_format,
            "expected_start": spec.expected_start.isoformat(),
            "minimum_observations": spec.minimum_observations,
            "max_latest_lag_days": spec.max_latest_lag_days,
            "research_role": spec.research_role,
            "perimeter": spec.perimeter,
            "definition_notes": spec.definition_notes,
            "parent_native_series_id": spec.parent_native_series_id,
            "non_additive_groups": list(spec.non_additive_groups),
        }
        for spec in sorted(specs, key=lambda item: item.native_series_id)
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MoneyLiquiditySource:
    """Fetch the complete history of one pinned native series."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
        artifact_dir: Path | None = None,
    ) -> None:
        self._client = client or requests.Session()
        set_default_headers(
            self._client,
            user_agent,
            "text/csv, application/json;q=0.9",
        )
        self._fetcher = CachedTextFetcher(
            self._client,
            CachedTextFetcher.resolve_cache_dir(
                cache_dir,
                "DALIO_MONEY_LIQUIDITY_CACHE",
                "data/cache/money_liquidity",
            ),
            cache_ttl_hours,
            label="money/liquidity source",
            timeout=DEFAULT_TIMEOUT,
            suffix=".txt",
            forbidden_hint="official public endpoint refused the request",
        )
        self._artifact_dir = artifact_dir or Path(
            os.environ.get(
                "DALIO_MONEY_LIQUIDITY_ARTIFACTS",
                "data/artifacts/money_liquidity",
            )
        )

    def fetch(
        self,
        spec: MoneyLiquiditySeries,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        text = self._fetcher.fetch(spec.url, use_cache=use_cache)
        # CachedTextFetcher exposes text and uses universal-newline reads. Read
        # back the UTF-8 cache bytes so a cache hit preserves the exact decoded
        # response, including CRLF versus LF, in the durable artifact identity.
        cache_path = self._fetcher.cache_path_for(spec.url)
        try:
            text = cache_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"{spec.delivery_service} response cache is not readable UTF-8: {cache_path}"
            ) from exc
        if spec.response_format == "fred_csv":
            frame = parse_fred_csv(text, spec)
        elif spec.response_format == "ecb_csv":
            frame = parse_ecb_csv(text, spec)
        elif spec.response_format == "scb_jsonstat2":
            frame = parse_scb_jsonstat(text, spec)
        elif spec.response_format == "boe_csv":
            frame = parse_boe_csv(text, spec)
        elif spec.response_format == "boj_csv":
            frame = parse_boj_csv(text, spec)
        else:
            raise ValueError(f"unsupported money/liquidity format: {spec.response_format!r}")
        frame.attrs["source_url"] = spec.url
        # Empty but syntactically valid responses remain available to the
        # pipeline's fail-closed history guard, but never enter durable evidence.
        if frame.empty:
            return frame

        missing_records = _missing_period_records(text, spec)
        missing_payload = {
            "schema_version": MONEY_LIQUIDITY_MISSINGNESS_SCHEMA_VERSION,
            "native_series_id": spec.native_series_id,
            "source_family": spec.source_family,
            "response_format": spec.response_format,
            "missing_value_policy": _MISSING_VALUE_POLICY,
            "records": missing_records,
        }
        missing_bytes = _canonical_json_bytes(missing_payload)
        provider_dir = self._artifact_dir / _provider_directory(spec)
        source_bytes = text.encode("utf-8")
        source_suffix = ".json" if spec.response_format == "scb_jsonstat2" else ".csv"

        # Both documents are computed and validated before the first durable
        # write. A parser or missingness-contract failure therefore cannot
        # admit an invalid publisher response to the archive.
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        missing_sha256 = hashlib.sha256(missing_bytes).hexdigest()
        source_path, archived_source_sha256 = _archive_content_addressed(
            source_bytes,
            provider_dir / "source-responses",
            suffix=source_suffix,
            provider=spec.delivery_service,
        )
        missing_path, archived_missing_sha256 = _archive_content_addressed(
            missing_bytes,
            provider_dir / "missingness-ledgers",
            suffix=".json",
            provider=spec.delivery_service,
        )
        if archived_source_sha256 != source_sha256:
            raise ValueError(
                f"{spec.delivery_service} source artifact hash changed while archiving"
            )
        if archived_missing_sha256 != missing_sha256:
            raise ValueError(
                f"{spec.delivery_service} missingness artifact hash changed while archiving"
            )

        frame.attrs.update(
            {
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_sha256,
                # Every endpoint is pinned to exactly one native series, so the
                # complete response is also the native payload artifact.
                "native_payload_artifact_path": str(source_path),
                "native_payload_sha256": source_sha256,
                "missing_period_records": missing_records,
                "missing_provenance_json": missing_bytes.decode("utf-8"),
                "missing_provenance_artifact_path": str(missing_path),
                "missing_provenance_sha256": missing_sha256,
                "missing_value_policy": _MISSING_VALUE_POLICY,
            }
        )
        return frame


def _provider_directory(spec: MoneyLiquiditySeries) -> str:
    provider = spec.source_family.lower()
    if not re.fullmatch(r"[a-z0-9_]+", provider):
        raise ValueError(f"unsafe money/liquidity source family: {spec.source_family!r}")
    return provider


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _archive_content_addressed(
    payload: bytes,
    artifact_dir: Path,
    *,
    suffix: str,
    provider: str,
) -> tuple[Path, str]:
    """Atomically retain validated bytes and reject a corrupt prior artifact."""
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_dir / digest[:2] / f"{digest}{suffix}"
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"{provider} content-addressed artifact cannot be read: {destination}"
            ) from exc
        if existing != payload:
            raise ValueError(
                f"{provider} artifact hash collision or corrupt archive: {destination}"
            )
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


def _missing_period_records(
    text: str,
    spec: MoneyLiquiditySeries,
) -> tuple[dict[str, object], ...]:
    """Describe explicit publisher missing cells without turning them into zeroes."""
    cells = _native_value_cells(text, spec)
    records: list[dict[str, object]] = []
    for position, (native_period, native_value, evidence, forced_kind) in enumerate(cells):
        record = _missing_cell_record(
            native_period=native_period,
            native_value=native_value,
            native_position=position,
            evidence=evidence,
            forced_kind=forced_kind,
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def _missing_cell_record(
    *,
    native_period: object,
    native_value: object,
    native_position: int,
    evidence: str,
    forced_kind: str | None,
) -> dict[str, object] | None:
    period = str(native_period).strip()
    if not period:
        raise ValueError("money/liquidity missingness ledger has an empty native period")

    kind = forced_kind
    token: object
    if kind in {"sparse_absent", "absent_field", "json_null"}:
        token = None
    elif isinstance(native_value, str):
        token = native_value
        normalized = native_value.strip()
        if not normalized:
            kind = "blank"
        elif normalized.lower() in _NULL_TEXT_TOKENS:
            kind = "null_token"
        else:
            return None
    else:
        return None

    reasons = {
        "blank": "publisher_blank_not_zero",
        "null_token": "publisher_null_not_zero",
        "json_null": "publisher_null_not_zero",
        "sparse_absent": "publisher_sparse_absence_not_zero",
        "absent_field": "publisher_absent_field_not_zero",
    }
    if kind not in reasons:
        raise ValueError(f"unsupported money/liquidity missing-value kind: {kind!r}")
    return {
        "native_period": period,
        "native_position": native_position,
        "missing_kind": kind,
        "native_token": token,
        "reason": reasons[kind],
        "evidence": evidence,
    }


def _native_value_cells(
    text: str,
    spec: MoneyLiquiditySeries,
) -> tuple[tuple[object, object, str, str | None], ...]:
    if spec.response_format == "fred_csv":
        return _dict_csv_value_cells(
            text,
            period_column="observation_date",
            value_column=spec.native_series_id,
            evidence="fred_csv_value_cell",
        )
    if spec.response_format == "ecb_csv":
        return _dict_csv_value_cells(
            text,
            period_column="TIME_PERIOD",
            value_column="OBS_VALUE",
            evidence="ecb_csv_obs_value_cell",
        )
    if spec.response_format == "scb_jsonstat2":
        return _scb_value_cells(text)
    if spec.response_format == "boe_csv":
        rows = list(csv.reader(StringIO(text)))
        return tuple((row[0], row[1], "boe_csv_value_cell", None) for row in rows[4:] if row)
    if spec.response_format == "boj_csv":
        rows = [row for row in csv.reader(StringIO(text)) if row]
        header_position = next(
            index for index, row in enumerate(rows) if row and row[0] == "SERIES_CODE"
        )
        header = rows[header_position]
        period_index = header.index("SURVEY_DATES")
        value_index = header.index("VALUES")
        return tuple(
            (row[period_index], row[value_index], "boj_csv_values_cell", None)
            for row in rows[header_position + 1 :]
        )
    raise ValueError(f"unsupported money/liquidity format: {spec.response_format!r}")


def _dict_csv_value_cells(
    text: str,
    *,
    period_column: str,
    value_column: str,
    evidence: str,
) -> tuple[tuple[object, object, str, str | None], ...]:
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None or not {period_column, value_column}.issubset(reader.fieldnames):
        raise ValueError("validated money/liquidity CSV cannot build a missingness ledger")
    cells = []
    for row in reader:
        value = row.get(value_column)
        cells.append(
            (
                row.get(period_column),
                value,
                evidence,
                "absent_field" if value is None else None,
            )
        )
    return tuple(cells)


def _scb_value_cells(
    text: str,
) -> tuple[tuple[object, object, str, str | None], ...]:
    payload = json.loads(text)
    time_index = payload["dimension"]["Tid"]["category"]["index"]
    periods = [period for period, _ in sorted(time_index.items(), key=lambda item: item[1])]
    values = payload["value"]
    cells = []
    for position, period in enumerate(periods):
        if isinstance(values, list):
            value = values[position]
            forced_kind = "json_null" if value is None else None
        else:
            key = str(position)
            value = values.get(key, _SPARSE_ABSENT)
            forced_kind = "sparse_absent" if value is _SPARSE_ABSENT else None
            if value is None:
                forced_kind = "json_null"
        cells.append((period, value, "scb_jsonstat_value_cell", forced_kind))
    return tuple(cells)


def parse_fred_csv(text: str, spec: MoneyLiquiditySeries) -> pd.DataFrame:
    """Parse FRED's public graph CSV while preserving its native levels."""
    if spec.response_format != "fred_csv":
        raise ValueError(f"{spec.native_series_id} is not a FRED CSV series")
    raw = _read_csv(text, "FRED")
    required = {"observation_date", spec.native_series_id}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"FRED response missing columns: {sorted(missing)}")
    if raw.empty:
        return _empty_long()

    dates = _parse_date_column(raw["observation_date"], "%Y-%m-%d", "FRED")
    values = _parse_numeric_column(raw[spec.native_series_id], "FRED")
    return _make_long(spec, dates, values)


def parse_boe_csv(text: str, spec: MoneyLiquiditySeries) -> pd.DataFrame:
    """Parse one exact Bank of England IADB monthly or quarterly stock series.

    IADB reports stocks on the native period end. The observatory stores the
    first day of that same month/quarter, while retaining the native labels as
    frame provenance. No interpolation, splicing or resampling occurs.
    """
    if spec.response_format != "boe_csv":
        raise ValueError(f"{spec.native_series_id} is not a Bank of England CSV series")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Bank of England returned an empty response body")
    try:
        rows = list(csv.reader(StringIO(text)))
    except csv.Error as exc:
        raise ValueError("Bank of England returned malformed CSV") from exc
    if len(rows) < 5 or rows[0] != ["SERIES", "DESCRIPTION"]:
        raise ValueError("Bank of England response lacks the exact series-description preamble")
    if rows[1] != [spec.native_series_id, spec.title]:
        raise ValueError(
            f"Bank of England series description mismatch for {spec.native_series_id}: {rows[1]!r}"
        )
    if rows[2] or rows[3] != ["DATE", spec.native_series_id]:
        raise ValueError(
            "Bank of England response columns do not match exact native series "
            f"{spec.native_series_id!r}"
        )
    data_rows = [row for row in rows[4:] if row]
    if any(len(row) != 2 for row in data_rows):
        raise ValueError("Bank of England returned malformed observation rows")
    raw = pd.DataFrame(data_rows, columns=rows[3])
    if raw.empty:
        return _empty_long()

    native_periods = tuple(raw["DATE"].astype(str).str.strip())
    period_ends = _parse_date_column(raw["DATE"], "%d %b %Y", "Bank of England")
    if any(
        observed_on != (pd.Timestamp(observed_on) + pd.offsets.MonthEnd(0)).date()
        for observed_on in period_ends
    ):
        raise ValueError("Bank of England stock date is not a calendar month end")
    if spec.frequency == "monthly":
        dates = period_ends.map(lambda observed_on: observed_on.replace(day=1))
        native_period_format = "calendar month end"
    elif spec.frequency == "quarterly":
        if any(observed_on.month not in {3, 6, 9, 12} for observed_on in period_ends):
            raise ValueError("Bank of England quarterly stock date is not a quarter end")
        dates = period_ends.map(
            lambda observed_on: date(
                observed_on.year,
                ((observed_on.month - 1) // 3) * 3 + 1,
                1,
            )
        )
        native_period_format = "calendar quarter end"
    else:
        raise ValueError(f"unsupported Bank of England stock frequency: {spec.frequency!r}")
    values = _parse_numeric_column(raw[spec.native_series_id], "Bank of England")
    if (values.dropna() <= 0).any():
        raise ValueError("Bank of England broad-money stock must be positive")

    frame = _make_long(spec, dates, values)
    frame.attrs["native_periods"] = native_periods
    frame.attrs["native_period_format"] = native_period_format
    return frame


def parse_boj_csv(text: str, spec: MoneyLiquiditySeries) -> pd.DataFrame:
    """Parse one exact BOJ Time-Series Data Search monthly stock series.

    The API's status/request preamble and native series metadata are validated
    before observations are accepted. A non-empty ``NEXTPOSITION`` means the
    response is incomplete and cannot become a complete-history release.
    """
    if spec.response_format != "boj_csv":
        raise ValueError(f"{spec.native_series_id} is not a Bank of Japan CSV series")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Bank of Japan returned an empty response body")
    try:
        rows = [row for row in csv.reader(StringIO(text)) if row]
    except csv.Error as exc:
        raise ValueError("Bank of Japan returned malformed CSV") from exc

    header = [
        "SERIES_CODE",
        "NAME_OF_TIME_SERIES",
        "UNIT",
        "FREQUENCY",
        "CATEGORY",
        "LAST_UPDATE",
        "SURVEY_DATES",
        "VALUES",
    ]
    header_positions = [index for index, row in enumerate(rows) if row == header]
    if len(header_positions) != 1:
        raise ValueError(
            "Bank of Japan response does not contain exactly one expected CSV data header"
        )
    header_position = header_positions[0]
    preamble = rows[:header_position]

    status = _single_boj_preamble_value(preamble, "STATUS")
    if status != "200":
        raise ValueError(f"Bank of Japan API status is not successful: {status!r}")
    parameters: dict[str, str] = {}
    for row in preamble:
        if row[0].strip().upper() != "PARAMETER":
            continue
        if len(row) != 3:
            raise ValueError("Bank of Japan returned malformed API parameter metadata")
        key = row[1].strip().upper()
        if key in parameters:
            raise ValueError(f"Bank of Japan repeated API parameter metadata: {key!r}")
        parameters[key] = row[2].strip()
    expected_parameters = {"FORMAT": "CSV", "LANG": "EN", "DB": "MD02"}
    for key, expected in expected_parameters.items():
        actual = parameters.get(key, "")
        if actual.upper() != expected:
            label = "database" if key == "DB" else key.lower()
            raise ValueError(
                f"Bank of Japan API {label} mismatch: expected {expected!r}, got {actual!r}"
            )

    next_position = _single_boj_preamble_value(preamble, "NEXTPOSITION")
    if next_position.strip():
        raise ValueError(
            "Bank of Japan response is truncated; NEXTPOSITION must be empty for "
            "a complete native history"
        )

    generated_at_text = _single_boj_preamble_value(preamble, "DATE")
    try:
        api_generated_at = pd.Timestamp(generated_at_text)
        if api_generated_at.tzinfo is None:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Bank of Japan returned an invalid API generation timestamp: {generated_at_text!r}"
        ) from exc

    data_rows = rows[header_position + 1 :]
    if any(len(row) != len(header) for row in data_rows):
        raise ValueError("Bank of Japan returned malformed CSV observation rows")
    if not data_rows:
        return _empty_long()
    raw = pd.DataFrame(data_rows, columns=header, dtype="object")

    series_codes = set(raw["SERIES_CODE"].astype(str).str.strip())
    if series_codes != {spec.native_series_id}:
        raise ValueError(
            "Bank of Japan response does not match native series "
            f"{spec.native_series_id!r}: {sorted(series_codes)!r}"
        )
    names = set(raw["NAME_OF_TIME_SERIES"].astype(str).str.strip())
    if names != {spec.title}:
        raise ValueError(
            f"Bank of Japan series name mismatch for {spec.native_series_id}: {sorted(names)!r}"
        )
    units = set(raw["UNIT"].astype(str).str.strip())
    if units != {spec.native_unit}:
        raise ValueError(
            f"Bank of Japan currency/unit mismatch for {spec.native_series_id}: {sorted(units)!r}"
        )
    frequencies = set(raw["FREQUENCY"].astype(str).str.strip())
    if spec.frequency != "monthly" or frequencies != {"MONTHLY"}:
        raise ValueError(
            f"Bank of Japan frequency mismatch for {spec.native_series_id}: {sorted(frequencies)!r}"
        )
    categories = set(raw["CATEGORY"].astype(str).str.strip())
    if categories != {"Money Stock"}:
        raise ValueError(
            f"Bank of Japan category mismatch for {spec.native_series_id}: {sorted(categories)!r}"
        )

    update_values = set(raw["LAST_UPDATE"].astype(str).str.strip())
    if len(update_values) != 1:
        raise ValueError(
            f"Bank of Japan returned inconsistent last-update dates: {sorted(update_values)!r}"
        )
    update_text = next(iter(update_values))
    update_match = _BOJ_LAST_UPDATE_RE.fullmatch(update_text)
    if not update_match:
        raise ValueError(f"Bank of Japan returned an invalid last-update date: {update_text!r}")
    try:
        publisher_last_updated_on = date(
            int(update_match.group(1)),
            int(update_match.group(2)),
            int(update_match.group(3)),
        )
    except ValueError as exc:
        raise ValueError(
            f"Bank of Japan returned an invalid last-update date: {update_text!r}"
        ) from exc

    native_periods = tuple(raw["SURVEY_DATES"].astype(str).str.strip())
    dates = raw["SURVEY_DATES"].map(_parse_boj_month)
    native_values = raw["VALUES"].astype("string").str.strip()
    native_values = native_values.mask(native_values.str.lower() == "null", pd.NA)
    values = _parse_numeric_column(native_values, "Bank of Japan")
    if (values.dropna() <= 0).any():
        raise ValueError("Bank of Japan money stock must be positive")

    frame = _make_long(spec, dates, values)
    frame.attrs["native_periods"] = native_periods
    frame.attrs["native_period_format"] = "YYYYMM"
    frame.attrs["publisher_last_updated_on"] = publisher_last_updated_on
    frame.attrs["api_generated_at"] = api_generated_at.to_pydatetime()
    return frame


def parse_ecb_csv(text: str, spec: MoneyLiquiditySeries) -> pd.DataFrame:
    """Parse one exact ECB SDMX-CSV series without frequency conversion."""
    if spec.response_format != "ecb_csv":
        raise ValueError(f"{spec.native_series_id} is not an ECB CSV series")
    raw = _read_csv(text, "ECB")
    required = {
        "KEY",
        "FREQ",
        "TIME_PERIOD",
        "OBS_VALUE",
        "OBS_STATUS",
        "TIME_FORMAT",
        "UNIT",
        "UNIT_MULT",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"ECB response missing columns: {sorted(missing)}")
    if raw.empty:
        return _empty_long(include_status=True)

    keys = set(raw["KEY"].dropna().astype(str).str.strip())
    if keys != {spec.native_series_id}:
        raise ValueError(
            f"ECB response does not match native series {spec.native_series_id!r}: {sorted(keys)!r}"
        )
    expected_frequency = {"monthly": "M", "weekly": "W"}.get(spec.frequency)
    frequencies = set(raw["FREQ"].dropna().astype(str).str.strip())
    if expected_frequency is None or frequencies != {expected_frequency}:
        raise ValueError(
            f"ECB frequency mismatch for {spec.native_series_id}: {sorted(frequencies)!r}"
        )

    expected_period_format = {"monthly": "P1M", "weekly": "P7D"}.get(spec.frequency)
    period_formats = set(raw["TIME_FORMAT"].dropna().astype(str).str.strip())
    if expected_period_format is None or period_formats != {expected_period_format}:
        raise ValueError(
            f"ECB period format mismatch for {spec.native_series_id}: {sorted(period_formats)!r}"
        )

    units = set(raw["UNIT"].dropna().astype(str).str.strip())
    multipliers = pd.to_numeric(raw["UNIT_MULT"], errors="coerce")
    expected_power = int(round(math.log10(spec.unit_multiplier)))
    if (
        units != {spec.currency}
        or multipliers.isna().any()
        or set(multipliers) != {float(expected_power)}
    ):
        raise ValueError(
            f"ECB currency/unit mismatch for {spec.native_series_id}: "
            f"UNIT={sorted(units)!r}, UNIT_MULT={sorted(set(multipliers.dropna()))!r}"
        )

    native_periods = tuple(raw["TIME_PERIOD"].astype(str).str.strip())
    dates = raw["TIME_PERIOD"].map(
        _parse_ecb_month if spec.frequency == "monthly" else _parse_ecb_week
    )
    values = _parse_numeric_column(raw["OBS_VALUE"], "ECB")
    statuses = raw["OBS_STATUS"].map(_parse_ecb_status)
    frame = _make_long(spec, dates, values, statuses=statuses)
    frame.attrs["native_periods"] = native_periods
    frame.attrs["native_period_format"] = expected_period_format
    return frame


def parse_scb_jsonstat(text: str, spec: MoneyLiquiditySeries = SCB_M3) -> pd.DataFrame:
    """Parse the exact M3/stock selection from SCB table TAB6541."""
    if spec.response_format != "scb_jsonstat2":
        raise ValueError(f"{spec.native_series_id} is not an SCB JSON-stat series")
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("SCB returned malformed JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("SCB returned an unsupported JSON-stat payload")
    if raw.get("class") != "dataset":
        raise ValueError("SCB response class is not a JSON-stat dataset")
    if _normalized_text(raw.get("source")) != "the riksbank":
        raise ValueError(f"SCB publisher mismatch: {raw.get('source')!r}")
    if _normalized_text(raw.get("label")) != (
        "outstanding money supply, sek millions by monetary aggregate and month"
    ):
        raise ValueError(f"SCB dataset label mismatch: {raw.get('label')!r}")
    role = raw.get("role")
    if not isinstance(role, dict) or role.get("time") != ["Tid"]:
        raise ValueError("SCB response has invalid native time-role metadata")
    if role.get("metric") != ["ContentsCode"]:
        raise ValueError("SCB response has invalid native metric-role metadata")
    values = raw.get("value")
    if values in (None, [], {}):
        return _empty_long()

    dimensions = raw.get("id")
    sizes = raw.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list):
        raise ValueError("SCB response is missing JSON-stat dimensions")
    if len(dimensions) != len(sizes) or "Tid" not in dimensions:
        raise ValueError("SCB response has inconsistent JSON-stat dimensions")
    expected_codes = {
        "Penningm": "5LLM3a.1E.NEP.V.A",
        "ContentsCode": "000007WQ",
    }
    try:
        dimension_map = raw["dimension"]
        for dimension, code in expected_codes.items():
            selected = dimension_map[dimension]["category"]["index"]
            if set(selected) != {code}:
                raise ValueError(
                    f"SCB response does not match {dimension} native selection {code!r}"
                )
        time_index = dimension_map["Tid"]["category"]["index"]
        money_label = dimension_map["Penningm"]["category"]["label"]["5LLM3a.1E.NEP.V.A"]
        contents = dimension_map["ContentsCode"]
        content_category = contents["category"]
        content_label = content_category["label"]["000007WQ"]
        native_unit = content_category["unit"]["000007WQ"]["base"]
        extension = contents["extension"]
        reference_period = extension["refperiod"]["000007WQ"]
        measure_type = extension["measuringType"]["000007WQ"]
        price_type = extension["priceType"]["000007WQ"]
        adjustment = extension["adjustment"]["000007WQ"]
    except (KeyError, TypeError) as exc:
        raise ValueError("SCB response is missing required M3 stock metadata") from exc

    if _normalized_text(money_label) != "m3":
        raise ValueError(f"SCB M3 label mismatch: {money_label!r}")
    if _normalized_text(content_label) != "outstanding money supply, sek millions":
        raise ValueError(f"SCB contents label mismatch: {content_label!r}")
    normalized_unit = " ".join(str(native_unit).strip().lower().split())
    if normalized_unit not in {"mnkr", "sek million", "sek millions"}:
        raise ValueError(f"SCB currency/unit mismatch: {native_unit!r}")
    if _normalized_text(reference_period) != "month":
        raise ValueError(f"SCB reference period mismatch: {reference_period!r}")
    if _normalized_text(measure_type) != "stock":
        raise ValueError(f"SCB measure type mismatch: {measure_type!r}")
    if _normalized_text(price_type) != "current":
        raise ValueError(f"SCB price type mismatch: {price_type!r}")
    if _normalized_text(adjustment) != "none":
        raise ValueError(f"SCB adjustment mismatch: {adjustment!r}")
    if not isinstance(time_index, dict):
        raise ValueError("SCB response uses an unsupported time index")
    for dimension, size in zip(dimensions, sizes, strict=True):
        if dimension != "Tid" and int(size) != 1:
            raise ValueError("SCB response has multiple measures; exact filters are required")

    periods = [label for label, _position in sorted(time_index.items(), key=lambda item: item[1])]
    if int(sizes[dimensions.index("Tid")]) != len(periods):
        raise ValueError("SCB response time dimension size mismatch")
    dense_values = _jsonstat_values(values, len(periods))
    dates = pd.Series(periods, dtype="object").map(_parse_scb_month)
    numeric = _parse_numeric_column(pd.Series(dense_values, dtype="object"), "SCB")
    frame = _make_long(spec, dates, numeric)
    frame.attrs["native_periods"] = tuple(periods)
    frame.attrs["native_period_format"] = "monthly"

    updated = raw.get("updated")
    if updated:
        try:
            parsed = pd.Timestamp(updated)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize(UTC)
            frame.attrs["published_at"] = parsed.to_pydatetime()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SCB returned invalid updated timestamp: {updated!r}") from exc
    return frame


def _read_csv(text: str, provider: str) -> pd.DataFrame:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{provider} returned an empty response body")
    try:
        return pd.read_csv(StringIO(text), dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"{provider} returned malformed CSV") from exc


def _parse_numeric_column(values: pd.Series, provider: str) -> pd.Series:
    normalized = values.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA, "..": pd.NA})
    parsed = pd.to_numeric(normalized, errors="coerce")
    invalid = normalized.notna() & parsed.isna()
    if invalid.any():
        bad = normalized.loc[invalid].iloc[0]
        raise ValueError(f"{provider} returned a non-numeric observation: {bad!r}")
    finite = parsed.dropna().map(math.isfinite)
    if not finite.all():
        raise ValueError(f"{provider} returned a non-finite observation")
    return parsed


def _parse_date_column(values: pd.Series, date_format: str, provider: str) -> pd.Series:
    try:
        return pd.to_datetime(values, format=date_format, errors="raise").dt.date
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{provider} returned an invalid observation date") from exc


def _parse_ecb_month(value: object) -> date:
    match = _MONTH_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"ECB returned an invalid monthly period: {value!r}")
    return date(int(match.group(1)), int(match.group(2)), 1)


def _parse_boj_month(value: object) -> date:
    match = _BOJ_MONTH_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Bank of Japan returned an invalid monthly period: {value!r}")
    return date(int(match.group(1)), int(match.group(2)), 1)


def _parse_ecb_week(value: object) -> date:
    match = _WEEK_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"ECB returned an invalid weekly period: {value!r}")
    try:
        # ECB exposes only an ISO week, not its exact WFS reporting day. Friday
        # is the consistent end-of-business-week representative and matches the
        # normal reporting rule. Holiday and quarter-end exceptions remain a
        # documented limitation of the native weekly key.
        return date.fromisocalendar(int(match.group(1)), int(match.group(2)), 5)
    except ValueError as exc:
        raise ValueError(f"ECB returned an invalid ISO week: {value!r}") from exc


def _parse_scb_month(value: object) -> date:
    match = _SCB_MONTH_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"SCB returned an invalid monthly period: {value!r}")
    return date(int(match.group(1)), int(match.group(2)), 1)


def _parse_ecb_status(value: object) -> str:
    if pd.isna(value):
        raise ValueError("ECB response has a missing observation status")
    code = str(value).strip().upper()
    if not code:
        raise ValueError("ECB response has a missing observation status")
    return _ECB_STATUS.get(code, f"native_{code.lower()}"[:24])


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _single_boj_preamble_value(rows: list[list[str]], label: str) -> str:
    matches = [row for row in rows if row and row[0].strip().upper() == label]
    if len(matches) != 1 or len(matches[0]) != 2:
        raise ValueError(f"Bank of Japan response has invalid {label} preamble metadata")
    return matches[0][1].strip()


def _jsonstat_values(values: object, expected_count: int) -> list[object]:
    if isinstance(values, list):
        if len(values) != expected_count:
            raise ValueError(
                f"SCB response size mismatch: {len(values)} values vs {expected_count} time labels"
            )
        return values
    if isinstance(values, dict):
        unexpected = set(values) - {str(index) for index in range(expected_count)}
        if unexpected:
            raise ValueError("SCB sparse values contain an out-of-range position")
        return [values.get(str(index)) for index in range(expected_count)]
    raise ValueError("SCB response uses an unsupported JSON-stat value shape")


def _make_long(
    spec: MoneyLiquiditySeries,
    dates: pd.Series,
    values: pd.Series,
    *,
    statuses: pd.Series | None = None,
) -> pd.DataFrame:
    if len(dates) != len(values) or (statuses is not None and len(statuses) != len(values)):
        raise ValueError("source response has misaligned observation columns")
    valid = values.notna()
    if not valid.any():
        return _empty_long(include_status=statuses is not None)
    output = pd.DataFrame(
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": dates.loc[valid].tolist(),
            "value": values.loc[valid].astype(float).tolist(),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
    )
    if statuses is not None:
        output["status"] = statuses.loc[valid].tolist()
    if output["date"].duplicated().any():
        raise ValueError(f"{spec.delivery_service} returned duplicate observation periods")
    return output.sort_values("date").reset_index(drop=True)


def _empty_long(*, include_status: bool = False) -> pd.DataFrame:
    columns = [*_LONG_COLUMNS, *(["status"] if include_status else [])]
    return pd.DataFrame(columns=columns)
