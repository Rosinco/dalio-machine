"""Pure original Destatis/Bundesbank replay plus Germany's shared ECB policy.

Native scopes intentionally remain distinct from other countries' selections.
No HTTP, inference of missing values, or mutation of existing evidence stores.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import math
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

INDUSTRY_CSV = "https://www.destatis.de/static/de_/opendata/data/produktionsindex_industrie_x13.csv"
INDUSTRY_PAGE = (
    "https://www.destatis.de/EN/Themes/Economy/Short-Term-Indicators/Production/kpi117.html"
)
BBK = "https://api.statistiken.bundesbank.de/rest/data"
LOAN_KEY = "M.DE.B.A2A.A.R.A.2240.EUR.N"
YIELD_KEY = "D.REN.EUR.A630.000000WT1010.A"
ECB_KEY = "D.U2.EUR.4F.KR.DFR.LEV"
G = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}"
M = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message}"
LOAN_TITLE = "Effective interest rates of German banks / New business / Loans to non-financial corporations, total / SUD939A"
YIELD_TITLE = "Daily yield of the current 10 year federal bond"
LOAN_DIMENSIONS = dict(
    zip(
        (
            "BBK_STD_FREQ",
            "BBK_STD_AREA",
            "BBK_IRM_REF_SEC",
            "BBK_IRM_ITEM",
            "BBK_BS_MATURITY_ORIG",
            "BBK_IRM_DATA_TYPE",
            "BBK_IRM_AMOUNT_CAT",
            "BBK_BS_COUNT_SEC",
            "BBK_STD_CURRENCY",
            "BBK_IRM_BUS_COV",
        ),
        LOAN_KEY.split("."),
        strict=True,
    )
)
YIELD_DIMENSIONS = dict(
    zip(
        (
            "BBK_STD_FREQ",
            "BBK_SEIS_ITEM",
            "BBK_STD_CURRENCY",
            "BBK_SEIS_SECURITY_CLASS",
            "BBK_SEIS_ISIN",
            "BBK_SEIS_RATING",
        ),
        YIELD_KEY.split("."),
        strict=True,
    )
)
ECB_IDENTITY = dict(
    KEY="FM." + ECB_KEY,
    FREQ="D",
    REF_AREA="U2",
    CURRENCY="EUR",
    PROVIDER_FM="4F",
    INSTRUMENT_FM="KR",
    PROVIDER_FM_ID="DFR",
    DATA_TYPE_FM="LEV",
)
INDUSTRY_COLUMNS = [
    "Datum",
    "Originalwert, 2021=100",
    "Originalwert, Veränderung gegenüber Vorjahresmonat in %",
    "Saison- und kalenderbereinigte Werte nach X 13 JDemetra+, 2021=100",
    "Saison- und kalenderbereinigte Werte nach X 13 JDemetra+, Veränderung gegenüber Vorsmonat in %",
]


def specs(end: date) -> list[dict]:
    """Four fixed original source selections, bounded by acquisition UTC date."""
    if type(end) is not date or end < date(2025, 1, 1):
        raise ValueError("Germany monitoring requires a date from 2025 onwards")
    common = {"country": "DE", "end": end.isoformat()}
    industry = {
        **common,
        "indicator": "industrial_production",
        "kind": "destatis_industry_csv",
        "series_id": "Destatis/produktionsindex_industrie_x13/2021=100/adjusted",
        "requests": [
            {"role": "metadata", "method": "GET", "url": INDUSTRY_PAGE},
            {"role": "data", "method": "GET", "url": INDUSTRY_CSV},
        ],
        "definition": "Destatis production volume index for German industry (except energy and construction), 2021=100, calendar and seasonally adjusted using X13 JDemetra+.",
        "frequency": "monthly",
        "unit": "index",
        "adjustment": "seasonally_adjusted",
        "comparison": "three_month_means",
        "label": "Industry production excluding energy and construction",
        "limits": [
            "This industry aggregate is not GDP and excludes energy and construction; it is not identical to every country's manufacturing aggregate.",
            "Seasonal adjustment and source revisions can change the historical path. The CSV supplies no per-point preliminary/final status.",
            "Destatis will stop updating this original short-term-indicator CSV in October 2026. Migration to GENESIS is required for subsequent releases; retained values must not be represented as newly updated.",
        ],
        "publisher": "Federal Statistical Office (Destatis)",
        "source": "DESTATIS_MONITORING",
        "documentation_url": INDUSTRY_PAGE,
    }
    loan = {
        **common,
        "indicator": "corporate_new_lending_rate",
        "kind": "bundesbank_generic_sdmx",
        "series_id": "BBIM1." + LOAN_KEY,
        "requests": [
            {
                "role": "data",
                "method": "GET",
                "url": f"{BBK}/BBIM1/{LOAN_KEY}?"
                + urlencode(
                    {
                        "format": "sdmx",
                        "lang": "en",
                        "startPeriod": "2025-01",
                        "endPeriod": end.isoformat()[:7],
                    }
                ),
            }
        ],
        "definition": "Bundesbank volume-weighted effective annual interest rate on new EUR loan agreements, including renegotiated existing contracts, by German reporting MFIs with euro-area non-financial corporations. All loan sizes and initial fixation periods; excluding overdrafts. Stratified sample survey, not solely domestic German borrowers.",
        "frequency": "monthly",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "three_month_rate",
        "label": "German-bank NFC new-business lending rate (EUR)",
        "limits": [
            "New agreements include renegotiations and differ from actual drawdowns, outstanding-loan rates and exclusively domestic-borrower measures.",
            "The AAR/NDER effective rate includes interest payments and discounts but excludes other administration, documentation, guarantee and insurance charges.",
            "Changing loan composition affects this volume-weighted average; it is not a quoted rate for an individual company. Native preliminary flags remain explicit.",
        ],
        "publisher": "Deutsche Bundesbank",
        "source": "BUNDESBANK_MONITORING",
        "documentation_url": "https://www.bundesbank.de/en/statistics/money-and-capital-markets/interest-rates-and-yields/mfi-interest-rate-statistics-amounts-outstanding-new-business--651536",
    }
    ecb_url = f"https://data-api.ecb.europa.eu/service/data/FM/{ECB_KEY}"
    policy = {
        **common,
        "indicator": "policy_rate",
        "kind": "ecb_deposit_facility_csv",
        "series_id": "FM." + ECB_KEY,
        "requests": [
            {
                "role": "metadata",
                "method": "GET",
                "url": ecb_url + "?detail=serieskeysonly&format=csvdata",
            },
            {
                "role": "data",
                "method": "GET",
                "url": ecb_url
                + "?"
                + urlencode(
                    {"startPeriod": "2025-01-01", "endPeriod": end.isoformat(), "format": "csvdata"}
                ),
            },
        ],
        "definition": "ECB deposit-facility percent-per-annum rate: shared euro-area monetary policy applicable to Germany as a euro-area member. Native daily observations are retained without interpolation; this is not a separate German national policy instrument.",
        "frequency": "daily",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "ninety_day_rate",
        "label": "ECB deposit facility (shared euro-area policy)",
        "limits": [
            "Shared euro-area policy does not imply identical financing conditions across member countries.",
            "The deposit-facility rate is not a company lending rate or an individual borrower's refinancing cost.",
        ],
        "publisher": "European Central Bank",
        "source": "ECB_MONITORING",
        "documentation_url": "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html",
    }
    yield_spec = {
        **common,
        "indicator": "yield_10y",
        "kind": "bundesbank_generic_sdmx",
        "series_id": "BBSSY." + YIELD_KEY,
        "requests": [
            {
                "role": "data",
                "method": "GET",
                "url": f"{BBK}/BBSSY/{YIELD_KEY}?"
                + urlencode(
                    {
                        "format": "sdmx",
                        "lang": "en",
                        "startPeriod": "2025-01-01",
                        "endPeriod": end.isoformat(),
                    }
                ),
            }
        ],
        "definition": "Bundesbank daily yield of the current 10 year federal bond, in percent per annum: the current German Federal bond with an agreed original ten-year maturity.",
        "frequency": "daily",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "ninety_day_rate",
        "label": "Current German ten-year federal-bond yield",
        "limits": [
            "This is a current-security benchmark, not a constant-maturity interpolation or a Svensson zero-coupon yield-curve estimate; benchmark security changes can affect comparisons.",
            "The original ten-year term is not an assertion of exactly ten years of remaining maturity on every observation date. Native missing slots remain null; their status codes are retained without inferring a cause from the date.",
            "Government benchmark yields do not measure a company's credit spread or borrowing cost.",
        ],
        "publisher": "Deutsche Bundesbank",
        "source": "BUNDESBANK_MONITORING",
        "documentation_url": "https://www.bundesbank.de/en/statistics/money-and-capital-markets/interest-rates-and-yields/daily-yields-of-current-federal-securities-772220",
    }
    return [industry, loan, policy, yield_spec]


def _number(value: str) -> float:
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError("Invalid source number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite source number")
    return result


def _month(value: str) -> tuple[date, date]:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Invalid source month")
    year, month = map(int, value.split("-"))
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _row(start, end, value, flag, locator, *, daily=False, attributes=None):
    return {
        "date": end.isoformat(),
        "period": end.isoformat() if daily else end.isoformat()[:7],
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "value": value,
        "status": "not_reported" if value is None else "observed",
        "native_status": flag,
        "source_locator": locator,
        "native_attributes": attributes or {},
    }


class _Tables(HTMLParser):
    """Read native HTML tables without browser execution or external dependencies."""

    def __init__(self):
        super().__init__()
        self.tables, self.text = [], []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.table is not None:
                raise ValueError("Nested source table")
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _industry(bodies):
    parser = _Tables()
    parser.feed(bodies["metadata"].decode("utf-8"))
    text = " ".join(" ".join(parser.text).split())
    if not all(
        part in text
        for part in (
            "Production (except energy and construction).",
            "2021=100",
            "X 13 JDemetra+ Table",
            "October 2026",
        )
    ):
        raise ValueError("Destatis scope, index base or source maintenance notice changed")
    tables = [
        t
        for t in parser.tables
        if t
        and t[0]
        == [
            "Year, month",
            "Non-adjusted value",
            "Calendar adjusted and seasonally adjusted using X13 JDemetra+",
        ]
    ]
    if len(tables) != 1:
        raise ValueError("Destatis X13 table missing or ambiguous")
    updated = re.findall(r"As at ([A-Za-z]+ \d{1,2}, \d{4})", text)
    if not updated or len(set(updated)) != 1:
        raise ValueError("Destatis as-at dates missing or inconsistent")
    updated_date = datetime.strptime(updated[0], "%B %d, %Y").date()
    native = list(csv.reader(io.StringIO(bodies["data"].decode("utf-8-sig")), delimiter=";"))
    if native[:2] != [["Produktionsindex, Industrie"], INDUSTRY_COLUMNS]:
        raise ValueError("Destatis native CSV title, units or columns changed")
    rows, by_period = [], {}
    for line, record in enumerate(native[2:], 3):
        if len(record) != 5 or not re.fullmatch(r"01/\d{2}/\d{4}", record[0]):
            raise ValueError("Destatis CSV width or monthly reference date changed")
        start = datetime.strptime(record[0], "%d/%m/%Y").date()
        start, stop = _month(start.isoformat()[:7])
        values = [None if v == "." else _number(v.replace(",", ".")) for v in record[1:]]
        rows.append(_row(start, stop, values[2], None, f"data#csv_row[{line}],column[4]"))
        if rows[-1]["period"] in by_period:
            raise ValueError("Duplicate Destatis month")
        by_period[rows[-1]["period"]] = values
    year, common = None, []
    months = {calendar.month_abbr[m]: m for m in range(1, 13)}
    for record in tables[0]:
        if len(record) == 6 and re.fullmatch(r"\d{4}", record[0]):
            year = int(record[0])
            record = record[1:]
        if len(record) == 5 and record[0] in months and year is not None:
            period = f"{year:04d}-{months[record[0]]:02d}"
            values = [None if v in {".", "-", "…"} else _number(v) for v in record[1:]]
            if period not in by_period or by_period[period] != values or period in common:
                raise ValueError("Destatis HTML and CSV releases disagree")
            common.append(period)
    if (
        not common
        or max(common) != max(by_period)
        or max(r["date"] for r in rows) > updated_date.isoformat()
    ):
        raise ValueError("Destatis source reference/update dates disagree")
    return (
        rows,
        {
            "publisher": "Federal Statistical Office (Destatis)",
            "native_title": "Produktionsindex, Industrie",
            "native_columns": INDUSTRY_COLUMNS,
            "native_adjustment": "X13 JDemetra+, calendar and seasonal",
            "base_period": "2021",
            "industry_scope": "Production except energy and construction",
            "report_updated_date": updated_date.isoformat(),
            "update_precision": "date",
            "source_updates_stop": "2026-10",
            "csv_html_reconciled_months": len(common),
            "status_convention": "Native CSV supplies no per-point status; published non-forecast values may be preliminary or revised.",
        },
        None,
    )


def _attributes(parent):
    if parent is None:
        return {}
    values = list(parent)
    result = {v.get("id"): v.get("value") for v in values}
    if len(result) != len(values) or None in result or any(v.tag != G + "Value" for v in values):
        raise ValueError("Duplicate or malformed SDMX attributes")
    return result


def _aware(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Native update clock requires timezone")
    return parsed.astimezone(UTC)


def _bundesbank(spec, bodies):
    body = bodies["data"]
    if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise ValueError("SDMX entities are not allowed")
    root = ET.fromstring(body)
    is_loan = spec["indicator"] == "corporate_new_lending_rate"
    flow, structure = ("BBIM1", "BBK_IRM") if is_loan else ("BBSSY", "BBK_SESY")
    if root.tag != M + "GenericData" or len(root.findall(M + "DataSet")) != 1:
        raise ValueError("Unexpected Bundesbank SDMX envelope")
    dataset = root.find(M + "DataSet")
    if (
        dataset.get("setID") != flow
        or dataset.get("structureRef") != structure
        or dataset.get("action") != "Replace"
    ):
        raise ValueError("Bundesbank dataset identity changed")
    if root.find(M + "Header/" + M + "Sender").get("id") != "BBK":
        raise ValueError("Bundesbank sender changed")
    updated = _aware(dataset.get("validFromDate"))
    prepared = _aware(root.findtext(M + "Header/" + M + "Prepared"))
    if abs((updated - prepared).total_seconds()) >= 1:
        raise ValueError("Bundesbank dataset update clocks disagree")
    series_list = dataset.findall(G + "Series")
    if len(series_list) != 1:
        raise ValueError("Expected one selected Bundesbank series")
    series = series_list[0]
    if len(series.findall(G + "SeriesKey")) != 1 or len(series.findall(G + "Attributes")) != 1:
        raise ValueError("Duplicate or missing Bundesbank series metadata")
    dimensions = _attributes(series.find(G + "SeriesKey"))
    attrs = _attributes(series.find(G + "Attributes"))
    expected = {
        "BBK_ID": spec["series_id"],
        "TIME_FORMAT": "P1M" if is_loan else "P1D",
        "BBK_UNIT": "% p.a." if is_loan else "PROZENT",
        "BBK_UNIT_MULT": "0",
        "BBK_DECIMALS": "2",
        "BBK_TITLE_ENG": LOAN_TITLE if is_loan else YIELD_TITLE,
    }
    if dimensions != (LOAN_DIMENSIONS if is_loan else YIELD_DIMENSIONS) or any(
        attrs.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Bundesbank native identity, unit or title changed")
    if is_loan and not all(
        p in attrs.get("BBK_COMM_GEN_ENG", "")
        for p in (
            "sample basis (stratified sample)",
            "Volume-weighted average rates of all new agreements",
            "Excluding overdrafts.",
        )
    ):
        raise ValueError("Bundesbank lending definition changed")
    rows = []
    for line, obs in enumerate(series.findall(G + "Obs"), 1):
        if len(obs.findall(G + "ObsDimension")) != 1 or len(obs.findall(G + "Attributes")) > 1:
            raise ValueError("Duplicate or missing Bundesbank observation metadata")
        when = obs.find(G + "ObsDimension").get("value")
        if is_loan:
            start, stop = _month(when)
        else:
            start = stop = date.fromisoformat(when)
            if start.isoformat() != when:
                raise ValueError("Invalid native daily date")
        if start < date(2025, 1, 1) or stop > updated.date():
            raise ValueError("Bundesbank observation outside request or after dataset update")
        flags = _attributes(obs.find(G + "Attributes"))
        if set(flags) - {"OBS_STATUS", "BBK_DIFF"}:
            raise ValueError("Unreviewed Bundesbank observation attributes")
        flag = flags.get("OBS_STATUS")
        if flag not in {None, "A", "P", "R", "K", "M"}:
            raise ValueError("Unreviewed or forecast Bundesbank observation status")
        values = obs.findall(G + "ObsValue")
        if len(values) > 1:
            raise ValueError("Duplicate source value")
        value = _number(values[0].get("value")) if values else None
        if (value is None) != (flag in {"K", "M"}):
            raise ValueError("Bundesbank status/value contradiction")
        rows.append(
            _row(
                start,
                stop,
                value,
                flag,
                f"data#Series[1]/Obs[{line}]",
                daily=not is_loan,
                attributes=flags,
            )
        )
    metadata = {
        "publisher": "Deutsche Bundesbank",
        "native_dimensions": dimensions,
        "native_attributes": attrs,
        "native_dataset_attributes": dataset.attrib,
        "prepared": prepared.isoformat(),
        "update_clock": "SDMX dataset validFromDate; not separately established first-publication time.",
        "status_convention": "Absent/A is a published observation without a finality assertion; P preliminary, R revised. Native K/M slots have no value; their codes are retained without inferring a missingness cause.",
    }
    if is_loan:
        metadata.update(
            borrower_area="Euro area",
            borrower_sector="Non-financial corporations (2240)",
            currency_scope="EUR",
            reporting_population="German MFIs, stratified sample",
            transaction_type="New agreements including renegotiated existing loans",
        )
    return rows, metadata, updated.isoformat()


def _csv(body):
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Duplicate or empty source CSV header")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed source CSV row width")
    return rows


def _policy(bodies):
    if _csv(bodies["metadata"]) != [ECB_IDENTITY]:
        raise ValueError("ECB policy metadata key changed")
    expected = {
        **ECB_IDENTITY,
        "UNIT": "PCPA",
        "UNIT_MULT": "0",
        "TIME_FORMAT": "P1D",
        "COLLECTION": "E",
        "COMPILING_ORG": "4F0",
        "TITLE": "Deposit facility - date of changes (raw data) - Level",
        "TITLE_COMPL": "Euro area (changing composition) - Key interest rate - Deposit facility - date of changes (raw data) - Level - Euro, provided by ECB",
    }
    rows = []
    for line, row in enumerate(_csv(bodies["data"]), 2):
        if (
            any(row.get(k) != v for k, v in expected.items())
            or row["OBS_STATUS"] not in {"A", "M"}
            or row["OBS_CONF"] != "F"
        ):
            raise ValueError("ECB policy native identity, unit or status changed")
        when = date.fromisoformat(row["TIME_PERIOD"])
        if when.isoformat() != row["TIME_PERIOD"] or when < date(2025, 1, 1):
            raise ValueError("ECB date outside native request")
        value = None if row["OBS_VALUE"] == "" else _number(row["OBS_VALUE"])
        if (row["OBS_STATUS"] == "M") != (value is None):
            raise ValueError("ECB native status/value contradiction")
        rows.append(_row(when, when, value, row["OBS_STATUS"], f"data#csv_row[{line}]", daily=True))
    return (
        rows,
        {
            "publisher": "European Central Bank",
            "policy_area": "Euro area (changing composition)",
            "applicable_country": "Germany, euro-area member; not a national German series",
            "native_identity": ECB_IDENTITY,
            "native_attributes": expected,
            "status_convention": "A is native normal observation, not an assertion of finality; M denotes a missing value.",
            "update_clock": "Source CSV has no publisher update timestamp.",
        },
        None,
    )


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Replay exact original bytes against the pinned native source contract."""
    try:
        expected = next(
            s for s in specs(date.fromisoformat(spec["end"])) if s["indicator"] == spec["indicator"]
        )
        if spec != expected or set(bodies) != {r["role"] for r in spec["requests"]}:
            raise ValueError("German source contract or response roles differ")
        if any(not isinstance(b, bytes) or not b for b in bodies.values()):
            raise ValueError("Original response bytes required")
        if spec["indicator"] == "industrial_production":
            rows, metadata, updated = _industry(bodies)
        elif spec["indicator"] == "policy_rate":
            rows, metadata, updated = _policy(bodies)
        else:
            rows, metadata, updated = _bundesbank(spec, bodies)
        dates = [r["date"] for r in rows]
        if not rows or dates != sorted(set(dates)) or any(d > spec["end"] for d in dates):
            raise ValueError("Native dates empty, duplicate, unsorted or after capture date")
        if metadata.get("report_updated_date", spec["end"]) > spec["end"]:
            raise ValueError("Publisher report date is after acquisition date")
        metadata.update(
            documentation_url=spec["documentation_url"],
            raw_sha256={r: hashlib.sha256(b).hexdigest() for r, b in bodies.items()},
            observed_convention="Published non-forecast value, not a claim of finality. Native missing values and statuses are preserved.",
        )
        return {
            **{
                k: spec[k]
                for k in (
                    "country",
                    "indicator",
                    "source",
                    "series_id",
                    "definition",
                    "frequency",
                    "unit",
                    "adjustment",
                    "comparison",
                    "label",
                    "limits",
                )
            },
            "source_url": next(r["url"] for r in spec["requests"] if r["role"] == "data"),
            "metadata_url": next(
                (r["url"] for r in spec["requests"] if r["role"] == "metadata"),
                spec["requests"][0]["url"],
            ),
            "published_at": None,
            "publisher_updated_at": updated,
            "publisher_metadata": metadata,
            "observations": rows,
            "missingness": {
                "explicit_nulls": sum(r["value"] is None for r in rows),
                "policy": "Retain native missing slots; do not interpolate or replace null with zero.",
            },
        }
    except (
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        StopIteration,
        ET.ParseError,
        UnicodeError,
        csv.Error,
    ) as exc:
        raise ValueError("Malformed original German source response or contract") from exc
