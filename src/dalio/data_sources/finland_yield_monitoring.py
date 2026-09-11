"""Pure extraction of the Bank of Finland's original daily benchmark table.

The report embeds its displayed history in ordinary HTML. We select the named
ten-year benchmark column, preserving empty cells rather than flattening text
across individual securities. No transient ReportViewer session is required.
"""

from __future__ import annotations

import math
import re
from datetime import date
from html.parser import HTMLParser

REPORT_URL = (
    "https://reports.suomenpankki.fi/WebForms/ReportViewerPage.aspx?"
    "report=%2Ftilastot%2Farvopaperimarkkinat%2Fvelkapaperit%2Fviitelainojen_korot_v2_en"
)
TITLE = "Yields on Finnish benchmark government bonds"
COLUMN = "Yield on goverment bonds, 10 year"  # Native spelling.
FIVE_YEAR_COLUMN = "Yield on goverment bonds, 5 year"
SERIES_ID = "viitelainojen_korot_v2_en/10_year/daily"
_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
_MONTHS = {
    name: index
    for index, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
    )
}


def specs(end: date) -> list[dict]:
    if type(end) is not date:
        raise ValueError("Finland yield end must be a date")
    return [
        {
            "country": "FI",
            "indicator": "yield_10y",
            "kind": "bank_of_finland_benchmark_html",
            "end": end.isoformat(),
            "series_id": SERIES_ID,
            "requests": [{"role": "data", "method": "GET", "url": REPORT_URL}],
            "publisher": "Bank of Finland",
            "source": "Bank of Finland",
            "label": "Finnish ten-year government benchmark yield",
            "definition": (
                "Finnish ten-year government benchmark bond yield published by the Bank of Finland, "
                "calculated from LSEG primary dealers' daily average selling prices at 13:00 Finnish time. "
                "The named ten-year benchmark column is separate from the individual-security columns."
            ),
            "frequency": "daily",
            "unit": "percent",
            "adjustment": "not_seasonally_adjusted",
            "comparison": "ninety_day_rate",
            "limits": [
                "The official HTML report's default current-year window supplies the displayed history only; "
                "an early-year capture may lack the 90-day anchor. No missing dates are filled.",
                "A benchmark bond can change; this is not a constant-maturity interpolation or a company borrowing rate.",
                "Underlying market quotations are from LSEG, compiled and published by the Bank of Finland.",
                "Values use the conventional percentage-yield scale; percent is not a separate literal unit field in this HTML table.",
                "The report update is supplied only as a calendar date; first-publication time and per-point finality are unavailable.",
                "The report warns of a technical data gap on 2 September 2025; native blanks remain missing.",
            ],
        }
    ]


class _Tables(HTMLParser):
    """Retain actual table cells, including empty cells and nested table boundaries."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.tables = []
        self.text = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "select"}:
            self.suppressed += 1
        if tag == "table":
            self.stack.append({"rows": [], "cell": None})
        elif tag == "tr" and self.stack:
            self.stack[-1]["rows"].append([])
        elif tag in {"td", "th"} and self.stack and self.stack[-1]["rows"]:
            cell = {"attrs": dict(attrs), "text": ""}
            self.stack[-1]["cell"] = cell
            self.stack[-1]["rows"][-1].append(cell)

    def handle_data(self, text):
        if self.suppressed:
            return
        if text.strip():
            self.text.append(text.strip())
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["text"] += text

    def handle_endtag(self, tag):
        if tag in {"script", "style", "select"}:
            self.suppressed -= 1
        if tag in {"td", "th"} and self.stack:
            self.stack[-1]["cell"] = None
        elif tag == "table":
            if not self.stack:
                raise ValueError("Unbalanced Bank of Finland table markup")
            self.tables.append(self.stack.pop()["rows"])


def _cells(row):
    return [" ".join(cell["text"].split()) for cell in row]


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Revalidate a pinned request and derive observations from its original bytes."""
    try:
        end = date.fromisoformat(spec["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid Finland yield descriptor") from exc
    if spec != specs(end)[0]:
        raise ValueError("Finland yield descriptor differs from the pinned original source")
    if set(bodies) != {"data"} or not isinstance(bodies["data"], bytes) or not bodies["data"]:
        raise ValueError("Finland yield requires exactly the original HTML data body")
    parser = _Tables()
    parser.feed(bodies["data"].decode("utf-8-sig"))
    parser.close()
    if parser.stack or parser.suppressed:
        raise ValueError("Incomplete Bank of Finland report markup")
    text = " ".join(parser.text)
    required = (
        TITLE,
        "Statistics on the yields on Finnish government benchmark bonds, issued by the Bank of Finland",
        "calculated from LSEG's data on primary dealers' daily average selling prices of debt instruments as of 13.00 hours.",
    )
    if any(term not in text for term in required):
        raise ValueError("Bank of Finland benchmark definition or attribution changed")
    updates = re.findall(r"Report updated (\d{1,2}) ([A-Za-z]{3}) (\d{4})\b", text)
    if len(updates) != 1:
        raise ValueError("A unique native Bank of Finland report update date is required")
    day, month, year = updates[0]
    try:
        updated = date(int(year), _MONTHS[month], int(day))
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid Bank of Finland report update date") from exc
    if updated > end:
        raise ValueError("Bank of Finland report update postdates the requested end")
    candidates = []
    for table in parser.tables:
        for index, row in enumerate(table):
            if COLUMN in _cells(row):
                candidates.append((table, index))
    if len(candidates) != 1:
        raise ValueError("A unique original ten-year benchmark column is required")
    table, header_index = candidates[0]
    header = _cells(table[header_index])
    if header[:4] != ["", "Period", FIVE_YEAR_COLUMN, COLUMN] or any(
        not name.startswith("Loan period ") for name in header[4:]
    ):
        raise ValueError("Bank of Finland benchmark table dimensions changed")
    observations, missingness, seen = [], [], set()
    for row_index, raw_row in enumerate(table[header_index:], header_index + 1):
        if any(
            str(cell["attrs"].get(span, "1")) != "1"
            for cell in raw_row
            for span in ("colspan", "rowspan")
        ):
            raise ValueError("Unsupported merged Bank of Finland benchmark cells")
        cells = _cells(raw_row)
        if len(cells) != len(header):
            raise ValueError("Bank of Finland benchmark row width changed")
        if row_index == header_index + 1:
            continue
        match = _DATE.fullmatch(cells[1])
        if not match:
            raise ValueError("Bank of Finland observation is not an explicit daily period")
        day, month, year = map(int, match.groups())
        observed = date(year, month, day)
        if observed in seen or observed > min(end, updated):
            raise ValueError("Duplicate or future Bank of Finland observation")
        seen.add(observed)
        native = cells[3]
        if native and not _NUMBER.fullmatch(native):
            raise ValueError("Unexpected Bank of Finland yield value or native status")
        value = float(native) if native else None
        if value is not None and not math.isfinite(value):
            raise ValueError("Nonfinite Bank of Finland yield")
        period = observed.isoformat()
        observation = {
            "date": period,
            "period": period,
            "period_start": period,
            "period_end": period,
            "value": value,
            "status": "observed" if value is not None else "not_reported",
            "native_status": None,
            "native_value": native,
            "source_locator": f"HTML benchmark table row {row_index}, column '{COLUMN}'",
        }
        observations.append(observation)
        if value is None:
            missingness.append({"date": period, "native_value": native, "reason": "native_blank"})
    if not observations:
        raise ValueError("Bank of Finland report contains no daily benchmark rows")
    observations.sort(key=lambda row: row["date"])
    return {
        **{
            key: spec[key]
            for key in (
                "country",
                "indicator",
                "series_id",
                "publisher",
                "source",
                "definition",
                "frequency",
                "unit",
                "adjustment",
                "comparison",
                "label",
                "limits",
            )
        },
        "source_url": REPORT_URL,
        "metadata_url": REPORT_URL,
        "published_at": None,
        "publisher_updated_at": None,
        "publisher_metadata": {
            "report_title": TITLE,
            "native_column": COLUMN,
            "underlying_source": "LSEG",
            "report_updated_date": updated.isoformat(),
            "update_precision": "date",
            "update_timezone": "Europe/Helsinki",
            "quote_time": "13:00 Finnish time",
            "native_definition": required[1] + ", " + required[2],
            "unit_basis": "Percentage yield convention; the original HTML column has no separate unit field.",
            "history_scope": "Displayed rows in the original report's default current-year window",
            "observed_status_basis": "Published daily numeric quote; the source supplies no per-point finality flag",
        },
        "observations": observations,
        "missingness": missingness,
    }
