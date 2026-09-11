"""Pure original-source US monitoring adapters, including an evidenced loan gap.

Full retained responses are validated before observations are returned. G.17
uses its native seasonally adjusted index; EFFR is a market implementation
rate; Treasury CMT is a fitted par yield, not a single bond transaction.
"""

from __future__ import annotations

import calendar
import hashlib
import html
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import urlencode

FED = "https://www.federalreserve.gov/releases"
EFFR_DOC = "https://www.newyorkfed.org/markets/reference-rates/effr"
TREASURY_BASE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)
TREASURY_DOC = "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology"
GAP = "No current broad US corporate new-loan-rate counterpart: the Federal Reserve E.2 survey ended in May 2017 (final release August 2, 2017). Its quarterly small-business successor has a narrower scope; prime rates and lending standards are not substituted."
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
}


def specs(end: date) -> list[dict]:
    """Return the fixed four US attempts, including original gap documentation."""
    if type(end) is not date or end.year < 2024:
        raise ValueError("US monitoring end must be a date from 2024 onwards")
    common = {"country": "US", "end": end.isoformat()}
    industry = {
        **common,
        "indicator": "industrial_production",
        "kind": "fed_g17_text",
        "source": "FED_G17_MONITORING",
        "publisher": "Federal Reserve Board",
        "series_id": "G17/B50001/SA",
        "requests": [
            {"role": "metadata", "method": "GET", "url": f"{FED}/g17/current/default.htm"},
            {"role": "catalogue", "method": "GET", "url": f"{FED}/g17/Current/ipdisk/g17tab1.txt"},
            {"role": "data", "method": "GET", "url": f"{FED}/g17/Current/ipdisk/ip_sa.txt"},
        ],
        "frequency": "monthly",
        "unit": "index",
        "adjustment": "seasonally_adjusted",
        "comparison": "three_month_means",
        "label": "Total industrial production (manufacturing, mining and utilities)",
        "definition": "Federal Reserve G.17 B50001 total industrial production, seasonally adjusted native index with 2017=100, covering manufacturing, mining and electric/gas utilities.",
        "limits": [
            "This output index is not whole-economy GDP; its native scope differs from other countries' industrial measures.",
            "Current histories incorporate revisions. The text file has no per-observation preliminary/final flags; observed is not a claim of finality.",
        ],
    }
    lending = {
        **common,
        "indicator": "corporate_new_lending_rate",
        "kind": "fed_e2_structural_gap",
        "source": "FED_E2_MONITORING",
        "publisher": "Federal Reserve Board",
        "series_id": "E2/discontinued",
        "requests": [{"role": "documentation", "method": "GET", "url": f"{FED}/e2/current/"}],
        "frequency": "monthly",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "three_month_rate",
        "label": "Corporate new-loan rate: current comparable source unavailable",
        "definition": GAP,
        "limits": ["The 2017 historical loan rate is not a current monitoring observation."],
        "unavailable_reason": GAP,
    }
    policy = {
        **common,
        "indicator": "policy_rate",
        "kind": "nyfed_effr_json",
        "source": "NYFED_MONITORING",
        "publisher": "Federal Reserve Bank of New York",
        "series_id": "EFFR",
        "requests": [
            {"role": "documentation", "method": "GET", "url": EFFR_DOC},
            {
                "role": "data",
                "method": "GET",
                "url": "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?"
                + urlencode({"startDate": f"{end.year - 1}-01-01", "endDate": end.isoformat()}),
            },
        ],
        "frequency": "daily",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "ninety_day_rate",
        "label": "Effective federal funds rate (market implementation rate)",
        "definition": "New York Fed EFFR: volume-weighted median percent rate on overnight federal funds transactions reported in FR 2420. This is the effective market rate, not the FOMC target range or its midpoint.",
        "limits": [
            "The prior business day's rate is normally published around 09:00 New York time; observation and acquisition dates are distinct.",
            "An overnight interbank rate is not a company lending rate or a measure of credit availability.",
        ],
    }
    treasury_requests = [{"role": "documentation", "method": "GET", "url": TREASURY_DOC}]
    treasury_requests += [
        {
            "role": role,
            "method": "GET",
            "url": TREASURY_BASE
            + "?"
            + urlencode({"data": "daily_treasury_yield_curve", "field_tdr_date_value": year}),
        }
        for role, year in [("data_previous", end.year - 1), ("data", end.year)]
    ]
    yield_ = {
        **common,
        "indicator": "yield_10y",
        "kind": "treasury_par_xml",
        "source": "US_TREASURY_MONITORING",
        "publisher": "US Department of the Treasury",
        "series_id": "DailyTreasuryYieldCurveRateData/BC_10YEAR",
        "requests": treasury_requests,
        "frequency": "daily",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "ninety_day_rate",
        "label": "Ten-year Treasury constant-maturity par yield",
        "definition": "US Treasury nominal ten-year constant-maturity par yield in percent, fitted with the monotone convex method to indicative bid-side prices of recently auctioned Treasury securities collected near 15:30 New York time.",
        "limits": [
            "This fitted constant-maturity par yield is not a single-bond transaction yield, a zero-coupon rate or an auction funding cost.",
            "Indicative prices are not actual transactions; methodology and market conditions can affect the curve.",
        ],
    }
    return [industry, lending, policy, yield_]


def _text(body):
    value = body.decode("utf-8-sig")
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _require_text(body, phrases):
    value = _text(body)
    if any(phrase.casefold() not in value.casefold() for phrase in phrases):
        raise ValueError("Original source documentation no longer supports pinned definition")
    return value


def _json(body):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate original JSON key")
            value[key] = item
        return value

    return json.loads(body, object_pairs_hook=unique)


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid original numeric value")
    return float(value)


def _row(when, value, locator, *, monthly=False, flag=None):
    start = when.replace(day=1) if monthly else when
    return {
        "date": when.isoformat(),
        "period_start": start.isoformat(),
        "period_end": when.isoformat(),
        "period": when.isoformat()[:7] if monthly else when.isoformat(),
        "value": value,
        "status": "not_reported" if value is None else "observed",
        "native_status": flag,
        "source_locator": locator,
    }


def _industry(spec, bodies):
    page = _require_text(
        bodies["metadata"],
        ["Industrial Production and Capacity Utilization", "Seasonally adjusted", "2017=100"],
    )
    catalogue = bodies["catalogue"].decode("utf-8-sig")
    if not re.search(r"^Total index\s+B50001\s", catalogue, re.M):
        raise ValueError("G.17 native total-index code changed")
    release_match = re.search(r"Release Date: ([A-Z][a-z]+ \d{1,2}, \d{4})", page)
    latest_match = re.search(
        r"At ([\d.]+) percent of its 2017 average, total IP in ([A-Z][a-z]+)", page
    )
    if not release_match or not latest_match:
        raise ValueError("G.17 release date or latest-level binding unavailable")
    release = datetime.strptime(release_match[1], "%B %d, %Y").date()
    month = datetime.strptime(latest_match[2], "%B").month
    latest_year = release.year if month < release.month else release.year - 1
    expected_end = date(latest_year, month, calendar.monthrange(latest_year, month)[1])
    if release > date.fromisoformat(spec["end"]):
        raise ValueError("G.17 release is beyond capture date")
    lines = bodies["data"].decode("utf-8-sig").splitlines()
    if lines.count('"B50001: Total index"') != 1:
        raise ValueError("G.17 native total-index block changed")
    rows, years = [], []
    for line_number, line in enumerate(lines, 1):
        if not line.startswith('"B50001"'):
            continue
        fields = line.split()
        year = int(fields[1])
        values = fields[2:]
        expected_count = month if year == latest_year else 12
        if year > latest_year or len(values) != expected_count:
            raise ValueError("G.17 selected year/month axis incomplete")
        years.append(year)
        for current_month, native_value in enumerate(values, 1):
            value = _number(float(native_value))
            when = date(year, current_month, calendar.monthrange(year, current_month)[1])
            rows.append(
                _row(
                    when,
                    value,
                    f"G17/B50001/SA/{year}-{current_month:02d}#line[{line_number}]",
                    monthly=True,
                )
            )
    if (
        years != list(range(1919, latest_year + 1))
        or not rows
        or rows[-1]["date"] != expected_end.isoformat()
    ):
        raise ValueError("G.17 full selected history is incomplete or duplicated")
    displayed = Decimal(str(rows[-1]["value"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if displayed != Decimal(latest_match[1]):
        raise ValueError("G.17 data and current release levels disagree")
    return (
        rows,
        {
            "publisher": "Federal Reserve Board",
            "native_code": "B50001",
            "native_base": "2017=100",
            "report_updated_date": release.isoformat(),
            "update_precision": "date",
            "release_date_native": release_match[1],
            "latest_reference_period": expected_end.isoformat()[:7],
            "status_convention": "The data text supplies levels without per-observation flags; current release labels some recent estimates preliminary, and historical revisions remain possible.",
        },
        None,
    )


def _policy(spec, bodies):
    _require_text(
        bodies["documentation"], ["volume-weighted median", "overnight federal funds", "FR 2420"]
    )
    raw = _json(bodies["data"])
    if (
        not isinstance(raw, dict)
        or set(raw) != {"refRates"}
        or not isinstance(raw["refRates"], list)
    ):
        raise ValueError("NY Fed response is not one rate collection")
    expected_fields = {
        "effectiveDate",
        "type",
        "percentRate",
        "percentPercentile1",
        "percentPercentile25",
        "percentPercentile75",
        "percentPercentile99",
        "targetRateFrom",
        "targetRateTo",
        "volumeInBillions",
        "revisionIndicator",
    }
    rows = []
    for i, native in enumerate(raw["refRates"]):
        if (
            set(native) != expected_fields
            or native["type"] != "EFFR"
            or native["revisionIndicator"] != ""
        ):
            raise ValueError("NY Fed native rate identity or unreviewed revision flag")
        when = date.fromisoformat(native["effectiveDate"])
        if when.isoformat() != native["effectiveDate"] or when < date(
            int(spec["end"][:4]) - 1, 1, 1
        ):
            raise ValueError("NY Fed date outside requested range")
        rows.append(
            _row(
                when,
                _number(native["percentRate"]),
                f"EFFR/{when.isoformat()}#refRates[{i}]",
                flag=native["revisionIndicator"],
            )
        )
    dates = [r["date"] for r in rows]
    if dates != sorted(set(dates), reverse=True):
        raise ValueError("NY Fed native rows duplicate or unordered")
    return (
        list(reversed(rows)),
        {
            "publisher": "Federal Reserve Bank of New York",
            "native_rate_type": "EFFR",
            "policy_instrument": "Effective overnight federal funds market rate; not the target range",
            "publication_schedule": "Prior business day, approximately 09:00 New York time; no original publication timestamp in the JSON.",
            "native_unit": "percentRate",
        },
        None,
    )


def _treasury(spec, bodies):
    _require_text(
        bodies["documentation"],
        ["par yield curve", "monotone convex", "indicative", "bid-side", "not actual transactions"],
    )
    rows, updates = [], []
    for role, year in [("data_previous", int(spec["end"][:4]) - 1), ("data", int(spec["end"][:4]))]:
        if b"<!DOCTYPE" in bodies[role].upper() or b"<!ENTITY" in bodies[role].upper():
            raise ValueError("Unsupported XML entity declaration")
        root = ET.fromstring(bodies[role])
        if (
            root.tag != f"{{{NS['a']}}}feed"
            or root.findtext("a:title", namespaces=NS) != "DailyTreasuryYieldCurveRateData"
            or root.attrib.get("{http://www.w3.org/XML/1998/namespace}base") != TREASURY_BASE
        ):
            raise ValueError("Treasury native nominal par-yield feed changed")
        if any(link.attrib.get("rel") == "next" for link in root.findall("a:link", NS)):
            raise ValueError("Treasury year feed is paginated unexpectedly")
        updated = datetime.fromisoformat(
            root.findtext("a:updated", namespaces=NS).replace("Z", "+00:00")
        )
        if updated.tzinfo is None:
            raise ValueError("Treasury feed update lacks timezone")
        updates.append(updated)
        entries = root.findall("a:entry", NS)
        if not entries:
            raise ValueError("Treasury selected year is empty")
        for i, entry in enumerate(entries):
            properties = entry.findall("a:content/m:properties", NS)
            if len(properties) != 1:
                raise ValueError("Treasury native entry properties changed")
            values = properties[0].findall("d:BC_10YEAR", NS)
            dates = properties[0].findall("d:NEW_DATE", NS)
            if (
                len(values) != 1
                or len(dates) != 1
                or values[0].attrib.get(f"{{{NS['m']}}}type") != "Edm.Double"
            ):
                raise ValueError("Treasury native ten-year field missing or duplicated")
            native_date = datetime.fromisoformat(dates[0].text)
            if native_date.time().isoformat() != "00:00:00" or native_date.year != year:
                raise ValueError("Treasury native reference date differs from requested year")
            null = values[0].attrib.get(f"{{{NS['m']}}}null")
            if null not in (None, "true") or (null == "true" and values[0].text not in (None, "")):
                raise ValueError("Treasury native null contradicts reported value")
            value = None if null == "true" else _number(float(values[0].text))
            rows.append(
                _row(
                    native_date.date(),
                    value,
                    f"DailyTreasuryYieldCurveRateData/{year}/BC_10YEAR/{native_date.date().isoformat()}#entry[{i}]",
                    flag=null,
                )
            )
    return (
        rows,
        {
            "publisher": "US Department of the Treasury",
            "native_field": "BC_10YEAR",
            "native_type": "Nominal constant-maturity par yield in percent",
            "feed_updated_at": [u.isoformat() for u in updates],
            "clock_convention": "Feed update timestamps are not per-observation first-publication times.",
        },
        max(updates).astimezone(UTC).isoformat(),
    )


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Reconstruct the pinned selection from every required original response."""
    try:
        expected = next(
            s for s in specs(date.fromisoformat(spec["end"])) if s["indicator"] == spec["indicator"]
        )
        if (
            spec != expected
            or set(bodies) != {r["role"] for r in spec["requests"]}
            or any(not isinstance(b, bytes) or not b for b in bodies.values())
        ):
            raise ValueError("US source contract or evidence roles differ")
        if "unavailable_reason" in spec:
            _require_text(
                bodies["documentation"],
                [
                    "has discontinued",
                    "Survey of Terms of Business Lending",
                    "May 2017",
                    "August 2, 2017",
                    "Small Business Lending Survey",
                ],
            )
            return {
                "country": "US",
                "indicator": spec["indicator"],
                "unavailable_reason": spec["unavailable_reason"],
            }
        parser = {
            "industrial_production": _industry,
            "policy_rate": _policy,
            "yield_10y": _treasury,
        }[spec["indicator"]]
        rows, metadata, updated = parser(spec, bodies)
        dates = [r["date"] for r in rows]
        if not rows or dates != sorted(set(dates)) or any(d > spec["end"] for d in dates):
            raise ValueError("US observations duplicate, unordered or beyond capture date")
        metadata["raw_sha256"] = {r: hashlib.sha256(b).hexdigest() for r, b in bodies.items()}
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
            "published_at": None,
            "publisher_updated_at": updated,
            "publisher_metadata": metadata,
            "observations": rows,
            "missingness": {
                "total": len(rows),
                "not_reported": sum(r["value"] is None for r in rows),
                "periods": [r["period"] for r in rows if r["value"] is None],
            },
        }
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, ET.ParseError) as exc:
        raise ValueError("Malformed or incompatible US original source") from exc
