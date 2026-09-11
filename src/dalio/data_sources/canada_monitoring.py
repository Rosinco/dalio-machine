"""Pure original Statistics Canada and Bank of Canada monitoring selections.

Canada's industrial measure is native real industrial GDP, not a rebased
output index. The selected funds-advanced loan rate has broader borrower and
renewal coverage than an NFC-only new-business interest rate.
"""

from __future__ import annotations

import calendar
import hashlib
import html
import json
import math
import re
from datetime import date, datetime
from urllib.parse import urlencode

WDS = "https://www150.statcan.gc.ca/t1/wds/rest"
VALET = "https://www.bankofcanada.ca/valet"
COORD = "1.1.1.10.0.0.0.0.0.0"
VECTOR = 65201219
VOLUME_UNIT = "millions of chained 2017 Canadian dollars"
VOLUME_KIND = "real_industrial_value_added_volume"
LENDING_DOC = "https://www.bankofcanada.ca/rates/banking-and-financial-statistics/interest-rates-for-new-and-existing-lending-by-chartered-banks/"
YIELD_DOC = "https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/"


def specs(end: date) -> list[dict]:
    """Four explicit native inputs, using actual units and no artificial index."""
    if type(end) is not date or end.year < 2024:
        raise ValueError("Canada monitoring end must be a date from 2024 onwards")
    common = {"country": "CA", "end": end.isoformat()}
    query = {"productId": 36100434, "coordinate": COORD}
    industry = {
        **common,
        "indicator": "industrial_production",
        "kind": "statistics_canada_wds",
        "source": "STATISTICS_CANADA_MONITORING",
        "publisher": "Statistics Canada",
        "series_id": f"36100434/{COORD}/v{VECTOR}",
        "requests": [
            {
                "role": "metadata",
                "method": "POST",
                "url": WDS + "/getCubeMetadata",
                "json": [{"productId": 36100434}],
            },
            {
                "role": "series",
                "method": "POST",
                "url": WDS + "/getSeriesInfoFromCubePidCoord",
                "json": [query],
            },
            {
                "role": "data",
                "method": "POST",
                "url": WDS + "/getDataFromCubePidCoordAndLatestNPeriods",
                "json": [{**query, "latestN": 36}],
            },
        ],
        "frequency": "monthly",
        "unit": VOLUME_UNIT,
        "measure_kind": VOLUME_KIND,
        "adjustment": "seasonally_adjusted",
        "comparison": "three_month_means",
        "label": "Real industrial GDP volume (T010, annual rates)",
        "definition": "Statistics Canada T010 industrial production real value added at basic prices, Canada, seasonally adjusted at annual rates, in millions of chained 2017 Canadian dollars. The industrial aggregate comprises mining, quarrying, oil/gas extraction, manufacturing and utilities under NAICS Canada 2022.",
        "limits": [
            "This is industrial real GDP value added, not a physical output index or whole-economy GDP; it is not rebased into an invented index.",
            "The three-month comparison uses a ratio of native volume averages, not nominal revenue growth or GDP forecast error.",
            "Annualized levels and chained-volume components must not be added across sectors or compared as like-for-like index levels.",
            "The current vintage can revise historical values; native release times lack a timezone offset and are retained without inventing an availability clock.",
        ],
    }
    configs = [
        (
            "corporate_new_lending_rate",
            "V122667819",
            "Loans to individuals and others for business purposes",
            "Credit extended to corporate sector - Funds advanced - Business loans, Total - non-mortgage loans",
            "Business-purpose non-mortgage funds-advanced rate",
            "Bank of Canada volume-weighted percent rate on funds advanced for non-mortgage loans to individuals and others for business purposes, booked in Canada in Canadian dollars by banks and foreign bank branches. Funds advanced includes new credit, draws on existing facilities, renewals and refinancing.",
            [
                "The borrower scope includes individuals and others for business purposes; it is not a pure non-financial-corporation sample.",
                "Separate regulated non-bank financial-institution, lease-receivable and non-residential mortgage categories are not selected.",
                "Funds-advanced rates differ from outstanding-balance rates and from other countries' new-agreement or actual-drawdown definitions.",
                "Borrower composition, loan mix and refinancing can change the average without measuring credit availability.",
            ],
            LENDING_DOC,
        ),
        (
            "policy_rate",
            "V39079",
            "Target for the overnight rate (business daily)",
            "Also called the policy interest rate, the average rate that the Bank of Canada wants to see in the market for overnight money market financing. (V39079)",
            "Bank of Canada target overnight rate",
            "Bank of Canada target for the overnight rate, percent, native business-daily observations. This is the policy target, not the realized overnight market rate or the Bank Rate.",
            [
                "A policy target is not a company borrowing rate; contractual repricing and risk affect transmission."
            ],
            None,
        ),
        (
            "yield_10y",
            "BD.CDN.10YR.DQ.YLD",
            "Benchmark bond yield: 10 year",
            "The yield on the Government of Canada 10-year benchmark bond. This series represents the prevailing market interest rate for the 10-year Government of Canada bond and is used as a reference for financial market analysis and reporting.",
            "Government of Canada ten-year benchmark yield",
            "Bank of Canada ten-year Government of Canada benchmark bond yield in percent, based on mid-market closing yields of selected issues maturing approximately at the stated term.",
            [
                "The benchmark can roll between bonds and is not necessarily the issue closest to the stated maturity.",
                "This is a selected-bond benchmark, not an interpolated constant-maturity rate, auction funding cost or company borrowing rate.",
            ],
            YIELD_DOC,
        ),
    ]
    result = [industry]
    for indicator, series, native_label, description, label, definition, limits, doc in configs:
        monthly = indicator == "corporate_new_lending_rate"
        requests = [{"role": "metadata", "method": "GET", "url": f"{VALET}/series/{series}/json"}]
        if doc:
            requests.append({"role": "documentation", "method": "GET", "url": doc})
        requests.append(
            {
                "role": "data",
                "method": "GET",
                "url": f"{VALET}/observations/{series}/json?"
                + urlencode({"start_date": f"{end.year - 2}-01-01", "end_date": end.isoformat()}),
            }
        )
        result.append(
            {
                **common,
                "indicator": indicator,
                "kind": "bank_of_canada_valet",
                "source": "BANK_OF_CANADA_MONITORING",
                "publisher": "Bank of Canada",
                "series_id": series,
                "native_label": native_label,
                "native_description": description,
                "requests": requests,
                "frequency": "monthly" if monthly else "daily",
                "unit": "percent",
                "adjustment": "not_adjusted",
                "comparison": "three_month_rate" if monthly else "ninety_day_rate",
                "label": label,
                "definition": definition,
                "limits": limits,
            }
        )
    return result


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate original Canadian JSON key")
            result[key] = value
        return result

    return json.loads(body, object_pairs_hook=unique)


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid Canadian native numeric value")
    return float(value)


def _row(start, stop, value, locator, flag=None, daily=False):
    return {
        "date": stop.isoformat(),
        "period_start": start.isoformat(),
        "period_end": stop.isoformat(),
        "period": stop.isoformat() if daily else stop.isoformat()[:7],
        "value": value,
        "status": "not_reported" if value is None else "observed",
        "native_status": flag,
        "source_locator": locator,
    }


def _object(body):
    raw = _json(body)
    if (
        not isinstance(raw, list)
        or len(raw) != 1
        or raw[0]["status"] != "SUCCESS"
        or raw[0]["object"]["responseStatusCode"] != 0
    ):
        raise ValueError("Statistics Canada response failed or is not one selected result")
    return raw[0]["object"]


def _month_axis(last, count):
    result = []
    for offset in range(count - 1, -1, -1):
        number = last.year * 12 + last.month - 1 - offset
        result.append(date(number // 12, number % 12 + 1, 1).isoformat())
    return result


def _industry(spec, bodies):
    meta, series, raw = (_object(bodies[r]) for r in ("metadata", "series", "data"))
    if (
        meta["productId"] != "36100434"
        or meta["cubeTitleEn"]
        != "Gross domestic product (GDP) at basic prices, by industry, monthly"
        or meta["frequencyCode"] != 6
        or meta["archiveStatusCode"] != "2"
    ):
        raise ValueError("Statistics Canada table identity or current status changed")
    expected_members = [
        (1, "Geography", 1, "Canada"),
        (2, "Seasonal adjustment", 1, "Seasonally adjusted at annual rates"),
        (3, "Prices", 1, "Chained (2017) dollars"),
        (4, "North American Industry Classification System (NAICS)", 10, "Industrial production"),
    ]
    if len(meta["dimension"]) != 4:
        raise ValueError("Statistics Canada native dimension set changed")
    selected = []
    for dimension, (position, name, member_id, label) in zip(
        meta["dimension"], expected_members, strict=True
    ):
        members = [m for m in dimension["member"] if m["memberId"] == member_id]
        if (
            dimension["dimensionPositionId"] != position
            or dimension["dimensionNameEn"] != name
            or len(members) != 1
            or members[0]["memberNameEn"] != label
            or members[0]["terminated"] != 0
        ):
            raise ValueError("Statistics Canada selected native scope changed")
        selected.append(members[0])
    if selected[-1]["classificationCode"] != "T010" or selected[-1]["memberUomCode"] != 81:
        raise ValueError("Statistics Canada industrial classification or unit changed")
    expected = {"productId": 36100434, "coordinate": COORD, "vectorId": VECTOR}
    if any(series.get(k) != v or raw.get(k) != v for k, v in expected.items()):
        raise ValueError("Statistics Canada series coordinate or vector changed")
    if any(
        series.get(k) != v
        for k, v in {
            "frequencyCode": 6,
            "scalarFactorCode": 6,
            "memberUomCode": 81,
            "terminated": 0,
            "SeriesTitleEn": "Canada;Seasonally adjusted at annual rates;Chained (2017) dollars;Industrial production",
        }.items()
    ):
        raise ValueError("Statistics Canada series units, scale or definition changed")
    end = date.fromisoformat(meta["cubeEndDate"])
    native_update = datetime.fromisoformat(meta["releaseTime"])
    if (
        end.day != 1
        or native_update.tzinfo is not None
        or native_update.date() > date.fromisoformat(spec["end"])
    ):
        raise ValueError("Statistics Canada native release date or precision changed")
    axis = _month_axis(end, 36)
    observations = raw["vectorDataPoint"]
    if [row["refPer"] for row in observations] != axis:
        raise ValueError("Statistics Canada requested latest-36 monthly axis is incomplete")
    rows = []
    for i, native in enumerate(observations):
        if (
            any(
                native.get(k) != v
                for k, v in {
                    "scalarFactorCode": 6,
                    "frequencyCode": 6,
                    "symbolCode": 0,
                    "statusCode": 0,
                    "securityLevelCode": 0,
                    "refPer2": "",
                    "refPerRaw2": "",
                }.items()
            )
            or native["refPerRaw"] != native["refPer"]
        ):
            raise ValueError(
                "Unreviewed Statistics Canada units, status, suppression or reference period"
            )
        released = datetime.fromisoformat(native["releaseTime"])
        start = date.fromisoformat(native["refPer"])
        stop = start.replace(day=calendar.monthrange(start.year, start.month)[1])
        if released.tzinfo is not None or released > native_update or stop > released.date():
            raise ValueError("Statistics Canada point release clock contradicts dataset")
        rows.append(
            _row(
                start,
                stop,
                _number(native["value"]),
                f"36100434/v{VECTOR}/{native['refPer']}#vectorDataPoint[{i}]",
                flag={
                    k: native[k]
                    for k in ("symbolCode", "statusCode", "securityLevelCode", "releaseTime")
                },
            )
        )
    if observations[-1]["releaseTime"] != meta["releaseTime"]:
        raise ValueError("Statistics Canada metadata and latest data vintage disagree")
    return rows, {
        "publisher": "Statistics Canada",
        "product_id": "36100434",
        "native_coordinate": COORD,
        "vector_id": VECTOR,
        "native_series": series,
        "selected_members": selected,
        "native_footnotes": meta["footnote"],
        "native_corrections": meta.get("correction", []),
        "report_updated_date": native_update.date().isoformat(),
        "update_precision": "native timestamp without timezone offset",
        "release_time_native": meta["releaseTime"],
        "native_update_timezone": "Not supplied in response; no UTC timestamp inferred.",
        "scale_convention": "Native scalarFactorCode6 denotes millions; values remain in millions rather than multiplied or rebased.",
        "status_convention": "Original statusCode0/symbolCode0/securityLevelCode0 retained; observed does not imply finality.",
    }


def _document(body, phrases):
    value = body.decode("utf-8-sig")
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S | re.I)
    value = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value)))
    if any(p.casefold() not in value.casefold() for p in phrases):
        raise ValueError("Bank of Canada documentation no longer supports native selection")


def _valet(spec, bodies):
    meta, raw = _json(bodies["metadata"]), _json(bodies["data"])
    series = spec["series_id"]
    if (
        meta["terms"]["url"] != "https://www.bankofcanada.ca/terms/"
        or raw["terms"] != meta["terms"]
    ):
        raise ValueError("Bank of Canada publisher terms identity changed")
    expected = {
        "name": series,
        "label": spec["native_label"],
        "description": spec["native_description"],
    }
    if meta["seriesDetails"] != expected or set(raw["seriesDetail"]) != {series}:
        raise ValueError("Bank of Canada native series identity changed")
    detail = raw["seriesDetail"][series]
    if detail != {
        "label": expected["label"],
        "description": expected["description"],
        "dimension": {"key": "d", "name": "Date"},
    }:
        raise ValueError("Bank of Canada data and metadata definitions disagree")
    if spec["indicator"] == "corporate_new_lending_rate":
        _document(
            bodies["documentation"],
            [
                "V122667819",
                "rates in percentage",
                "booked in Canada",
                "Canadian dollars only",
                "renewals and refinancing",
                "volume-weighted average",
            ],
        )
    elif spec["indicator"] == "yield_10y":
        _document(
            bodies["documentation"],
            ["Selected benchmark bond yields", "mid-market closing yields", "Government of Canada"],
        )
    rows = []
    monthly = spec["frequency"] == "monthly"
    for i, native in enumerate(raw["observations"]):
        if (
            set(native) != {"d", series}
            or not isinstance(native[series], dict)
            or set(native[series]) != {"v"}
        ):
            raise ValueError("Bank of Canada observation identity or unreviewed native flag")
        start = date.fromisoformat(native["d"])
        if (
            start.isoformat() != native["d"]
            or start < date(int(spec["end"][:4]) - 2, 1, 1)
            or (monthly and start.day != 1)
        ):
            raise ValueError("Bank of Canada native date outside requested range or frequency")
        stop = (
            start.replace(day=calendar.monthrange(start.year, start.month)[1]) if monthly else start
        )
        native_value = native[series]["v"]
        if native_value is not None and not isinstance(native_value, str):
            raise ValueError("Bank of Canada native value encoding changed")
        value = None if native_value in (None, "") else _number(float(native_value))
        rows.append(
            _row(start, stop, value, f"{series}/{native['d']}#observations[{i}]", daily=not monthly)
        )
    return rows, {
        "publisher": "Bank of Canada",
        "native_series": meta["seriesDetails"],
        "native_unit": "percent",
        "status_convention": "Valet supplies no per-point finality flag; null/empty native values remain not_reported.",
        "update_clock": "No original publication or update timestamp is supplied in these Valet responses.",
    }


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Verify complete selected native histories and their definitions."""
    try:
        expected = next(
            s for s in specs(date.fromisoformat(spec["end"])) if s["indicator"] == spec["indicator"]
        )
        if (
            spec != expected
            or set(bodies) != {r["role"] for r in spec["requests"]}
            or any(not isinstance(b, bytes) or not b for b in bodies.values())
        ):
            raise ValueError("Canadian source contract or evidence roles differ")
        rows, metadata = (
            _industry(spec, bodies)
            if spec["kind"] == "statistics_canada_wds"
            else _valet(spec, bodies)
        )
        dates = [r["date"] for r in rows]
        if not rows or dates != sorted(set(dates)) or any(d > spec["end"] for d in dates):
            raise ValueError("Canadian observations duplicate, unordered or beyond capture date")
        metadata["raw_sha256"] = {r: hashlib.sha256(b).hexdigest() for r, b in bodies.items()}
        result = {
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
            "publisher_updated_at": None,
            "publisher_metadata": metadata,
            "observations": rows,
            "missingness": {
                "total": len(rows),
                "not_reported": sum(r["value"] is None for r in rows),
                "periods": [r["period"] for r in rows if r["value"] is None],
            },
        }
        if "measure_kind" in spec:
            result["measure_kind"] = spec["measure_kind"]
        return result
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError) as exc:
        raise ValueError("Malformed or incompatible Canadian original source") from exc
