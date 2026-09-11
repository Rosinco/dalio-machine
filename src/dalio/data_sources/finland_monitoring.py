"""Original Finnish industry/drawdown statistics and applicable ECB policy.

Pure replay adapters, with the Bank of Finland benchmark report delegated to
its dedicated HTML parser. Native currency, borrower and industry scopes are
retained rather than represented as identical Nordic measures.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import re
from datetime import UTC, date, datetime
from urllib.parse import urlencode

from dalio.data_sources import finland_yield_monitoring

PX = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ttvi/14mh.px"
BOF = "https://api.boffsaopendata.fi/v4"
MFI_KEY = "M.A.1.B.A2A.A.A.U6.2241.ZZ.Z01.A.ANR.0.A.0.A.0"
ECB_KEY = "D.U2.EUR.4F.KR.DFR.LEV"
INDUSTRY_TITLE = "Volume index of industrial output (2021=100) by Month, Standard Industrial Classification (TOL 2008) and Information"
MFI_DIMENSIONS = [
    ("FREQ", "M", "Monthly"),
    ("REPORTER_GROUP_MFI", "A", "MFIs excl. Bank of Finland"),
    ("MEASURE_MFI", "1", "Annualised agreed rate"),
    ("TRANSACTION_TYPE_MFI", "B", "New drawdown"),
    ("BS_ITEM_MFI", "A2A", "Loans excl. overdrafts and credit card credit"),
    ("MATURITY_ORIG_MFI", "A", "Maturities total"),
    ("MATURITY_REM_MFI", "A", "Maturities total"),
    ("COUNT_AREA", "U6", "Domestic (home or reference area)"),
    ("COUNT_SECTOR_MFI", "2241", "Non-financial corporations"),
    ("COUNT_INDUSTRY_MFI", "ZZ", "Industries total"),
    ("CURRENCY_TRANS", "Z01", "All currencies combined"),
    ("LOAN_PURPOSE_MFI", "A", "Total"),
    ("COLLATERAL_TYPE_MFI", "ANR", "Total excluding non-recourse factoring"),
    ("LOAN_SIZE_MFI", "0", "Total"),
    ("INT_RATE_LINK_MFI", "A", "Total"),
    ("INIT_FIX_PERIOD_MFI", "0", "Total"),
    ("INT_RESET_TYPE_MFI", "A", "Total"),
    ("NOTICE_TYPE_MFI", "0", "Total"),
]
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


def specs(end: date) -> list[dict]:
    """Four explicit native source selections; end is the acquisition UTC date."""
    if type(end) is not date or end.year < 2024:
        raise ValueError("Finland monitoring end must be a date from 2024 onwards")
    query = {
        "query": [
            {"code": "timeperiod_m", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "toimiala_78_20180201", "selection": {"filter": "item", "values": ["BTD"]}},
            {
                "code": "contentscode",
                "selection": {"filter": "item", "values": ["ttvi-Kausitasoitettu"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }
    common = {"country": "FI", "end": end.isoformat()}
    industry = {
        **common,
        "indicator": "industrial_production",
        "kind": "statistics_finland_jsonstat",
        "series_id": "14mh/BTD/ttvi-Kausitasoitettu",
        "requests": [
            {"role": "metadata", "method": "GET", "url": PX},
            {"role": "data", "method": "POST", "url": PX, "json": query},
        ],
        "definition": "Statistics Finland seasonally adjusted volume index of industrial output, 2021=100, TOL 2008 BCD total industries, including mining, manufacturing and energy supply.",
        "frequency": "monthly",
        "unit": "index",
        "adjustment": "seasonally_adjusted",
        "comparison": "three_month_means",
        "label": "Total BCD industry production (including energy)",
        "limits": [
            "The native aggregate includes energy supply; it is broader than the selected BC aggregates and is not GDP.",
            "The official scope can include production abroad and merchanting, with incomplete global-production coverage; it is not a map of domestic factory output.",
            "Seasonal adjustment and revised histories can change recent momentum.",
        ],
        "publisher": "Statistics Finland",
        "source": "STATISTICS_FINLAND_MONITORING",
        "documentation_url": "https://stat.fi/en/documentation/documentation-of-statistics/ttvi",
    }
    loan = {
        **common,
        "indicator": "corporate_new_lending_rate",
        "kind": "bank_of_finland_mfi",
        "series_id": "MFI_PUBL/" + MFI_KEY,
        "requests": [
            {"role": "structure", "method": "GET", "url": f"{BOF}/structures/MFI_PUBL"},
            {
                "role": "metadata",
                "method": "GET",
                "url": f"{BOF}/series/MFI_PUBL?"
                + urlencode({"seriesName": MFI_KEY, "pageNumber": 1, "pageSize": 10}),
            },
            {
                "role": "data",
                "method": "GET",
                "url": f"{BOF}/observations/MFI_PUBL?"
                + urlencode({"seriesName": MFI_KEY, "pageNumber": 1, "pageSize": 10000}),
            },
        ],
        "definition": "Bank of Finland annualised agreed rate, weighted across the reference month's actual new drawdowns by domestic Finnish non-financial corporations, excluding housing corporations, from Finnish MFIs excluding the Bank of Finland. All currencies combined; loans exclude overdrafts, credit card credit and non-recourse factoring; all maturities, sizes and fixation periods.",
        "frequency": "monthly",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "three_month_rate",
        "label": "Domestic NFC new-drawdown lending rate (all currencies)",
        "limits": [
            "Native new drawdown (B) differs from new business (C) and completely new loans (C01); this is not a stock-loan rate.",
            "Counterparty U6 is domestic Finland and sector 2241 excludes the separately coded housing corporations (2242).",
            "Currency and loan composition affect the average; native coverage differs from other Nordic rates.",
            "Native UNIT PC has an English 'Percentage change' label, but MEASURE 1 and Finnish/Swedish unit labels establish a percent interest-rate level; no growth transformation is applied.",
        ],
        "publisher": "Bank of Finland",
        "source": "BANK_OF_FINLAND_MONITORING",
        "documentation_url": "https://www.suomenpankki.fi/globalassets/bof/en/statistics/to-the-reporter/mfi/instructions/mfi_reporting_instructions_v4.8.pdf",
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
                    {"startPeriod": "2005-01-01", "endPeriod": end.isoformat(), "format": "csvdata"}
                ),
            },
        ],
        "definition": "ECB deposit-facility percent-per-annum rate for the euro area (changing composition), applicable to Finland as a euro-area member; not a separate Finnish national policy rate. Native daily observations are retained without interpolation.",
        "frequency": "daily",
        "unit": "percent",
        "adjustment": "not_adjusted",
        "comparison": "ninety_day_rate",
        "label": "ECB deposit facility (shared euro-area policy)",
        "limits": [
            "Shared euro-area monetary policy does not imply identical financing conditions across members.",
            "The deposit-facility policy rate is not a company lending rate or an individual borrower's refinancing cost.",
        ],
        "publisher": "European Central Bank",
        "source": "ECB_MONITORING",
        "documentation_url": "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html",
    }
    return [industry, loan, policy, *finland_yield_monitoring.specs(end)]


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate native JSON key")
            result[key] = value
        return result

    result = json.loads(body, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("Expected native JSON object")
    return result


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid native numeric value")
    return float(value)


def _month(period):
    match = re.fullmatch(r"(\d{4})M(\d{1,2})", period)
    if not match:
        raise ValueError("Invalid native month")
    year, month = map(int, match.groups())
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _row(start, stop, value, flag, locator, daily=False):
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


def _industry(bodies):
    meta, raw = _json(bodies["metadata"]), _json(bodies["data"])
    if (
        meta["title"] != INDUSTRY_TITLE
        or raw["label"] != INDUSTRY_TITLE
        or raw["source"] != "Statistics Finland, volume index of industrial output"
    ):
        raise ValueError("Finnish production definition, base or publisher changed")
    ids = ["timeperiod_m", "toimiala_78_20180201", "contentscode"]
    definitions = {v["code"]: v for v in meta["variables"]}
    if (
        len(definitions) != len(meta["variables"])
        or set(definitions) != set(ids)
        or raw["id"] != ids
        or set(raw["dimension"]) != set(ids)
    ):
        raise ValueError("Finnish production dimensions changed")
    if (
        raw["class"] != "dataset"
        or raw["version"] != "2.0"
        or raw["extension"]["px"]["tableid"] != "14mh"
        or raw["role"] != {"time": ["timeperiod_m"], "metric": ["contentscode"]}
    ):
        raise ValueError("Finnish native table or roles changed")
    for dimension, code, label in [
        ("toimiala_78_20180201", "BTD", "BCD Total industries"),
        ("contentscode", "ttvi-Kausitasoitettu", "Seasonally adjusted index series"),
    ]:
        cat = raw["dimension"][dimension]["category"]
        source = definitions[dimension]
        if len(source["values"]) != len(set(source["values"])) or len(source["values"]) != len(
            source["valueTexts"]
        ):
            raise ValueError("Finnish metadata category axis changed")
        labels = dict(zip(source["values"], source["valueTexts"], strict=True))
        if labels.get(code) != label or cat["index"] != {code: 0} or cat["label"] != {code: label}:
            raise ValueError("Finnish industry or seasonal category changed")
    unit = raw["dimension"]["contentscode"]["category"]["unit"]
    if unit["ttvi-Kausitasoitettu"]["base"] != "index point":
        raise ValueError("Finnish production unit changed")
    periods = definitions["timeperiod_m"]["values"]
    cat = raw["dimension"]["timeperiod_m"]["category"]
    if (
        not periods
        or any(type(i) is not int for i in cat["index"].values())
        or cat["index"] != {p: i for i, p in enumerate(periods)}
        or len(periods) != len(set(periods))
        or set(cat["label"]) != set(periods)
        or raw["size"] != [len(periods), 1, 1]
    ):
        raise ValueError("Finnish production time axis incomplete")
    if not isinstance(raw["value"], list) or len(raw["value"]) != len(periods):
        raise ValueError("Finnish production values incomplete")
    statuses = raw.get("status", {})
    if not isinstance(statuses, dict) or any(
        k not in {str(i) for i in range(len(periods))} for k in statuses
    ):
        raise ValueError("Finnish production native status axis changed")
    updated = datetime.fromisoformat(raw["updated"].replace("Z", "+00:00"))
    if updated.tzinfo is None:
        raise ValueError("Finnish dataset update must supply a timezone")
    rows = []
    for i, period in enumerate(periods):
        start, stop = _month(period)
        value = _number(raw["value"][i])
        flag = statuses.get(str(i))
        if (
            flag not in (None, "..")
            or (flag == ".." and value is not None)
            or stop > updated.date()
        ):
            raise ValueError("Finnish production status or publication date invalid")
        rows.append(
            _row(
                start,
                stop,
                value,
                flag,
                f"14mh/BTD/ttvi-Kausitasoitettu/timeperiod_m={period}#value[{i}]",
            )
        )
    return (
        rows,
        {
            "publisher": "Statistics Finland",
            "native_title": raw["label"],
            "native_source": raw["source"],
            "native_selection": {
                "toimiala_78_20180201": "BTD",
                "contentscode": "ttvi-Kausitasoitettu",
            },
            "native_notes": raw.get("note", []),
            "native_unit": unit,
            "metadata_update_clock": "Table metadata supplies no release clock; UTC dataset update comes from the data response.",
        },
        updated.astimezone(UTC).isoformat(),
    )


def _single(raw, page_size):
    if (
        any(
            raw.get(k) != v
            for k, v in {
                "currentPage": 1,
                "totalPages": 1,
                "pageSize": page_size,
                "totalCount": 1,
            }.items()
        )
        or len(raw["items"]) != 1
    ):
        raise ValueError("Finnish API selected-series pagination incomplete")
    item = raw["items"][0]
    if item["dataset"] != "MFI_PUBL" or item["name"] != MFI_KEY:
        raise ValueError("Finnish API native series identity changed")
    return item


def _lending(bodies):
    meta = _single(_json(bodies["metadata"]), 10)
    raw = _single(_json(bodies["data"]), 10000)
    structure = _json(bodies["structure"])
    if structure["dataset"] != "MFI_PUBL" or len(meta["dimensions"]) != len(MFI_DIMENSIONS):
        raise ValueError("Finnish MFI structure changed")
    structures = {d["name"]: d for d in structure["dimensions"]}
    if len(structures) != len(structure["dimensions"]) or set(structures) != {
        d[0] for d in MFI_DIMENSIONS
    }:
        raise ValueError("Finnish MFI native dimension set changed")
    for position, ((name, value, label), observed) in enumerate(
        zip(MFI_DIMENSIONS, meta["dimensions"], strict=True), 1
    ):
        descriptions = {x["lang"]: x["description"] for x in observed["descriptions"]}
        codes = {x["value"]: x["description"] for x in structures[name]["codelistValues"]}
        if (
            any(
                observed.get(k) != v
                for k, v in {"name": name, "value": value, "position": position}.items()
            )
            or descriptions.get("en") != label
            or codes.get(value) != label
            or structures[name]["position"] != position
        ):
            raise ValueError("Finnish MFI native scope or definition changed")
    sector_codes = {
        x["value"]: x["description"] for x in structures["COUNT_SECTOR_MFI"]["codelistValues"]
    }
    if (
        sector_codes.get("2240") != "Non-financial corporations and housing corporations"
        or sector_codes.get("2242") != "Housing corporations"
    ):
        raise ValueError("Finnish housing sector distinction changed")
    attrs = {x["name"]: x["value"] for x in meta["metadatas"]}
    if len(attrs) != len(meta["metadatas"]) or any(
        attrs.get(k) != v for k, v in {"UNIT": "PC", "UNIT_MULT": "0", "COLLECTION": "E"}.items()
    ):
        raise ValueError("Finnish lending units, scale or collection metadata changed")
    rows = []
    for i, observation in enumerate(raw["observations"]):
        if set(observation) != {"period", "periodCode", "value"}:
            raise ValueError("Unreviewed Finnish observation fields/status")
        start, stop = _month(observation["periodCode"])
        if observation["period"] != stop.isoformat():
            raise ValueError("Finnish MFI month-end/date mismatch")
        rows.append(
            _row(
                start,
                stop,
                _number(observation["value"]),
                None,
                f"MFI_PUBL/{MFI_KEY}/{observation['periodCode']}#observations[{i}]",
            )
        )
    return (
        rows,
        {
            "publisher": "Bank of Finland",
            "native_title": meta.get("title"),
            "native_dimensions": meta["dimensions"],
            "native_attributes": meta["metadatas"],
            "borrower_area": "Finland domestic (U6)",
            "borrower_sector": "NFC excluding housing corporations (2241)",
            "currency_scope": "All currencies combined (Z01)",
            "transaction_type": "New drawdown (B)",
            "pagination": "One selected series on one page; totalCount counts series, not observations.",
            "update_clock": "Original API supplies no publication/update timestamp.",
        },
        None,
    )


def _csv(body):
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Duplicate or empty source CSV header")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed source CSV row width")
    return rows


def _policy(bodies):
    meta = _csv(bodies["metadata"])
    if meta != [ECB_IDENTITY]:
        raise ValueError("ECB policy metadata key changed")
    raw = _csv(bodies["data"])
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
    for i, row in enumerate(raw):
        if (
            any(row.get(k) != v for k, v in expected.items())
            or row["OBS_STATUS"] not in {"A", "M"}
            or row["OBS_CONF"] != "F"
        ):
            raise ValueError("ECB policy native identity, unit or status changed")
        when = date.fromisoformat(row["TIME_PERIOD"])
        if when.isoformat() != row["TIME_PERIOD"] or when < date(2005, 1, 1):
            raise ValueError("ECB policy native date invalid")
        value = None if row["OBS_VALUE"] == "" else _number(float(row["OBS_VALUE"]))
        if (row["OBS_STATUS"] == "M") != (value is None):
            raise ValueError("ECB native status/value contradiction")
        rows.append(
            _row(
                when,
                when,
                value,
                row["OBS_STATUS"],
                f"FM.{ECB_KEY}/{when.isoformat()}#csv_row[{i + 2}]",
                daily=True,
            )
        )
    return (
        rows,
        {
            "publisher": "European Central Bank",
            "policy_area": "Euro area (changing composition)",
            "applicable_country": "Finland, euro-area member; not a national Finnish series",
            "native_identity": ECB_IDENTITY,
            "native_attributes": expected,
            "status_convention": "A is native normal observation, not an assertion of finality; M denotes a missing value.",
            "update_clock": "Source CSV has no publisher update timestamp.",
        },
        None,
    )


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Validate whole native responses before returning selected evidence rows."""
    try:
        expected = next(
            s for s in specs(date.fromisoformat(spec["end"])) if s["indicator"] == spec["indicator"]
        )
        if spec != expected or set(bodies) != {r["role"] for r in spec["requests"]}:
            raise ValueError("Finnish source contract or evidence roles differ")
        if any(not isinstance(b, bytes) or not b for b in bodies.values()):
            raise ValueError("Original Finnish response bytes required")
        if spec["indicator"] == "yield_10y":
            return finland_yield_monitoring.parse_input(spec, bodies)
        parser = {
            "industrial_production": _industry,
            "corporate_new_lending_rate": _lending,
            "policy_rate": _policy,
        }[spec["indicator"]]
        rows, metadata, updated = parser(bodies)
        dates = [r["date"] for r in rows]
        if not rows or dates != sorted(set(dates)) or any(d > spec["end"] for d in dates):
            raise ValueError(
                "Finnish native observation dates duplicate, unsorted or beyond capture"
            )
        metadata.update(
            documentation_url=spec["documentation_url"],
            raw_sha256={r: hashlib.sha256(b).hexdigest() for r, b in bodies.items()},
            observed_convention="Observed denotes a published non-forecast value, not a claim of finality. Native nulls/flags remain explicit.",
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
            "source_url": spec["requests"][-1]["url"],
            "metadata_url": next(r["url"] for r in spec["requests"] if r["role"] == "metadata"),
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
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError) as exc:
        raise ValueError("Malformed or incompatible Finnish original source") from exc
