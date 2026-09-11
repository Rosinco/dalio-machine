"""Pinned original Norwegian inputs for the bounded Nordic monitoring pilot.

No network or storage writes occur here. The shared acquisition layer retains
original requests/bytes/receipts and enforces the SSB maintenance exclusion.
Native update time, observation time and acquisition time remain distinct.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from urllib.parse import urlencode

SSB_BASE = "https://data.ssb.no/api/v0/en/table"
NB_BASE = "https://data.norges-bank.no/api"
SSB_PRODUCTION_DOC = (
    "https://www.ssb.no/en/energi-og-industri/industri-og-bergverksdrift/statistikk/"
    "produksjonsindeks-for-olje-og-gass-industri-bergverk-og-kraftforsyning"
)
SSB_RATE_DOC = (
    "https://www.ssb.no/en/bank-og-finansmarked/finansinstitusjoner-og-andre-finansielle-foretak/"
    "statistikk/renter-i-banker-og-kredittforetak"
)
NB_YIELD_DOC = (
    "https://www.norges-bank.no/en/topics/statistics/norwegian-government-securities/"
    "generiske-statsrenter/"
)
NB_POLICY_DOC = "https://www.norges-bank.no/en/topics/Monetary-policy/Policy-rate/"
XML_NS = {
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}


def specs(end: date) -> list[dict]:
    """Return four complete source contracts with explicit original scopes."""
    if type(end) is not date or end.year < 2024:
        raise ValueError("Norway monitoring end must be a date from 2024 onwards")
    result = []
    common = dict(country="NO", end=end.isoformat(), published_at=None)
    for indicator, table, selection, unit, adjustment, label, definition, limits, doc in (
        (
            "industrial_production",
            "07095",
            (("PKoder", "P103"), ("ContentsCode", "Sesongjustert")),
            "index",
            "seasonally_adjusted",
            "Manufacturing, mining and quarrying production",
            "SSB P103 production volume chain index, 2021=100, calendar and seasonally adjusted; "
            "manufacturing, mining and quarrying, excluding oil/gas extraction, extraction support "
            "and electricity. Petroleum-related manufacturing remains included.",
            [
                "The scope excludes petroleum extraction but includes petroleum-related manufacturing; it is not oil-independent.",
                "This industry index is not whole-economy or mainland GDP, and is not identical to other countries' industry aggregates.",
                "Seasonal adjustment and revised historical observations can change recent momentum; no per-point finality is supplied.",
            ],
            SSB_PRODUCTION_DOC,
        ),
        (
            "corporate_new_lending_rate",
            "10729",
            (("Utlanstype", "02"), ("Sektor", "03"), ("ContentsCode", "RenterNyeUtlan")),
            "per cent",
            "not_adjusted",
            "NFC rate on new repayment loans",
            "SSB monthly weighted rate on new NOK repayment loans to Norwegian non-financial "
            "corporations, total repayment-loan types, from a sample of banks and mortgage "
            "companies. Credit lines are excluded; this is not a mortgage-only or outstanding-loan rate.",
            [
                "New agreements set the rate for the first time; purchased/transferred loans count as new only if terms change.",
                "Recurring commissions and current administration fees are included; non-recurring commissions, setup and instalment fees are excluded.",
                "Coverage, fee treatment and new-business definitions differ from Sweden; rate levels must not be ranked as like-for-like borrowing costs.",
                "Borrower and loan composition can change the average; the series does not measure credit availability or an individual company's terms.",
            ],
            SSB_RATE_DOC,
        ),
    ):
        query = {
            "query": [
                {"code": name, "selection": {"filter": "item", "values": [code]}}
                for name, code in selection
            ]
            + [{"code": "Tid", "selection": {"filter": "all", "values": ["*"]}}],
            "response": {"format": "json-stat2"},
        }
        url = f"{SSB_BASE}/{table}"
        result.append(
            dict(
                common,
                indicator=indicator,
                kind="ssb",
                table_id=table,
                series_id="/".join([table, *(code for _, code in selection)]),
                selection=dict(selection),
                requests=[
                    {"role": "metadata", "method": "GET", "url": url},
                    {"role": "data", "method": "POST", "url": url, "json": query},
                ],
                definition=definition,
                frequency="monthly",
                unit=unit,
                adjustment=adjustment,
                comparison="three_month_means"
                if indicator == "industrial_production"
                else "three_month_rate",
                label=label,
                limits=limits,
                publisher="Statistics Norway",
                source="SSB_MONITORING",
                documentation_url=doc,
                capture_exclusion={
                    "timezone": "Europe/Oslo",
                    "start_hour": 5,
                    "end_hour": 8,
                    "reason": "SSB warns figures under revision can temporarily appear as 0 or . between 05:00 and 08:00 local time.",
                    "source_url": "https://www.ssb.no/en/statbank/",
                },
            )
        )
    for indicator, flow, series, label, definition, limits, doc in (
        (
            "policy_rate",
            "IR",
            {"FREQ": "B", "INSTRUMENT_TYPE": "KPRA", "TENOR": "SD", "UNIT_MEASURE": "R"},
            "Norges Bank effective policy rate",
            "Norges Bank key policy rate, business-day end-of-day level in percent; the effective rate, not the announcement or forecast path.",
            [
                "Policy rates are not company lending rates or credit-availability measures.",
                "Contractual repricing and borrower risk affect transmission to companies.",
            ],
            NB_POLICY_DOC,
        ),
        (
            "yield_10y",
            "GOVT_GENERIC_RATES",
            {"FREQ": "B", "TENOR": "10Y", "INSTRUMENT_TYPE": "GBON"},
            "Norwegian generic ten-year government yield",
            "Norges Bank generic ten-year government mid-yield in percent: the security nearest the stated maturity, "
            "using best interdealer bid and ask yields at 16:00 market close. Published after 09:00 on business days.",
            [
                "This nearest-maturity benchmark can roll between securities; it is not an interpolated constant-maturity yield or a same-bond return.",
                "Secondary-market yield is not an auction funding cost, company borrowing cost or proof of refinancing stress.",
                "Market-close observation date and subsequent publication/receipt time are separate; no original first-publication timestamp is supplied in the CSV.",
            ],
            NB_YIELD_DOC,
        ),
    ):
        start = date(end.year - 2, 1, 1)
        key = ".".join(series.values())
        url = f"{NB_BASE}/data/{flow}/{key}?" + urlencode(
            {
                "format": "csv",
                "startPeriod": start.isoformat(),
                "endPeriod": end.isoformat(),
                "locale": "en",
            }
        )
        result.append(
            dict(
                common,
                indicator=indicator,
                kind="norges_bank",
                flow=flow,
                native_series=series,
                series_id=f"{flow}/{key}",
                history_start=start.isoformat(),
                requests=[
                    {
                        "role": "metadata",
                        "method": "GET",
                        "url": f"{NB_BASE}/dataflow/NB/{flow}/latest?references=all",
                    },
                    {"role": "data", "method": "GET", "url": url},
                ],
                definition=definition,
                frequency="daily",
                unit="percent",
                adjustment="not_adjusted",
                comparison="ninety_day_rate",
                label=label,
                limits=limits,
                publisher="Norges Bank",
                source="NORGES_BANK_MONITORING",
                documentation_url=doc,
            )
        )
    return result


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Independently reconstruct a pinned selection from the original bytes."""
    try:
        expected = next(
            item
            for item in specs(date.fromisoformat(spec["end"]))
            if item["indicator"] == spec["indicator"]
        )
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise ValueError("Invalid Norway source spec") from exc
    if spec != expected:
        raise ValueError("Norway source spec differs from pinned contract")
    if set(bodies) != {"metadata", "data"} or any(
        not isinstance(b, bytes) or not b for b in bodies.values()
    ):
        raise ValueError("Norway source bodies must contain exact metadata and data bytes")
    rows, metadata, updated = _ssb(spec, bodies) if spec["kind"] == "ssb" else _nb(spec, bodies)
    metadata.update(
        metadata_sha256=hashlib.sha256(bodies["metadata"]).hexdigest(),
        response_sha256=hashlib.sha256(bodies["data"]).hexdigest(),
        documentation_url=spec["documentation_url"],
        status_convention="observed is a published non-forecast statistical value, not a claim of finality; native flags are retained.",
    )
    return {
        **{
            k: spec[k]
            for k in (
                "country",
                "indicator",
                "definition",
                "frequency",
                "unit",
                "adjustment",
                "comparison",
                "label",
                "limits",
                "source",
                "series_id",
            )
        },
        "source_url": spec["requests"][-1]["url"],
        "metadata_url": spec["requests"][0]["url"],
        "published_at": None,
        "publisher_updated_at": updated,
        "publisher_metadata": metadata,
        "observations": rows,
        "missingness": {
            "total": len(rows),
            "not_reported": sum(row["value"] is None for row in rows),
            "periods": [row["period"] for row in rows if row["value"] is None],
        },
    }


def _json(body: bytes) -> dict:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate source JSON key")
            result[name] = value
        return result

    try:
        raw = json.loads(body, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Malformed Norwegian source JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Expected source JSON object")
    return raw


def _ssb(spec, bodies):
    meta, raw = _json(bodies["metadata"]), _json(bodies["data"])
    try:
        definitions = {item["code"]: item for item in meta["variables"]}
        expected_ids = set(spec["selection"]) | {"Tid"}
        if len(definitions) != len(meta["variables"]) or set(definitions) != expected_ids:
            raise ValueError("SSB metadata dimensions changed")
        if not meta["title"].startswith(spec["table_id"] + ":"):
            raise ValueError("Wrong SSB metadata table")
        if (
            raw["class"] != "dataset"
            or raw["version"] != "2.0"
            or raw["extension"]["px"]["tableid"] != spec["table_id"]
            or raw["source"] != "Statistics Norway"
        ):
            raise ValueError("Wrong SSB native table/source")
        ids, sizes = raw["id"], raw["size"]
        if (
            set(ids) != expected_ids
            or len(ids) != len(expected_ids)
            or len(ids) != len(sizes)
            or set(raw["dimension"]) != expected_ids
        ):
            raise ValueError("SSB response dimensions changed")
        categories = {}
        dimensions = {}
        for name, size in zip(ids, sizes, strict=True):
            index = raw["dimension"][name]["category"]["index"]
            if (
                type(size) is not int
                or size < 1
                or len(index) != size
                or any(type(position) is not int for position in index.values())
                or set(index.values()) != set(range(size))
            ):
                raise ValueError("SSB category positions/count invalid")
            categories[name] = sorted(index, key=index.get)
            declared = definitions[name]
            if len(declared["values"]) != len(set(declared["values"])):
                raise ValueError("Duplicate SSB metadata categories")
            labels = dict(zip(declared["values"], declared["valueTexts"], strict=True))
            if name == "Tid":
                if categories[name] != declared["values"]:
                    raise ValueError("Incomplete SSB requested time axis")
            else:
                code = spec["selection"][name]
                if categories[name] != [code] or code not in labels:
                    raise ValueError("SSB selected native category changed")
                if raw["dimension"][name]["category"]["label"][code] != labels[code]:
                    raise ValueError("SSB native label disagrees with metadata")
                dimensions[name] = dict(code=code, label=labels[code])
        metric = spec["selection"]["ContentsCode"]
        contents = raw["dimension"]["ContentsCode"]
        unit = contents["category"]["unit"][metric]
        extension = contents["extension"]
        adjustment = "WorkAndSes" if spec["indicator"] == "industrial_production" else "None"
        base = "2021" if spec["indicator"] == "industrial_production" else None
        if (
            unit["base"] != spec["unit"]
            or extension["adjustment"][metric] != adjustment
            or extension.get("basePeriod", {}).get(metric) != base
            or extension["priceType"][metric] != "NotApplicable"
        ):
            raise ValueError("SSB native unit/adjustment/base/price type changed")
        updated = datetime.fromisoformat(raw["updated"].replace("Z", "+00:00"))
        if updated.tzinfo is None or updated.utcoffset() is None:
            raise ValueError("SSB update clock lacks timezone")
        updated = updated.astimezone(UTC)
        periods = categories["Tid"]
        values = _cells(raw["value"], len(periods))
        statuses = _cells(raw.get("status", {}), len(periods))
        rows = []
        for position, period in enumerate(periods):
            match = re.fullmatch(r"(\d{4})M(\d{2})", period)
            if not match:
                raise ValueError("SSB monthly period malformed")
            year, month = int(match[1]), int(match[2])
            start, end = (
                date(year, month, 1),
                date(year, month, calendar.monthrange(year, month)[1]),
            )
            if end > updated.date() or end > date.fromisoformat(spec["end"]):
                raise ValueError("SSB observation is after native update or requested end")
            status = statuses[position]
            if status not in (None, "", ".", "..", ":"):
                raise ValueError("Unknown SSB native observation status")
            value = _number(values[position])
            if value is not None and status not in (None, ""):
                raise ValueError("SSB missing status conflicts with numeric value")
            rows.append(
                _row(
                    start,
                    end,
                    value,
                    status,
                    f"/value/{position}; {spec['series_id']}/{period}",
                    monthly=True,
                )
            )
        metadata = dict(
            publisher="Statistics Norway",
            source=raw["source"],
            title=raw["label"],
            table_id=spec["table_id"],
            dimensions=dimensions,
            native_adjustment=adjustment,
            base_period=base,
            native_unit=unit,
            native_measure_metadata=contents,
            notes=raw.get("note", []),
            native_extension=raw.get("extension", {}),
            metadata=meta,
            dataset_updated_native=raw["updated"],
        )
        if spec["indicator"] == "corporate_new_lending_rate":
            metadata["currency"] = "NOK"
        return rows, metadata, updated.isoformat()
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Incomplete or invalid SSB native response") from exc


def _nb(spec, bodies):
    if b"<!DOCTYPE" in bodies["metadata"].upper() or b"<!ENTITY" in bodies["metadata"].upper():
        raise ValueError("Unsupported XML source declaration")
    try:
        structure = ET.fromstring(bodies["metadata"])
    except ET.ParseError as exc:
        raise ValueError("Invalid Norges Bank structure XML") from exc
    flow = structure.find(f".//s:Dataflow[@id='{spec['flow']}']", XML_NS)
    if flow is None or flow.get("agencyID") != "NB":
        raise ValueError("Wrong Norges Bank metadata flow")
    dsd_id = "DSD_IR" if spec["flow"] == "IR" else "DSD_GOVT_GENERIC_RATES"
    dsd = structure.find(f".//s:DataStructure[@id='{dsd_id}']", XML_NS)
    if dsd is None:
        raise ValueError("Norges Bank data structure missing")
    dimensions = dsd.findall(".//s:DimensionList/s:Dimension", XML_NS)
    dimensions.sort(key=lambda element: int(element.get("position", "0")))
    if [item.get("id") for item in dimensions] != list(spec["native_series"]):
        raise ValueError("Norges Bank native dimension order changed")
    native_labels = {}
    for item in dimensions:
        name = item.get("id")
        ref = item.find(".//s:Enumeration/Ref", XML_NS)
        if ref is None:
            raise ValueError("Norges Bank dimension has no codelist")
        code = spec["native_series"][name]
        entry = structure.find(f".//s:Codelist[@id='{ref.get('id')}']/s:Code[@id='{code}']", XML_NS)
        if entry is None:
            raise ValueError("Norges Bank native code absent from metadata")
        english = [
            label.text
            for label in entry.findall("c:Name", XML_NS)
            if label.get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
        ]
        if len(english) != 1:
            raise ValueError("Norges Bank native English label absent")
        native_labels[name] = english[0]
    reader = csv.DictReader(io.StringIO(bodies["data"].decode("utf-8-sig")), delimiter=";")
    fields = reader.fieldnames
    required = set(spec["native_series"]) | {"TIME_PERIOD", "OBS_VALUE", "DECIMALS"}
    if not fields or len(fields) != len(set(fields)) or not required <= set(fields):
        raise ValueError("Norges Bank CSV lacks required native dimensions")
    start, end = date.fromisoformat(spec["history_start"]), date.fromisoformat(spec["end"])
    rows, seen = [], set()
    for position, raw in enumerate(reader, 2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("Norges Bank CSV row width mismatch")
        if any(raw[name] != value for name, value in spec["native_series"].items()):
            raise ValueError("Norges Bank CSV has a different native series")
        when = date.fromisoformat(raw["TIME_PERIOD"])
        if not start <= when <= end or when in seen:
            raise ValueError("Norges Bank date outside request or duplicated")
        seen.add(when)
        if raw["DECIMALS"] != ("2" if spec["flow"] == "IR" else "3"):
            raise ValueError("Norges Bank native decimals changed")
        if spec["flow"] == "IR" and (
            raw.get("COLLECTION") != "E" or raw.get("CALC_METHOD") not in ("", "N", "KP")
        ):
            raise ValueError("Norges Bank collection/calculation method changed")
        status = raw.get("OBS_STATUS") or None
        if status not in (None, "A"):
            raise ValueError("Unsupported Norges Bank observation status")
        text = raw["OBS_VALUE"].strip()
        if text in ("", ".", "..", "NaN"):
            if text == "NaN":
                raise ValueError("Norges Bank non-finite value")
            value = None
        else:
            try:
                value = _number(float(text))
            except (ValueError, OverflowError) as exc:
                raise ValueError("Invalid Norges Bank value") from exc
        row = _row(
            when,
            when,
            value,
            status,
            f"CSV line{position}; {spec['series_id']}/{when}",
            monthly=False,
        )
        row["native_attributes"] = {
            name: value for name, value in raw.items() if name not in {"OBS_VALUE", "TIME_PERIOD"}
        }
        rows.append(row)
    if not rows:
        raise ValueError("Norges Bank response has no observations")
    return (
        sorted(rows, key=lambda row: row["date"]),
        dict(
            publisher="Norges Bank",
            flow=spec["flow"],
            native_series=spec["native_series"],
            native_labels=native_labels,
            decimals=2 if spec["flow"] == "IR" else 3,
            source_clock_note="SDMX structure header preparation time is not a data publication or revision clock; CSV supplies no such clock.",
        ),
        None,
    )


def _cells(value, count):
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError("SSB value/status count mismatch")
        return value
    if isinstance(value, dict):
        if any(not re.fullmatch(r"0|[1-9]\d*", k) or int(k) >= count for k in value):
            raise ValueError("SSB sparse cell position invalid")
        return [value.get(str(i)) for i in range(count)]
    raise ValueError("SSB unsupported cell structure")


def _number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Native value must be finite numeric or null")
    return float(value)


def _row(start, end, value, status, locator, *, monthly):
    return dict(
        date=end.isoformat(),
        period=start.strftime("%Y-%m") if monthly else end.isoformat(),
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        value=value,
        status="not_reported" if value is None else "observed",
        native_status=status,
        source_locator=locator,
    )
