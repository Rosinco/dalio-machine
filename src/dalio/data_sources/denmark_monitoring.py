"""Pure parsers for four pinned Danish original-publisher Statbank selections.

The capture layer retains exact table metadata, POST selection and JSON-stat
bytes. A monthly bond series is kept monthly; its August 2026 revision is
explicit. No live requests, cache fallback, daily interpolation or storage.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
from datetime import UTC, date, datetime

BASE = "https://api.statbank.dk/v1"
REVISION = (
    "August 2026 publication corrected the ten-year bond series for 2021M1–2026M6; "
    "this capture retains the corrected vintage, not the values previously published."
)


def specs(end: date) -> list[dict]:
    """Return four fixed selections; the daily policy window is three calendar years."""
    if type(end) is not date or end.year < 2024:
        raise ValueError("Denmark monitoring end must be a date from 2024 onwards")
    configs = [
        (
            "industrial_production",
            "IPOP21",
            "Industrial production index",
            "Index",
            {
                "SÆSON": ("SÆSON", "Seasonally adjusted"),
                "BRANCHEDB25UDVALG": ("BC", "Mining, quarrying and manufacturing"),
            },
            "Statistics Denmark",
            "Mining, quarrying and manufacturing production",
            "Statistics Denmark IPOP21 seasonally adjusted production volume index, 2021=100, "
            "DB25 mining, quarrying and manufacturing (BC); energy supply is excluded.",
            [
                "The DB25 series starts in 2021; discontinued DB07 histories are not spliced into it.",
                "BC includes oil and gas extraction but excludes energy supply; some production can occur abroad for Danish industrial companies, so this is not a map of domestic factory output.",
                "This industry aggregate is not GDP and differs from the other Nordic native scopes.",
                "Seasonal adjustment and historical revisions can change recent momentum.",
            ],
            "https://www.dst.dk/en/Statistik/dokumentation/documentationofstatistics/production-and-turnover-in-manufacturing-industries",
        ),
        (
            "corporate_new_lending_rate",
            "DNRNUPI",
            "New domestic loans excl. revolving loans, etc. from banks",
            "-",
            {
                "FORMÅL": ("ALLE", "All purposes"),
                "DATA": ("EFFR", "Annualised agreed rate (per cent)"),
                "INDSEK": ("1100", "1100: Non-financial corporations"),
                "VALUTA": ("DKK", "- of which DKK"),
                "LØBETID1": ("ALLE", "All maturities"),
                "RENTFIX": ("ALLE", "All interest rate fixation periods"),
                "LAANSTRREPO": (
                    "ALLEAL00EXAL40",
                    " - Of which loans excl. repo business to non-financial corporations",
                ),
            },
            "Danmarks Nationalbank",
            "NFC rate on new domestic DKK bank loans",
            "Danmarks Nationalbank annualised agreed rate on new domestic DKK bank loans to "
            "non-financial corporations, all purposes, maturities and fixation periods; excluding "
            "revolving loans and repo business. Published through Statistics Denmark's Statbank.",
            [
                "This is the native new-loan series, not an outstanding-loan rate or credit-availability measure.",
                "Borrower, currency and loan composition affect the average; definitions differ across countries.",
                "Observations through September 2013 use an older methodology and selected dimensions lack values.",
            ],
            "https://www.nationalbanken.dk/en/news-and-knowledge/data-and-statistics",
        ),
        (
            "policy_rate",
            "DNRENTD",
            "Interest rates",
            "-",
            {
                "INSTRUMENT": (
                    "OIBNAA",
                    "The Nationalbanks official rates - Certificates of deposit (Jan 1992-)",
                ),
                "LAND": ("DK", "DK: Denmark"),
                "OPGOER": ("E", "Daily interest rates (per cent)"),
            },
            "Danmarks Nationalbank",
            "Certificates of deposit: official Danish policy rate",
            "Danmarks Nationalbank official certificates-of-deposit interest rate, native daily "
            "percent level. It is one of Denmark's monetary-policy rates, not a company lending rate.",
            [
                "Denmark uses several official monetary-policy rates; this selection is specifically certificates of deposit.",
                "Only actual published daily rows are retained; missing days are not forward-filled.",
            ],
            "https://www.nationalbanken.dk/en/what-we-do/stable-prices-monetary-policy-and-the-danish-economy/official-interest-rates",
        ),
        (
            "yield_10y",
            "MPK3",
            "Interest rates, by type",
            "Per cent per annum",
            {"TYPE": ("5500701004", "10-years central government bond (redemption yield)")},
            "Statistics Denmark",
            "Ten-year central-government redemption yield (monthly)",
            "Statistics Denmark MPK3 monthly ten-year central-government bond redemption yield, "
            "percent per annum. This is the published monthly series, not a daily or constant-maturity interpolation.",
            [
                REVISION,
                "DST republishes financial-market statistics compiled by other producers; its documentation attributes bond-yield source data to the Copenhagen Stock Exchange.",
                "A secondary-market benchmark yield is not an auction funding cost or a company's borrowing rate.",
            ],
            "https://www.dst.dk/en/Statistik/dokumentation/documentationofstatistics/interest-and-share-price-indices/statistical-processing",
        ),
    ]
    result = []
    for (
        indicator,
        table,
        title,
        native_unit,
        values,
        publisher,
        label,
        definition,
        limits,
        doc,
    ) in configs:
        daily = indicator == "policy_rate"
        time_values = [f"{year}*" for year in range(end.year - 2, end.year + 1)] if daily else ["*"]
        query = {
            "table": table,
            "format": "JSONSTAT",
            "lang": "en",
            "variables": [{"code": k, "values": [v[0]]} for k, v in values.items()]
            + [{"code": "Tid", "values": time_values}],
        }
        result.append(
            {
                "country": "DK",
                "indicator": indicator,
                "end": end.isoformat(),
                "kind": "danish_statbank",
                "table_id": table,
                "native_title": title,
                "native_unit": native_unit,
                "selection": {k: v[0] for k, v in values.items()},
                "selection_labels": {k: v[1] for k, v in values.items()},
                "series_id": "/".join([table, *(v[0] for v in values.values())]),
                "requests": [
                    {
                        "role": "metadata",
                        "method": "GET",
                        "url": f"{BASE}/tableinfo/{table}?format=JSON&lang=en",
                    },
                    {"role": "data", "method": "POST", "url": f"{BASE}/data", "json": query},
                ],
                "definition": definition,
                "frequency": "daily" if daily else "monthly",
                "unit": "index" if indicator == "industrial_production" else "percent",
                "adjustment": "seasonally_adjusted"
                if indicator == "industrial_production"
                else "not_adjusted",
                "comparison": "ninety_day_rate"
                if daily
                else "three_month_means"
                if indicator == "industrial_production"
                else "three_month_rate",
                "label": label,
                "limits": limits,
                "publisher": publisher,
                "source": "DST_MONITORING"
                if publisher == "Statistics Denmark"
                else "DANMARKS_NATIONALBANK_MONITORING",
                "documentation_url": doc,
            }
        )
    return result


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate source JSON key")
            result[key] = value
        return result

    result = json.loads(body, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("Expected source JSON object")
    return result


def _ordered(category):
    index = category["index"]
    if (
        not isinstance(index, dict)
        or not index
        or any(type(v) is not int for v in index.values())
        or sorted(index.values()) != list(range(len(index)))
    ):
        raise ValueError("Native category positions are not unique and complete")
    if set(category["label"]) != set(index):
        raise ValueError("Native category labels are incomplete")
    return sorted(index, key=index.get)


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Non-finite or nonnumeric source observation")
    return float(value)


def table_is_industry(spec, dimension):
    return spec["table_id"] == "IPOP21" and dimension == "BRANCHEDB25UDVALG"


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    """Replay complete selected axes and native identities before exposing facts."""
    try:
        expected = next(
            s for s in specs(date.fromisoformat(spec["end"])) if s["indicator"] == spec["indicator"]
        )
        if spec != expected or set(bodies) != {"metadata", "data"}:
            raise ValueError("Danish source contract or evidence roles differ")
        if any(not isinstance(b, bytes) or not b for b in bodies.values()):
            raise ValueError("Original source bytes required")
        meta, envelope = _json(bodies["metadata"]), _json(bodies["data"])
        if set(envelope) != {"dataset"}:
            raise ValueError("Expected one Danish dataset")
        raw = envelope["dataset"]
        dims = raw["dimension"]
        if (
            meta["id"] != spec["table_id"]
            or meta["text"] != spec["native_title"]
            or meta["unit"] != spec["native_unit"]
            or meta["active"] is not True
        ):
            raise ValueError("Danish metadata table, unit or activity changed")
        if raw["source"] != spec["publisher"]:
            raise ValueError("Danish original producer changed")
        if (
            "tableid" in raw.get("extension", {}).get("px", {})
            and raw["extension"]["px"]["tableid"] != spec["table_id"]
        ):
            raise ValueError("Danish native table changed")
        variables = {v["id"]: v for v in meta["variables"]}
        if len(variables) != len(meta["variables"]) or set(variables) != set(spec["selection"]) | {
            "Tid"
        }:
            raise ValueError("Danish metadata dimensions changed")
        expected_ids = [*spec["selection"], "ContentsCode", "Tid"]
        if (
            dims["id"] != expected_ids
            or set(dims) != set(expected_ids) | {"id", "size", "role"}
            or dims["role"] != {"metric": ["ContentsCode"], "time": ["Tid"]}
        ):
            raise ValueError("Danish data dimensions changed")
        for dimension, code in spec["selection"].items():
            cat = dims[dimension]["category"]
            native = {v["id"]: v["text"] for v in variables[dimension]["values"]}
            expected_label = spec["selection_labels"][dimension]
            # Tableinfo prefixes DB25 category codes; the JSON-stat data label does not.
            metadata_label = (
                "BC " + expected_label if table_is_industry(spec, dimension) else expected_label
            )
            if (
                native.get(code) != metadata_label
                or _ordered(cat) != [code]
                or cat["label"][code] != expected_label
            ):
                raise ValueError("Danish selected native category or definition changed")
        measure = dims["ContentsCode"]["category"]
        table = spec["table_id"]
        if (
            _ordered(measure) != [table]
            or measure["label"][table] != spec["native_title"]
            or measure["unit"][table]["base"] != spec["native_unit"]
        ):
            raise ValueError("Danish measure or unit changed")
        periods = _ordered(dims["Tid"]["category"])
        expected_periods = [v["id"] for v in variables["Tid"]["values"]]
        if spec["frequency"] == "daily":
            expected_periods = [
                p for p in expected_periods if int(p[:4]) >= int(spec["end"][:4]) - 2
            ]
        if (
            periods != expected_periods
            or len(set(periods)) != len(periods)
            or dims["size"] != [1] * (len(expected_ids) - 1) + [len(periods)]
        ):
            raise ValueError("Danish time axis is incomplete")
        if len(raw["value"]) != len(periods) or not isinstance(raw["value"], list):
            raise ValueError("Danish native value axis is incomplete")
        updated = datetime.fromisoformat(raw["updated"].replace("Z", "+00:00"))
        if (
            updated.tzinfo is None
            or datetime.fromisoformat(meta["updated"]).date() != updated.date()
        ):
            raise ValueError("Danish dataset update date/clock differs")
        status = raw.get("status", {})
        if not isinstance(status, dict) or any(
            k not in {str(i) for i in range(len(periods))} for k in status
        ):
            raise ValueError("Danish native status axis differs")
        rows = []
        for i, period in enumerate(periods):
            match = re.fullmatch(r"(\d{4})M(\d{2})(?:D(\d{2}))?", period)
            if not match or (match[3] is not None) != (spec["frequency"] == "daily"):
                raise ValueError("Invalid Danish native period")
            year, month = int(match[1]), int(match[2])
            start = date(year, month, int(match[3]) if match[3] else 1)
            stop = start if match[3] else date(year, month, calendar.monthrange(year, month)[1])
            if stop > date.fromisoformat(spec["end"]) or stop > updated.date():
                raise ValueError("Danish observed date exceeds release/capture date")
            value = _number(raw["value"][i])
            flag = status.get(str(i))
            if flag not in (None, "..") or (flag == ".." and value is not None):
                raise ValueError("Unreviewed or contradictory Danish native status")
            rows.append(
                {
                    "date": stop.isoformat(),
                    "period_start": start.isoformat(),
                    "period_end": stop.isoformat(),
                    "period": stop.isoformat() if match[3] else f"{year:04d}-{month:02d}",
                    "value": value,
                    "status": "not_reported" if value is None else "observed",
                    "native_status": flag,
                    "source_locator": f"{spec['series_id']}/Tid={period}#value[{i}]",
                }
            )
        if [r["date"] for r in rows] != sorted({r["date"] for r in rows}):
            raise ValueError("Danish observation dates are duplicated or unsorted")
        metadata = {
            "publisher": spec["publisher"],
            "delivery_host": "Statistics Denmark Statbank",
            "table": table,
            "native_title": meta["text"],
            "native_unit": meta["unit"],
            "native_selection": spec["selection"],
            "native_selection_labels": spec["selection_labels"],
            "footnote": meta.get("footnote"),
            "documentation": meta.get("documentation"),
            "documentation_url": spec["documentation_url"],
            "tableinfo_updated_native": meta["updated"],
            "tableinfo_update_timezone": "not supplied; not converted to an availability timestamp",
            "jsonstat_updated": raw["updated"],
            "clock_check": "Metadata and response update dates must agree; metadata supplies no timezone for an exact timestamp comparison.",
            "status_convention": "Observed means a published statistical value, not a claim of finality. Nulls and native missing flags are retained.",
            "raw_sha256": {r: hashlib.sha256(b).hexdigest() for r, b in bodies.items()},
        }
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
            "metadata_url": spec["requests"][0]["url"],
            "published_at": None,
            "publisher_updated_at": updated.astimezone(UTC).isoformat(),
            "publisher_metadata": metadata,
            "observations": rows,
            "missingness": {
                "total": len(rows),
                "not_reported": sum(r["value"] is None for r in rows),
                "periods": [r["period"] for r in rows if r["value"] is None],
            },
        }
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError) as exc:
        raise ValueError("Malformed or incompatible Danish original source") from exc
