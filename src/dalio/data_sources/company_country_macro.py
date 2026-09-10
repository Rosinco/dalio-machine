"""Artifact-bound official macro acquisition for the saved company country universe.

This is an acquisition registry, never a replacement for the scoring population.
World Bank source organizations/notes and IMF dataset/vintage metadata remain
attached. WDI is an official harmonized publication, not necessarily the original
national producer. DataMapper supplies no observation-specific estimate cutoff;
the inherited current-calendar-year forecast convention is explicitly labelled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from dalio.data_sources.imf_datamapper import IMF_DM_BASE, IMF_FUNDAMENTALS
from dalio.data_sources.worldbank import WB_BASE_URL, WB_FUNDAMENTALS, WorldBankSource

logger = logging.getLogger(__name__)
MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data/reference/company_listing_countries_v1.json"
MANIFEST_SHA256 = "91965456fa9387ad687ccd0ad971229ddeba68f88e2ccea27a4dbdc7a7e3abd3"
COUNTRY_CODES = ("BE", "CA", "CH", "DE", "DK", "EE", "ES", "FI", "FR", "GB",
                 "IT", "LT", "LV", "NL", "NO", "PL", "PT", "SE", "US")
COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]
METHOD = "company-country-official-macro-v2"


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode()


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def archive(body: bytes, directory: Path) -> Path:
    sha = digest(body)
    path = directory.resolve() / sha[:2] / f"{sha}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != body:
        raise ValueError(f"Artifact hash collision or corruption: {path}")
    if not path.exists():
        path.write_bytes(body)
    return path


def company_country_manifest() -> dict:
    body = MANIFEST_PATH.read_bytes()
    if digest(body) != MANIFEST_SHA256:
        raise ValueError("Company-country acquisition manifest hash mismatch")
    manifest = json.loads(body)
    rows = manifest["countries"]
    if (tuple(row["listing_iso2"] for row in rows) != COUNTRY_CODES
            or sum(row["listing_count"] for row in rows) != 19140
            or manifest["scoring_population_change"] is not False):
        raise ValueError("Company-country acquisition manifest mismatch")
    return manifest


def selected_countries(codes=None) -> list[dict]:
    normalized = {"GB" if code == "UK" else code for code in (codes or COUNTRY_CODES)}
    if not normalized or not normalized <= set(COUNTRY_CODES):
        raise ValueError("Unknown company listing country selection")
    return [row for row in company_country_manifest()["countries"]
            if row["listing_iso2"] in normalized]


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Publisher value must be numeric or null")
    if not math.isfinite(value):
        raise ValueError("Publisher value must be finite")
    return float(value)


def _utc(value):
    timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("Acquisition clocks must be timezone-aware")
    return timestamp.astimezone(UTC)


def _publisher_clock(value, received_at):
    """Validate provider updates without inventing a timezone for date-only metadata."""
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Invalid publisher update clock")
    update = datetime.fromisoformat(value)
    if update.tzinfo is None:
        # WB dates and IMF DataMapper last-modified timestamps carry no timezone.
        # Their calendar date is checked; exact within-day ordering stays unknown.
        future = update.date() > received_at.date()
    else:
        future = update.astimezone(UTC) > received_at
    if future:
        raise ValueError("Source receipt precedes publisher update")


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["country", "date"]).reset_index(drop=True)


def parse_worldbank(pages, spec, country_map, end_year):
    """Validate all pages and cells, retaining null and unreturned periods separately."""
    rows, seen, nulls, statuses = [], set(), {}, {}
    if not pages:
        raise ValueError("Missing World Bank response pages")
    total = None
    update = None
    received = 0
    for number, page in enumerate(pages, 1):
        if (not isinstance(page, list) or len(page) != 2 or not isinstance(page[0], dict)
                or not isinstance(page[1], list)):
            raise ValueError("World Bank source error or malformed response")
        meta, cells = page
        if (int(meta.get("page", 0)) != number or int(meta.get("pages", 0)) != len(pages)
                or str(meta.get("sourceid")) != str(spec.source_id)):
            raise ValueError("World Bank pagination/source mismatch")
        if total is None:
            total = int(meta["total"])
            update = meta.get("lastupdated")
        if int(meta["total"]) != total:
            raise ValueError("World Bank total changed across pages")
        if meta.get("lastupdated") != update:
            raise ValueError("World Bank source vintage changed across pages")
        received += len(cells)
        for cell in cells:
            if (not isinstance(cell, dict) or cell.get("indicator", {}).get("id") != spec.wb_code
                    or cell.get("countryiso3code") not in country_map):
                raise ValueError("World Bank country/indicator identity mismatch")
            iso2 = country_map[cell["countryiso3code"]]
            native_year = cell.get("date", "")
            if not isinstance(native_year, str) or len(native_year) != 4 or not native_year.isdigit():
                raise ValueError("World Bank invalid annual period")
            year = int(native_year)
            if not spec.start_year <= year <= end_year or (iso2, year) in seen:
                raise ValueError("World Bank duplicate or out-of-range period")
            seen.add((iso2, year))
            if "value" not in cell:
                raise ValueError("World Bank cell lacks value field")
            statuses.setdefault(iso2, []).append({"year": year,
                "native_status": cell.get("obs_status"), "native_unit": cell.get("unit")})
            if cell["value"] is None:
                nulls.setdefault(iso2, []).append(year)
                continue
            rows.append(dict(country=iso2, indicator=spec.indicator, date=date(year, 12, 31),
                             value=_finite(cell["value"]), source=spec.source_label,
                             series_id=spec.wb_code, status="published_statistic"))
    if total != received:
        raise ValueError("World Bank incomplete response total")
    missing = {iso2: {"null_years": sorted(nulls.get(iso2, [])),
        "not_returned_years": [year for year in range(spec.start_year, end_year + 1)
                               if (iso2, year) not in seen],
        "native_status_records": sorted(statuses.get(iso2, []), key=lambda row: row["year"])}
        for iso2 in country_map.values()}
    return _frame(rows), missing


def parse_imf(payload, metadata, spec, country_map, cutoff):
    expected_dataset = "FM" if spec == IMF_FUNDAMENTALS[3] else "WEO"
    if metadata.get("dataset") != expected_dataset or not metadata.get("source"):
        raise ValueError("IMF dataset metadata mismatch")
    values = payload.get("values", {}).get(spec.imf_code) if isinstance(payload, dict) else None
    if not isinstance(values, dict) or not values:
        raise ValueError("IMF source error or malformed indicator response")
    source = "IMF_FISCAL_MONITOR" if expected_dataset == "FM" else "IMF_WEO"
    rows, missing = [], {}
    for iso3, iso2 in country_map.items():
        series = values.get(iso3)
        missing[iso2] = {"null_years": [], "entity_not_returned": series is None,
                        "forecast_policy": "Current-calendar-year convention; no native cutoff",
                        "earlier_years_policy": "May include estimates as well as outturns"}
        if series is None:
            continue
        if not isinstance(series, dict):
            raise ValueError("IMF malformed country history")
        for native_year, value in sorted(series.items()):
            if not isinstance(native_year, str) or len(native_year) != 4 or not native_year.isdigit():
                raise ValueError("IMF invalid annual period")
            year = int(native_year)
            if year < spec.start_year:
                continue
            if not year <= cutoff + 10:
                raise ValueError("IMF unexpectedly distant projection")
            if value is None:
                missing[iso2]["null_years"].append(year)
                continue
            projected = year >= cutoff
            rows.append(dict(country=iso2, indicator=spec.indicator, date=date(year, 12, 31),
                value=_finite(value), source=source + ("_FCST" if projected else ""),
                series_id=spec.imf_code,
                status="forecast_calendar_convention" if projected else "estimate_or_outturn"))
    return _frame(rows), missing


def _specs(families, indicators):
    if not families or not set(families) <= {"wb", "imf"}:
        raise ValueError("Source families must be wb and/or imf")
    specs = [(family, spec) for family, group in (("wb", WB_FUNDAMENTALS),
                                                 ("imf", IMF_FUNDAMENTALS))
             if family in families for spec in group
             if not indicators or spec.indicator in indicators]
    if not specs or (indicators and set(indicators) - {spec.indicator for _, spec in specs}):
        raise ValueError("Unknown indicator selection")
    return specs


def collect_bundle(*, artifact_dir: Path, countries=None, families=("wb", "imf"),
                   indicators=None, client=None, clock=None) -> Path:
    """Collect official bytes, then write a complete offline-replayable batch manifest.

    Each failed publisher request is a source error, never evidence of no data.
    Every successful series is preflighted before the returned bundle can be stored.
    """
    selection = selected_countries(countries)
    specs = _specs(families, indicators)
    country_map = {row["iso3"]: row["country"] for row in selection}
    now = clock or (lambda: datetime.now(UTC))
    year = _utc(now()).year
    session = client or requests.Session()
    if client is None:
        session.headers.update({"User-Agent": "DalioMacroResearch/1.0 (official statistics research)"})
    requests_log, series = [], []

    def obtain(url):
        # Keep the evidence delivery boundary explicit. A future publisher
        # redirect requires review of its destination rather than automatic follow.
        response = session.get(url, timeout=45, allow_redirects=False)
        body = response.content
        path = archive(body, artifact_dir / "responses")
        index = len(requests_log)
        requests_log.append(dict(url=url, path=str(path), sha256=digest(body),
                                 http_status=response.status_code,
                                 obtained_at=_utc(now()).isoformat(),
                                 final_url=response.url,
                                 redirect_history=[step.url for step in response.history],
                                 headers={key: value for key, value in response.headers.items()
                                          if key.lower() in {"content-type", "last-modified",
                                                             "etag", "location"}}))
        response.raise_for_status()
        if response.status_code != 200 or response.history or response.url != url:
            raise ValueError("Unsupported HTTP status or redirected delivery")
        return index, json.loads(body)

    imf_metadata = None
    for family, spec in specs:
        logger.info("Collecting %s/%s for %d listing countries", family, spec.indicator, len(selection))
        record = dict(family=family, indicator=spec.indicator, spec=asdict(spec), request_indices=[])
        try:
            if family == "wb":
                url = f"{WB_BASE_URL}/indicator/{spec.wb_code}?format=json&source={spec.source_id}"
                index, metadata = obtain(url)
                record["request_indices"].append(index)
                if (not isinstance(metadata, list) or len(metadata) != 2
                        or not isinstance(metadata[1], list) or len(metadata[1]) != 1
                        or metadata[1][0].get("id") != spec.wb_code
                        or str(metadata[1][0].get("source", {}).get("id")) != str(spec.source_id)):
                    raise ValueError("World Bank indicator metadata mismatch")
                pages = []
                page = 1
                while True:
                    url = WorldBankSource._url(spec, list(country_map), page, year)
                    index, payload = obtain(url)
                    record["request_indices"].append(index)
                    pages.append(payload)
                    if (not isinstance(payload, list) or len(payload) != 2
                            or not isinstance(payload[0], dict)):
                        raise ValueError("World Bank source error or malformed response")
                    count = int(payload[0].get("pages", 0))
                    if not 1 <= count <= 100:
                        raise ValueError("World Bank invalid page count")
                    if page >= count:
                        break
                    page += 1
                parse_worldbank(pages, spec, country_map, year)
            else:
                if imf_metadata is None:
                    imf_metadata = obtain(f"{IMF_DM_BASE}/indicators")
                index, metadata = imf_metadata
                record["request_indices"].append(index)
                index, payload = obtain(f"{IMF_DM_BASE}/{spec.imf_code}")
                record["request_indices"].append(index)
                parse_imf(payload, metadata.get("indicators", {}).get(spec.imf_code, {}),
                          spec, country_map, year)
            record["status"] = "validated"
        except Exception as exc:  # noqa: BLE001 - retain an explicit failed source partition
            logger.warning("Source error %s/%s: %s", family, spec.indicator, exc)
            record.update(status="source_error", error=f"{type(exc).__name__}: {exc}")
        series.append(record)
    body = canonical(dict(schema_version=2, method=METHOD, manifest=company_country_manifest(),
        countries=[row["listing_iso2"] for row in selection], year=year,
        families=list(families), indicators=list(indicators) if indicators else None,
        retrieved_at=_utc(now()).isoformat(),
        requests=requests_log, series=series))
    path = archive(body, artifact_dir / "bundles")
    load_bundle(path)  # Full replay preflight before returning a usable batch.
    return path


def load_bundle(path: Path) -> dict:
    """Re-hash every retained response and rebuild observations solely from raw bytes."""
    body = path.read_bytes()
    if path.stem != digest(body):
        raise ValueError("Batch bundle hash mismatch")
    bundle = json.loads(body)
    if (bundle.get("schema_version") != 2 or bundle.get("method") != METHOD
            or bundle.get("manifest") != company_country_manifest()):
        raise ValueError("Batch method/country manifest mismatch")
    selection = selected_countries(bundle["countries"])
    if bundle["countries"] != [row["listing_iso2"] for row in selection]:
        raise ValueError("Batch country ordering/identity mismatch")
    specs = _specs(bundle["families"], bundle["indicators"])
    if len(specs) != len(bundle["series"]):
        raise ValueError("Batch series denominator mismatch")
    country_map = {row["iso3"]: row["country"] for row in selection}
    retrieved_at = _utc(bundle["retrieved_at"])
    if retrieved_at.year != bundle["year"]:
        raise ValueError("Batch retrieval clock mismatch")
    payloads = []
    for request in bundle["requests"]:
        if _utc(request["obtained_at"]) > retrieved_at:
            raise ValueError("Batch retrieval precedes source receipt")
        raw = Path(request["path"]).read_bytes()
        if digest(raw) != request["sha256"]:
            raise ValueError("Source response artifact hash mismatch")
        # Failed HTTP responses may be HTML; successful series below require valid JSON.
        try:
            payloads.append(json.loads(raw))
        except (ValueError, UnicodeDecodeError):
            payloads.append(None)
    ready, reports = [], []
    for (family, spec), record in zip(specs, bundle["series"], strict=True):
        if (record["family"] != family or record["indicator"] != spec.indicator
                or record["spec"] != asdict(spec)):
            raise ValueError("Batch source specification mismatch")
        if record["status"] == "source_error":
            if not record.get("error"):
                raise ValueError("Source error lacks diagnostic")
            reports.extend(dict(country=iso2, indicator=spec.indicator, family=family,
                                status="source_error", error=record["error"])
                           for iso2 in country_map.values())
            continue
        if record["status"] != "validated":
            raise ValueError("Unknown source partition status")
        indices = record["request_indices"]
        refs = [bundle["requests"][index] for index in indices]
        contents = [payloads[index] for index in indices]
        if not refs or any(ref["http_status"] != 200 for ref in refs):
            raise ValueError("Validated series binds unsuccessful HTTP response")
        if any(ref.get("final_url") != ref["url"] or ref.get("redirect_history") != []
               for ref in refs):
            raise ValueError("Unverified or redirected source delivery")
        for ref in refs:
            for key, value in ref["headers"].items():
                if key.lower() == "last-modified":
                    update = parsedate_to_datetime(value)
                    if _utc(update) > _utc(ref["obtained_at"]):
                        raise ValueError("Source receipt precedes HTTP publisher update")
        if family == "wb":
            urls = [f"{WB_BASE_URL}/indicator/{spec.wb_code}?format=json&source={spec.source_id}"]
            urls += [WorldBankSource._url(spec, list(country_map), page, bundle["year"])
                     for page in range(1, len(refs))]
            meta = contents[0]
            if (not isinstance(meta, list) or len(meta) != 2 or len(meta[1]) != 1
                    or meta[1][0].get("id") != spec.wb_code
                    or str(meta[1][0].get("source", {}).get("id")) != str(spec.source_id)):
                raise ValueError("World Bank indicator metadata mismatch")
            metadata = meta[1][0]
            frame, missing = parse_worldbank(contents[1:], spec, country_map, bundle["year"])
            for payload, ref in zip(contents[1:], refs[1:], strict=True):
                _publisher_clock(payload[0].get("lastupdated"), _utc(ref["obtained_at"]))
            sources = (spec.source_label,)
        else:
            urls = [f"{IMF_DM_BASE}/indicators", f"{IMF_DM_BASE}/{spec.imf_code}"]
            metadata = contents[0].get("indicators", {}).get(spec.imf_code, {})
            frame, missing = parse_imf(contents[1], metadata, spec, country_map, bundle["year"])
            for ref in refs:
                _publisher_clock(metadata.get("last-modified"), _utc(ref["obtained_at"]))
            source = "IMF_FISCAL_MONITOR" if spec == IMF_FUNDAMENTALS[3] else "IMF_WEO"
            sources = (source, source + "_FCST")
        if [ref["url"] for ref in refs] != urls:
            raise ValueError("Source URL or response order mismatch")
        for iso2 in country_map.values():
            country_frame = frame[frame.country == iso2].reset_index(drop=True)
            report = dict(country=iso2, indicator=spec.indicator, family=family,
                          status="ready" if len(country_frame) else "missing",
                          rows=len(country_frame), missingness=missing[iso2],
                          publisher_metadata=metadata, source_urls=urls,
                          first_period=str(country_frame.date.min()) if len(country_frame) else None,
                          last_period=str(country_frame.date.max()) if len(country_frame) else None)
            reports.append(report)
            if len(country_frame):
                ready.append(dict(frame=country_frame, report=report, refs=refs,
                                  sources=sources, source_url=urls[1],
                                  series_id=spec.wb_code if family == "wb" else spec.imf_code))
    summary = dict(requested_countries=len(selection), requested_partitions=len(reports),
        ready_partitions=sum(row["status"] == "ready" for row in reports),
        missing_partitions=sum(row["status"] == "missing" for row in reports),
        source_error_partitions=sum(row["status"] == "source_error" for row in reports),
        observation_count=sum(row.get("rows", 0) for row in reports),
        retrieved_at=retrieved_at.isoformat(), bundle_path=str(path.resolve()),
        bundle_sha256=digest(body), partitions=reports)
    return dict(ready=ready, summary=summary, retrieved_at=retrieved_at,
                bundle_path=path.resolve(), bundle_sha256=digest(body))
