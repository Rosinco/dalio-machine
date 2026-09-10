"""Immutable, complete five-signal Swedish monitoring supplements.

The batch completion clock is conservative availability for every series.
Raw bytes and exact requests are retained even for failed responses; missing
attempts never inherit facts from an older capture. This module never opens a DB.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from dalio.data_sources.riksbank import RIKSBANK_SERIES, RIKSBANK_SOURCE, RiksbankSource

METHOD = "sweden-monitoring-v1"
INDICATORS = ("industrial_production", "industrial_orders", "corporate_new_lending_rate",
              "policy_rate", "yield_10y")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def utc(value: datetime | str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(UTC)
    except (TypeError, ValueError) as exc:
        raise ValueError("Monitoring clock must be a timezone-aware datetime") from exc


def _scb_specs():
    from dalio.data_sources.scb_monitoring import SCB_MONITORING_SPECS
    return SCB_MONITORING_SPECS


def _scb_parse(key, metadata, response, request):
    from dalio.data_sources.scb_monitoring import parse_scb_input
    return parse_scb_input(key, metadata, response, request)


def _catalogue(end: date) -> list[dict]:
    scb = {spec.key: spec for spec in _scb_specs()}
    if set(scb) != set(INDICATORS[:3]):
        raise ValueError("SCB monitoring catalogue differs from fixed pilot selection")
    items = []
    for indicator in INDICATORS[:3]:
        spec = scb[indicator]
        items.append(dict(indicator=indicator, kind="scb", query=spec.request, requests=[
            dict(id=f"{indicator}:metadata", url=spec.metadata_url),
            dict(id=f"{indicator}:data", url=spec.source_url)]))
    by_indicator = {spec.indicator: spec for spec in RIKSBANK_SERIES}
    for indicator in INDICATORS[3:]:
        spec = by_indicator[indicator]
        items.append(dict(indicator=indicator, kind="riksbank", query=None, requests=[
            dict(id=f"{indicator}:data", url=RiksbankSource.url_for(
                spec, from_date=spec.history_start, to_date=end))]))
    return items


def _archive(root: Path, body: bytes, *, bundle: bool = False) -> Path:
    folder = root / ("bundles" if bundle else "responses")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest(body)}.{ 'json' if bundle else 'bin'}"
    try:
        with path.open("xb") as output:
            output.write(body)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != body:
            raise ValueError(f"Existing immutable artifact differs: {path}") from None
    return path.resolve()


def collect_bundle(*, artifact_root: Path, client=None, clock=None,
                   pacing_seconds: float = 13.0) -> Path:
    """Capture all five fixed inputs and return the immutable bundle path.

    A failed publisher request remains a recorded attempt. No mutable source
    cache is read or retroactively attributed to an earlier release clock.
    ``client`` and ``clock`` are injectable solely for deterministic HTTP tests.
    """
    if not math.isfinite(pacing_seconds) or pacing_seconds < 0:
        raise ValueError("pacing_seconds must be finite and non-negative")
    now = clock or (lambda: datetime.now(UTC))
    started = utc(now())
    last_clock = started
    root = Path(artifact_root).expanduser().resolve()
    http = client or requests.Session()
    signals = []
    swea_calls = 0
    for spec in _catalogue(started.date()):
        captured = dict(indicator=spec["indicator"], kind=spec["kind"], query=spec["query"], requests=[])
        for request in spec["requests"]:
            if spec["kind"] == "riksbank":
                if swea_calls and pacing_seconds:
                    time.sleep(pacing_seconds)
                swea_calls += 1
            requested = utc(now())
            if requested < last_clock:
                raise ValueError("Monitoring acquisition clock regressed")
            record = dict(**request, method="GET", requested_at=requested.isoformat(),
                          http_status=None, final_url=None, redirect_history=[], headers={},
                          path=None, sha256=None, error=None)
            try:
                response = http.get(request["url"], headers={"Accept": "application/json"},
                                    timeout=60, allow_redirects=False)
                record.update(http_status=int(response.status_code), final_url=str(response.url),
                    redirect_history=[str(item.url) for item in response.history],
                    headers={key.lower(): str(value) for key, value in response.headers.items()
                             if key.lower() in {"content-type", "last-modified", "etag", "date"}})
                body = response.content
                if not isinstance(body, bytes):
                    raise ValueError("HTTP client must return exact response bytes")
                record.update(path=str(_archive(root, body)), sha256=digest(body))
                if (record["http_status"] != 200 or record["final_url"] != request["url"]
                        or record["redirect_history"]):
                    record["error"] = "Non-200 response or unexpected delivery URL/redirect"
            except Exception as exc:  # noqa: BLE001 - preserve each independently failed attempt
                record["error"] = f"{type(exc).__name__}: {exc}"
            received = utc(now())
            if received < requested:
                raise ValueError("Monitoring response clock regressed")
            record["received_at"] = received.isoformat()
            last_clock = received
            captured["requests"].append(record)
        signals.append(captured)
    completed = utc(now())
    if completed < last_clock:
        raise ValueError("Monitoring batch completion clock regressed")
    batch = dict(method=METHOD, started_at=started.isoformat(), available_at=completed.isoformat(),
                 retrieved_at=completed.isoformat(), history_end=started.date().isoformat(), signals=signals)
    return _archive(root, canonical(batch), bundle=True)


def read_header(path: Path) -> tuple[dict, str]:
    """Read exact content-addressed envelope; source artifacts are checked later."""
    path = Path(path)
    body = path.read_bytes()
    sha256 = digest(body)
    if path.is_symlink() or path.stem != sha256:
        raise ValueError("Monitoring bundle hash does not match its immutable filename")
    try:
        bundle = json.loads(body)
        if not isinstance(bundle, dict) or canonical(bundle) != body:
            raise ValueError
        if bundle["method"] != METHOD:
            raise ValueError
        utc(bundle["available_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid monitoring bundle envelope") from exc
    return bundle, sha256


def _checked_request(record: dict, expected: dict, *, started: datetime, completed: datetime) -> bytes | None:
    if record.get("id") != expected["id"] or record.get("url") != expected["url"] or record.get("method") != "GET":
        raise ValueError("Monitoring request identity differs from fixed official catalogue")
    requested, received = utc(record["requested_at"]), utc(record["received_at"])
    if not started <= requested <= received <= completed:
        raise ValueError("Monitoring request clock lies outside its batch")
    body = None
    if record["path"] is not None:
        path = Path(record["path"])
        if not path.is_absolute() or path.is_symlink() or str(path.resolve()) != str(path):
            raise ValueError("Monitoring raw artifact path is not a canonical regular file")
        body = path.read_bytes()
        if digest(body) != record["sha256"]:
            raise ValueError("Monitoring raw artifact hash mismatch")
    elif record["sha256"] is not None:
        raise ValueError("Monitoring artifact hash has no retained bytes")
    if record["error"] is None:
        if (record["http_status"] != 200 or record["final_url"] != expected["url"]
                or record["redirect_history"] or body is None):
            raise ValueError("Successful monitoring request lacks exact official delivery evidence")
    elif not isinstance(record["error"], str) or not record["error"]:
        raise ValueError("Monitoring source error must be explicitly recorded")
    return body


def _parse_riksbank(indicator: str, body: bytes, end: date) -> dict:
    spec = next(spec for spec in RIKSBANK_SERIES if spec.indicator == indicator)
    url = RiksbankSource.url_for(spec, from_date=spec.history_start, to_date=end)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("SWEA JSON contains an ambiguous duplicate key")
            result[key] = value
        return result
    json.loads(body, object_pairs_hook=unique)
    payload = RiksbankSource._parse_payload(body.decode("utf-8"), url)
    finite = [row for row in payload if row.get("value") is not None]
    # Reuse the official-series parser for non-null numeric/date validation.
    RiksbankSource._to_long(finite, spec)
    rows, seen = [], set()
    for row in payload:
        when = date.fromisoformat(str(row["date"]))
        if when in seen or not spec.history_start <= when <= end or "value" not in row:
            raise ValueError("SWEA observation date/identity outside fixed history request")
        seen.add(when)
        value = None if row["value"] is None else float(row["value"])
        rows.append(dict(date=when.isoformat(), period=when.isoformat(),
            period_start=when.isoformat(), period_end=when.isoformat(),
            value=value, status="not_reported" if value is None else "observed", native_status=None,
            source_locator=f"SWEA {spec.series_id} {when.isoformat()}"))
    if not rows:
        raise ValueError("SWEA returned no history observations")
    return dict(indicator=indicator, source=RIKSBANK_SOURCE, series_id=spec.series_id,
        unit=spec.unit, frequency="daily", definition=spec.description,
        publisher_metadata={"publisher": "Sveriges Riksbank", "native_series_id": spec.series_id,
            "definition": spec.description, "unit": spec.unit,
            "observation_status_convention": "Reported historical value; SWEA does not supply per-point revision/provisional status."},
        observations=sorted(rows, key=lambda row: row["date"]),
        missingness={"explicit_null_observations": sum(row["value"] is None for row in rows)})


def load_bundle(path: Path) -> dict:
    """Verify a full selected bundle and reconstruct facts from retained bytes.

    Artifact/envelope corruption raises. A recorded failed HTTP attempt or a
    malformed publisher response is an explicit source gap, never older facts.
    """
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Monitoring bundle symlink is not permitted")
    path = path.resolve()
    bundle, sha256 = read_header(path)
    started, completed = utc(bundle["started_at"]), utc(bundle["available_at"])
    if started > completed or utc(bundle["retrieved_at"]) != completed:
        raise ValueError("Monitoring batch clock ordering is invalid")
    if bundle["history_end"] != started.date().isoformat():
        raise ValueError("Monitoring history window differs from capture date")
    expected = _catalogue(started.date())
    if not isinstance(bundle["signals"], list) or len(bundle["signals"]) != len(expected):
        raise ValueError("Monitoring bundle must contain the complete fixed five attempts")
    result = dict(bundle_path=str(path), bundle_sha256=sha256, available_at=completed.isoformat(), series=[], gaps=[])
    protected = {str(path)}
    for item, spec in zip(bundle["signals"], expected, strict=True):
        if any(item.get(key) != spec[key] for key in ("indicator", "kind", "query")):
            raise ValueError("Monitoring signal identity differs from fixed catalogue")
        if len(item["requests"]) != len(spec["requests"]):
            raise ValueError("Monitoring signal has an incomplete request attempt set")
        bodies, artifacts, errors = [], [], []
        for record, request in zip(item["requests"], spec["requests"], strict=True):
            bodies.append(_checked_request(record, request, started=started, completed=completed))
            if record["path"]:
                protected.add(record["path"])
                artifacts.append(dict(role=record["id"].split(":")[1], sha256=record["sha256"], path=record["path"]))
            if record["error"]:
                errors.append(dict(request_id=record["id"], error=record["error"]))
        if errors:
            result["gaps"].append(dict(indicator=item["indicator"], reason="source_error", errors=errors))
            continue
        try:
            parsed = (_scb_parse(item["indicator"], bodies[0], bodies[1], item["query"])
                      if item["kind"] == "scb" else _parse_riksbank(item["indicator"], bodies[0], started.date()))
            if parsed["indicator"] != item["indicator"]:
                raise ValueError("Parsed signal differs from requested native identity")
            # The source parser validates native update clocks where supplied;
            # they cannot postdate the exact retained response receipt.
            updated = parsed.get("publisher_updated_at")
            if updated and utc(updated) > min(utc(r["received_at"]) for r in item["requests"]):
                raise ValueError("Publisher update clock is later than source receipt")
            canonical(parsed)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            result["gaps"].append(dict(indicator=item["indicator"], reason="source_validation_error", error=str(exc)))
            continue
        artifacts.append(dict(role="acquisition_bundle", sha256=sha256, path=str(path)))
        result["series"].append(dict(parsed, source_url=spec["requests"][-1]["url"],
            available_at=completed.isoformat(), retrieved_at=completed.isoformat(), published_at=None,
            bundle_sha256=sha256, evidence_ref=f"monitoring:{sha256}:{item['indicator']}", artifacts=artifacts))
    result["protected_artifact_paths"] = sorted(protected)
    return result
