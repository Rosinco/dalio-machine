"""Capture and replay a complete twelve-signal national-source supplement."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as day_time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from dalio.monitoring.acquisition import _archive, canonical, digest, utc
from dalio.monitoring.evidence import _select_bundle

METHOD = "nordic-monitoring-v1"
COUNTRIES = ("NO", "DK", "FI")
INDICATORS = ("industrial_production", "corporate_new_lending_rate", "policy_rate", "yield_10y")


def catalogue(end: date) -> list[dict]:
    from dalio.data_sources import denmark_monitoring, finland_monitoring, norway_monitoring

    result = [spec for module in (norway_monitoring, denmark_monitoring, finland_monitoring)
              for spec in module.specs(end)]
    expected = {(country, indicator) for country in COUNTRIES for indicator in INDICATORS}
    if len(result) != len(expected) or {(s["country"], s["indicator"]) for s in result} != expected:
        raise ValueError("Nordic catalogue must contain the fixed twelve country/signals")
    return result


def parse_input(spec: dict, bodies: dict[str, bytes]) -> dict:
    from dalio.data_sources import denmark_monitoring, finland_monitoring, norway_monitoring

    module = {"NO": norway_monitoring, "DK": denmark_monitoring, "FI": finland_monitoring}[spec["country"]]
    return module.parse_input(spec, bodies)


def _check_capture_window(spec, records):
    rule = spec.get("capture_exclusion")
    if rule is None:
        return
    zone = ZoneInfo(rule["timezone"])
    for record in records:
        start, end = (utc(record[k]).astimezone(zone) for k in ("requested_at", "received_at"))
        day = start.date()
        while day <= end.date():
            excluded_start = datetime.combine(day, day_time(rule["start_hour"]), zone)
            excluded_end = datetime.combine(day, day_time(rule["end_hour"]), zone)
            if start < excluded_end and end >= excluded_start:
                raise ValueError("Publisher maintenance capture window: " + rule["reason"])
            day += timedelta(days=1)


def collect_bundle(*, artifact_root: Path, client=None, clock=None, pacing_seconds=1.0) -> Path:
    if not math.isfinite(pacing_seconds) or pacing_seconds < 0:
        raise ValueError("Pacing must be finite and non-negative")
    now = clock or (lambda: datetime.now(UTC))
    started = utc(now())
    last = started
    http = client or requests.Session()
    root = Path(artifact_root).expanduser().resolve()
    signals = []
    calls = 0
    for spec in catalogue(started.date()):
        item = {"country": spec["country"], "indicator": spec["indicator"],
                "spec_sha256": digest(canonical(spec)), "requests": []}
        for request in spec["requests"]:
            if calls and pacing_seconds:
                time.sleep(pacing_seconds)
            calls += 1
            requested = utc(now())
            if requested < last:
                raise ValueError("Acquisition clock regressed")
            record = {**request, "requested_at": requested.isoformat(), "http_status": None,
                      "final_url": None, "redirect_history": [], "headers": {},
                      "path": None, "sha256": None, "error": None}
            try:
                kwargs = {"timeout": 60, "allow_redirects": False}
                if "json" in request:
                    kwargs["json"] = request["json"]
                response = http.request(request["method"], request["url"], **kwargs)
                record.update(http_status=int(response.status_code), final_url=str(response.url),
                    redirect_history=[str(r.url) for r in response.history],
                    headers={k.lower(): str(v) for k, v in response.headers.items()
                             if k.lower() in {"content-type", "date", "last-modified", "etag"}})
                body = response.content
                if not isinstance(body, bytes):
                    raise ValueError("HTTP client must return original bytes")
                record.update(path=str(_archive(root, body)), sha256=digest(body))
                if record["http_status"] != 200 or record["final_url"] != request["url"] or record["redirect_history"]:
                    record["error"] = "Non-200 response or unexpected delivery URL/redirect"
            except Exception as exc:  # noqa: BLE001 - retain independently failed attempts
                record["error"] = f"{type(exc).__name__}: {exc}"
            received = utc(now())
            if received < requested:
                raise ValueError("Response clock regressed")
            record["received_at"] = received.isoformat()
            last = received
            item["requests"].append(record)
        signals.append(item)
    completed = utc(now())
    if completed < last:
        raise ValueError("Batch completion clock regressed")
    batch = {"method": METHOD, "started_at": started.isoformat(), "available_at": completed.isoformat(),
             "retrieved_at": completed.isoformat(), "history_end": started.date().isoformat(), "signals": signals}
    return _archive(root, canonical(batch), bundle=True)


def load_bundle(path: Path) -> dict:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Nordic bundle symlink is not permitted")
    path = path.resolve()
    body = path.read_bytes()
    sha = digest(body)
    bundle = json.loads(body)
    if path.stem != sha or canonical(bundle) != body or bundle.get("method") != METHOD:
        raise ValueError("Nordic bundle hash/contract mismatch")
    started, completed = utc(bundle["started_at"]), utc(bundle["available_at"])
    if started > completed or utc(bundle["retrieved_at"]) != completed:
        raise ValueError("Nordic batch clock ordering is invalid")
    if bundle["history_end"] != started.date().isoformat():
        raise ValueError("Nordic history end differs from capture date")
    specs = catalogue(started.date())
    if len(bundle["signals"]) != len(specs):
        raise ValueError("Nordic capture must contain every fixed signal attempt")
    result = {"bundle_path": str(path), "bundle_sha256": sha, "available_at": completed.isoformat(),
              "series": [], "gaps": []}
    protected = {str(path)}
    for item, spec in zip(bundle["signals"], specs, strict=True):
        identity = {key: spec[key] for key in ("country", "indicator")}
        if any(item.get(k) != v for k, v in identity.items()) or item["spec_sha256"] != digest(canonical(spec)):
            raise ValueError("Nordic signal identity or semantic catalogue differs")
        if len(item["requests"]) != len(spec["requests"]):
            raise ValueError("Nordic request set is incomplete")
        bodies, artifacts, errors = {}, [], []
        for record, request in zip(item["requests"], spec["requests"], strict=True):
            if any(record.get(key) != request.get(key) for key in ("role", "method", "url", "json")):
                raise ValueError("Nordic request identity differs from official catalogue")
            requested, received = utc(record["requested_at"]), utc(record["received_at"])
            if not started <= requested <= received <= completed:
                raise ValueError("Nordic request clock lies outside batch")
            raw = None
            if record["path"] is not None:
                raw_path = Path(record["path"])
                if not raw_path.is_absolute() or raw_path.is_symlink() or str(raw_path.resolve()) != str(raw_path):
                    raise ValueError("Nordic artifact path must be a canonical regular file")
                raw = raw_path.read_bytes()
                if digest(raw) != record["sha256"]:
                    raise ValueError("Nordic raw artifact hash mismatch")
                protected.add(str(raw_path))
                artifacts.append({"role": request["role"], "sha256": record["sha256"], "path": str(raw_path)})
            elif record["sha256"] is not None:
                raise ValueError("Nordic artifact hash has no retained bytes")
            if record["error"] is None:
                if (record["http_status"] != 200 or record["final_url"] != request["url"]
                        or record["redirect_history"] or raw is None):
                    raise ValueError("Nordic successful request lacks original delivery evidence")
            elif not isinstance(record["error"], str) or not record["error"]:
                raise ValueError("Nordic source error must be explicit")
            else:
                errors.append({"role": request["role"], "error": record["error"]})
            if request["role"] in bodies:
                raise ValueError("Duplicate Nordic request role")
            bodies[request["role"]] = raw
        if errors:
            result["gaps"].append({**identity, "reason": "source_error", "errors": errors})
            continue
        try:
            _check_capture_window(spec, item["requests"])
            parsed = parse_input(spec, bodies)
            if any(parsed.get(k) != v for k, v in identity.items()):
                raise ValueError("Parsed Nordic identity differs from requested source")
            updated = parsed.get("publisher_updated_at")
            if updated and utc(updated) > min(utc(r["received_at"]) for r in item["requests"]):
                raise ValueError("Publisher update postdates source receipt")
            canonical(parsed)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            result["gaps"].append({**identity, "reason": "source_validation_error", "error": str(exc)})
            continue
        artifacts.append({"role": "acquisition_bundle", "sha256": sha, "path": str(path)})
        result["series"].append({**parsed, "spec": spec, "available_at": completed.isoformat(),
            "retrieved_at": completed.isoformat(), "published_at": None, "bundle_sha256": sha,
            "source_url": next(r["url"] for r in spec["requests"] if r["role"] == "data"),
            "artifacts": artifacts})
    result["protected_artifact_paths"] = sorted(protected)
    return result


def load_evidence(*, bundle_paths, as_known_at: datetime) -> dict:
    cutoff = utc(as_known_at)
    selected = _select_bundle(bundle_paths, cutoff)
    if selected is None:
        result = {"series": [], "gaps": [
            {"country": s["country"], "indicator": s["indicator"], "reason": "no_bundle_as_known_at"}
            for s in catalogue(cutoff.date())], "protected_artifact_paths": []}
    else:
        result = load_bundle(selected)
    result["as_known_at"] = cutoff.isoformat()
    result["evidence_digest"] = digest(canonical(result))
    return result
