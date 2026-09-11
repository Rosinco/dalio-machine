"""Original delivery, documented structural gaps and whole-batch selection."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from dalio.national_monitoring import acquisition as a


class Client:
    def __init__(self, status=200, body=b'{"rate":4.25}'):
        self.status, self.body, self.calls = status, body, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return type("Response", (), {"status_code": self.status, "url": url, "content": self.body,
                                     "history": [], "headers": {"Content-Type": "application/json"}})()


@pytest.fixture
def specs(monkeypatch):
    values = [
        {"country": "US", "indicator": "policy_rate", "kind": "test", "requests": [
            {"role": "data", "method": "POST", "url": "https://example.test/rate", "json": {"series": "EFFR"}}]},
        {"country": "US", "indicator": "corporate_new_lending_rate", "kind": "unavailable",
         "definition": "New business lending rate", "unavailable_reason": "Discontinued survey; no current counterpart.",
         "requests": [{"role": "documentation", "method": "GET", "url": "https://example.test/discontinued"}]},
    ]
    monkeypatch.setattr(a, "catalogue", lambda end: values)
    def parse(spec, bodies):
        if spec.get("unavailable_reason"):
            assert bodies["documentation"]
            return {k: spec[k] for k in ("country", "indicator", "unavailable_reason")}
        return {"country": spec["country"], "indicator": spec["indicator"],
                "observations": [{"value": json.loads(bodies["data"])["rate"]}]}
    monkeypatch.setattr(a, "parse_input", parse)
    return values


def capture(tmp_path, client=None, start=None):
    current = start or datetime(2026, 9, 11, 12, tzinfo=UTC)
    def clock():
        nonlocal current
        current += timedelta(seconds=1)
        return current
    return a.collect_bundle(artifact_root=tmp_path, client=client or Client(), clock=clock, pacing_seconds=0)


def test_documented_gap_retains_source_bytes_and_honest_clocks(tmp_path, specs):
    client = Client()
    path = capture(tmp_path, client)
    result = a.load_bundle(path)
    assert len(result["series"]) == 1
    gap = result["gaps"][0]
    assert gap["country"] == "US" and gap["reason"] == "structural_gap"
    assert gap["error"] == specs[1]["unavailable_reason"]
    assert gap["available_at"] == result["available_at"]
    assert {r["role"] for r in gap["artifacts"]} == {"documentation", "acquisition_bundle"}
    assert client.calls[0][2]["json"] == {"series": "EFFR"}
    assert result["series"][0]["published_at"] is None


def test_newest_failed_batch_never_reuses_old_success(tmp_path, specs):
    old = capture(tmp_path)
    new = capture(tmp_path, Client(status=503), datetime(2026, 9, 12, tzinfo=UTC))
    latest = a.load_evidence(bundle_paths=[old, new], as_known_at=datetime(2026, 9, 13, tzinfo=UTC))
    assert latest["series"] == []
    assert all(g["reason"] == "source_error" for g in latest["gaps"])
    earlier = a.load_evidence(bundle_paths=[old, new], as_known_at=datetime(2026, 9, 11, 13, tzinfo=UTC))
    assert len(earlier["series"]) == 1


def test_rehashed_request_or_method_substitution_is_rejected(tmp_path, specs):
    path = capture(tmp_path)
    batch = json.loads(path.read_bytes())
    batch["signals"][0]["requests"][0]["json"]["series"] = "prime"
    altered = a._archive(tmp_path, a.canonical(batch), bundle=True)
    with pytest.raises(ValueError, match="request identity"):
        a.load_bundle(altered)
    batch = json.loads(path.read_bytes())
    batch["method"] = "nordic-monitoring-v1"
    with pytest.raises(ValueError, match="contract"):
        a.load_bundle(a._archive(tmp_path, a.canonical(batch), bundle=True))


def test_document_tampering_does_not_become_structural_gap(tmp_path, specs):
    path = capture(tmp_path)
    batch = json.loads(path.read_bytes())
    raw = a.Path(batch["signals"][1]["requests"][0]["path"])
    raw.write_bytes(b"altered documentation")
    with pytest.raises(ValueError, match="hash"):
        a.load_bundle(path)


def test_unavailable_claim_requires_adapter_validation(tmp_path, specs, monkeypatch):
    monkeypatch.setattr(a, "parse_input", lambda spec, bodies: {"country": "US", "indicator": spec["indicator"]})
    result = a.load_bundle(capture(tmp_path))
    assert result["gaps"][0]["reason"] == "source_validation_error"


def test_naive_cutoff_and_request_clock_outside_capture_fail(tmp_path, specs):
    path = capture(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        a.load_evidence(bundle_paths=[path], as_known_at=datetime(2026, 9, 12))
    batch = json.loads(path.read_bytes())
    batch["signals"][0]["requests"][0]["received_at"] = "2027-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="clock"):
        a.load_bundle(a._archive(tmp_path, a.canonical(batch), bundle=True))


def test_date_only_publisher_update_is_not_promoted_before_receipt_day(tmp_path, specs, monkeypatch):
    parse = a.parse_input
    def dated(spec, bodies):
        value = parse(spec, bodies)
        if not spec.get("unavailable_reason"):
            value["publisher_metadata"] = {"report_updated_date": "2026-09-12", "update_precision": "date"}
        return value
    monkeypatch.setattr(a, "parse_input", dated)
    result = a.load_bundle(capture(tmp_path))
    assert result["series"] == []
    assert "postdates source receipt day" in result["gaps"][0]["error"]


def test_date_only_check_respects_explicit_native_timezone(tmp_path, specs, monkeypatch):
    parse = a.parse_input
    def dated(spec, bodies):
        value = parse(spec, bodies)
        if not spec.get("unavailable_reason"):
            value["publisher_metadata"] = {"report_updated_date": "2026-09-12", "update_precision": "date",
                                           "update_timezone": "Europe/Berlin"}
        return value
    monkeypatch.setattr(a, "parse_input", dated)
    result = a.load_bundle(capture(tmp_path, start=datetime(2026, 9, 11, 22, tzinfo=UTC)))
    assert len(result["series"]) == 1
    assert result["series"][0]["published_at"] is None
