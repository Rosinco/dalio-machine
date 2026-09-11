"""Offline acquisition boundaries; failures cannot resurrect an older input."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from dalio.nordic_monitoring import acquisition as a


@pytest.fixture
def source(monkeypatch):
    spec = {"country": "NO", "indicator": "policy_rate", "kind": "test",
            "requests": [{"role": "data", "method": "POST", "url": "https://example.test/data",
                          "json": {"selection": "native"}}]}
    monkeypatch.setattr(a, "catalogue", lambda end: [spec])
    monkeypatch.setattr(a, "parse_input", lambda spec, bodies: {
        "country": "NO", "indicator": "policy_rate", "frequency": "daily", "unit": "percent",
        "observations": json.loads(bodies["data"]), "definition": "Native policy rate"})
    return spec


class Client:
    def __init__(self, status=200, body=b'[{"value": 3.0}]'):
        self.status, self.body = status, body
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return type("Response", (), {"status_code": self.status, "url": url, "content": self.body,
                                     "history": [], "headers": {"Content-Type": "application/json"}})()


def capture(tmp_path, client=None, start=None):
    current = start or datetime(2026, 9, 11, tzinfo=UTC)
    def clock():
        nonlocal current
        current += timedelta(seconds=1)
        return current
    return a.collect_bundle(artifact_root=tmp_path, client=client or Client(), clock=clock, pacing_seconds=0)


def test_original_post_body_and_exact_receipts_round_trip(tmp_path, source):
    client = Client()
    path = capture(tmp_path, client)
    loaded = a.load_bundle(path)
    assert len(loaded["series"]) == 1
    assert not loaded["gaps"]
    assert client.calls[0][0] == "POST"
    assert client.calls[0][2]["json"] == source["requests"][0]["json"]
    assert loaded["series"][0]["available_at"] == loaded["available_at"]
    assert loaded["series"][0]["published_at"] is None


def test_newest_failed_whole_capture_does_not_fallback(tmp_path, source):
    old = capture(tmp_path)
    new = capture(tmp_path, Client(503), datetime(2026, 9, 12, tzinfo=UTC))
    result = a.load_evidence(bundle_paths=[old, new], as_known_at=datetime(2026, 9, 13, tzinfo=UTC))
    assert result["series"] == []
    assert result["gaps"][0]["reason"] == "source_error"
    prior = a.load_evidence(bundle_paths=[old, new], as_known_at=datetime(2026, 9, 11, 12, tzinfo=UTC))
    assert len(prior["series"]) == 1


def test_raw_tampering_and_native_request_change_fail(tmp_path, source):
    path = capture(tmp_path)
    batch = json.loads(path.read_bytes())
    raw = a.Path(batch["signals"][0]["requests"][0]["path"])
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        a.load_bundle(path)


def test_semantic_request_tamper_rehashed_envelope_still_fails(tmp_path, source):
    path = capture(tmp_path)
    batch = json.loads(path.read_bytes())
    batch["signals"][0]["requests"][0]["json"]["selection"] = "wrong"
    altered = a._archive(tmp_path, a.canonical(batch), bundle=True)
    with pytest.raises(ValueError, match="request identity"):
        a.load_bundle(altered)


def test_invalid_publisher_payload_is_explicit_gap(tmp_path, source):
    result = a.load_bundle(capture(tmp_path, Client(body=b"not JSON")))
    assert not result["series"]
    assert result["gaps"][0]["reason"] == "source_validation_error"


def test_future_capture_raw_corruption_does_not_affect_prior_cutoff(tmp_path, source):
    old = capture(tmp_path)
    future = capture(tmp_path, Client(body=b'[{"value": 4.0}]'), datetime(2026, 9, 12, tzinfo=UTC))
    raw = a.Path(json.loads(future.read_bytes())["signals"][0]["requests"][0]["path"])
    raw.write_bytes(b"bad")
    assert len(a.load_evidence(bundle_paths=[old, future],
                              as_known_at=datetime(2026, 9, 11, 12, tzinfo=UTC))["series"]) == 1


def test_naive_cutoff_and_symlink_rejected(tmp_path, source):
    path = capture(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        a.load_bundle(link)
    with pytest.raises(ValueError, match="timezone"):
        a.load_evidence(bundle_paths=[path], as_known_at=datetime(2026, 9, 12))


def test_request_clock_outside_batch_rejected(tmp_path, source):
    path = capture(tmp_path)
    batch = json.loads(path.read_bytes())
    batch["signals"][0]["requests"][0]["received_at"] = "2027-01-01T00:00:00+00:00"
    altered = a._archive(tmp_path, a.canonical(batch), bundle=True)
    with pytest.raises(ValueError, match="clock"):
        a.load_bundle(altered)


def test_no_eligible_batch_is_explicit_gap(tmp_path, source):
    path = capture(tmp_path)
    result = a.load_evidence(bundle_paths=[path], as_known_at=datetime(2026, 9, 10, tzinfo=UTC))
    assert not result["series"]
    assert result["gaps"][0]["reason"] == "no_bundle_as_known_at"


def test_publisher_update_window_retains_bytes_but_suppresses_placeholder_facts(tmp_path, source):
    source["capture_exclusion"] = {"timezone": "Europe/Oslo", "start_hour": 5, "end_hour": 8,
                                   "reason": "Temporary zeros during table update"}
    path = capture(tmp_path, start=datetime(2026, 9, 11, 4, tzinfo=UTC))
    result = a.load_bundle(path)
    assert not result["series"]
    assert "maintenance" in result["gaps"][0]["error"]
    assert len(result["protected_artifact_paths"]) == 2


def test_publisher_window_handles_dst_and_boundary_crossing(source):
    source["capture_exclusion"] = {"timezone": "Europe/Oslo", "start_hour": 5, "end_hour": 8,
                                   "reason": "Temporary zeros"}
    with pytest.raises(ValueError, match="maintenance"):
        a._check_capture_window(source, [{"requested_at": "2026-12-01T03:59:59+00:00",
                                         "received_at": "2026-12-01T04:00:01+00:00"}])
    a._check_capture_window(source, [{"requested_at": "2026-12-01T07:00:00+00:00",
                                      "received_at": "2026-12-01T07:00:01+00:00"}])
