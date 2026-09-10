"""Immutable five-signal supplement boundaries; HTTP is always mocked."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dalio.monitoring import acquisition

NOW = datetime(2026, 9, 10, 22, tzinfo=UTC)
SCB_KEYS = ("industrial_production", "industrial_orders", "corporate_new_lending_rate")


@pytest.fixture(autouse=True)
def scb_contract(monkeypatch):
    specs = [SimpleNamespace(key=key, metadata_url=f"https://api.scb.se/{key}/metadata",
        source_url=f"https://api.scb.se/{key}/data", request={"selection": key}) for key in SCB_KEYS]
    monkeypatch.setattr(acquisition, "_scb_specs", lambda: specs)
    def parse(key, metadata, data, request):
        assert request == {"selection": key}
        if json.loads(data).get("invalid"):
            raise ValueError("wrong native selection")
        return dict(indicator=key, source="SCB_MONITORING", series_id=key, unit="index",
            frequency="monthly", adjustment="seasonally_adjusted", definition="Native definition",
            publisher_metadata=json.loads(metadata), observations=[
                dict(date="2026-06-30", period_start="2026-06-01", period_end="2026-06-30",
                     value=100.0, status="observed", native_status=None),
                dict(date="2026-07-31", period_start="2026-07-01", period_end="2026-07-31",
                     value=None, status="not_reported", native_status=None)], missingness={"nulls": 1})
    monkeypatch.setattr(acquisition, "_scb_parse", parse)


class Client:
    def __init__(self, *, fail=None, redirect=None, invalid=None):
        self.fail, self.redirect, self.invalid, self.calls = fail, redirect, invalid, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail and self.fail in url:
            raise OSError("synthetic source unavailable")
        if "/Observations/" in url:
            body = [{"date": "2026-09-09", "value": 1.75},
                    {"date": "2026-09-10", "value": 1.75}]
        else:
            body = {"invalid": bool(self.invalid and self.invalid in url)}
        return SimpleNamespace(status_code=200, content=json.dumps(body).encode(),
            url="https://third-party.invalid/data" if self.redirect and self.redirect in url else url,
            history=[], headers={"Content-Type": "application/json"})


def collect(tmp_path, **kwargs):
    return acquisition.collect_bundle(artifact_root=tmp_path, client=Client(**kwargs),
                                      clock=lambda: NOW, pacing_seconds=0)


def rewritten(path, change):
    data = json.loads(path.read_bytes())
    change(data)
    body = acquisition.canonical(data)
    target = path.parent / f"{hashlib.sha256(body).hexdigest()}.json"
    target.write_bytes(body)
    return target


def test_complete_capture_retains_exact_requests_raw_bytes_nulls_and_replays(tmp_path):
    client = Client()
    path = acquisition.collect_bundle(artifact_root=tmp_path, client=client,
                                      clock=lambda: NOW, pacing_seconds=0)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    result = acquisition.load_bundle(path)
    assert len(client.calls) == 8
    assert all(call[1]["allow_redirects"] is False for call in client.calls)
    assert len(result["series"]) == 5 and result["gaps"] == []
    assert result["available_at"] == NOW.isoformat()
    assert result["series"][0]["observations"][-1]["value"] is None
    assert all(s["available_at"] == NOW.isoformat() for s in result["series"])
    assert all(s["published_at"] is None for s in result["series"])
    assert len(result["protected_artifact_paths"]) == len(before)
    assert acquisition.load_bundle(path) == result
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())


def test_http_failure_is_explicit_gap_in_complete_attempt_set(tmp_path):
    path = collect(tmp_path, fail="industrial_orders/data")
    result = acquisition.load_bundle(path)
    assert len(json.loads(path.read_bytes())["signals"]) == 5
    assert len(result["series"]) == 4
    assert result["gaps"][0]["indicator"] == "industrial_orders"
    assert result["gaps"][0]["reason"] == "source_error"


def test_nonofficial_delivery_never_becomes_calculation_evidence(tmp_path):
    result = acquisition.load_bundle(collect(tmp_path, redirect="SECBREPOEFF"))
    assert "policy_rate" not in {s["indicator"] for s in result["series"]}
    assert any(g["indicator"] == "policy_rate" and g["reason"] == "source_error" for g in result["gaps"])


def test_native_parser_failure_remains_gap_without_dropping_attempt(tmp_path):
    result = acquisition.load_bundle(collect(tmp_path, invalid="industrial_orders/data"))
    assert any(g["indicator"] == "industrial_orders" and g["reason"] == "source_validation_error"
               for g in result["gaps"])


@pytest.mark.parametrize("mutation", [
    lambda b: b["signals"].pop(),
    lambda b: b["signals"].append(b["signals"][0]),
    lambda b: b["signals"][0]["requests"][0].update(id="wrong"),
    lambda b: b["signals"][0]["requests"][0].update(url="https://third-party.invalid/data"),
    lambda b: b["signals"][0].update(query={"selection": "wrong"}),
    lambda b: b.update(available_at=(NOW - timedelta(seconds=1)).isoformat()),
    lambda b: b["signals"][0]["requests"][0].update(received_at=(NOW + timedelta(seconds=1)).isoformat()),
    lambda b: b.update(available_at="2026-09-10T22:00:00"),
])
def test_rehashed_invalid_envelope_rejected(tmp_path, mutation):
    with pytest.raises(ValueError):
        acquisition.load_bundle(rewritten(collect(tmp_path), mutation))


def test_changed_raw_artifact_fails_closed(tmp_path):
    path = collect(tmp_path)
    request = json.loads(path.read_bytes())["signals"][0]["requests"][0]
    Path(request["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        acquisition.load_bundle(path)


def test_bundle_itself_requires_exact_content_address(tmp_path):
    path = collect(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        acquisition.load_bundle(path)


def test_bundle_symlink_is_rejected_before_resolution(tmp_path):
    path = collect(tmp_path / "capture")
    alias = tmp_path / path.name
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        acquisition.load_bundle(alias)


def test_exact_capture_replay_preserves_content_addresses_and_modification_times(tmp_path):
    first = collect(tmp_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob("*") if path.is_file()}
    assert collect(tmp_path) == first
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == state for path, state in before.items())


def test_publisher_update_later_than_capture_cannot_be_backdated(tmp_path, monkeypatch):
    parser = acquisition._scb_parse
    def future(*args):
        return dict(parser(*args), publisher_updated_at=(NOW + timedelta(seconds=1)).isoformat())
    monkeypatch.setattr(acquisition, "_scb_parse", future)
    result = acquisition.load_bundle(collect(tmp_path))
    assert len(result["series"]) == 2
    assert {gap["indicator"] for gap in result["gaps"]} == set(SCB_KEYS)
    assert all(gap["reason"] == "source_validation_error" for gap in result["gaps"])


def test_swea_ambiguous_json_and_boolean_values_are_not_numeric_evidence():
    for body in (b'[{"date":"2026-09-10","value":1,"value":2}]',
                 b'[{"date":"2026-09-10","value":true}]'):
        with pytest.raises(ValueError):
            acquisition._parse_riksbank("policy_rate", body, NOW.date())


def test_collector_rejects_clock_regression(tmp_path):
    ticks = iter([NOW, NOW - timedelta(seconds=1)])
    with pytest.raises(ValueError, match="clock"):
        acquisition.collect_bundle(artifact_root=tmp_path, client=Client(),
                                   clock=lambda: next(ticks), pacing_seconds=0)


def test_original_scb_fixtures_integrate_without_losing_definitions_or_null_slots(tmp_path, monkeypatch):
    from dalio.data_sources.scb_monitoring import SCB_MONITORING_SPECS, parse_scb_input
    monkeypatch.setattr(acquisition, "_scb_specs", lambda: SCB_MONITORING_SPECS)
    monkeypatch.setattr(acquisition, "_scb_parse", parse_scb_input)
    fixtures = Path(__file__).parent / "fixtures" / "scb_monitoring"
    responses = {}
    for spec in SCB_MONITORING_SPECS:
        responses[spec.metadata_url] = (fixtures / f"{spec.key}_metadata.json").read_bytes()
        responses[spec.source_url] = (fixtures / f"{spec.key}_response.json").read_bytes()
    class Originals(Client):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url in responses:
                response.content = responses[url]
            return response
    bundle = acquisition.collect_bundle(artifact_root=tmp_path, client=Originals(),
                                         clock=lambda: NOW, pacing_seconds=0)
    checked = acquisition.load_bundle(bundle)
    assert checked["gaps"] == [] and len(checked["series"]) == 5
    daily = [item for item in checked["series"] if item["frequency"] == "daily"]
    assert len(daily) == 2
    assert all(row["period"] == row["date"] == row["period_end"]
               for item in daily for row in item["observations"])
    for spec, latest, count in zip(SCB_MONITORING_SPECS, (108.9, 99.9, 3.5912), (319, 319, 325), strict=True):
        series = next(item for item in checked["series"] if item["indicator"] == spec.key)
        assert series["source_url"] == spec.source_url and series["series_id"] == spec.series_id
        assert len(series["observations"]) == count and series["observations"][-1]["value"] == latest
        assert series["observations"][-1]["date"] == "2026-07-31"
        assert series["publisher_metadata"]["native_dimension_metadata"]
