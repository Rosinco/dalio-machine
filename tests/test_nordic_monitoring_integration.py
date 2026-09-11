"""All twelve official fixture selections pass the complete capture/replay path."""

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dalio.assessments.core import content_hash
from dalio.nordic_monitoring import acquisition
from dalio.nordic_monitoring.core import build_snapshot
from tests.test_nordic_monitoring_core import inputs  # noqa: F401 - shared fixture

FIX = Path(__file__).parent / "fixtures"


def body_for(spec, request):
    country, indicator, role = spec["country"], spec["indicator"], request["role"]
    if country == "NO":
        suffix = "json" if spec["kind"] == "ssb" else "xml" if role == "metadata" else "csv"
        return (FIX / "norway_monitoring" / f"{indicator}_{role}.{suffix}").read_bytes()
    if country == "FI" and indicator == "yield_10y":
        return gzip.decompress((FIX / "finland_yield_monitoring/benchmark_report.html.gz").read_bytes())
    name = {"industrial_production": "industry", "corporate_new_lending_rate": "lending",
            "policy_rate": "policy", "yield_10y": "yield"}[indicator]
    directory = "denmark_monitoring" if country == "DK" else "finland_monitoring"
    return (FIX / directory / f"{name}_{role}.bin").read_bytes()


def test_all_twelve_originals_survive_http_capture_and_offline_reparse(tmp_path, inputs, monkeypatch):  # noqa: F811
    # Replace the synthetic core-test catalogue with the complete production registry.
    from dalio.nordic_monitoring import core
    monkeypatch.setattr(core, "catalogue", acquisition.catalogue)
    current = datetime(2026, 9, 11, 12, tzinfo=UTC)
    expected = acquisition.catalogue(current.date())
    deliveries = {}
    for spec in expected:
        for request in spec["requests"]:
            identity = request["method"], request["url"], content_hash(request.get("json"))
            body = body_for(spec, request)
            if identity in deliveries:
                assert deliveries[identity] == body
            deliveries[identity] = body

    class Client:
        def request(self, method, url, **kwargs):
            body = deliveries[method, url, content_hash(kwargs.get("json"))]
            return type("Response", (), {"status_code": 200, "url": url, "content": body,
                                         "history": [], "headers": {}})()

    def clock():
        nonlocal current
        current += timedelta(seconds=1)
        return current

    path = acquisition.collect_bundle(artifact_root=tmp_path, client=Client(), clock=clock, pacing_seconds=0)
    captured = acquisition.load_bundle(path)
    assert not captured["gaps"]
    assert {(s["country"], s["indicator"]) for s in captured["series"]} == {
        (c, i) for c in ("NO", "DK", "FI") for i in acquisition.INDICATORS}
    evidence = acquisition.load_evidence(bundle_paths=[path], as_known_at=current)
    sweden, parent = inputs[1:]
    sweden["as_known_at"] = current.isoformat()
    sweden["evidence_digest"] = content_hash({k: v for k, v in sweden.items() if k != "evidence_digest"})
    parent["as_known_at"] = current.isoformat()
    parent["as_of"] = current.date().isoformat()
    parent["snapshot_sha256"] = content_hash({k: v for k, v in parent.items() if k != "snapshot_sha256"})
    result = build_snapshot(evidence, sweden, parent, as_of=current.date())
    assert [c["coverage"]["signals_source_bound"] for c in result["countries"]] == [5, 4, 4, 4]
    assert [c["coverage"]["signals_with_comparison"] for c in result["countries"]] == [5, 4, 4, 4]
    finland = next(c for c in result["countries"] if c["country"] == "FI")
    yield_ = next(s for s in finland["signals"] if s["indicator"] == "yield_10y")
    assert round(yield_["comparison"]["value"], 8) == 0.41
    assert yield_["latest"]["period"] == "2026-09-10"
    assert "not a separate literal unit field" in " ".join(yield_["limits"])
