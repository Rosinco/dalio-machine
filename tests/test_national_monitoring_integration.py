"""Complete original-source capture feeds native comparisons and a documented gap."""

from datetime import UTC, datetime, timedelta

from dalio.assessments.core import content_hash
from dalio.national_monitoring import acquisition, core
from tests.test_national_monitoring_core import inputs  # noqa: F401 - shared fixture


def fixture_body(spec, request):
    from tests.test_canada_monitoring import example as canada
    from tests.test_germany_monitoring import capture as germany
    from tests.test_us_monitoring import example as us
    return {"US": us, "DE": germany, "CA": canada}[spec["country"]](spec["indicator"])[1][request["role"]]


def test_complete_three_country_capture_preserves_native_measures_and_documented_gap(tmp_path, inputs, monkeypatch):  # noqa: F811
    monkeypatch.setattr(core, "catalogue", acquisition.catalogue)
    current = datetime(2026, 9, 11, 12, tzinfo=UTC)
    deliveries = {}
    for spec in acquisition.catalogue(current.date()):
        for request in spec["requests"]:
            key = request["method"], request["url"], content_hash(request.get("json"))
            body = fixture_body(spec, request)
            if key in deliveries:
                assert deliveries[key] == body
            deliveries[key] = body

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
    assert len(captured["series"]) == 11
    assert len(captured["gaps"]) == 1 and captured["gaps"][0]["reason"] == "structural_gap"
    assert captured["gaps"][0]["country"] == "US"
    parent = inputs[1]
    parent.update(as_known_at=current.isoformat(), as_of=current.date().isoformat())
    parent["snapshot_sha256"] = content_hash({k: v for k, v in parent.items() if k != "snapshot_sha256"})
    evidence = acquisition.load_evidence(bundle_paths=[path], as_known_at=current)
    result = core.build_snapshot(evidence, parent, as_of=current.date())
    assert [c["coverage"]["signals_source_bound"] for c in result["countries"]] == [3, 4, 4]
    assert [c["coverage"]["signals_with_comparison"] for c in result["countries"]] == [3, 4, 4]
    canadian = next(c for c in result["countries"] if c["country"] == "CA")
    industry = next(s for s in canadian["signals"] if s["indicator"] == "industrial_production")
    assert industry["latest"]["unit"] == "millions of chained 2017 Canadian dollars"
    assert industry["latest"]["value"] == 392804
    assert "value added" in industry["definition"]
