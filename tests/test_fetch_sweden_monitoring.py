"""Supplement CLI never initializes or mutates the live database."""

import json

from dalio.pipelines import fetch_sweden_monitoring


def test_cli_reports_retained_bundle_and_nonzero_on_source_gap(tmp_path, monkeypatch, capsys):
    calls = []
    path = tmp_path / ("a" * 64 + ".json")
    monkeypatch.setattr(fetch_sweden_monitoring, "collect_bundle",
                        lambda **kwargs: calls.append(kwargs) or path)
    monkeypatch.setattr(fetch_sweden_monitoring, "load_bundle", lambda _: {
        "bundle_path": str(path), "bundle_sha256": "a" * 64, "available_at": "2026-09-10T22:00:00+00:00",
        "series": [{"indicator": "policy_rate"}], "gaps": [{"indicator": "industrial_orders", "reason": "source_error"}]})
    assert fetch_sweden_monitoring.main(["--artifact-root", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert calls == [{"artifact_root": tmp_path}]
    assert report["bundle_path"] == str(path) and report["successful_series"] == 1
    assert report["requested_series"] == 5
