"""Offline evidence projection: identity, native dates and whole-country selection."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "country_evidence", Path(__file__).parents[1] / "scripts/export_country_evidence.py"
)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def seal(value, key="snapshot_sha256"):
    value[key] = model.sha(model.canonical({k: v for k, v in value.items() if k != key}))
    return value


def assessment():
    point = {
        "year": 2025,
        "value": 0,
        "unit": "%",
        "status": "estimate_or_outturn",
        "evidence_ref": "annual",
        "release_id": 1,
    }
    series = {
        "indicator": "real_gdp_growth",
        "release_id": 1,
        "series_id": "GDP",
        "source": "IMF_WEO",
        "source_url": "https://example.test/gdp",
        "available_at": "2026-09-10T20:00:00+00:00",
        "observations": [
            {
                "year": 2023,
                "date": "2023-12-31",
                "value": 1,
                "status": "estimate_or_outturn",
                "source": "IMF_WEO",
            },
            {
                "year": 2025,
                "date": "2025-12-31",
                "value": 0,
                "status": "estimate_or_outturn",
                "source": "IMF_WEO",
            },
        ],
    }
    profile = {
        "country": "SE",
        "name": "Sweden",
        "listing_iso2": "SE",
        "baseline": {"real_gdp_growth": point},
        "structural": {},
        "projections": [],
        "findings": [],
        "scenarios": [],
        "national_debt_context": [],
    }
    return seal(
        {
            "schema_version": 1,
            "as_of": "2026-09-10",
            "as_known_at": "2026-09-10T21:00:00+00:00",
            "baseline_year": 2025,
            "horizon_end_year": 2031,
            "methodology": {"metrics": {"real_gdp_growth": {"label": "Real growth", "unit": "%"}}},
            "countries": [profile],
            "citations": {"annual": {**point, "country": "SE"}},
            "source_evidence": {"countries": [{"country": "SE", "series": [series]}]},
        }
    )


def monitoring(known="2026-09-11T06:00:00+00:00", missing=False):
    row = {
        "date": "2026-09-10",
        "period": "2026-09-10",
        "value": None if missing else 2.0,
        "status": "not_reported" if missing else "observed",
    }
    point = {**row, "evidence_ref": "daily", "unit": "percent"}
    series = {
        "country": "SE",
        "indicator": "policy_rate",
        "series_id": "POLICY",
        "source": "Central bank",
        "source_url": "https://example.test/rate",
        "available_at": known,
        "frequency": "daily",
        "unit": "percent",
        "observations": [row],
    }
    signal = {
        "indicator": "policy_rate",
        "label": "Policy rate",
        "latest": point,
        "comparison": None,
        "status": "missing_latest" if missing else "no_comparison",
        "evidence_refs": ["daily"],
        "scenario_links": [],
        "source": {k: v for k, v in series.items() if k != "observations"},
        "reading": "No complete comparison.",
        "limits": [],
        "gaps": [],
    }
    return seal(
        {
            "schema_version": 1,
            "as_of": "2026-09-11",
            "as_known_at": known,
            "methodology": {},
            "countries": [
                {
                    "country": "SE",
                    "name": "Sweden",
                    "signals": [signal],
                    "scenarios": [],
                    "national_debt_context": [],
                    "coverage": {},
                }
            ],
            "citations": {
                "daily": {**point, **{k: v for k, v in series.items() if k != "observations"}}
            },
            "source_evidence": {"national": {"series": [series]}},
            "input_gaps": [],
            "remaining_gaps": [],
        }
    )


def test_projection_preserves_zero_missing_years_native_rows_and_separate_clocks():
    result = model.project(assessment(), [monitoring()])["SE"]
    assert result["assessment"]["profile"]["baseline"]["real_gdp_growth"]["value"] == 0
    assert [r["year"] for r in result["assessment"]["histories"][0]["observations"]] == [2023, 2025]
    assert result["assessment"]["as_of"] == "2026-09-10"
    assert result["monitoring"]["as_of"] == "2026-09-11"
    assert result["monitoring"]["histories"][0]["observations"][0]["date"] == "2026-09-10"


def test_latest_whole_country_keeps_gap_and_rejects_ambiguous_same_clock():
    old, new = monitoring(), monitoring("2026-09-11T07:00:00+00:00", missing=True)
    assert (
        model.project(assessment(), [new, old])["SE"]["monitoring"]["profile"]["signals"][0][
            "latest"
        ]["value"]
        is None
    )
    conflict = monitoring(missing=True)
    with pytest.raises(ValueError, match="same.*cutoff|ambiguous"):
        model.project(assessment(), [old, conflict])


def test_tampered_snapshots_cross_country_citations_and_missing_native_history_fail():
    changed = assessment()
    changed["countries"][0]["baseline"]["real_gdp_growth"]["value"] = 99
    with pytest.raises(ValueError, match="hash"):
        model.project(changed, [])
    changed = monitoring()
    changed["citations"]["daily"]["country"] = "FI"
    seal(changed)
    with pytest.raises(ValueError, match="citation|country"):
        model.project(assessment(), [changed])
    changed = monitoring()
    changed["source_evidence"]["national"]["series"] = []
    seal(changed)
    with pytest.raises(ValueError, match="history|source"):
        model.project(assessment(), [changed])


def test_no_monitoring_is_explicit_and_documented_gaps_are_retained():
    assert model.project(assessment(), [])["SE"]["monitoring"] is None
    changed = monitoring()
    changed["countries"][0]["signals"] = []
    changed["source_evidence"]["national"]["series"] = []
    changed["input_gaps"] = [
        {
            "country": "SE",
            "indicator": "credit",
            "reason": "not collected",
            "artifacts": [{"sha256": "a" * 64, "role": "documentation"}],
        }
    ]
    seal(changed)
    result = model.project(assessment(), [changed])["SE"]["monitoring"]
    assert result["input_gaps"] == changed["input_gaps"]


def test_newest_embedded_annual_profile_keeps_its_own_clock_and_provenance():
    base = assessment()
    parent = assessment()
    parent["as_of"] = "2026-09-11"
    parent["as_known_at"] = "2026-09-11T06:00:00+00:00"
    parent["countries"][0]["findings"] = [
        {"text": "New dated reading", "evidence_refs": ["annual"]}
    ]
    seal(parent)
    native = monitoring()
    native["country_assessment"] = parent
    seal(native)
    result = model.project(base, [native])["SE"]
    assert result["assessment"]["snapshot_sha256"] == parent["snapshot_sha256"]
    assert result["assessment"]["as_of"] == "2026-09-11"
    assert result["assessment"]["profile"]["findings"] == parent["countries"][0]["findings"]
    assert result["monitoring"]["snapshot_sha256"] == native["snapshot_sha256"]
    parent["as_known_at"] = "2026-09-11T07:00:00+00:00"
    seal(parent)
    seal(native)
    with pytest.raises(ValueError, match="after monitoring"):
        model.project(base, [native])


def test_atomic_export_replay_and_source_protection(tmp_path):
    source = tmp_path / "assessment.json"
    source.write_bytes(model.canonical(assessment()))
    monitor = tmp_path / "monitoring.json"
    monitor.write_bytes(model.canonical(monitoring()))
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [source, monitor]}
    target = tmp_path / "pack"
    index = model.export_pack(source, [monitor], target)
    assert model.verify_pack(target)["countries"] == 1
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.rglob("*") if p.is_file()}
    assert model.export_pack(source, [monitor], target) == index
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == old for p, old in {**original, **before}.items()
    )
    entry = json.loads(index.read_bytes())["countries"]["SE"]
    (target / entry["file"]).write_text("corrupt")
    with pytest.raises(ValueError, match="hash|immutable"):
        model.verify_pack(target)
    with pytest.raises(ValueError, match="protected|source"):
        model.export_pack(source, [], source)


def test_output_cannot_contain_input_index_or_embedded_artifact_index(tmp_path):
    source = tmp_path / "index.json"
    source.write_bytes(model.canonical(assessment()))
    before = source.read_bytes(), source.stat().st_mtime_ns
    with pytest.raises(ValueError, match="protected"):
        model.export_pack(source, [], tmp_path)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert list(tmp_path.iterdir()) == [source]

    artifact = tmp_path / "evidence/index.json"
    artifact.parent.mkdir()
    artifact.write_text("irreplaceable original response")
    doc = assessment()
    doc["source_evidence"]["countries"][0]["series"][0]["artifacts"] = [
        {"path": str(artifact), "role": "original"}
    ]
    source.write_bytes(model.canonical(seal(doc)))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (source, artifact)}
    with pytest.raises(ValueError, match="protected"):
        model.export_pack(source, [], artifact.parent)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    assert list(artifact.parent.iterdir()) == [artifact]


def test_protected_artifact_namespaces_and_symlink_destinations_are_rejected(tmp_path):
    artifact = tmp_path / "artifacts/responses/source.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("retained bytes")
    source = tmp_path / "assessment.json"
    doc = assessment()
    doc["protected_artifact_paths"] = [str(artifact)]
    source.write_bytes(model.canonical(seal(doc)))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (source, artifact)}
    target = tmp_path / "artifacts/unused-export"
    with pytest.raises(ValueError, match="namespace"):
        model.export_pack(source, [], target)
    assert not target.exists()
    alias = tmp_path / "alias"
    alias.symlink_to(artifact.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        model.export_pack(source, [], alias / "export")
    assert not (artifact.parent / "export").exists()
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
