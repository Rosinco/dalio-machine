"""Read-only source access and deterministic, atomic country assessment export."""

import copy
import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from dalio.assessments.render import render_country, render_index
from dalio.pipelines import build_country_assessments as export

KNOWN = datetime(2026, 9, 10, 12, tzinfo=UTC)
AS_OF = date(2026, 9, 10)


def _hash(snapshot):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in snapshot.items() if k != "snapshot_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _snapshot(protected=None):
    countries, citations = [], {}
    for code, iso, name in (("SE", "SE", "Sweden"), ("UK", "GB", "United Kingdom")):
        ref = f"{code}-debt-2025"
        point = {
            "value": 35.1,
            "year": 2025,
            "unit": "% GDP",
            "status": "estimate",
            "evidence_ref": ref,
            "source_url": "https://www.imf.org/data/weo/2026-04",
            "source": "IMF WEO",
            "release_id": 42,
            "available_at": KNOWN.isoformat(),
            "series_id": "GGXWDG_NGDP",
        }
        citations[ref] = {"country": code, "indicator": "govt_debt", **point}
        countries.append(
            {
                "country": code,
                "listing_iso2": iso,
                "name": name,
                "listing_count": 5,
                "status": "partial",
                "coverage": {"core_available": 1, "core_expected": 2},
                "baseline": {"govt_debt": point, "inflation": None},
                "projections": [{"year": 2027, "metrics": {"govt_debt": None}}],
                "structural": {},
                "findings": [
                    {
                        "id": "debt-level",
                        "title": "Debt baseline",
                        "kind": "observation",
                        "text": "The publisher estimates debt at 35.1% of GDP.",
                        "evidence_refs": [ref],
                        "limits": ["Publisher estimate, not final outturn."],
                    }
                ],
                "scenarios": [
                    {
                        "id": "funding-pressure",
                        "title": "Funding pressure",
                        "kind": "scenario_hypothesis",
                        "horizon": "1–3 years",
                        "assumptions": ["Funding costs remain elevated."],
                        "pathway": ["Interest expense rises", "Budget flexibility falls"],
                        "signposts": [
                            {
                                "text": "To monitor: financing cost revisions.",
                                "evidence_refs": [ref],
                            }
                        ],
                        "invalidators": ["Debt-service costs decline."],
                        "evidence_refs": [ref],
                        "company_checks": ["Check actual foreign-currency debt exposure."],
                        "limitations": ["No company exposure is established by domicile alone."],
                    }
                ],
                "gaps": ["Inflation baseline unavailable."],
            }
        )
    result = {
        "schema_version": 1,
        "as_of": AS_OF.isoformat(),
        "as_known_at": KNOWN.isoformat(),
        "baseline_year": 2025,
        "horizon_end_year": 2031,
        "methodology": {"version": "test-v1"},
        "countries": countries,
        "citations": citations,
        "full_source_evidence": {},
        "protected_artifact_paths": list(protected or []),
    }
    result["snapshot_sha256"] = _hash(result)
    return result


@pytest.fixture
def setup_export(monkeypatch, tmp_path):
    db = tmp_path / "source.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE sentinel (id INTEGER)")
        con.execute("INSERT INTO sentinel VALUES (1)")
    artifact = tmp_path / "evidence" / "source.json"
    artifact.parent.mkdir()
    artifact.write_text('{"original":true}')
    snapshot = _snapshot([str(artifact)])

    def load(engine, *, as_known_at, countries):
        assert as_known_at == KNOWN
        with engine.connect() as con, pytest.raises(OperationalError, match="readonly"):
            con.exec_driver_sql("INSERT INTO sentinel VALUES (2)")
        return {"protected_artifact_paths": [str(artifact)]}

    loader = MagicMock(side_effect=load)
    builder = MagicMock(side_effect=lambda evidence, *, as_of: copy.deepcopy(snapshot))
    monkeypatch.setattr(export, "load_evidence", loader)
    monkeypatch.setattr(export, "build_snapshot", builder)
    return db, artifact, snapshot, loader, builder


def test_country_markdown_preserves_estimates_sources_gaps_and_conditional_scenarios():
    snapshot = _snapshot()
    text = render_country(snapshot["countries"][0], snapshot)
    assert "35.1" in text and "Estimate" in text
    assert text.index("## At a glance") < text.index("## Source baseline")
    assert "The publisher estimates debt at 35.1% of GDP." in text.split("## Source baseline")[0]
    assert "https://www.imf.org/data/weo/2026-04" in text
    assert "42" in text and "GGXWDG_NGDP" in text
    assert "Inflation baseline unavailable." in text
    assert "Scenario hypothesis" in text
    assert "To monitor: financing cost revisions." in text
    assert "Check actual foreign-currency debt exposure." in text
    assert "No company exposure is established by domicile alone." in text
    assert "2027" in text and "Not available" in text
    assert "probability: 0" not in text.lower()
    index = render_index(snapshot)
    assert "countries/UK.md" in index and "countries/GB.md" not in index
    assert "2025" in index and "Estimate" in index


def test_full_publish_is_read_only_complete_and_identical_on_repeat(setup_export, tmp_path):
    db, artifact, snapshot, loader, _ = setup_export
    original_db, original_artifact = db.read_bytes(), artifact.read_bytes()
    output = tmp_path / "assessments"
    result = export.export_assessments(
        db_path=db, as_of=AS_OF, as_known_at=KNOWN, countries=["SE", "GB"], output_dir=output
    )
    assert result == output / snapshot["snapshot_sha256"]
    assert {str(p.relative_to(result)) for p in result.rglob("*") if p.is_file()} == {
        "snapshot.json",
        "index.md",
        "countries/SE.md",
        "countries/UK.md",
    }
    assert json.loads((result / "snapshot.json").read_text()) == snapshot
    pointer = json.loads((output / "LATEST.json").read_text())
    assert pointer["snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert loader.call_args.kwargs["countries"] == ["SE", "UK"]
    before = {
        str(p.relative_to(output)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert (
        export.export_assessments(
            db_path=db, as_of=AS_OF, as_known_at=KNOWN, countries=["SE", "GB"], output_dir=output
        )
        == result
    )
    after = {
        str(p.relative_to(output)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert before == after
    assert db.read_bytes() == original_db and artifact.read_bytes() == original_artifact


def test_missing_database_does_not_create_database_or_output(tmp_path):
    db, output = tmp_path / "absent.sqlite", tmp_path / "reports"
    with pytest.raises(ValueError, match="database"):
        export.export_assessments(db_path=db, output_dir=output)
    assert not db.exists() and not output.exists()


def test_output_cannot_overwrite_database_or_verified_artifact(setup_export):
    db, artifact, _, _, _ = setup_export
    for output in (db, artifact):
        before = output.read_bytes()
        with pytest.raises(ValueError, match="output"):
            export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
        assert output.read_bytes() == before


def test_invalid_hash_or_unsafe_country_stops_before_any_publication(setup_export, tmp_path):
    db, _, snapshot, _, builder = setup_export
    for change in (
        lambda s: s.update(snapshot_sha256="a" * 64),
        lambda s: s["countries"][0].update(country="../SE"),
    ):
        broken = copy.deepcopy(snapshot)
        change(broken)
        if broken["countries"][0]["country"].startswith("../"):
            broken["snapshot_sha256"] = _hash(broken)
        builder.side_effect = None
        builder.return_value = broken
        output = tmp_path / "unpublished"
        with pytest.raises(ValueError):
            export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
        assert not output.exists()


def test_render_failure_preserves_previous_latest_and_all_existing_files(
    setup_export, monkeypatch, tmp_path
):
    db, _, snapshot, _, builder = setup_export
    output = tmp_path / "reports"
    export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    previous = {
        str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    changed = copy.deepcopy(snapshot)
    changed["methodology"]["version"] = "test-v2"
    changed["snapshot_sha256"] = _hash(changed)
    builder.side_effect = None
    builder.return_value = changed
    monkeypatch.setattr(
        export,
        "render_country",
        MagicMock(side_effect=ValueError("Cannot render incomplete evidence")),
    )
    with pytest.raises(ValueError, match="Cannot render"):
        export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    assert previous == {
        str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }


def test_mid_write_failure_never_publishes_partial_directory_or_latest(
    setup_export, monkeypatch, tmp_path
):
    db, _, snapshot, _, _ = setup_export
    output = tmp_path / "reports"
    original_write = export._write_file
    calls = 0

    def fail_second(path, body):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        original_write(path, body)

    monkeypatch.setattr(export, "_write_file", fail_second)
    with pytest.raises(OSError, match="disk full"):
        export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    assert not (output / snapshot["snapshot_sha256"]).exists()
    assert not (output / "LATEST.json").exists()
    assert not list(output.glob(".build-*"))


def test_modified_existing_snapshot_is_not_repaired_or_republished(setup_export, tmp_path):
    db, _, _, _, _ = setup_export
    output = tmp_path / "reports"
    directory = export.export_assessments(
        db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output
    )
    index = directory / "index.md"
    index.write_text("locally changed")
    pointer = (output / "LATEST.json").read_bytes()
    with pytest.raises(ValueError, match="existing"):
        export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    assert index.read_text() == "locally changed"
    assert (output / "LATEST.json").read_bytes() == pointer


def test_naive_known_at_and_duplicate_country_names_are_rejected(setup_export, tmp_path):
    db, _, _, loader, _ = setup_export
    with pytest.raises(ValueError, match="timezone"):
        export.export_assessments(
            db_path=db,
            as_of=AS_OF,
            as_known_at=KNOWN.replace(tzinfo=None),
            output_dir=tmp_path / "reports",
        )
    loader.assert_not_called()
    with pytest.raises(ValueError, match="Duplicate"):
        export.export_assessments(
            db_path=db,
            as_of=AS_OF,
            as_known_at=KNOWN,
            countries=["GB", "UK"],
            output_dir=tmp_path / "reports",
        )


def test_cli_uses_explicit_arguments_and_optional_latest(setup_export, tmp_path, capsys):
    db, _, snapshot, _, _ = setup_export
    output = tmp_path / "reports"
    assert (
        export.main(
            [
                "--db",
                str(db),
                "--as-of",
                "2026-09-10",
                "--known-at",
                KNOWN.isoformat(),
                "--countries",
                "SE,GB",
                "--output-dir",
                str(output),
                "--no-latest",
            ]
        )
        == 0
    )
    assert snapshot["snapshot_sha256"] in capsys.readouterr().out
    assert not (output / "LATEST.json").exists()


def test_missing_citation_and_infinite_source_value_cannot_be_published(setup_export, tmp_path):
    db, _, snapshot, _, builder = setup_export
    broken = copy.deepcopy(snapshot)
    del broken["citations"]["SE-debt-2025"]
    broken["snapshot_sha256"] = _hash(broken)
    builder.side_effect = None
    builder.return_value = broken
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="evidence reference"):
        export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    assert not output.exists()
    broken = copy.deepcopy(snapshot)
    broken["countries"][0]["baseline"]["govt_debt"]["value"] = float("inf")
    builder.return_value = broken
    with pytest.raises(ValueError):
        export.export_assessments(db_path=db, as_of=AS_OF, as_known_at=KNOWN, output_dir=output)
    assert not output.exists()


def test_publisher_forecast_and_stale_history_labels_remain_explicit():
    snapshot = _snapshot()
    country = snapshot["countries"][0]
    ref = "SE-debt-2027"
    point = {
        **country["baseline"]["govt_debt"],
        "evidence_ref": ref,
        "year": 2027,
        "value": 38.96,
        "status": "forecast",
    }
    snapshot["citations"][ref] = point
    country["projections"][0]["metrics"]["govt_debt"] = point
    country["structural"]["dependency_ratio"] = {
        **country["baseline"]["govt_debt"],
        "stale": True,
        "age_years": 4,
    }
    text = render_country(country, snapshot)
    assert "38.96" in text and "Forecast" in text
    assert "stale (4 years old)" in text


def test_native_issuer_amounts_and_refixing_stay_outside_general_debt_baseline():
    snapshot = _snapshot()
    point = {
        "metric": "gross_borrowing_requirement",
        "native_label": "Gross borrowing requirement",
        "value": 655.6,
        "unit": "SEK_bn",
        "period_start": "2026-01-01",
        "period_end": "2026-12-31",
        "status": "forecast",
        "source": "RIKSGALDEN_DEBT",
        "release_id": 314,
        "source_url": "https://www.riksgalden.se/data.xlsx",
        "available_at": KNOWN.isoformat(),
        "source_locator": "XLSX F10!F11",
        "evidence_ref": "SE-native-314",
        "dimensions": {"scope": "central_government", "aggregation": "calendar_year"},
    }
    snapshot["countries"][0]["national_debt_context"] = [point]
    snapshot["citations"][point["evidence_ref"]] = point
    text = render_country(snapshot["countries"][0], snapshot)
    before, native = text.split("## Original national debt-office evidence", 1)
    assert "35.1 % GDP" in before and "655.6" not in before
    assert "655.6 billion SEK" in native and "2026-01-01 to 2026-12-31" in native
    assert point["unit"] == "SEK_bn"
    assert "central government; calendar year" in native and "Forecast" in native
    assert "XLSX F10!F11" in native and "314" in native
    for aggregation, value in (("monthly_mean", 4.85), ("reference_date", 5.05)):
        atr = {
            **point,
            "metric": "average_time_to_refixing",
            "native_label": "total",
            "value": value,
            "unit": "years",
            "status": "observed",
            "evidence_ref": "SE-ATR-" + aggregation,
            "dimensions": {"scope": "central_government", "aggregation": aggregation},
        }
        snapshot["countries"][0]["national_debt_context"].append(atr)
        snapshot["citations"][atr["evidence_ref"]] = atr
    text = render_country(snapshot["countries"][0], snapshot)
    assert "4.85 years" in text and "monthly mean" in text
    assert "5.05 years" in text and "reference date" in text
    assert "Average time to refixing — total" in text


def test_imf_status_convention_is_explained_without_claiming_native_point_status():
    snapshot = _snapshot()
    for text in (render_country(snapshot["countries"][0], snapshot), render_index(snapshot)):
        assert "IMF DataMapper does not supply a native per-point status" in text
        assert "estimate/outturn" in text
        assert "current and future years" in text
        assert "documented calendar convention" in text
        assert "publisher's status" not in text
        assert "publisher status" not in text
        assert "Publisher statuses distinguish" not in text


def test_whole_sek_amounts_are_grouped_and_refixing_explained_without_json_changes():
    snapshot = _snapshot()
    point = {
        "metric": "central_gov_gross_debt_sek",
        "native_label": "Central government gross debt",
        "value": 1249558872203.0,
        "unit": "SEK",
        "period_start": "2026-08-31",
        "period_end": "2026-08-31",
        "status": "observed",
        "source": "RIKSGALDEN_DEBT",
        "release_id": 314,
        "source_url": "https://www.riksgalden.se/data.pdf",
        "available_at": KNOWN.isoformat(),
        "source_locator": "PDF page 1",
        "evidence_ref": "SE-native-debt",
        "dimensions": {"scope": "central_government", "aggregation": "reference_date"},
    }
    snapshot["countries"][0]["national_debt_context"] = [point]
    snapshot["citations"][point["evidence_ref"]] = point
    original = copy.deepcopy(snapshot)
    text = render_country(snapshot["countries"][0], snapshot)
    assert "1,249,558,872,203 SEK" in text
    assert "interest-rate resets" in text and "principal repayment dates" in text
    assert "derivative and accounting bases" in text
    assert snapshot == original
