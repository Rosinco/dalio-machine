"""First-hand national debt context preserves exact availability and native scope."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.assessments.debt_evidence import load_national_debt_context
from dalio.assessments.evidence import load_evidence
from dalio.storage import national_debt
from dalio.storage.db import DataRelease, NationalDebtFact, make_engine
from tests.test_national_debt_storage import Document

KNOWN = datetime(2026, 9, 10, 10, tzinfo=UTC)
MONTHLY = "se_central_government_debt_monthly_report"
FUNDING = "se_central_government_funding_plan"


@pytest.fixture(autouse=True)
def synthetic_native_parser(monkeypatch):
    def reparse(doc):
        facts = json.loads(doc.source_bytes)
        for fact in facts:
            fact["period_start"] = date.fromisoformat(fact["period_start"])
            fact["period_end"] = date.fromisoformat(fact["period_end"])
        return replace(doc, facts=tuple(facts))
    monkeypatch.setattr(national_debt, "_reparse_document", reparse)
    monkeypatch.setattr(national_debt, "_construct_document", lambda fields: Document(**fields))


def fact(metric, value, *, year=2026, month=8, status="observed", unit="SEK", dims=None,
         kind="debt_stock", locator="PDF page 1 headline", start=None, end=None):
    return dict(fact_type=kind, metric=metric, value=value, unit=unit,
        period_start=start or date(year, month, 1), period_end=end or date(year, month, 28),
        status=status, dimensions=dims or {"scope": "central_government", "aggregation": "reference_date"},
        source_locator=locator, native_label=metric, native_value=str(value))


def monthly(*, month=8, gross=100.0):
    ref = date(2026, month, 28)
    facts = [fact("central_gov_gross_debt_sek", gross, month=month, start=ref, end=ref)]
    for aggregation, value in (("monthly_mean", 5.2), ("reference_date", 4.8)):
        facts.append(fact("average_time_to_refixing", value, month=month, unit="years",
            kind="portfolio_risk", locator=f"PDF total {aggregation}",
            dims={"debt_class": "total", "aggregation": aggregation,
                  "scope": "central_government_including_on_lending_and_assets"}))
    facts.append(fact("average_time_to_refixing", 99, month=month, unit="years",
        kind="portfolio_risk", locator="PDF one class",
        dims={"debt_class": "Nominal krona debt", "aggregation": "monthly_mean"}))
    facts.append(fact("debt_including_on_lending_and_assets_sek", 150, month=month))
    return document(MONTHLY, ref.isoformat(), ref, date(2026, month + 1, 7), facts)


def funding():
    facts = []
    for year in (2025, 2026, 2027):
        for metric, column, value in (("gross_borrowing_requirement", "F", 300.0),
                                      ("redemptions", "C", 100.0)):
            facts.append(fact(metric, value + year - 2026, year=year,
                start=date(year, 1, 1), end=date(year, 12, 31), unit="SEK_bn",
                kind="funding_flow", status="forecast" if year >= 2026 else "observed",
                dims={"forecast_vintage": "2026-1", "native_column": column,
                      "scope": "central_government", "aggregation": "calendar_year"},
                locator=f"XLSX F10!{column}{year - 2015}"))
    facts.append(fact("gross_borrowing_requirement", 999, year=2027, unit="SEK_bn",
        kind="funding_flow", status="forecast", start=date(2027, 1, 1), end=date(2027, 12, 31),
        dims={"forecast_vintage": "2025-2", "native_column": "G"}, locator="XLSX F10!G12"))
    facts.append(fact("debt_outstanding", 9999, year=2027, unit="SEK_bn",
        kind="funding_plan_stock", status="forecast", start=date(2027, 1, 1), end=date(2027, 12, 31),
        dims={"forecast_vintage": "2026-1"}, locator="XLSX F11!K12"))
    return document(FUNDING, "2026-1", date(2026, 5, 13), date(2026, 5, 28), facts)


def document(stream, snapshot, reference, published, facts):
    body = json.dumps(facts, default=str, sort_keys=True).encode()
    return Document(stream, snapshot, "SE", f"https://www.riksgalden.se/contentassets/test/{snapshot}.pdf",
        datetime.combine(published, datetime.min.time(), UTC), reference, body, tuple(facts),
        {"publisher": "Swedish National Debt Office", "publication_precision": "date"})


def store(engine, tmp_path, docs, when):
    batch = national_debt.prepare_native_batch(docs, artifact_root=tmp_path, retrieved_at=when)
    national_debt.ingest_native_batch(batch, engine=engine)
    return batch


def test_missing_native_database_returns_empty_context_without_creating_it(tmp_path):
    path = tmp_path / "absent.db"
    result = load_national_debt_context(make_engine(path), as_known_at=KNOWN, countries=["SE", "BE"])
    assert result == {"contexts": {"SE": [], "BE": []}, "protected_artifact_paths": []}
    assert not path.exists()


def test_exact_instant_selects_eligible_older_revision_not_end_of_day_latest(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    store(engine, tmp_path / "first", [monthly(gross=100)], KNOWN - timedelta(hours=1))
    future = store(engine, tmp_path / "later", [monthly(gross=200)], KNOWN + timedelta(hours=1))
    source = next(a for a in future[0].artifacts if a.role == "source_response")
    Path(source.artifact_path).write_bytes(b"corrupt future source")
    result = load_national_debt_context(engine, as_known_at=KNOWN, countries=["SE"])
    gross = next(f for f in result["contexts"]["SE"] if f["metric"] == "central_gov_gross_debt_sek")
    assert gross["value"] == 100
    assert gross["available_at"] == (KNOWN - timedelta(hours=1)).isoformat()
    assert source.artifact_path not in result["protected_artifact_paths"]
    with pytest.raises(ValueError):
        load_national_debt_context(engine, as_known_at=KNOWN + timedelta(hours=1), countries=["SE"])


def test_latest_eligible_monthly_reference_and_bounded_native_facts(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    batch = store(engine, tmp_path / "artifacts", [monthly(month=7), monthly(), funding()], KNOWN)
    result = load_national_debt_context(engine, as_known_at=KNOWN, countries=["SE", "BE"])
    facts = result["contexts"]["SE"]
    assert len(facts) == 7
    assert result["contexts"]["BE"] == []
    gross = next(f for f in facts if f["metric"] == "central_gov_gross_debt_sek")
    assert gross["period_end"] == "2026-08-28"
    assert gross["unit"] == "SEK"
    atr = [f for f in facts if f["metric"] == "average_time_to_refixing"]
    assert {f["dimensions"]["aggregation"] for f in atr} == {"monthly_mean", "reference_date"}
    assert {f["dimensions"]["debt_class"] for f in atr} == {"total"}
    plan = [f for f in facts if f["status"] == "forecast"]
    assert len(plan) == 4 and {f["year"] for f in plan} == {2026, 2027}
    assert {f["dimensions"]["forecast_vintage"] for f in plan} == {"2026-1"}
    assert {f["unit"] for f in plan} == {"SEK_bn"}
    assert all(f["source_locator"].startswith("XLSX F10!") for f in plan)
    selected_paths = {a.artifact_path for item in batch[1:] for a in item.artifacts}
    assert set(result["protected_artifact_paths"]) == selected_paths
    assert all(f["evidence_ref"].startswith(f"native:r{f['release_id']}:") for f in facts)
    assert len({f["evidence_ref"] for f in facts}) == 7
    assert json.loads(json.dumps(result)) == result


def test_funding_forecast_year_filter_uses_known_at_year(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    store(engine, tmp_path / "artifacts", [funding()], KNOWN)
    result = load_national_debt_context(engine, as_known_at=datetime(2027, 1, 1, tzinfo=UTC), countries=["SE"])
    assert len(result["contexts"]["SE"]) == 2
    assert {f["year"] for f in result["contexts"]["SE"]} == {2027}


def test_latest_funding_publication_wins_over_more_recent_old_document_capture(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    original = funding()
    updated_facts = []
    for row in original.facts:
        row = dict(row, dimensions=dict(row["dimensions"]))
        if row["dimensions"].get("forecast_vintage") == "2026-1":
            row["dimensions"]["forecast_vintage"] = "2026-2"
        updated_facts.append(row)
    updated = document(FUNDING, "2026-2", date(2026, 9, 6), date(2026, 9, 8), updated_facts)
    store(engine, tmp_path / "newer", [updated], KNOWN - timedelta(hours=2))
    store(engine, tmp_path / "older", [original], KNOWN - timedelta(hours=1))
    result = load_national_debt_context(engine, as_known_at=KNOWN, countries=["SE"])
    assert len(result["contexts"]["SE"]) == 4
    assert {row["publisher_metadata"]["snapshot_key"] for row in result["contexts"]["SE"]} == {"2026-2"}


def test_future_monthly_reference_is_excluded_before_source_restoration(tmp_path):
    from dalio.storage.db import init_db
    engine = make_engine(tmp_path / "native.db")
    init_db(engine)
    with Session(engine) as session:
        session.add(DataRelease(partition_key=f"{national_debt.PREFIX}{MONTHLY}:2027-01-31",
            source_family=national_debt.SOURCE, available_at=KNOWN, retrieved_at=KNOWN,
            published_at=datetime(2027, 2, 1, tzinfo=UTC), vintage_label="future-unverified",
            source_url="https://www.riksgalden.se/future.pdf", content_sha256="0" * 64, row_count=0))
        session.commit()
    assert load_national_debt_context(engine, as_known_at=KNOWN, countries=["SE"])["contexts"]["SE"] == []


def test_full_selected_native_snapshot_corruption_fails_before_fact_filter(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    store(engine, tmp_path / "artifacts", [monthly()], KNOWN)
    with Session(engine) as session:
        release = session.scalar(select(DataRelease))
        row = session.scalars(select(NationalDebtFact)).first()
        values = {column.name: getattr(row, column.name) for column in NationalDebtFact.__table__.columns
                  if column.name not in {"id", "fact_key"}}
        values.update(fact_key="f" * 64, metric="otherwise_unselected_fact", release_id=release.id)
        session.add(NationalDebtFact(**values))
        session.commit()
    with pytest.raises(ValueError, match="rows"):
        load_national_debt_context(engine, as_known_at=KNOWN, countries=["SE"])


def test_native_context_integrates_without_flattening_macro_series_and_protects_files(tmp_path):
    engine = make_engine(tmp_path / "native.db")
    batch = store(engine, tmp_path / "artifacts", [monthly(), funding()], KNOWN)
    result = load_evidence(engine, as_known_at=KNOWN, countries=["SE", "BE"])
    sweden = next(country for country in result["countries"] if country["country"] == "SE")
    belgium = next(country for country in result["countries"] if country["country"] == "BE")
    assert len(sweden["national_debt_context"]) == 7
    assert sweden["series"] == []
    assert belgium["national_debt_context"] == []
    expected = {a.artifact_path for item in batch for a in item.artifacts}
    assert expected <= set(result["protected_artifact_paths"])
