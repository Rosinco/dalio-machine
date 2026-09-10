"""Typed national-debt evidence: immutable snapshots, atomic batches and replay."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dalio.storage import national_debt
from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    NationalDebtFact,
    Observation,
    init_db,
    make_engine,
)

RUN_AT = datetime(2026, 9, 10, 21, tzinfo=UTC)


@dataclass(frozen=True)
class Document:
    stream_id: str
    snapshot_key: str
    country: str
    source_url: str
    published_at: datetime
    reference_date: date
    source_bytes: bytes
    facts: tuple[dict, ...]
    metadata: dict


def fact(metric="nominal_amount", value=100.0, status="observed", year=2026):
    return {
        "fact_type": "security_position",
        "metric": metric,
        "value": value,
        "unit": "SEK",
        "period_start": date(year, 8, 31),
        "period_end": date(year, 8, 31),
        "status": status,
        "dimensions": {
            "instrument": "SGB 1059",
            "maturity_date": "2026-11-12",
            "issuer_scope": "central_government",
            "basis": "nominal_face_value",
        },
        "source_locator": "pdf:page=2;row=1",
        "native_label": "SGB 1059",
        "native_value": str(value) if value is not None else "..",
    }


def document(snapshot="2026-08-31", facts=None):
    facts = facts or (fact(), fact("time_to_refixing", 0.2))
    body = json.dumps(facts, default=str, sort_keys=True).encode()
    return Document(
        "se_central_government_debt_monthly_report",
        snapshot,
        "SE",
        "https://www.riksgalden.se/contentassets/test/debt.pdf",
        datetime(2026, 9, 7, tzinfo=UTC),
        date(2026, 8, 31),
        body,
        tuple(facts),
        {"parser_version": "test-v1", "publication_precision": "date"},
    )


@pytest.fixture(autouse=True)
def synthetic_source_parser(monkeypatch):
    """The real source parser has separate PDF/XLSX fixture tests."""

    def reparse(doc):
        facts = json.loads(doc.source_bytes)
        for row in facts:
            row["period_start"] = date.fromisoformat(row["period_start"])
            row["period_end"] = date.fromisoformat(row["period_end"])
        return replace(doc, facts=tuple(facts))

    monkeypatch.setattr(national_debt, "_reparse_document", reparse)
    monkeypatch.setattr(national_debt, "_construct_document", lambda fields: Document(**fields))


def counts(engine):
    with Session(engine) as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (
                DataRelease,
                NationalDebtFact,
                DataReleaseArtifact,
                Observation,
            )
        )


def test_native_fields_periods_and_missingness_round_trip_without_flat_projection(tmp_path):
    docs = (
        document(
            facts=(
                fact(),
                fact("missing", None, "not_reported"),
                fact("planned_amount", 0.0, "forecast", year=2027),
            )
        ),
    )
    batch = national_debt.prepare_native_batch(
        docs, artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    engine = make_engine(tmp_path / "native.sqlite")
    result = national_debt.ingest_native_batch(batch, engine=engine)
    assert result["created_releases"] == 1
    assert counts(engine) == (1, 3, 4, 0)
    audit = national_debt.audit_national_debt(engine, as_of=RUN_AT.date())
    assert audit["ready_snapshots"] == 1
    assert audit["fact_count"] == 3
    assert audit["streams"][0]["stream_id"] == docs[0].stream_id
    with Session(engine) as session:
        missing = session.scalar(
            select(NationalDebtFact).where(NationalDebtFact.metric == "missing")
        )
        assert missing.value is None
        assert missing.status == "not_reported"
        plan = session.scalar(
            select(NationalDebtFact).where(NationalDebtFact.metric == "planned_amount")
        )
        assert plan.value == 0.0
        assert plan.status == "forecast" and plan.period_end.year == 2027
        assert json.loads(plan.dimensions_json)["maturity_date"] == "2026-11-12"


def test_repeat_deduplicates_revision_appends_and_preserves_old_source(tmp_path):
    engine = make_engine(tmp_path / "native.sqlite")
    original = document()
    first = national_debt.prepare_native_batch(
        (original,), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(first, engine=engine)
    repeat = national_debt.prepare_native_batch(
        (original,), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT + timedelta(hours=1)
    )
    assert national_debt.ingest_native_batch(repeat, engine=engine)["created_releases"] == 0
    revised = document(facts=(fact(value=125.0), fact("time_to_refixing", 0.2)))
    changed = national_debt.prepare_native_batch(
        (revised,), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT + timedelta(hours=2)
    )
    assert national_debt.ingest_native_batch(changed, engine=engine)["created_releases"] == 1
    assert counts(engine) == (2, 4, 8, 0)
    with Session(engine) as session:
        assert (
            session.scalar(
                select(NationalDebtFact.value).where(
                    NationalDebtFact.release_id == 1, NationalDebtFact.metric == "nominal_amount"
                )
            )
            == 100.0
        )
    audit = national_debt.audit_national_debt(engine, as_of=RUN_AT.date())
    assert audit["ready_snapshots"] == 1 and audit["historical_releases"] == 2


@pytest.mark.parametrize(
    "damage", ["value", "unit", "period", "source", "catalogue", "subset", "clock"]
)
def test_reject_changed_prepared_evidence_before_creating_database(tmp_path, damage):
    batch = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    if damage in {"value", "unit", "period"}:
        changed = dict(batch[0].document.facts[0])
        changed[{"value": "value", "unit": "unit", "period": "period_end"}[damage]] = {
            "value": 999.0,
            "unit": "USD",
            "period": date(2028, 1, 1),
        }[damage]
        batch = (
            replace(
                batch[0],
                document=replace(batch[0].document, facts=(changed, *batch[0].document.facts[1:])),
            ),
        )
    elif damage == "subset":
        batch = (
            replace(
                batch[0], document=replace(batch[0].document, facts=batch[0].document.facts[:1])
            ),
        )
    elif damage == "clock":
        batch = (replace(batch[0], retrieved_at=RUN_AT.replace(year=2025)),)
    else:
        role = "source_response" if damage == "source" else "catalogue_manifest"
        artifact = next(a for a in batch[0].artifacts if a.role == role)
        Path(artifact.artifact_path).write_bytes(b"changed")
    engine = make_engine(tmp_path / "rejected.sqlite")
    with pytest.raises(ValueError):
        national_debt.ingest_native_batch(batch, engine=engine)
    assert not (tmp_path / "rejected.sqlite").exists()


def test_late_failure_rolls_back_all_native_releases(tmp_path, monkeypatch):
    batch = national_debt.prepare_native_batch(
        (document(), document("2026-07-31")),
        artifact_root=tmp_path / "evidence",
        retrieved_at=RUN_AT,
    )
    engine = make_engine(tmp_path / "atomic.sqlite")
    init_db(engine)
    original = national_debt._ingest_one
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("late failure")
        return result

    monkeypatch.setattr(national_debt, "_ingest_one", failing)
    with pytest.raises(RuntimeError, match="late failure"):
        national_debt.ingest_native_batch(batch, engine=engine)
    assert counts(engine) == (0, 0, 0, 0)


def test_sql_immutability_and_offline_audit_replay(tmp_path):
    source = make_engine(tmp_path / "stage.sqlite")
    batch = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(batch, engine=source)
    with pytest.raises(IntegrityError, match="immutable"), source.begin() as connection:
        connection.execute(text("UPDATE national_debt_facts SET value = 999"))
    with pytest.raises(IntegrityError, match="immutable"), source.begin() as connection:
        connection.execute(text("DELETE FROM national_debt_facts"))
    promoted = national_debt.load_native_batch(source, as_of=RUN_AT.date())
    target = make_engine(tmp_path / "live.sqlite")
    national_debt.ingest_native_batch(promoted, engine=target)
    assert counts(source) == counts(target)
    artifact = next(a for a in batch[0].artifacts if a.role == "native_payload")
    Path(artifact.artifact_path).write_bytes(b"corrupted")
    audit = national_debt.audit_national_debt(target, as_of=RUN_AT.date())
    assert audit["ready_snapshots"] == 0
    with pytest.raises(ValueError, match="verified"):
        national_debt.load_native_batch(source, as_of=RUN_AT.date())


def test_empty_read_only_audit_does_not_initialize_tables(tmp_path):
    engine = make_engine(tmp_path / "empty.sqlite")
    audit = national_debt.audit_national_debt(engine, as_of=RUN_AT.date())
    assert audit["ready_snapshots"] == 0
    assert inspect(engine).get_table_names() == []


def test_national_stream_counts_inside_fixed_refinancing_denominator(tmp_path):
    from dalio.storage.refinancing import audit_refinancing

    engine = make_engine(tmp_path / "native-coverage.sqlite")
    batch = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(batch, engine=engine)
    audit = audit_refinancing(engine, as_of=RUN_AT.date())
    assert audit["expected_partitions"] == 48
    assert audit["ready_partitions"] == 1
    assert audit["harmonized_ready"] == 0
    assert audit["national_native_ready"] == 1
    assert audit["national_native_planned"] == 16
    assert audit["native_fact_count"] == 2
    assert audit["observation_count"] == 0


def test_native_audit_selects_the_revision_known_at_the_requested_cutoff(tmp_path):
    engine = make_engine(tmp_path / "vintages.sqlite")
    earlier = RUN_AT - timedelta(days=2)
    first = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=earlier
    )
    national_debt.ingest_native_batch(first, engine=engine)
    revised = document(facts=(fact(value=125.0), fact("time_to_refixing", 0.2)))
    second = national_debt.prepare_native_batch(
        (revised,), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(second, engine=engine)
    audit = national_debt.audit_national_debt(engine, as_of=earlier.date())
    assert audit["ready_snapshots"] == 1
    old = national_debt.load_native_batch(engine, as_of=earlier.date())
    assert old[0].document.facts[0]["value"] == 100.0
    assert old[0].retrieved_at == earlier


def test_offline_native_replay_keeps_each_snapshots_original_availability(tmp_path):
    engine = make_engine(tmp_path / "mixed-clocks.sqlite")
    first = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT - timedelta(days=2)
    )
    second = national_debt.prepare_native_batch(
        (document("2026-07-31"),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(first, engine=engine)
    national_debt.ingest_native_batch(second, engine=engine)
    loaded = national_debt.load_native_batch(engine, as_of=RUN_AT.date())
    assert sorted(item.retrieved_at for item in loaded) == [RUN_AT - timedelta(days=2), RUN_AT]
    target = make_engine(tmp_path / "promoted.sqlite")
    national_debt.ingest_native_batch(loaded, engine=target)
    assert (
        national_debt.audit_national_debt(target, as_of=(RUN_AT - timedelta(days=1)).date())[
            "ready_snapshots"
        ]
        == 1
    )


def test_deduplication_rejects_augmented_native_rows(tmp_path):
    engine = make_engine(tmp_path / "augmented.sqlite")
    batch = national_debt.prepare_native_batch(
        (document(),), artifact_root=tmp_path / "evidence", retrieved_at=RUN_AT
    )
    national_debt.ingest_native_batch(batch, engine=engine)
    with Session(engine) as session:
        prior = session.scalar(select(NationalDebtFact))
        fields = {
            column.name: getattr(prior, column.name)
            for column in NationalDebtFact.__table__.columns
            if column.name != "id"
        }
        fields["fact_key"] = "f" * 64
        session.add(NationalDebtFact(**fields))
        session.commit()
    with pytest.raises(ValueError, match="rows differ"):
        national_debt.ingest_native_batch(batch, engine=engine)
