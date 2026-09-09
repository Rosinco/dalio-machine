"""Strict human decisions over provenance-bound official-report candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select

import dalio.reports.review_decisions as decisions_module
from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import (
    REPORT_REVIEW_METHODOLOGY_VERSION,
    CandidateCatalogue,
    CandidateCitation,
    ClaimCandidate,
)
from dalio.reports.review_decisions import (
    REPORT_DECISION_KIND,
    REPORT_DECISION_SCHEMA_VERSION,
    apply_review_decisions,
    build_decision_template,
    decision_file_sha256,
    load_decision_file,
    parse_decision_payload,
    validate_review_decisions,
)
from dalio.storage.db import (
    Claim,
    ClaimCitation,
    ReportCandidateReview,
    init_db,
    make_engine,
    make_session_factory,
)
from dalio.storage.reports import PageText, ReportMeta, ingest_report


class FakeExtractor:
    name = "fixture-pages"
    version = "1"

    def extract(self, _pdf_bytes: bytes) -> tuple[PageText, ...]:
        return (
            PageText(1, "The policy rate was held at 1.75 percent.", "1"),
            PageText(2, "Growth is expected to reach 2 percent in 2100.", "2"),
            PageText(3, "Financial stability risks remain elevated.", "3"),
        )


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


@pytest.fixture
def decision_store(tmp_path):
    db_path = tmp_path / "reports.db"
    engine = make_engine(db_path)
    init_db(engine)
    factory = make_session_factory(engine)
    source = REPORT_SOURCES[0]
    pdf = b"%PDF-1.7\nhuman review decision fixture\n%%EOF\n"
    with factory() as session:
        report = ingest_report(
            session,
            pdf,
            ReportMeta(
                source_id=source.source_id,
                report_family=source.report_family,
                issue_key="2099-01",
                publisher=source.publisher,
                jurisdiction=source.jurisdiction,
                title="Monetary Policy Report January 2099",
                language=source.language,
                document_date=date(2099, 1, 1),
                published_at=None,
                available_at=_at(2099, 1, 1),
                retrieved_at=_at(2099, 1, 1),
                landing_url=source.landing_url,
                artifact_url="https://www.riksbank.se/report-2099-01.pdf",
                allowed_domains=source.official_domains,
                expected_sha256=hashlib.sha256(pdf).hexdigest(),
            ),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "archive",
        )

    common = {
        "source_id": source.source_id,
        "issue_key": "2099-01",
        "document_sha256": report.content_sha256,
        "extraction_name": FakeExtractor.name,
        "extraction_version": FakeExtractor.version,
        "claim_series_key": None,
        "geographies": ("SE",),
        "reference_start": None,
        "reference_end": None,
        "numeric_value": None,
        "lower_bound": None,
        "upper_bound": None,
        "unit": None,
        "condition_text": None,
        "importance_rationale": "Model-selected for a bounded human review fixture.",
    }
    candidates = (
        ClaimCandidate(
            **common,
            extraction_corpus_sha256=_extraction_sha(factory),
            claim_type="fact",
            statement="The Riksbank held its policy rate at 1.75 percent.",
            topic_key="monetary_policy",
            target_start=None,
            target_end=None,
            horizon="0-1y",
            citations=(
                CandidateCitation(
                    1,
                    1,
                    "The policy rate was held at 1.75 percent.",
                    "direct",
                    printed_locator="p. 1",
                ),
            ),
        ),
        ClaimCandidate(
            **common,
            extraction_corpus_sha256=_extraction_sha(factory),
            claim_type="forecast",
            statement="The Riksbank expected growth to reach 2 percent in 2100.",
            topic_key="growth",
            target_start=date(2100, 1, 1),
            target_end=date(2100, 12, 31),
            horizon="1-3y",
            citations=(
                CandidateCitation(
                    2,
                    2,
                    "Growth is expected to reach 2 percent in 2100.",
                    "direct",
                    printed_locator="p. 2",
                ),
            ),
        ),
        ClaimCandidate(
            **common,
            extraction_corpus_sha256=_extraction_sha(factory),
            claim_type="judgment",
            statement="The Riksbank judged financial stability risks to remain elevated.",
            topic_key="risks",
            target_start=None,
            target_end=None,
            horizon="3-5y",
            citations=(
                CandidateCitation(
                    3,
                    3,
                    "Financial stability risks remain elevated.",
                    "direct",
                    printed_locator="p. 3",
                ),
            ),
        ),
    )
    catalogue = CandidateCatalogue(
        schema_version=1,
        methodology_version=REPORT_REVIEW_METHODOLOGY_VERSION,
        created_by="model:fixture-v1",
        created_at=_at(2099, 1, 2),
        as_known_at=_at(2099, 1, 1),
        candidates=candidates,
    )
    try:
        yield db_path, factory, catalogue
    finally:
        engine.dispose()


def _extraction_sha(factory) -> str:
    from dalio.storage.db import DocumentExtraction

    with factory() as session:
        return session.scalar(select(DocumentExtraction.corpus_sha256))


def _all_true() -> dict[str, bool]:
    return {
        "attribution_fair": True,
        "type_correct": True,
        "scope_periods_units_conditions_correct": True,
        "important_for_macro_risk": True,
    }


def _revision(candidate: ClaimCandidate) -> dict[str, object]:
    return {
        "claim_type": candidate.claim_type,
        "statement": candidate.statement + " This wording was narrowed by the reviewer.",
        "topic_key": candidate.topic_key,
        "geographies": list(candidate.geographies),
        "claim_series_key": candidate.claim_series_key,
        "reference_start": (
            candidate.reference_start.isoformat() if candidate.reference_start else None
        ),
        "reference_end": candidate.reference_end.isoformat() if candidate.reference_end else None,
        "target_start": candidate.target_start.isoformat() if candidate.target_start else None,
        "target_end": candidate.target_end.isoformat() if candidate.target_end else None,
        "numeric_value": candidate.numeric_value,
        "lower_bound": candidate.lower_bound,
        "upper_bound": candidate.upper_bound,
        "unit": candidate.unit,
        "condition_text": candidate.condition_text,
    }


def _completed_payload(session, catalogue) -> dict[str, object]:
    payload = build_decision_template(session, catalogue)
    approve, revise, reject = payload["decisions"]
    candidates_by_id = {candidate.candidate_id: candidate for candidate in catalogue.candidates}
    approve.update(
        outcome="approve",
        attestations=_all_true(),
        review_note="The exact statement is fairly supported by the cited page.",
    )
    revise.update(
        outcome="revise",
        attestations=_all_true(),
        reason_code="wrong_scope",
        review_note="The final wording is narrower while retaining the cited evidence.",
        revision=_revision(candidates_by_id[revise["candidate_id"]]),
    )
    reject.update(
        outcome="reject",
        attestations={**_all_true(), "attribution_fair": False},
        reason_code="unsupported",
        review_note="The proposed attribution goes beyond the bounded excerpt.",
    )
    return payload


def test_template_is_exactly_packet_bound_pending_and_has_no_identity_or_clock(decision_store):
    db_path, factory, catalogue = decision_store
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    with factory() as session:
        template = build_decision_template(session, catalogue)
        parsed = parse_decision_payload(template)
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()

    assert template["schema_version"] == REPORT_DECISION_SCHEMA_VERSION
    assert template["decision_kind"] == REPORT_DECISION_KIND
    assert len(template["decisions"]) == len(catalogue.candidates) == 3
    assert {item["candidate_id"] for item in template["decisions"]} == {
        candidate.candidate_id for candidate in catalogue.candidates
    }
    assert all(item["outcome"] is None for item in template["decisions"])
    assert "reviewer" not in json.dumps(template)
    assert "reviewed_at" not in json.dumps(template)
    assert len(parsed.packet_sha256) == len(parsed.candidate_catalogue_sha256) == 64
    assert before == after


def test_decision_fingerprint_binds_candidate_outcomes_not_only_packet(decision_store):
    _db_path, factory, catalogue = decision_store
    with factory() as session:
        payload = _completed_payload(session, catalogue)

    parsed = parse_decision_payload(payload)
    changed = copy.deepcopy(payload)
    outcome_fields = ("outcome", "attestations", "reason_code", "review_note", "revision")
    first = {field: copy.deepcopy(changed["decisions"][0][field]) for field in outcome_fields}
    third = {field: copy.deepcopy(changed["decisions"][2][field]) for field in outcome_fields}
    changed["decisions"][0].update(third)
    changed["decisions"][2].update(first)

    assert changed["packet_sha256"] == payload["packet_sha256"]
    assert decision_file_sha256(parsed) == decision_file_sha256(payload)
    assert decision_file_sha256(changed) != decision_file_sha256(payload)


def test_loader_rejects_duplicate_keys_unknown_fields_and_nonfinite_numbers(
    decision_store, tmp_path
):
    _db_path, factory, catalogue = decision_store
    with factory() as session:
        template = build_decision_template(session, catalogue)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"decision_kind":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_decision_file(duplicate)

    unknown = copy.deepcopy(template)
    unknown["reviewer"] = "human:forbidden-in-editable-file"
    with pytest.raises(ValueError, match="unknown fields"):
        parse_decision_payload(unknown)

    non_integer_version = copy.deepcopy(template)
    non_integer_version["schema_version"] = 1.0
    with pytest.raises(ValueError, match="schema_version"):
        parse_decision_payload(non_integer_version)

    unknown_reason = copy.deepcopy(template)
    unknown_reason["decisions"][0].update(
        outcome="reject",
        attestations={**_all_true(), "attribution_fair": False},
        reason_code="unsupported_paraphrase",
        review_note="Reviewed with an unsupported reason code.",
    )
    with pytest.raises(ValueError, match="reason_code must be one of"):
        parse_decision_payload(unknown_reason)

    nonfinite = tmp_path / "nonfinite.json"
    template["decisions"][0]["revision"] = {
        **_revision(catalogue.candidates[0]),
        "numeric_value": float("nan"),
    }
    nonfinite.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported JSON numeric constant"):
        load_decision_file(nonfinite)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda item, _candidate: item.update(
                outcome="approve",
                attestations={**_all_true(), "type_correct": False},
                review_note="Reviewed but inconsistent.",
            ),
            "approval requires all four",
        ),
        (
            lambda item, candidate: item.update(
                outcome="revise",
                attestations=_all_true(),
                reason_code="wrong_scope",
                review_note="Reviewed and revised.",
                revision=_revision(candidate) | {"statement": candidate.statement},
            ),
            "revision is a no-op",
        ),
        (
            lambda item, _candidate: item.update(
                outcome="reject",
                attestations=_all_true(),
                reason_code="not_material",
                review_note="Reviewed and rejected.",
            ),
            "rejection requires at least one false",
        ),
    ],
)
def test_outcome_contracts_fail_closed(decision_store, mutate, message):
    _db_path, factory, catalogue = decision_store
    with factory() as session:
        payload = build_decision_template(session, catalogue)
        candidate = next(
            candidate
            for candidate in catalogue.candidates
            if candidate.candidate_id == payload["decisions"][0]["candidate_id"]
        )
        mutate(payload["decisions"][0], candidate)
        if "no-op" in message:
            parsed = parse_decision_payload(payload)
            with pytest.raises(ValueError, match=message):
                validate_review_decisions(session, catalogue, parsed)
        else:
            with pytest.raises(ValueError, match=message):
                parse_decision_payload(payload)


def test_validation_requires_exact_candidates_identity_hashes_and_complete_apply(decision_store):
    _db_path, factory, catalogue = decision_store
    with factory() as session:
        template = build_decision_template(session, catalogue)

        missing = copy.deepcopy(template)
        missing["decisions"].pop()
        with pytest.raises(ValueError, match="exactly every packet candidate"):
            validate_review_decisions(session, catalogue, missing)

        changed = copy.deepcopy(template)
        changed["decisions"][0]["proposed_statement"] += " changed"
        with pytest.raises(ValueError, match="identity fields"):
            validate_review_decisions(session, catalogue, changed)

        stale = copy.deepcopy(template)
        stale["packet_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="packet_sha256"):
            validate_review_decisions(session, catalogue, stale)

        with pytest.raises(ValueError, match="has not been decided"):
            validate_review_decisions(session, catalogue, template, require_complete=True)


def test_apply_approve_revise_reject_is_atomic_auditable_and_idempotent(decision_store):
    _db_path, factory, catalogue = decision_store
    reviewed_at = _at(2099, 1, 3)
    with factory() as session:
        payload = _completed_payload(session, catalogue)
        result = apply_review_decisions(
            session,
            catalogue,
            payload,
            reviewer="human:adam",
            reviewed_at=reviewed_at,
        )
        rows = (
            session.execute(
                select(ReportCandidateReview).order_by(ReportCandidateReview.candidate_id)
            )
            .scalars()
            .all()
        )
        claims = session.execute(select(Claim).order_by(Claim.id)).scalars().all()
        citations = session.execute(select(ClaimCitation)).scalars().all()

    assert not result.replayed
    assert {item.outcome for item in result.results} == {"approve", "revise", "reject"}
    assert len(rows) == 3
    assert len(claims) == 6
    assert sum(claim.status == "verified" for claim in claims) == 2
    assert sum(citation.semantic_verified_by == "human:adam" for citation in citations) == 2
    by_outcome = {row.outcome: row for row in rows}
    assert by_outcome["approve"].verified_claim_id is not None
    assert by_outcome["approve"].revised_draft_claim_id is None
    assert by_outcome["revise"].verified_claim_id is not None
    assert by_outcome["revise"].revised_draft_claim_id is not None
    assert by_outcome["reject"].verified_claim_id is None
    assert by_outcome["reject"].revised_draft_claim_id is None
    assert by_outcome["revise"].reason_code == "wrong_scope"
    assert json.loads(by_outcome["approve"].candidate_json)["document_sha256"]
    assert json.loads(by_outcome["approve"].checklist_json) == _all_true()

    with factory() as session:
        replay = apply_review_decisions(
            session,
            catalogue,
            payload,
            reviewer="human:adam",
            reviewed_at=_at(2099, 1, 4),
        )
        assert session.scalar(select(func.count()).select_from(Claim)) == 6
        assert session.scalar(select(func.count()).select_from(ReportCandidateReview)) == 3
    assert replay.replayed


def test_changed_replay_conflicts_without_appending_rows(decision_store):
    _db_path, factory, catalogue = decision_store
    reviewed_at = _at(2099, 1, 3)
    with factory() as session:
        payload = _completed_payload(session, catalogue)
        apply_review_decisions(
            session,
            catalogue,
            payload,
            reviewer="human:adam",
            reviewed_at=reviewed_at,
        )
    changed = copy.deepcopy(payload)
    changed["decisions"][0]["review_note"] = "A materially different review note."
    with factory() as session:
        with pytest.raises(ValueError, match="already has a different review"):
            apply_review_decisions(
                session,
                catalogue,
                changed,
                reviewer="human:adam",
                reviewed_at=reviewed_at,
            )
        assert session.scalar(select(func.count()).select_from(Claim)) == 6
        assert session.scalar(select(func.count()).select_from(ReportCandidateReview)) == 3


def test_apply_rolls_back_every_row_when_a_late_append_fails(decision_store, monkeypatch):
    _db_path, factory, catalogue = decision_store
    real_append = decisions_module._append_verified_claim
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced late verification failure")
        return real_append(*args, **kwargs)

    monkeypatch.setattr(decisions_module, "_append_verified_claim", fail_second)
    with factory() as session:
        payload = _completed_payload(session, catalogue)
        with pytest.raises(RuntimeError, match="forced late verification failure"):
            apply_review_decisions(
                session,
                catalogue,
                payload,
                reviewer="human:adam",
                reviewed_at=_at(2099, 1, 3),
            )
        assert session.scalar(select(func.count()).select_from(Claim)) == 0
        assert session.scalar(select(func.count()).select_from(ClaimCitation)) == 0
        assert session.scalar(select(func.count()).select_from(ReportCandidateReview)) == 0


@pytest.mark.parametrize(
    ("reviewer", "reviewed_at", "message"),
    [
        ("model:reviewer", _at(2099, 1, 3), "human:<id>"),
        ("human:adam", datetime(2099, 1, 3, 12), "timezone-aware"),
        ("human:adam", _at(2099, 1, 1), "earlier than proposal"),
    ],
)
def test_apply_requires_external_human_identity_and_valid_review_clock(
    decision_store, reviewer, reviewed_at, message
):
    _db_path, factory, catalogue = decision_store
    with factory() as session:
        payload = _completed_payload(session, catalogue)
        with pytest.raises(ValueError, match=message):
            apply_review_decisions(
                session,
                catalogue,
                payload,
                reviewer=reviewer,
                reviewed_at=reviewed_at,
            )
        assert session.scalar(select(func.count()).select_from(Claim)) == 0
