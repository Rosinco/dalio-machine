"""CLI boundary tests for packet-bound human report-claim decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

import dalio.pipelines.review_report_claims as review_pipeline
from dalio.pipelines.review_report_claims import apply, check, main, status
from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import REPORT_REVIEW_METHODOLOGY_VERSION
from dalio.storage.db import (
    Claim,
    DocumentExtraction,
    ReportCandidateReview,
    ReportDocument,
    init_db,
    make_engine,
    make_session_factory,
)
from dalio.storage.reports import PageText, ReportMeta, ingest_report


class FakeExtractor:
    name = "fixture-pages"
    version = "1"

    def extract(self, _pdf_bytes: bytes) -> tuple[PageText, ...]:
        return (PageText(1, "The policy rate was held at 1.75 percent.", "1"),)


def _at(day: int) -> datetime:
    return datetime(2099, 1, day, 12, tzinfo=UTC)


@pytest.fixture
def review_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    # Keep the fixture small while preserving the production all-enabled-family
    # guard: in this isolated manifest, the one real official family is all of
    # the enabled families.
    monkeypatch.setattr(review_pipeline, "REPORT_SOURCES", (REPORT_SOURCES[0],))
    db_path = tmp_path / "reports.db"
    engine = make_engine(db_path)
    init_db(engine)
    factory = make_session_factory(engine)
    source = REPORT_SOURCES[0]
    pdf = b"%PDF-1.7\nreview pipeline fixture\n%%EOF\n"

    with factory() as session:
        ingested = ingest_report(
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
                available_at=_at(1),
                retrieved_at=_at(1),
                landing_url=source.landing_url,
                artifact_url="https://www.riksbank.se/report-2099-01.pdf",
                allowed_domains=source.official_domains,
                expected_sha256=hashlib.sha256(pdf).hexdigest(),
            ),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "archive",
        )
        document = session.get(ReportDocument, ingested.document_id)
        extraction = session.scalar(
            select(DocumentExtraction).where(DocumentExtraction.document_id == ingested.document_id)
        )
        candidate = {
            "source_id": document.source_id,
            "issue_key": document.issue_key,
            "document_sha256": document.content_sha256,
            "extraction_name": extraction.extractor_name,
            "extraction_version": extraction.extractor_version,
            "extraction_corpus_sha256": extraction.corpus_sha256,
            "claim_type": "fact",
            "statement": "The Riksbank held its policy rate at 1.75 percent.",
            "topic_key": "monetary_policy",
            "geographies": ["SE"],
            "claim_series_key": None,
            "reference_start": None,
            "reference_end": None,
            "target_start": None,
            "target_end": None,
            "numeric_value": 1.75,
            "lower_bound": None,
            "upper_bound": None,
            "unit": "percent",
            "condition_text": None,
            "horizon": "0-1y",
            "importance_rationale": "A bounded policy-rate fact for human review.",
            "citations": [
                {
                    "pdf_page_start": 1,
                    "pdf_page_end": 1,
                    "printed_locator": "p. 1",
                    "section_title": "Policy decision",
                    "evidence_excerpt": "The policy rate was held at 1.75 percent.",
                    "support_role": "direct",
                }
            ],
        }

    engine.dispose()
    catalogue_path = tmp_path / "report_claim_candidates.json"
    catalogue_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "methodology_version": REPORT_REVIEW_METHODOLOGY_VERSION,
                "created_by": "model:fixture-v1",
                "created_at": "2099-01-04T12:00:00Z",
                "as_known_at": "2099-01-03T12:00:00Z",
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    return db_path, catalogue_path


def _complete_approval(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0].update(
        outcome="approve",
        attestations={
            "attribution_fair": True,
            "type_correct": True,
            "scope_periods_units_conditions_correct": True,
            "important_for_macro_risk": True,
        },
        review_note="The statement is fairly and directly supported by the cited page.",
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _replace_with_rejection(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0].update(
        outcome="reject",
        attestations={
            "attribution_fair": False,
            "type_correct": True,
            "scope_periods_units_conditions_correct": True,
            "important_for_macro_risk": True,
        },
        reason_code="unsupported",
        review_note="The proposed attribution is not fairly supported by the cited page.",
        revision=None,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def test_prepare_check_and_status_are_database_read_only_and_print_full_paths(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    output_dir = tmp_path / "human review"
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    monkeypatch.setenv("WSL_DISTRO_NAME", "AuditDistro")

    exit_code = main(
        [
            "prepare",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--out-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    files = list(output_dir.iterdir())
    assert len(files) == 1
    decision_path = files[0]
    template = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision_path.name == (
        f"report_decisions_2099-01-03_{template['packet_sha256'][:16]}.json"
    )
    assert f"POSIX path: {decision_path.resolve()}" in captured.out
    expected_windows = "\\\\wsl.localhost\\AuditDistro" + str(decision_path.resolve()).replace(
        "/", "\\"
    )
    assert f"Windows path: {expected_windows}" in captured.out
    assert "database writes: none" in captured.out
    assert "reviewer" not in template
    assert "reviewed_at" not in template

    plan = check(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
    )
    report = status(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
    )

    assert plan.decision_file.packet_sha256 == template["packet_sha256"]
    assert report["candidate_count"] == 1
    assert report["file"] == {
        "approve": 0,
        "revise": 0,
        "reject": 0,
        "pending": 1,
    }
    assert report["recorded"] == {
        "approve": 0,
        "revise": 0,
        "reject": 0,
        "total": 0,
    }
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha


def test_check_and_status_cli_report_completed_file_without_writing_database(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    check_code = main(
        [
            "check",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--require-complete",
        ]
    )
    check_output = capsys.readouterr()
    status_code = main(
        [
            "status",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
        ]
    )
    status_output = capsys.readouterr()

    assert check_code == status_code == 0
    assert "approve 1, revise 0, reject 0, pending 0" in check_output.out
    assert "Database writes: none" in check_output.out
    assert "Decision file: approve 1, revise 0, reject 0, pending 0" in status_output.out
    assert "Recorded ledger: approve 0, revise 0, reject 0, total 0" in status_output.out
    assert "Database writes: none" in status_output.out
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha


def test_apply_cli_refuses_non_tty_before_backup_or_database_write(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    backup_dir = tmp_path / "backups"
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    exit_code = main(
        [
            "apply",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Write plan: approve 1, revise 0, reject 0, pending 0" in captured.out
    assert "apply requires a TTY" in captured.err
    assert not backup_dir.exists()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert _row_count(db_path, "report_candidate_reviews") == 0
    assert _row_count(db_path, "claims") == 0


def test_apply_cli_rejects_decision_changes_after_the_previewed_plan(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    previewed = review_pipeline.load_decision_file(decision_path)
    expected_decision_hash = review_pipeline.decision_file_sha256(previewed)
    backup_dir = tmp_path / "backups"
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    monkeypatch.setattr(review_pipeline.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review_pipeline.sys.stdout, "isatty", lambda: True)

    class ReviewDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2099, 1, 5, 12, tzinfo=tz)

    monkeypatch.setattr(review_pipeline, "datetime", ReviewDatetime)

    def answer(prompt: str) -> str:
        if prompt.startswith("Human reviewer"):
            return "human:pipeline-test"
        _replace_with_rejection(decision_path)
        return expected_decision_hash

    monkeypatch.setattr("builtins.input", answer)

    exit_code = main(
        [
            "apply",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Write plan: approve 1, revise 0, reject 0, pending 0" in captured.out
    assert f"Decision SHA-256: {expected_decision_hash}" in captured.out
    assert f"{previewed.decisions[0].candidate_id}  " in captured.out
    assert "  approve" in captured.out
    assert "decision file changed after preview; nothing was written" in captured.err
    assert not backup_dir.exists()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert _row_count(db_path, "report_candidate_reviews") == 0
    assert _row_count(db_path, "claims") == 0


def test_apply_cli_requires_the_full_decision_fingerprint_before_backup(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    decision_hash = review_pipeline.decision_file_sha256(
        review_pipeline.load_decision_file(decision_path)
    )
    backup_dir = tmp_path / "backups"
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    monkeypatch.setattr(review_pipeline.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review_pipeline.sys.stdout, "isatty", lambda: True)
    answers = iter(("human:pipeline-test", "0" * 64))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    exit_code = main(
        [
            "apply",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert f"Decision SHA-256: {decision_hash}" in captured.out
    assert "decision confirmation did not match; nothing was written" in captured.err
    assert not backup_dir.exists()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert _row_count(db_path, "report_candidate_reviews") == 0
    assert _row_count(db_path, "claims") == 0


def test_apply_cli_records_the_confirmed_decision_set_and_prints_receipt(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    payload = _complete_approval(decision_path)
    decision_hash = review_pipeline.decision_file_sha256(
        review_pipeline.load_decision_file(decision_path)
    )
    backup_dir = tmp_path / "backups"
    monkeypatch.setattr(review_pipeline.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review_pipeline.sys.stdout, "isatty", lambda: True)

    class ReviewDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2099, 1, 5, 12, tzinfo=tz)

    monkeypatch.setattr(review_pipeline, "datetime", ReviewDatetime)
    answers = iter(("human:pipeline-test", decision_hash))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    exit_code = main(
        [
            "apply",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    captured = capsys.readouterr()

    backup_path = backup_dir / (
        f"dalio-before-report-review-20990105T120000000000Z-{payload['packet_sha256'][:16]}.db"
    )
    assert exit_code == 0
    assert "Recorded operator-attributed decisions: approve 1, revise 0, reject 0" in captured.out
    assert "Operator attribution: human:pipeline-test" in captured.out
    assert "Verified successors: 1" in captured.out
    assert "Review receipt IDs: 1" in captured.out
    assert "Replay: no" in captured.out
    assert f"Verified pre-write backup (POSIX): {backup_path.resolve()}" in captured.out
    assert backup_path.is_file()
    assert _row_count(db_path, "report_candidate_reviews") == 1
    assert _row_count(db_path, "claims") == 2

    before_replay_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    answers = iter(("human:pipeline-test", decision_hash))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    replay_code = main(
        [
            "apply",
            "--db",
            str(db_path),
            "--catalogue",
            str(catalogue_path),
            "--decisions",
            str(decision_path),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    replay_output = capsys.readouterr()
    assert replay_code == 0
    assert "Replay: yes" in replay_output.out
    assert "exact replay made no database write" in replay_output.out
    assert list(backup_dir.iterdir()) == [backup_path]
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_replay_sha


def test_programmatic_apply_creates_exact_prewrite_backup_then_records_batch(
    review_store: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    payload = _complete_approval(decision_path)
    backup_dir = tmp_path / "backups"

    result, backup_path = apply(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
        backup_dir=backup_dir,
        reviewer="human:pipeline-test",
        reviewed_at=_at(5),
    )

    assert backup_path == backup_dir / (
        f"dalio-before-report-review-20990105T120000000000Z-{payload['packet_sha256'][:16]}.db"
    )
    assert backup_path.is_file()
    assert _row_count(backup_path, "report_candidate_reviews") == 0
    assert _row_count(backup_path, "claims") == 0
    assert _row_count(db_path, "report_candidate_reviews") == 1
    assert _row_count(db_path, "claims") == 2
    assert not result.replayed
    assert result.results[0].outcome == "approve"

    before_replay_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    backup_files = sorted(backup_dir.iterdir())
    replay, replay_backup_path = apply(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
        backup_dir=backup_dir,
        reviewer="human:pipeline-test",
        reviewed_at=_at(5),
    )
    assert replay.replayed
    assert replay_backup_path is None
    assert sorted(backup_dir.iterdir()) == backup_files
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_replay_sha
    assert _row_count(db_path, "report_candidate_reviews") == 1
    assert _row_count(db_path, "claims") == 2

    with pytest.raises(ValueError, match="already has a different review"):
        apply(
            db_path=db_path,
            catalogue_path=catalogue_path,
            decision_path=decision_path,
            backup_dir=backup_dir,
            reviewer="human:different-reviewer",
            reviewed_at=_at(6),
        )
    assert sorted(backup_dir.iterdir()) == backup_files
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_replay_sha

    report = status(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
    )
    assert report["recorded"] == {
        "approve": 1,
        "revise": 0,
        "reject": 0,
        "total": 1,
    }

    engine = make_engine(db_path)
    factory = make_session_factory(engine)
    try:
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(ReportCandidateReview)) == 1
            assert session.scalar(select(func.count()).select_from(Claim)) == 2
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("reviewed_at", "message"),
    [
        (datetime(2099, 1, 5, 12), "timezone-aware"),
        (_at(3), "earlier than proposal"),
    ],
)
def test_invalid_review_clock_fails_before_backup_or_schema_write(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    reviewed_at: datetime,
    message: str,
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    backup_dir = tmp_path / "backups"
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        apply(
            db_path=db_path,
            catalogue_path=catalogue_path,
            decision_path=decision_path,
            backup_dir=backup_dir,
            reviewer="human:pipeline-test",
            reviewed_at=reviewed_at,
        )

    assert not backup_dir.exists()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert _row_count(db_path, "report_candidate_reviews") == 0
    assert _row_count(db_path, "claims") == 0


def test_backup_failure_prevents_any_review_write(
    review_store: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, catalogue_path = review_store
    _template, decision_path = review_pipeline.prepare(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "review",
    )
    _complete_approval(decision_path)
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    def fail_backup(_source: Path, _destination: Path) -> Path:
        assert _row_count(db_path, "report_candidate_reviews") == 0
        raise RuntimeError("forced backup failure")

    monkeypatch.setattr(review_pipeline, "create_verified_sqlite_backup", fail_backup)

    with pytest.raises(RuntimeError, match="forced backup failure"):
        apply(
            db_path=db_path,
            catalogue_path=catalogue_path,
            decision_path=decision_path,
            backup_dir=tmp_path / "backups",
            reviewer="human:pipeline-test",
            reviewed_at=_at(5),
        )

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert _row_count(db_path, "report_candidate_reviews") == 0
    assert _row_count(db_path, "claims") == 0
