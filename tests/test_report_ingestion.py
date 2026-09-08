"""Checked-in official-report catalogue, Poppler extraction, and local ingestion."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from dalio.pipelines.ingest_reports import issue_identity, run_pipeline
from dalio.reports.catalogue import load_issue_catalogue
from dalio.reports.poppler import PopplerPageExtractor
from dalio.storage.db import (
    DocumentExtraction,
    DocumentPage,
    ReportDocument,
    make_engine,
    make_session_factory,
)
from dalio.storage.reports import PageText

PDF_ONE = b"%PDF-1.7\nfirst fixture\n%%EOF\n"
PDF_TWO = b"%PDF-1.7\nsecond fixture\n%%EOF\n"
CHECKED_IN_CATALOGUE = Path(__file__).parents[1] / "data/reference/report_issues.json"


def _issue(
    *,
    source_id: str = "riksbank_mpr_en",
    issue_key: str = "2026-06",
    pdf: bytes = PDF_ONE,
    page_count: int = 2,
    artifact_filename: str = "riksbank-mpr-2026-06.pdf",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "issue_key": issue_key,
        "title": "Monetary Policy Report June 2026",
        "document_date": "2026-06-18",
        "published_at": None,
        "available_at": "2026-06-18T23:59:59Z",
        "retrieved_at": "2026-09-08T10:15:00Z",
        "issue_url": (
            "https://www.riksbank.se/en-gb/monetary-policy/monetary-policy-report/2026/june/"
        ),
        "artifact_url": "https://www.riksbank.se/globalassets/report.pdf",
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "page_count": page_count,
        "artifact_filename": artifact_filename,
    }


def _catalogue(tmp_path: Path, issues: list[dict[str, object]]) -> Path:
    path = tmp_path / "reports.json"
    path.write_text(json.dumps({"schema_version": 1, "issues": issues}), encoding="utf-8")
    return path


class FakeExtractor:
    name = "fixture-pages"
    version = "1"

    def __init__(self, pages_by_hash: dict[str, tuple[PageText, ...]]):
        self.pages_by_hash = pages_by_hash
        self.calls: list[str] = []

    def extract(self, pdf_bytes: bytes) -> tuple[PageText, ...]:
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        self.calls.append(digest)
        return self.pages_by_hash[digest]


def test_catalogue_loads_exact_metadata_and_derives_manifest_identity(tmp_path):
    path = _catalogue(tmp_path, [_issue()])

    issue = load_issue_catalogue(path)[0]
    meta = issue.to_report_meta()

    assert issue.identity == "riksbank_mpr_en:2026-06"
    assert issue.document_date == date(2026, 6, 18)
    assert issue.available_at == datetime(2026, 6, 18, 23, 59, 59, tzinfo=UTC)
    assert issue.published_at is None
    assert issue.page_count == 2
    assert issue.artifact_filename == "riksbank-mpr-2026-06.pdf"
    assert meta.publisher == "Sveriges Riksbank"
    assert meta.report_family == "Monetary Policy Report"
    assert meta.jurisdiction == "SE"
    assert meta.language == "en"
    assert meta.allowed_domains == ("riksbank.se",)
    assert meta.expected_sha256 == hashlib.sha256(PDF_ONE).hexdigest()


def test_checked_in_catalogue_has_current_and_prior_issue_for_each_manifest_source():
    issues = load_issue_catalogue(CHECKED_IN_CATALOGUE)

    source_counts: dict[str, int] = {}
    for issue in issues:
        source_counts[issue.source_id] = source_counts.get(issue.source_id, 0) + 1

    assert source_counts == {
        "riksbank_mpr_en": 2,
        "ecb_staff_projections_en": 2,
        "fed_mpr_en": 2,
        "imf_weo_en": 2,
        "bis_aer_en": 2,
    }
    assert len({issue.sha256 for issue in issues}) == 10


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version=True), "schema_version"),
        (lambda payload: payload.update(extra=True), "unknown fields"),
        (lambda payload: payload["issues"][0].update(extra=True), "unknown fields"),
        (lambda payload: payload["issues"][0].pop("sha256"), "missing fields"),
        (lambda payload: payload["issues"][0].update(source_id="unknown"), "manifest"),
        (
            lambda payload: payload["issues"][0].update(issue_url="https://example.com/report"),
            "allowlisted",
        ),
        (
            lambda payload: payload["issues"][0].update(
                artifact_url="http://www.riksbank.se/report.pdf"
            ),
            "HTTPS",
        ),
        (
            lambda payload: payload["issues"][0].update(available_at="2026-06-18T23:59:59"),
            "timezone",
        ),
        (
            lambda payload: payload["issues"][0].update(retrieved_at="2026-06-17T23:59:59Z"),
            "retrieved_at",
        ),
        (lambda payload: payload["issues"][0].update(page_count=0), "page_count"),
        (
            lambda payload: payload["issues"][0].update(artifact_filename="../report.pdf"),
            "artifact_filename",
        ),
        (
            lambda payload: payload["issues"][0].update(sha256="A" * 64),
            "sha256",
        ),
    ],
)
def test_catalogue_rejects_unsafe_or_ambiguous_metadata(tmp_path, mutate, message):
    payload = {"schema_version": 1, "issues": [_issue()]}
    mutate(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_issue_catalogue(path)


def test_catalogue_rejects_duplicate_issue_identity_and_artifact_filename(tmp_path):
    duplicate_identity = [_issue(), _issue(artifact_filename="another.pdf")]
    with pytest.raises(ValueError, match="duplicate issue identity"):
        load_issue_catalogue(_catalogue(tmp_path, duplicate_identity))

    duplicate_filename = [
        _issue(),
        _issue(
            issue_key="2026-03",
            pdf=PDF_TWO,
            artifact_filename="riksbank-mpr-2026-06.pdf",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate artifact_filename"):
        load_issue_catalogue(_catalogue(tmp_path, duplicate_filename))


def test_catalogue_rejects_duplicate_json_object_keys(tmp_path):
    path = tmp_path / "duplicate-key.json"
    path.write_text('{"schema_version":1,"schema_version":1,"issues":[]}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_issue_catalogue(path)


def test_poppler_extractor_uses_fixed_nonshell_command_and_keeps_blank_pages(monkeypatch, tmp_path):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "-v":
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"pdftotext version 24.02.0\n")
        assert Path(command[-2]).read_bytes() == PDF_ONE
        return SimpleNamespace(returncode=0, stdout=b"first page\n\f\fthird page\r\n\f", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    extractor = PopplerPageExtractor(executable="/usr/bin/pdftotext", temp_root=tmp_path)

    pages = extractor.extract(PDF_ONE)

    assert extractor.name == "poppler-pdftotext"
    assert extractor.version == "24.02.0-layout-utf8-eol-unix-v1"
    assert [(page.pdf_page, page.text, page.printed_page_label) for page in pages] == [
        (1, "first page", None),
        (2, "", None),
        (3, "third page", None),
    ]
    version_command, version_kwargs = calls[0]
    extract_command, extract_kwargs = calls[1]
    assert version_command == ["/usr/bin/pdftotext", "-v"]
    assert extract_command[:7] == [
        "/usr/bin/pdftotext",
        "-layout",
        "-enc",
        "UTF-8",
        "-eol",
        "unix",
        "-q",
    ]
    assert extract_command[-1] == "-"
    assert version_kwargs["shell"] is False and extract_kwargs["shell"] is False
    assert extract_kwargs["input"] is None


def test_poppler_extractor_reports_binary_and_utf8_failures(monkeypatch):
    def missing(_command, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="pdftotext executable"):
        PopplerPageExtractor()

    calls = 0

    def invalid_utf8(_command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"pdftotext version 1.2.3")
        return SimpleNamespace(returncode=0, stdout=b"\xff\f", stderr=b"")

    monkeypatch.setattr(subprocess, "run", invalid_utf8)
    extractor = PopplerPageExtractor()
    with pytest.raises(RuntimeError, match="UTF-8"):
        extractor.extract(PDF_ONE)


def test_pipeline_preflights_all_artifacts_then_ingests_idempotently(tmp_path):
    second_issue = _issue(
        issue_key="2026-03",
        pdf=PDF_TWO,
        artifact_filename="riksbank-mpr-2026-03.pdf",
    )
    catalogue_path = _catalogue(tmp_path, [_issue(), second_issue])
    artifact_dir = tmp_path / "downloads"
    artifact_dir.mkdir()
    (artifact_dir / "riksbank-mpr-2026-06.pdf").write_bytes(PDF_ONE)
    (artifact_dir / "riksbank-mpr-2026-03.pdf").write_bytes(PDF_TWO)
    pages = {
        hashlib.sha256(PDF_ONE).hexdigest(): (PageText(1, "one"), PageText(2, "two")),
        hashlib.sha256(PDF_TWO).hexdigest(): (
            PageText(1, "three"),
            PageText(2, "four"),
        ),
    }
    extractor = FakeExtractor(pages)
    engine = make_engine(tmp_path / "reports.db")
    kwargs = {
        "catalogue_path": catalogue_path,
        "artifact_dir": artifact_dir,
        "blob_root": tmp_path / "archive",
        "engine": engine,
        "extractor": extractor,
    }

    first = run_pipeline(**kwargs)
    second = run_pipeline(**kwargs)

    assert set(first) == {"riksbank_mpr_en:2026-06", "riksbank_mpr_en:2026-03"}
    assert all(result["created"] is True for result in first.values())
    assert all(result["created"] is False for result in second.values())
    assert len(extractor.calls) == 4  # one extraction per selected artifact and run
    assert all(result["page_count"] == 2 for result in first.values())

    factory = make_session_factory(engine)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(ReportDocument)) == 2
        assert session.scalar(select(func.count()).select_from(DocumentExtraction)) == 2
        assert session.scalar(select(func.count()).select_from(DocumentPage)) == 4
        documents = session.scalars(select(ReportDocument)).all()
    assert {document.available_at for document in documents} == {datetime(2026, 6, 18, 23, 59, 59)}
    assert all(Path(document.blob_path).is_file() for document in documents)


def test_pipeline_can_select_one_issue_with_exact_explicit_mapping(tmp_path):
    catalogue_path = _catalogue(
        tmp_path,
        [
            _issue(),
            _issue(
                issue_key="2026-03",
                pdf=PDF_TWO,
                artifact_filename="riksbank-mpr-2026-03.pdf",
            ),
        ],
    )
    selected = "riksbank_mpr_en:2026-03"
    artifact = tmp_path / "local.pdf"
    artifact.write_bytes(PDF_TWO)
    extractor = FakeExtractor(
        {hashlib.sha256(PDF_TWO).hexdigest(): (PageText(1, "a"), PageText(2, "b"))}
    )

    summary = run_pipeline(
        catalogue_path=catalogue_path,
        artifacts={selected: artifact},
        issue_ids=(selected,),
        blob_root=tmp_path / "archive",
        engine=make_engine(tmp_path / "reports.db"),
        extractor=extractor,
    )

    assert set(summary) == {selected}
    assert issue_identity("riksbank_mpr_en", "2026-03") == selected


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("both", "either artifacts or artifact_dir"),
        ("missing_mapping", "missing artifact mappings"),
        ("extra_mapping", "unexpected artifact mappings"),
        ("unknown_selection", "unknown issue"),
        ("wrong_hash", "sha256 mismatch"),
        ("wrong_pages", "physical page count"),
    ],
)
def test_pipeline_rejects_bad_selection_or_artifacts_before_database_write(tmp_path, mode, message):
    catalogue_path = _catalogue(tmp_path, [_issue()])
    key = "riksbank_mpr_en:2026-06"
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(PDF_ONE + (b"tampered" if mode == "wrong_hash" else b""))
    artifacts: dict[str, Path] | None = {key: artifact}
    artifact_dir: Path | None = None
    issue_ids: tuple[str, ...] | None = None
    extractor = FakeExtractor(
        {
            hashlib.sha256(PDF_ONE).hexdigest(): (
                (PageText(1, "only"),)
                if mode == "wrong_pages"
                else (PageText(1, "one"), PageText(2, "two"))
            )
        }
    )
    if mode == "both":
        artifact_dir = tmp_path
    elif mode == "missing_mapping":
        artifacts = {}
    elif mode == "extra_mapping":
        artifacts = {key: artifact, "riksbank_mpr_en:extra": artifact}
    elif mode == "unknown_selection":
        issue_ids = ("riksbank_mpr_en:missing",)

    db_path = tmp_path / "reports.db"
    with pytest.raises(ValueError, match=message):
        run_pipeline(
            catalogue_path=catalogue_path,
            artifacts=artifacts,
            artifact_dir=artifact_dir,
            issue_ids=issue_ids,
            blob_root=tmp_path / "archive",
            engine=make_engine(db_path),
            extractor=extractor,
        )

    assert not db_path.exists()
