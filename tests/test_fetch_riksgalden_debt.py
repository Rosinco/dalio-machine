"""Offline acquisition/replay tests; no actual HTTP is performed."""

import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dalio.data_sources.riksgalden_debt import RIKSGALDEN_DOCUMENT_SPECS, NationalDebtDocument
from dalio.pipelines import fetch_riksgalden_debt as pipeline
from dalio.storage.national_debt import PreparedNativePartition

RECEIPT = datetime(2026, 9, 10, 22, 0, tzinfo=UTC)


def fake_document(spec):
    return NationalDebtDocument(
        spec.stream_id,
        spec.snapshot_key,
        "SE",
        spec.source_url,
        spec.published_at,
        spec.reference_date,
        b"source bytes",
        ({"status": "observed", "value": 1.0},),
        {"publication_precision": "date"},
    )


def fake_batch():
    return tuple(
        PreparedNativePartition(fake_document(spec), RECEIPT, ())
        for spec in RIKSGALDEN_DOCUMENT_SPECS
    )


@pytest.fixture
def acquisition(monkeypatch):
    client = MagicMock()

    def get(url, **kwargs):
        is_pdf = url.endswith(".pdf")
        return SimpleNamespace(
            status_code=200,
            url=url,
            headers={
                "Content-Type": "application/pdf"
                if is_pdf
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            },
            content=b"%PDF-1.7\nmock" if is_pdf else b"PK\x03\x04mock",
        )

    client.get.side_effect = get
    parse_pdf = MagicMock(side_effect=lambda body, *, spec: fake_document(spec))
    parse_xlsx = MagicMock(side_effect=lambda body, *, spec: fake_document(spec))
    prepare = MagicMock(
        side_effect=lambda documents, *, artifact_root, retrieved_at: tuple(
            PreparedNativePartition(doc, retrieved_at, ()) for doc in documents
        )
    )
    monkeypatch.setattr(pipeline, "parse_monthly_report", parse_pdf)
    monkeypatch.setattr(pipeline, "parse_funding_workbook", parse_xlsx)
    monkeypatch.setattr(pipeline, "prepare_native_batch", prepare)
    return client, parse_pdf, parse_xlsx, prepare


def test_fetches_every_original_url_without_redirects_and_uses_receipt_clock(acquisition, tmp_path):
    client, pdf, xlsx, prepare = acquisition
    batch = pipeline.prepare_batch(client=client, artifact_root=tmp_path, retrieved_at=RECEIPT)
    assert [call.args[0] for call in client.get.call_args_list] == [
        s.source_url for s in RIKSGALDEN_DOCUMENT_SPECS
    ]
    assert all(call.kwargs["allow_redirects"] is False for call in client.get.call_args_list)
    assert pdf.call_count == 8 and xlsx.call_count == 1
    assert prepare.call_count == 1 and len(batch) == 9
    assert {item.retrieved_at for item in batch} == {RECEIPT}
    assert batch[0].document.published_at != RECEIPT
    client.close.assert_not_called()


@pytest.mark.parametrize(
    "problem", ["redirect", "wrong_origin", "wrong_file", "wrong_type", "oversize", "http_error"]
)
def test_failed_response_never_prepares_evidence_or_opens_target(
    acquisition, monkeypatch, tmp_path, problem
):
    client, _, _, prepare = acquisition
    response = client.get.side_effect(RIKSGALDEN_DOCUMENT_SPECS[0].source_url)
    if problem == "redirect":
        response.status_code = 302
    elif problem == "wrong_origin":
        response.url = "https://mirror.example/report.pdf"
    elif problem == "wrong_file":
        response.url = RIKSGALDEN_DOCUMENT_SPECS[1].source_url
    elif problem == "wrong_type":
        response.headers["Content-Type"] = "text/html"
    elif problem == "oversize":
        response.content = b"%PDF-" + b"0" * 10_000_000
    else:
        response.status_code = 503
    client.get.side_effect = lambda *args, **kwargs: response
    engine = MagicMock()
    monkeypatch.setattr(pipeline, "make_engine", engine)
    with pytest.raises(ValueError):
        pipeline.run_pipeline(
            db_path=tmp_path / "absent.sqlite",
            client=client,
            artifact_root=tmp_path / "evidence",
            retrieved_at=RECEIPT,
        )
    prepare.assert_not_called()
    engine.assert_not_called()
    assert not (tmp_path / "absent.sqlite").exists()


def test_last_document_parse_failure_keeps_database_unopened(acquisition, monkeypatch, tmp_path):
    client, _, xlsx, prepare = acquisition
    xlsx.side_effect = ValueError("Changed forecast workbook")
    engine = MagicMock()
    monkeypatch.setattr(pipeline, "make_engine", engine)
    with pytest.raises(ValueError, match="forecast workbook"):
        pipeline.run_pipeline(
            db_path=tmp_path / "new.sqlite", client=client, artifact_root=tmp_path
        )
    assert client.get.call_count == 9
    prepare.assert_not_called()
    engine.assert_not_called()


def test_all_sources_preflight_before_target_creation(acquisition, monkeypatch, tmp_path):
    client, _, _, prepare = acquisition
    events = []
    original = prepare.side_effect
    prepare.side_effect = lambda *args, **kwargs: (
        events.append("prepared"),
        original(*args, **kwargs),
    )[1]
    target = MagicMock()
    monkeypatch.setattr(pipeline, "make_engine", lambda path: (events.append("engine"), target)[1])
    monkeypatch.setattr(
        pipeline,
        "ingest_native_batch",
        lambda batch, *, engine: {"snapshots": len(batch), "created_releases": 9, "fact_count": 9},
    )
    result = pipeline.run_pipeline(
        db_path=tmp_path / "new.sqlite", client=client, artifact_root=tmp_path, retrieved_at=RECEIPT
    )
    assert events == ["prepared", "engine"]
    assert result["snapshots"] == 9
    target.dispose.assert_called_once()


def test_offline_replay_opens_source_read_only_and_preserves_availability(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "staging.sqlite"
    with sqlite3.connect(source) as con:
        con.execute("CREATE TABLE sentinel (id INTEGER)")
    original = source.read_bytes()
    batch = fake_batch()

    def load(engine, **kwargs):
        with engine.connect() as con, pytest.raises(Exception, match="readonly"):
            con.exec_driver_sql("INSERT INTO sentinel VALUES (1)")
        return batch

    monkeypatch.setattr(pipeline, "load_native_batch", load)
    fetch = MagicMock(side_effect=AssertionError("Offline replay must not use HTTP"))
    monkeypatch.setattr(pipeline, "prepare_batch", fetch)
    ingest = MagicMock(return_value={"snapshots": 9, "created_releases": 9, "fact_count": 9})
    monkeypatch.setattr(pipeline, "ingest_native_batch", ingest)
    report = tmp_path / "receipt.json"
    assert (
        pipeline.main(
            [
                "--db",
                str(tmp_path / "live.sqlite"),
                "--from-db",
                str(source),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert ingest.call_args.args[0] == batch
    assert source.read_bytes() == original
    assert json.loads(report.read_text())["snapshots"] == 9
    assert json.loads(capsys.readouterr().out)["snapshots"] == 9
    fetch.assert_not_called()


def test_partial_offline_batch_is_rejected_before_target_creation(monkeypatch, tmp_path):
    source = tmp_path / "staging.sqlite"
    source.touch()
    monkeypatch.setattr(pipeline, "load_native_batch", lambda *args, **kwargs: fake_batch()[:-1])
    target = MagicMock()
    monkeypatch.setattr(pipeline, "make_engine", target)
    assert pipeline.main(["--db", str(tmp_path / "live.sqlite"), "--from-db", str(source)]) == 1
    target.assert_not_called()


def test_report_cannot_overwrite_database_and_missing_staging_is_not_created(monkeypatch, tmp_path):
    target = tmp_path / "live.sqlite"
    target.write_bytes(b"database sentinel")
    assert pipeline.main(["--db", str(target), "--report", str(target)]) == 1
    assert target.read_bytes() == b"database sentinel"
    source = tmp_path / "absent.sqlite"
    assert pipeline.main(["--db", str(target), "--from-db", str(source)]) == 1
    assert not source.exists()


def test_owned_http_session_is_closed_on_preflight_failure(acquisition, monkeypatch, tmp_path):
    client, _, xlsx, _ = acquisition
    monkeypatch.setattr(pipeline.requests, "Session", lambda: client)
    xlsx.side_effect = ValueError("Invalid final document")
    with pytest.raises(ValueError, match="Invalid final document"):
        pipeline.prepare_batch(artifact_root=tmp_path)
    client.close.assert_called_once()
