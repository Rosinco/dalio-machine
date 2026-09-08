"""AP-fund pipeline preflights local official artifacts before any writes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from dalio.pipelines.ingest_ap_funds import partition_key_for, run_pipeline
from dalio.storage.allocators import load_allocator_vintage
from dalio.storage.db import AllocatorFact, DataRelease, make_engine, make_session_factory

AP2_PDF = b"%PDF-1.4\nAP2 fixture\n%%EOF\n"
AP3_PDF = b"%PDF-1.4\nAP3 fixture\n%%EOF\n"
AS_KNOWN = datetime(2026, 9, 8, 12, tzinfo=UTC)


def _row(item_code: str, amount: float) -> dict:
    return {
        "record_type": "asset_allocation",
        "item_code": item_code,
        "reported_amount": amount,
        "reported_unit": "SEK_bn",
        "amount_sek_mn": amount * 1_000,
        "exposure_pct": 50.0,
        "basis": "actual_portfolio_exposure",
        "row_role": "detail",
        "physical_page": 2,
        "table_heading": "Asset allocation",
    }


def _exposure_only_row(item_code: str, exposure_pct: float) -> dict:
    return {
        "record_type": "management",
        "item_code": item_code,
        "reported_amount": None,
        "reported_unit": None,
        "amount_sek_mn": None,
        "exposure_pct": exposure_pct,
        "basis": "fund_capital",
        "row_role": "metric",
        "physical_page": 3,
        "table_heading": "Portfolio management",
    }


def _release(fund: str, pdf: bytes, row: dict) -> dict:
    return {
        "fund": fund,
        "report_date": "2026-06-30",
        "title": f"{fund} Half-Year Report 2026",
        "source_url": f"https://{fund.lower()}.se/report.pdf",
        "official_domains": [f"{fund.lower()}.se"],
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "page_count": 4,
        "rows": [row],
    }


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.json"
    ap3 = _release("AP3", AP3_PDF, _row("fixed_income", 40.0))
    ap3["rows"].append(_exposure_only_row("external_management_share", 11.0))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method_note": "Verified direct table transcription.",
                "releases": [
                    _release("AP2", AP2_PDF, _row("equities", 50.0)),
                    ap3,
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = {"AP2": tmp_path / "source-ap2.pdf", "AP3": tmp_path / "source-ap3.pdf"}
    paths["AP2"].write_bytes(AP2_PDF)
    paths["AP3"].write_bytes(AP3_PDF)
    return paths


def test_pipeline_ingests_exact_mapping_and_is_idempotent(tmp_path):
    engine = make_engine(tmp_path / "allocators.db")
    artifacts = _artifacts(tmp_path)
    kwargs = {
        "reference_path": _reference(tmp_path),
        "artifacts": artifacts,
        "blob_root": tmp_path / "blobs",
        "engine": engine,
        "retrieved_at": AS_KNOWN,
    }

    first = run_pipeline(**kwargs)
    second = run_pipeline(**kwargs)

    assert {fund: result["created"] for fund, result in first.items()} == {
        "AP2": True,
        "AP3": True,
    }
    assert {fund: result["created"] for fund, result in second.items()} == {
        "AP2": False,
        "AP3": False,
    }
    assert {fund: result["rows"] for fund, result in first.items()} == {"AP2": 1, "AP3": 2}
    assert first["AP2"]["artifact_sha256"] == hashlib.sha256(AP2_PDF).hexdigest()

    factory = make_session_factory(engine)
    with factory() as session:
        releases = session.scalars(select(DataRelease).order_by(DataRelease.id)).all()
        facts = load_allocator_vintage(session, AS_KNOWN)
        fact_count = session.scalar(select(func.count()).select_from(AllocatorFact))

    assert fact_count == 3
    assert len(releases) == 2
    assert {release.source_family for release in releases} == {"AP_FUNDS"}
    assert {release.partition_key for release in releases} == {
        partition_key_for("AP2", date(2026, 6, 30)),
        partition_key_for("AP3", date(2026, 6, 30)),
    }
    assert {release.available_at for release in releases} == {AS_KNOWN.replace(tzinfo=None)}
    assert set(facts["parser_name"]) == {"direct-table-transcription"}
    assert set(facts["parser_version"]) == {"1"}
    assert set(facts["extraction_status"]) == {"model_visual_check"}
    assert set(facts["quality_flag"]) == {"none"}
    assert set(facts["source_url"]) == {
        "https://ap2.se/report.pdf",
        "https://ap3.se/report.pdf",
    }
    assert all(Path(path).is_file() for path in facts["artifact_path"])


def test_artifact_directory_uses_deterministic_reference_filenames(tmp_path):
    engine = make_engine(tmp_path / "allocators.db")
    artifact_dir = tmp_path / "downloads"
    artifact_dir.mkdir()
    (artifact_dir / "ap2-h1-2026.pdf").write_bytes(AP2_PDF)
    (artifact_dir / "ap3-h1-2026.pdf").write_bytes(AP3_PDF)

    summary = run_pipeline(
        reference_path=_reference(tmp_path),
        artifact_dir=artifact_dir,
        blob_root=tmp_path / "blobs",
        engine=engine,
        retrieved_at=AS_KNOWN,
    )

    assert set(summary) == {"AP2", "AP3"}
    assert all(result["created"] for result in summary.values())


@pytest.mark.parametrize(
    "mapping_factory, message",
    [
        (lambda paths: {"AP2": paths["AP2"]}, "missing artifact mappings"),
        (
            lambda paths: {**paths, "AP4": paths["AP2"]},
            "unexpected artifact mappings",
        ),
    ],
)
def test_pipeline_rejects_inexact_artifact_mapping_before_database_write(
    tmp_path, mapping_factory, message
):
    engine = make_engine(tmp_path / "allocators.db")
    artifacts = _artifacts(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_pipeline(
            reference_path=_reference(tmp_path),
            artifacts=mapping_factory(artifacts),
            blob_root=tmp_path / "blobs",
            engine=engine,
            retrieved_at=AS_KNOWN,
        )

    assert not (tmp_path / "allocators.db").exists()


def test_pipeline_preflights_all_hashes_before_writing_any_release(tmp_path):
    engine = make_engine(tmp_path / "allocators.db")
    artifacts = _artifacts(tmp_path)
    artifacts["AP3"].write_bytes(AP3_PDF + b"tampered")

    with pytest.raises(ValueError, match="AP3 artifact sha256 mismatch"):
        run_pipeline(
            reference_path=_reference(tmp_path),
            artifacts=artifacts,
            blob_root=tmp_path / "blobs",
            engine=engine,
            retrieved_at=AS_KNOWN,
        )

    assert not (tmp_path / "allocators.db").exists()


def test_pipeline_can_select_one_fund_and_rejects_both_artifact_modes(tmp_path):
    engine = make_engine(tmp_path / "allocators.db")
    artifacts = _artifacts(tmp_path)

    summary = run_pipeline(
        reference_path=_reference(tmp_path),
        artifacts={"AP3": artifacts["AP3"]},
        funds=("AP3",),
        blob_root=tmp_path / "blobs",
        engine=engine,
        retrieved_at=AS_KNOWN,
    )
    assert set(summary) == {"AP3"}

    with pytest.raises(ValueError, match="either artifacts or artifact_dir"):
        run_pipeline(
            reference_path=_reference(tmp_path),
            artifacts=artifacts,
            artifact_dir=tmp_path,
            blob_root=tmp_path / "other-blobs",
            engine=make_engine(tmp_path / "other.db"),
            retrieved_at=AS_KNOWN,
        )
