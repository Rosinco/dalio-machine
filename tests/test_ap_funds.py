"""Strict validation for page-located AP-fund reference facts."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.ap_funds import (
    ALLOCATOR_FACT_COLUMNS,
    AP_FUNDS_H1_2026_REFERENCE,
    load_ap_fund_reference,
)


def _minimal_payload(pdf: bytes = b"%PDF-1.4\nfixture\n%%EOF\n") -> dict:
    return {
        "schema_version": 1,
        "method_note": "Direct transcription checked against the rendered PDF.",
        "releases": [
            {
                "fund": "AP2",
                "report_date": "2026-06-30",
                "title": "AP2 Half-Year Report 2026",
                "source_url": "https://ap2.se/report.pdf",
                "official_domains": ["ap2.se"],
                "sha256": hashlib.sha256(pdf).hexdigest(),
                "page_count": 9,
                "rows": [
                    {
                        "record_type": "asset_allocation",
                        "item_code": "listed_equities",
                        "reported_amount": 10.5,
                        "reported_unit": "SEK_bn",
                        "amount_sek_mn": 10500,
                        "exposure_pct": 42.0,
                        "basis": "actual_portfolio_exposure",
                        "row_role": "detail",
                        "physical_page": 4,
                        "table_heading": "Asset-class exposure",
                    }
                ],
            }
        ],
    }


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "ap-funds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_h1_2026_reference_has_exact_provenance_and_normalized_rows():
    reference = load_ap_fund_reference(AP_FUNDS_H1_2026_REFERENCE)

    assert reference.schema_version == 1
    assert "model-checked, not a named human review" in reference.method_note
    assert [
        (release.fund, len(release.facts), release.page_count) for release in reference.releases
    ] == [
        ("AP2", 21, 9),
        ("AP3", 12, 7),
        ("AP4", 15, 13),
    ]
    assert {release.fund: release.sha256 for release in reference.releases} == {
        "AP2": "0014316e60401290e5d62913c2ec858fedb87e5c6dce538c60198589d6760abb",
        "AP3": "3372a2e27df9654af6b8183b83a08a6c42dcb7595da9736ca7996ad7095d95d7",
        "AP4": "275a5f72cc7fab87fc8356989eebf50cb5e0ecf960da42c1b59b875f3a26f6aa",
    }

    for release in reference.releases:
        assert list(release.facts.columns) == list(ALLOCATOR_FACT_COLUMNS)
        assert set(release.facts["fund"]) == {release.fund}
        assert set(release.facts["as_of_date"]) == {date(2026, 6, 30)}
        assert set(release.facts["extraction_status"]) == {"model_visual_check"}
        assert release.facts["physical_page"].between(1, release.page_count).all()
        assert release.artifact_filename == f"{release.fund.lower()}-h1-2026.pdf"

    ap2, ap3, ap4 = reference.releases
    ap2_transfer = ap2.facts.loc[ap2.facts["item_code"] == "transfer_in_from_ap6"].iloc[0]
    assert ap2_transfer["record_type"] == "reorganisation_transfer"
    assert ap2_transfer["amount_sek_mn"] == 31_796
    assert ap2_transfer["period_start"] == date(2026, 1, 1)
    assert ap2_transfer["period_end"] == date(2026, 6, 30)

    ap3_total = ap3.facts.loc[
        (ap3.facts["record_type"] == "asset_allocation") & (ap3.facts["item_code"] == "total")
    ].iloc[0]
    assert ap3_total["exposure_pct"] == 102.9
    assert ap3_total["quality_flag"] == "publisher_non_additive"

    ap4_zero = ap4.facts.loc[ap4.facts["item_code"] == "other_assets"].iloc[0]
    assert ap4_zero["reported_amount"] == 0.0
    assert ap4_zero["amount_sek_mn"] == 0.0
    assert ap4_zero["exposure_pct"] == 0.0


def test_loader_expands_defaults_and_preserves_page_locator(tmp_path):
    release = load_ap_fund_reference(_write_payload(tmp_path, _minimal_payload())).releases[0]
    row = release.facts.iloc[0]

    assert row.to_dict() == {
        "fund": "AP2",
        "as_of_date": date(2026, 6, 30),
        "period_start": None,
        "period_end": None,
        "record_type": "asset_allocation",
        "item_code": "listed_equities",
        "reported_amount": 10.5,
        "reported_unit": "SEK_bn",
        "amount_sek_mn": 10_500.0,
        "exposure_pct": 42.0,
        "basis": "actual_portfolio_exposure",
        "row_role": "detail",
        "physical_page": 4,
        "table_heading": "Asset-class exposure",
        "extraction_status": "model_visual_check",
        "quality_flag": "none",
        "notes": None,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version=1.0), "schema_version"),
        (lambda payload: payload.update(unexpected=True), "unexpected fields"),
        (lambda payload: payload["releases"][0].update(sha256="ABC"), "sha256"),
        (
            lambda payload: payload["releases"][0].update(
                source_url="https://untrusted.example/report.pdf"
            ),
            "official domain",
        ),
        (lambda payload: payload["releases"][0].update(page_count=True), "page_count"),
        (lambda payload: payload["releases"][0]["rows"][0].update(physical_page=10), "page"),
        (
            lambda payload: payload["releases"][0]["rows"][0].update(amount_sek_mn=10.5),
            "normalization",
        ),
        (
            lambda payload: payload["releases"][0]["rows"][0].update(period_start="2026-01-01"),
            "both be set",
        ),
        (
            lambda payload: payload["releases"][0]["rows"][0].update(hidden_field="x"),
            "unexpected fields",
        ),
    ],
)
def test_loader_fails_closed_on_malformed_reference(tmp_path, mutate, message):
    payload = copy.deepcopy(_minimal_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_ap_fund_reference(_write_payload(tmp_path, payload))


def test_loader_rejects_duplicate_release_and_fact_identity(tmp_path):
    duplicate_fund = _minimal_payload()
    duplicate_fund["releases"].append(copy.deepcopy(duplicate_fund["releases"][0]))
    with pytest.raises(ValueError, match="duplicate fund"):
        load_ap_fund_reference(_write_payload(tmp_path, duplicate_fund))

    duplicate_fact = _minimal_payload()
    duplicate_fact["releases"][0]["rows"].append(
        copy.deepcopy(duplicate_fact["releases"][0]["rows"][0])
    )
    with pytest.raises(ValueError, match="duplicate semantic fact"):
        load_ap_fund_reference(_write_payload(tmp_path, duplicate_fact))
