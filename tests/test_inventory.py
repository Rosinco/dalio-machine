"""Read-only coverage inventory for the macro observatory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.imf_bop import BOP_COUNTRIES, BOP_SERIES
from dalio.data_sources.imf_positions import (
    DIP_FREQUENCIES,
    DIP_SERIES,
    IMF_POSITION_COUNTRIES,
    PIP_FREQUENCIES,
    PIP_SERIES,
)
from dalio.data_sources.money_liquidity import (
    BOE_M4EX_QUARTERLY,
    FED_M2,
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    MONEY_LIQUIDITY_SERIES,
    money_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    OFR_SHADOW_LIQUIDITY_SERIES,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.data_sources.worldbank_commodities import (
    CATALOGUE_GENERATOR_VERSION,
    CATALOGUE_SCHEMA_VERSION,
    SOURCE_WORLD_BANK_COMMODITIES,
    canonical_series_id,
    pink_sheet_vintage_label,
)
from dalio.data_sources.worldbank_qpsd import QPSD_COUNTRIES, QPSD_SERIES
from dalio.storage import inventory as inventory_module
from dalio.storage.db import (
    AllocatorFact,
    Claim,
    ClaimCitation,
    DataRelease,
    DataReleaseArtifact,
    DebtHolderPosition,
    DocumentExtraction,
    DocumentPage,
    Observation,
    ReleaseObservation,
    ReportDocument,
    init_db,
    make_engine,
)
from dalio.storage.inventory import build_observatory_inventory, render_inventory_summary
from dalio.storage.releases import make_partition_key


def _release(**overrides) -> dict:
    values = {
        "partition_key": "test:partition",
        "source_family": "TEST",
        "published_at": None,
        "available_at": datetime(2026, 8, 1, tzinfo=UTC),
        "retrieved_at": datetime(2026, 8, 2, tzinfo=UTC),
        "vintage_label": None,
        "source_url": "https://example.test/data",
        "content_sha256": "1" * 64,
        "row_count": 1,
    }
    values.update(overrides)
    return values


def _monthly_dates(start: date, end: date) -> list[date]:
    dates = []
    cursor = start
    while cursor <= end:
        dates.append(cursor)
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return dates


def _money_observations(dates: list[date], spec=FED_M2) -> list[dict]:
    return [
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": observed_on,
            "value": float(index),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
        for index, observed_on in enumerate(dates, start=1)
    ]


def _shadow_observations(spec, dates: list[date]) -> list[dict]:
    return [
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": observed_on,
            "value": float(index),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
        for index, observed_on in enumerate(dates, start=1)
    ]


def _shadow_release(
    spec,
    prefix: str,
    digest: str,
    row_count: int,
    *,
    payload_tag: str | None = None,
) -> dict:
    vintage_label = f"{prefix}{digest}"
    if payload_tag is not None:
        vintage_label = f"{vintage_label};p:{payload_tag}"
    return _release(
        partition_key=make_partition_key(
            spec.source_family,
            spec.native_series_id,
            spec.country,
            spec.indicator,
        ),
        source_family=spec.source_family,
        vintage_label=vintage_label,
        content_sha256=spec.native_series_id.encode().hex().ljust(64, "0")[:64],
        row_count=row_count,
    )


def _insert_test_release_artifacts(
    connection, release_id: int, spec, artifact_root
) -> dict[str, Path]:
    series_root = artifact_root / spec.native_series_id
    series_root.mkdir(parents=True, exist_ok=True)
    provenance_json = json.dumps(
        {"native_series_id": spec.native_series_id, "records": []},
        sort_keys=True,
        separators=(",", ":"),
    )
    missing_hash = hashlib.sha256(provenance_json.encode()).hexdigest()
    native_bytes = json.dumps(
        {"native_series_id": spec.native_series_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    native_hash = hashlib.sha256(native_bytes).hexdigest()
    native_path = series_root / "native.json"
    native_path.write_bytes(native_bytes)
    source_bytes = native_bytes if spec.source_family == "BIS_GLI" else b'{"bulk":"response"}'
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_path = series_root / "source.payload"
    source_path.write_bytes(source_bytes)
    artifacts = [("source_response", source_hash, source_path)]
    if spec.source_family != "BIS_GLI":
        missing_path = series_root / "missing.json"
        missing_path.write_text(provenance_json)
        artifacts.extend(
            [
                ("native_series_payload", native_hash, native_path),
                ("missingness_ledger", missing_hash, missing_path),
            ]
        )
    connection.execute(
        insert(DataReleaseArtifact),
        [
            {
                "release_id": release_id,
                "role": role,
                "artifact_sha256": artifact_hash,
                "artifact_path": str(path.resolve()),
                "native_payload_sha256": native_hash,
                "missing_provenance_sha256": missing_hash,
                "provenance_json": provenance_json,
            }
            for role, artifact_hash, path in artifacts
        ],
    )
    return {role: path for role, _artifact_hash, path in artifacts}


def test_empty_inventory_is_json_safe_and_distinguishes_absent_tables(tmp_path):
    engine = make_engine(tmp_path / "empty.db")
    init_db(engine)

    inventory = build_observatory_inventory(engine)

    assert json.loads(json.dumps(inventory)) == inventory
    assert inventory["schema_version"] == 1
    assert inventory["observations"]["row_count"] == 0
    assert inventory["releases"]["release_count"] == 0
    assert inventory["qpsd"]["expected_partitions"] == len(QPSD_COUNTRIES) * len(QPSD_SERIES)
    assert inventory["qpsd"]["stored_partitions"] == 0
    assert inventory["qpsd"]["not_reported_partitions"] == 0
    assert inventory["qpsd"]["missing_partitions"] == len(QPSD_COUNTRIES) * len(QPSD_SERIES)
    assert inventory["imf_bop"]["expected_partitions"] == len(BOP_COUNTRIES) * len(BOP_SERIES)
    assert inventory["imf_bop"]["missing_partitions"] == len(BOP_COUNTRIES) * len(BOP_SERIES)
    assert inventory["cross_border_positions"]["table_present"] is True
    assert inventory["cross_border_positions"]["row_count"] == 0
    assert inventory["cross_border_positions"]["current_row_count"] == 0
    expected_positions = len(IMF_POSITION_COUNTRIES) * (
        len(PIP_SERIES) * len(PIP_FREQUENCIES) + len(DIP_SERIES) * len(DIP_FREQUENCIES)
    )
    assert inventory["cross_border_positions"]["expected_partitions"] == expected_positions
    assert inventory["cross_border_positions"]["stored_partitions"] == 0
    assert inventory["cross_border_positions"]["missing_partitions"] == expected_positions
    assert inventory["cross_border_positions"]["gap"] == "empty"
    assert inventory["market_history"]["commodities"]["series_count"] == 0
    assert inventory["market_history"]["commodities"]["stored_series"] == 0
    assert inventory["market_history"]["commodities"]["ready_series"] == 0
    assert inventory["market_history"]["commodities"]["guard_failures"] == ["missing"]
    assert inventory["market_history"]["money_liquidity"]["expected_series"] == len(
        MONEY_LIQUIDITY_SERIES
    )
    assert inventory["market_history"]["money_liquidity"]["stored_series"] == 0
    assert inventory["reports"]["review_count"] == 0
    assert inventory["reports"]["review_outcomes"] == []
    assert (
        inventory["market_history"]["money_liquidity"]["catalogue_semantic_sha256"]
        == money_liquidity_catalogue_sha256()
    )
    shadow = inventory["market_history"]["shadow_liquidity"]
    expected_shadow = len(BIS_GLOBAL_LIQUIDITY_SERIES) + len(OFR_SHADOW_LIQUIDITY_SERIES)
    assert shadow["expected_series"] == expected_shadow
    assert shadow["stored_series"] == 0
    assert shadow["ready_series"] == 0
    assert shadow["guard_failed_series"] == 0
    assert shadow["missing_series"] == expected_shadow
    assert shadow["payload_provenance_unrecorded"] == 0
    assert shadow["provider_catalogue_semantic_sha256"] == {
        "BIS_GLI": bis_global_liquidity_catalogue_sha256(),
        "OFR_STFM": ofr_shadow_liquidity_catalogue_sha256(),
    }
    assert all(item["status"] == "missing" for item in shadow["series"])
    assert inventory["readiness"]["commodity_history"] == "empty"
    assert inventory["readiness"]["money_liquidity"] == "empty"
    assert inventory["readiness"]["shadow_liquidity"] == "empty"
    cross_border_table = next(
        row for row in inventory["tables"] if row["name"] == "cross_border_positions"
    )
    assert cross_border_table == {
        "name": "cross_border_positions",
        "present": True,
        "row_count": 0,
    }
    assert inventory["readiness"]["report_evidence"] == "empty"


def test_inventory_reports_structured_coverage_and_evidence_without_mutating(tmp_path):
    engine = make_engine(tmp_path / "populated.db")
    init_db(engine)
    qpsd_spec = QPSD_SERIES[0]
    bop_spec = BOP_SERIES[0]

    with engine.begin() as connection:
        qpsd_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="qpsd:se:total",
                    source_family="WORLD_BANK_QPSD",
                    source_url=(
                        "https://api.worldbank.org/v2/country/USA;SWE/indicator/"
                        f"{qpsd_spec.wb_code}?source=20"
                    ),
                    content_sha256="2" * 64,
                    row_count=2,
                )
            )
        ).inserted_primary_key[0]
        bop_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="bop:se:direct",
                    source_family="IMF_BOP",
                    content_sha256="3" * 64,
                )
            )
        ).inserted_primary_key[0]
        debt_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="debt:se",
                    source_family="SCB_FINANCIAL_ACCOUNTS",
                    content_sha256="4" * 64,
                )
            )
        ).inserted_primary_key[0]
        allocator_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="allocator:ap2",
                    source_family="AP_FUNDS",
                    content_sha256="5" * 64,
                )
            )
        ).inserted_primary_key[0]

        connection.execute(
            insert(Observation),
            [
                {
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2025, 10, 1),
                    "value": 30.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                },
                {
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 31.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                },
                {
                    "country": "SE",
                    "indicator": bop_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 10.0,
                    "source": "IMF_BOP",
                    "series_id": bop_spec.series_id,
                },
            ],
        )
        connection.execute(
            insert(ReleaseObservation),
            [
                {
                    "release_id": qpsd_release,
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2025, 10, 1),
                    "value": 30.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                    "status": "observed",
                },
                {
                    "release_id": qpsd_release,
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 31.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                    "status": "observed",
                },
                {
                    "release_id": bop_release,
                    "country": "SE",
                    "indicator": bop_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 10.0,
                    "source": "IMF_BOP",
                    "series_id": bop_spec.series_id,
                    "status": "observed",
                },
            ],
        )
        connection.execute(
            insert(DebtHolderPosition).values(
                release_id=debt_release,
                country="SE",
                date=date(2026, 1, 1),
                issuer_sector_code="S1311",
                issuer_sector_label="Central government",
                instrument_code="FL3000",
                instrument_label="Debt securities",
                holder_sector_code="S122",
                holder_sector_label="Deposit-taking corporations",
                measure_code="FM0103AS",
                measure_label="Closing balance",
                unit="SEK million",
                value=100.0,
                source="SCB_FINANCIAL_ACCOUNTS",
                series_id="TAB1203",
                status="observed",
            )
        )
        connection.execute(
            insert(AllocatorFact).values(
                release_id=allocator_release,
                fact_key="6" * 64,
                fund="AP2",
                as_of_date=date(2026, 6, 30),
                period_start=None,
                period_end=None,
                record_type="asset_allocation",
                item_code="listed_equity",
                reported_amount=10.0,
                reported_unit="SEK_bn",
                amount_sek_mn=10_000.0,
                exposure_pct=20.0,
                basis="fund_capital",
                row_role="component",
                physical_page=4,
                table_heading="Asset allocation",
                extraction_status="model_visual_check",
                quality_flag="none",
                notes=None,
                artifact_sha256="7" * 64,
                artifact_path="/blobs/report.pdf",
                source_url="https://ap2.se/report.pdf",
                parser_name="direct-table-transcription",
                parser_version="1",
            )
        )

        document_id = connection.execute(
            insert(ReportDocument).values(
                source_id="riksbank",
                report_family="monetary_policy_report",
                issue_key="2026-06",
                publisher="Sveriges Riksbank",
                jurisdiction="SE",
                title="Monetary Policy Report",
                language="en",
                document_date=date(2026, 6, 18),
                published_at=datetime(2026, 6, 18, tzinfo=UTC),
                available_at=datetime(2026, 6, 18, tzinfo=UTC),
                retrieved_at=datetime(2026, 8, 2, tzinfo=UTC),
                landing_url="https://riksbank.se/report",
                artifact_url="https://riksbank.se/report.pdf",
                mime_type="application/pdf",
                content_sha256="8" * 64,
                size_bytes=100,
                page_count=1,
                blob_path="/blobs/riksbank.pdf",
            )
        ).inserted_primary_key[0]
        extraction_id = connection.execute(
            insert(DocumentExtraction).values(
                document_id=document_id,
                extractor_name="pdftotext",
                extractor_version="1",
                extracted_at=datetime(2026, 8, 2, tzinfo=UTC),
                status="success",
                extracted_page_count=1,
                corpus_sha256="9" * 64,
                error=None,
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(DocumentPage).values(
                extraction_id=extraction_id,
                pdf_page=1,
                printed_page_label="1",
                text="Policy is restrictive.",
                text_sha256="a" * 64,
                char_count=22,
            )
        )
        claim_id = connection.execute(
            insert(Claim).values(
                claim_type="publisher_statement",
                statement="Policy is restrictive.",
                attribution_document_id=document_id,
                claim_series_key=None,
                topic_key="monetary_policy",
                geographies_json='["SE"]',
                status="verified",
                available_at=datetime(2026, 6, 18, tzinfo=UTC),
                created_by="test",
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(ClaimCitation).values(
                claim_id=claim_id,
                extraction_id=extraction_id,
                pdf_page_start=1,
                pdf_page_end=1,
                printed_locator="p. 1",
                section_title="Summary",
                evidence_excerpt="Policy is restrictive.",
                excerpt_sha256="b" * 64,
                support_role="supports",
                locator_verified_at=datetime(2026, 8, 2, tzinfo=UTC),
            )
        )

        connection.exec_driver_sql(
            """
            INSERT INTO cross_border_positions
                (release_id, dataset, reporter_country, reporter_code,
                 counterpart_country, counterpart_code, date, direction,
                 accounting_basis, instrument_code, instrument_label, frequency,
                 unit, value, source, native_indicator, series_id, status)
            VALUES
                (?, 'PIP', 'SE', 'SWE', 'US', 'USA', '2025-01-01', 'outward_assets',
                 'assets', 'portfolio_total', 'Total portfolio investment', 'A',
                 'USD', 1.0, 'IMF_PIP', 'P_TOTINV_P_USD', 'x', 'observed'),
                (?, 'PIP', 'SE', 'SWE', 'DE', 'DEU', '2024-01-01', 'outward_assets',
                 'assets', 'portfolio_total', 'Total portfolio investment', 'A',
                 'USD', 2.0, 'IMF_PIP', 'P_TOTINV_P_USD', 'y', 'observed')
            """,
            (bop_release, bop_release),
        )

    before = {}
    with engine.connect() as connection:
        for table in (Observation, DataRelease, DebtHolderPosition, AllocatorFact):
            before[table.__tablename__] = len(connection.execute(select(table)).all())

    inventory = build_observatory_inventory(engine)

    with engine.connect() as connection:
        after = {
            table.__tablename__: len(connection.execute(select(table)).all())
            for table in (Observation, DataRelease, DebtHolderPosition, AllocatorFact)
        }
    assert after == before

    qpsd_partition = inventory["qpsd"]["stored"][0]
    assert qpsd_partition == {
        "country": "SE",
        "indicator": qpsd_spec.indicator,
        "row_count": 2,
        "first_date": "2025-10-01",
        "latest_date": "2026-01-01",
    }
    assert inventory["qpsd"]["stored_partitions"] == 1
    assert inventory["qpsd"]["not_reported_partitions"] == 1
    assert inventory["qpsd"]["not_reported"] == [
        {
            "country": "US",
            "indicator": qpsd_spec.indicator,
            "series_id": qpsd_spec.series_id,
            "status": "not_reported",
        }
    ]
    assert inventory["qpsd"]["missing_partitions"] == (len(QPSD_COUNTRIES) * len(QPSD_SERIES) - 2)
    assert inventory["imf_bop"]["stored_partitions"] == 1
    assert inventory["imf_bop"]["missing_partitions"] == (len(BOP_COUNTRIES) * len(BOP_SERIES) - 1)

    assert inventory["debt_holders"]["current_row_count"] == 1
    assert inventory["debt_holders"]["instruments"] == [
        {"code": "FL3000", "label": "Debt securities"}
    ]
    assert inventory["debt_holders"]["holder_sectors"] == [
        {"code": "S122", "label": "Deposit-taking corporations"}
    ]
    assert inventory["allocators"]["funds"] == [
        {
            "fund": "AP2",
            "row_count": 1,
            "first_as_of_date": "2026-06-30",
            "latest_as_of_date": "2026-06-30",
            "record_types": ["asset_allocation"],
            "artifact_count": 1,
            "quality_flags": ["none"],
        }
    ]
    assert inventory["reports"]["document_count"] == 1
    assert inventory["reports"]["page_count"] == 1
    assert inventory["reports"]["claim_count"] == 1
    assert inventory["reports"]["citation_count"] == 1
    assert inventory["reports"]["review_count"] == 0
    assert inventory["reports"]["review_outcomes"] == []
    assert inventory["reports"]["documents"] == [
        {
            "publisher": "Sveriges Riksbank",
            "report_family": "monetary_policy_report",
            "document_count": 1,
            "first_document_date": "2026-06-18",
            "latest_document_date": "2026-06-18",
            "page_count": 1,
        }
    ]

    cross_border = inventory["cross_border_positions"]
    assert cross_border["table_present"] is True
    assert cross_border["row_count"] == 2
    assert cross_border["datasets"] == ["PIP"]
    assert cross_border["stored_partitions"] == 1
    assert cross_border["expected_partitions"] == 294
    assert cross_border["missing_partitions"] == 293
    assert cross_border["by_dataset_reporter"] == [
        {
            "dataset": "PIP",
            "reporter_country": "SE",
            "row_count": 2,
            "counterpart_count": 2,
            "first_date": "2024-01-01",
            "latest_date": "2025-01-01",
        }
    ]
    assert inventory["readiness"] == {
        "current_observations": "available",
        "immutable_release_history": "available",
        "sovereign_debt_anatomy": "partial",
        "cross_border_transactions": "partial",
        "debt_holder_positions": "available",
        "allocator_disclosures": "available",
        "report_evidence": "available",
        "bilateral_positions": "partial",
        "commodity_history": "empty",
        "money_liquidity": "empty",
        "shadow_liquidity": "empty",
    }
    summary = render_inventory_summary(inventory)
    assert "QPSD: 1/264 stored" in summary
    assert "IMF BOP transactions: 1/525 stored" in summary
    assert "Cross-border positions: 2 rows; 1/294 partitions" in summary


def test_inventory_surfaces_commodity_and_money_history_without_comparing_levels(tmp_path):
    engine = make_engine(tmp_path / "markets.db")
    init_db(engine)
    catalogue_hash = money_liquidity_catalogue_sha256()
    with engine.begin() as connection:
        money_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="9" * 64,
                    row_count=len(_monthly_dates(FED_M2.expected_start, date(2026, 7, 1))),
                )
            )
        )
        _insert_test_release_artifacts(
            connection,
            int(money_release.inserted_primary_key[0]),
            FED_M2,
            tmp_path / "money-artifacts",
        )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": "commodity_price_copper",
                    "date": date(2026, 7, 1),
                    "value": 9_000.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Copper",
                },
                {
                    "country": "WLD",
                    "indicator": "commodity_price_copper",
                    "date": date(2026, 8, 1),
                    "value": 9_100.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Copper",
                },
                {
                    "country": "WLD",
                    "indicator": "commodity_index_energy",
                    "date": date(2026, 8, 1),
                    "value": 120.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Energy",
                },
                *_money_observations(_monthly_dates(FED_M2.expected_start, date(2026, 7, 1))),
            ],
        )

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    commodities = inventory["market_history"]["commodities"]
    money = inventory["market_history"]["money_liquidity"]

    assert commodities["row_count"] == 3
    assert commodities["series_count"] == 2
    assert commodities["price_series_count"] == 1
    assert commodities["index_series_count"] == 1
    assert commodities["first_date"] == "2026-07-01"
    assert commodities["latest_date"] == "2026-08-01"
    assert commodities["stored_series"] == 2
    assert commodities["ready_series"] == 0
    assert commodities["status"] == "guard_failed"
    assert "expected_catalogue" in commodities["guard_failures"]
    assert "latest_release_group" in commodities["guard_failures"]
    assert money["stored_series"] == 1
    assert money["ready_series"] == 1
    assert money["guard_failed_series"] == 0
    assert money["missing_series"] == len(MONEY_LIQUIDITY_SERIES) - 1
    assert money["coverage_pct"] == round(100 / len(MONEY_LIQUIDITY_SERIES), 3)
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")
    assert fed["status"] == "ready"
    assert fed["storage_status"] == "stored"
    assert fed["guard_failures"] == []
    assert fed["unit"] == "USD billion"
    assert fed["frequency"] == "monthly"
    assert fed["perimeter"] == FED_M2.perimeter
    assert fed["definition_notes"] == FED_M2.definition_notes
    assert fed["parent_native_series_id"] is None
    assert fed["non_additive_groups"] == list(FED_M2.non_additive_groups)
    assert fed["catalogue_semantic_sha256"] == catalogue_hash
    assert fed["release_catalogue_semantic_sha256"] == catalogue_hash
    assert fed["catalogue_semantic_status"] == "match"
    assert fed["artifact_manifest_status"] == "valid"
    assert fed["artifact_manifest"]["roles"] == [
        "missingness_ledger",
        "native_series_payload",
        "source_response",
    ]
    assert money["catalogue_semantic_sha256"] == catalogue_hash
    assert money["catalogue_semantic_mismatches"] == 0
    assert money["catalogue_semantic_unrecorded"] == 0
    assert inventory["readiness"]["commodity_history"] == "empty"
    assert inventory["readiness"]["money_liquidity"] == "partial"
    summary = render_inventory_summary(inventory)
    assert "Commodity history: 0/87 expected series ready; 2 stored" in summary
    assert (
        f"Money/liquidity: 1/{len(MONEY_LIQUIDITY_SERIES)} pinned series ready; 1 stored"
    ) in summary
    assert f"catalogue {catalogue_hash[:12]}" in summary


def test_money_inventory_requires_an_intact_release_artifact_manifest(tmp_path):
    dates = _monthly_dates(FED_M2.expected_start, date(2026, 7, 1))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "money-corrupt-artifact.db")
    init_db(engine)
    with engine.begin() as connection:
        release_result = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="6" * 64,
                    row_count=len(dates),
                )
            )
        )
        paths = _insert_test_release_artifacts(
            connection,
            int(release_result.inserted_primary_key[0]),
            FED_M2,
            tmp_path / "money-artifacts",
        )
        connection.execute(insert(Observation), _money_observations(dates))

    paths["missingness_ledger"].write_bytes(b"corrupt")
    money = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")

    assert fed["status"] == "guard_failed"
    assert fed["artifact_manifest_status"] == "invalid"
    assert "missingness_ledger:artifact_hash_mismatch" in fed["artifact_manifest"]["failures"]
    assert "artifact_manifest_invalid" in fed["guard_failures"]
    assert money["ready_series"] == 0


def test_commodity_inventory_qualifies_complete_group_and_allows_expansion(tmp_path, monkeypatch):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    new_id = canonical_series_id("New benchmark", "Monthly Prices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 7, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 2)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 2)
    series = [
        (copper_id, "commodity_price_copper"),
        (energy_id, "commodity_index_energy"),
        (new_id, "commodity_price_new_benchmark"),
    ]
    months = [date(2026, 7, 1), date(2026, 8, 1)]
    group_label = pink_sheet_vintage_label("c" * 64)
    engine = make_engine(tmp_path / "commodity-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        for series_id, indicator in series:
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=group_label,
                        content_sha256=indicator.encode().hex().ljust(64, "0")[:64],
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": float(100 + month_index),
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator in series
                for month_index, observed_on in enumerate(months)
            ],
        )

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    commodities = inventory["market_history"]["commodities"]

    assert commodities["status"] == "ready"
    assert commodities["expected_series"] == 2
    assert commodities["stored_series"] == 3
    assert commodities["ready_series"] == 2
    assert commodities["extra_series"] == 1
    assert commodities["extra_series_ids"] == [new_id]
    assert commodities["stored_coverage_pct"] == 100.0
    assert commodities["coverage_pct"] == 100.0
    assert commodities["price_series_count"] == 2
    assert commodities["index_series_count"] == 1
    assert commodities["workbook_sha256"] == "c" * 64
    assert commodities["catalogue_schema_version"] == CATALOGUE_SCHEMA_VERSION
    assert commodities["catalogue_generator_version"] == CATALOGUE_GENERATOR_VERSION
    assert commodities["catalogue_semantic_status"] == "match"
    assert commodities["guard_failures"] == []
    assert inventory["readiness"]["commodity_history"] == "complete"


def test_commodity_inventory_requires_identical_release_group_clocks(tmp_path, monkeypatch):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 7, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 2)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 2)
    series = [
        (copper_id, "commodity_price_copper"),
        (energy_id, "commodity_index_energy"),
    ]
    months = [date(2026, 7, 1), date(2026, 8, 1)]
    group_label = pink_sheet_vintage_label("f" * 64)
    engine = make_engine(tmp_path / "commodity-mixed-clocks.db")
    init_db(engine)
    with engine.begin() as connection:
        for index, (series_id, indicator) in enumerate(series):
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=group_label,
                        retrieved_at=datetime(2026, 8, 2 + index, tzinfo=UTC),
                        content_sha256=str(index + 1) * 64,
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": 100.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator in series
                for observed_on in months
            ],
        )

    commodities = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "commodities"
    ]

    assert commodities["guard_failures"] == ["latest_release_group"]
    assert commodities["status"] == "guard_failed"
    assert commodities["ready_series"] == 0
    assert commodities["release_group_vintage_label"] is None


def test_commodity_inventory_surfaces_value_cadence_and_release_group_failures(
    tmp_path, monkeypatch
):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 6, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 3)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 1)
    engine = make_engine(tmp_path / "commodity-bad.db")
    init_db(engine)
    series = [
        (copper_id, "commodity_price_gold", "d" * 64),
        (energy_id, "commodity_index_energy", "e" * 64),
    ]
    with engine.begin() as connection:
        for series_id, indicator, digest in series:
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=pink_sheet_vintage_label(digest),
                        content_sha256=digest,
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": value,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator, _digest in series
                for observed_on, value in (
                    (date(2026, 6, 1), 100.0),
                    (date(2026, 8, 1), 0.0),
                )
            ],
        )

    commodities = build_observatory_inventory(engine, as_of=date(2026, 10, 8))["market_history"][
        "commodities"
    ]

    assert commodities["status"] == "guard_failed"
    assert commodities["stored_series"] == 2
    assert commodities["ready_series"] == 0
    assert commodities["nonpositive_rows"] == 2
    assert set(commodities["guard_failures"]) >= {
        "nonpositive_values",
        "canonical_indicator_mapping",
        "monthly_cadence",
        "minimum_history",
        "latest_lag",
        "latest_release_group",
    }


@pytest.mark.parametrize(
    ("dates", "vintage_label", "failure"),
    [
        (
            _monthly_dates(date(1959, 2, 1), date(2026, 7, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "expected_start",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2024, 12, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "minimum_observations",
        ),
        (
            [
                observed_on
                for observed_on in _monthly_dates(FED_M2.expected_start, date(2026, 7, 1))
                if observed_on != date(2000, 1, 1)
            ],
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "cadence",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2025, 12, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "latest_lag",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2026, 7, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{'0' * 64}",
            "catalogue_semantic_mismatch",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2026, 7, 1)),
            None,
            "catalogue_semantic_unrecorded",
        ),
    ],
)
def test_money_inventory_readiness_requires_every_history_guard(
    tmp_path,
    dates,
    vintage_label,
    failure,
):
    engine = make_engine(tmp_path / f"money-{failure}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=vintage_label,
                    content_sha256="8" * 64,
                    row_count=len(dates),
                )
            )
        )
        connection.execute(insert(Observation), _money_observations(dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    money = inventory["market_history"]["money_liquidity"]
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")

    assert fed["storage_status"] == "stored"
    assert fed["status"] == "guard_failed"
    assert failure in fed["guard_failures"]
    assert money["stored_series"] == 1
    assert money["ready_series"] == 0
    assert money["guard_failed_series"] == 1
    assert money["coverage_pct"] == 0.0
    assert inventory["readiness"]["money_liquidity"] == "empty"


def test_money_inventory_qualifies_quarterly_bridge_and_preserves_semantics(
    tmp_path,
    monkeypatch,
):
    spec = replace(
        BOE_M4EX_QUARTERLY,
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    dates = [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
    monkeypatch.setattr(inventory_module, "MONEY_LIQUIDITY_SERIES", (spec,))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "money-quarterly-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        money_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        spec.source_family,
                        spec.native_series_id,
                        spec.country,
                        spec.indicator,
                    ),
                    source_family=spec.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="7" * 64,
                    row_count=len(dates),
                )
            )
        )
        _insert_test_release_artifacts(
            connection,
            int(money_release.inserted_primary_key[0]),
            spec,
            tmp_path / "money-artifacts",
        )
        connection.execute(insert(Observation), _money_observations(dates, spec))

    money = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]
    item = money["series"][0]

    assert item["status"] == "ready"
    assert item["guard_failures"] == []
    assert item["perimeter"] == spec.perimeter
    assert item["definition_notes"] == spec.definition_notes
    assert item["parent_native_series_id"] == spec.parent_native_series_id
    assert item["non_additive_groups"] == list(spec.non_additive_groups)
    assert item["artifact_manifest_status"] == "valid"


@pytest.mark.parametrize(
    "dates",
    [
        [date(2026, 1, 1), date(2026, 7, 1)],
        [date(2026, 1, 1), date(2026, 5, 1)],
    ],
)
def test_money_inventory_rejects_noncontinuous_or_nonstart_quarters(
    tmp_path,
    monkeypatch,
    dates,
):
    spec = replace(
        BOE_M4EX_QUARTERLY,
        expected_start=date(2026, 1, 1),
        minimum_observations=2,
        max_latest_lag_days=365,
    )
    monkeypatch.setattr(inventory_module, "MONEY_LIQUIDITY_SERIES", (spec,))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / f"money-quarterly-{dates[-1].month}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        spec.source_family,
                        spec.native_series_id,
                        spec.country,
                        spec.indicator,
                    ),
                    source_family=spec.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="6" * 64,
                    row_count=len(dates),
                )
            )
        )
        connection.execute(insert(Observation), _money_observations(dates, spec))

    item = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]["series"][0]

    assert item["status"] == "guard_failed"
    assert "cadence" in item["guard_failures"]


def test_shadow_liquidity_inventory_qualifies_bis_and_ofr_native_cadences(
    tmp_path,
    monkeypatch,
):
    bis_spec = replace(
        BIS_GLOBAL_LIQUIDITY_SERIES[0],
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_complete_monthly_spec = replace(
        next(
            spec
            for spec in OFR_SHADOW_LIQUIDITY_SERIES
            if spec.cadence_policy == "complete_monthly"
        ),
        expected_start=date(2026, 1, 31),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_sparse_monthly_spec = replace(
        next(
            spec
            for spec in OFR_SHADOW_LIQUIDITY_SERIES
            if spec.cadence_policy == "sparse_monthly" and spec.parent_native_series_id is not None
        ),
        expected_start=date(2026, 1, 31),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_daily_spec = replace(
        next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.frequency == "daily"),
        expected_start=date(2026, 8, 31),
        minimum_observations=3,
        max_latest_lag_days=30,
        minimum_weekday_coverage_ratio=0.5,
    )
    monkeypatch.setattr(inventory_module, "BIS_GLOBAL_LIQUIDITY_SERIES", (bis_spec,))
    monkeypatch.setattr(
        inventory_module,
        "OFR_SHADOW_LIQUIDITY_SERIES",
        (ofr_complete_monthly_spec, ofr_sparse_monthly_spec, ofr_daily_spec),
    )
    histories = {
        bis_spec: [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)],
        ofr_complete_monthly_spec: [
            date(2026, 1, 31),
            date(2026, 2, 28),
            date(2026, 3, 31),
        ],
        # Missing February is allowed; observed MMF dates must still be month-end.
        ofr_sparse_monthly_spec: [date(2026, 1, 31), date(2026, 3, 31), date(2026, 7, 31)],
        # Missing Tuesday/Thursday is allowed; stored repo observations stay weekdays.
        ofr_daily_spec: [date(2026, 8, 31), date(2026, 9, 2), date(2026, 9, 4)],
    }
    bis_hash = bis_global_liquidity_catalogue_sha256()
    ofr_hash = ofr_shadow_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "shadow-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        for spec, dates in histories.items():
            prefix, digest = (
                (BIS_GLI_CATALOGUE_VINTAGE_PREFIX, bis_hash)
                if spec.source_family == "BIS_GLI"
                else (OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX, ofr_hash)
            )
            release_result = connection.execute(
                insert(DataRelease).values(
                    **_shadow_release(
                        spec,
                        prefix,
                        digest,
                        len(dates),
                        payload_tag=("a" * 24 if spec.source_family == "OFR_STFM" else None),
                    )
                )
            )
            _insert_test_release_artifacts(
                connection,
                int(release_result.inserted_primary_key[0]),
                spec,
                tmp_path / "artifacts",
            )
            connection.execute(insert(Observation), _shadow_observations(spec, dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    shadow = inventory["market_history"]["shadow_liquidity"]

    assert shadow["expected_series"] == 4
    assert shadow["stored_series"] == 4
    assert shadow["ready_series"] == 4
    assert shadow["missing_series"] == 0
    assert shadow["guard_failed_series"] == 0
    assert shadow["coverage_pct"] == 100.0
    assert shadow["provider_catalogue_semantic_sha256"] == {
        "BIS_GLI": bis_hash,
        "OFR_STFM": ofr_hash,
    }
    assert shadow["providers"]["BIS_GLI"]["ready_series"] == 1
    assert shadow["providers"]["OFR_STFM"]["ready_series"] == 3
    assert inventory["readiness"]["shadow_liquidity"] == "complete"

    by_id = {item["native_series_id"]: item for item in shadow["series"]}
    bis_item = by_id[bis_spec.native_series_id]
    assert bis_item["status"] == "ready"
    assert bis_item["measure_kind"] == "credit_stock"
    assert bis_item["side"] == bis_spec.claim_side
    assert bis_item["claim_side"] == bis_spec.claim_side
    assert bis_item["from_sector"] == bis_spec.from_sector
    assert bis_item["to_sector"] == bis_spec.to_sector
    assert bis_item["instrument"] == bis_spec.instrument
    assert bis_item["aggregation_role"] == "currency_total"
    assert bis_item["parent_native_series_id"] is None
    assert bis_item["parents"] == []
    assert bis_item["non_additive_groups"] == [bis_spec.non_additive_group]
    assert bis_item["native_unit"] == bis_spec.native_unit
    assert bis_item["unit"] == bis_spec.unit
    assert bis_item["currency"] == "USD"
    assert bis_item["frequency"] == "quarterly"

    monthly_item = by_id[ofr_sparse_monthly_spec.native_series_id]
    assert monthly_item["status"] == "ready"
    assert monthly_item["measure_kind"] == ofr_sparse_monthly_spec.measure_kind
    assert monthly_item["side"] == ofr_sparse_monthly_spec.economic_side
    assert monthly_item["claim_side"] == ofr_sparse_monthly_spec.claim_side
    assert monthly_item["from_sector"] == ofr_sparse_monthly_spec.from_sector
    assert monthly_item["to_sector"] == ofr_sparse_monthly_spec.to_sector
    assert monthly_item["instrument"] == ofr_sparse_monthly_spec.instrument
    assert monthly_item["collateral_scope"] == (ofr_sparse_monthly_spec.collateral_scope)
    assert monthly_item["aggregation_role"] == "component"
    assert monthly_item["parent_native_series_id"] == (
        ofr_sparse_monthly_spec.parent_native_series_id
    )
    assert monthly_item["parents"] == [ofr_sparse_monthly_spec.parent_native_series_id]
    assert monthly_item["non_additive_groups"] == list(ofr_sparse_monthly_spec.non_additive_groups)
    assert monthly_item["required_notes_phrases"] == list(
        ofr_sparse_monthly_spec.required_notes_phrases
    )
    assert monthly_item["cadence_policy"] == "sparse_monthly"
    assert monthly_item["native_unit"] == ofr_sparse_monthly_spec.native_unit
    assert monthly_item["unit"] == ofr_sparse_monthly_spec.unit
    assert monthly_item["currency"] == ofr_sparse_monthly_spec.currency
    assert monthly_item["frequency"] == "monthly"
    assert monthly_item["release_payload_provenance_tag"] == "a" * 24

    summary = render_inventory_summary(inventory)
    assert "Shadow liquidity: 4/4 series ready; 4 stored" in summary
    assert f"BIS {bis_hash[:12]}" in summary
    assert f"OFR {ofr_hash[:12]}" in summary


@pytest.mark.parametrize(
    ("damage", "manifest_failure"),
    [
        ("corrupt", "source_response:artifact_hash_mismatch"),
        ("delete", "source_response:artifact_missing"),
    ],
)
def test_shadow_liquidity_inventory_rehashes_retained_artifacts(
    tmp_path,
    monkeypatch,
    damage,
    manifest_failure,
):
    spec = replace(
        BIS_GLOBAL_LIQUIDITY_SERIES[0],
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    monkeypatch.setattr(inventory_module, "BIS_GLOBAL_LIQUIDITY_SERIES", (spec,))
    monkeypatch.setattr(inventory_module, "OFR_SHADOW_LIQUIDITY_SERIES", ())
    dates = [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
    engine = make_engine(tmp_path / f"artifact-{damage}.db")
    init_db(engine)
    with engine.begin() as connection:
        release_result = connection.execute(
            insert(DataRelease).values(
                **_shadow_release(
                    spec,
                    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
                    bis_global_liquidity_catalogue_sha256(),
                    len(dates),
                )
            )
        )
        paths = _insert_test_release_artifacts(
            connection,
            int(release_result.inserted_primary_key[0]),
            spec,
            tmp_path / "artifacts",
        )
        connection.execute(insert(Observation), _shadow_observations(spec, dates))

    source_path = paths["source_response"]
    if damage == "corrupt":
        source_path.write_bytes(b"corrupt")
    else:
        source_path.unlink()

    item = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "shadow_liquidity"
    ]["series"][0]

    assert item["status"] == "guard_failed"
    assert item["artifact_manifest_status"] == "invalid"
    assert manifest_failure in item["artifact_manifest"]["failures"]
    assert "artifact_manifest_invalid" in item["guard_failures"]


_SMALL_BIS_SHADOW_SPEC = replace(
    BIS_GLOBAL_LIQUIDITY_SERIES[0],
    expected_start=date(2026, 1, 1),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_COMPLETE_MONTHLY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.cadence_policy == "complete_monthly"),
    expected_start=date(2026, 1, 31),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_SPARSE_MONTHLY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.cadence_policy == "sparse_monthly"),
    expected_start=date(2026, 1, 31),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_DAILY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.frequency == "daily"),
    expected_start=date(2026, 8, 31),
    minimum_observations=2,
    max_latest_lag_days=30,
)


@pytest.mark.parametrize(
    ("spec", "dates", "digest", "payload_tag", "failure"),
    [
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 4, 1), date(2026, 7, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "expected_start",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "minimum_observations",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1), date(2026, 7, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "quarterly_cadence",
        ),
        (
            replace(_SMALL_BIS_SHADOW_SPEC, max_latest_lag_days=30),
            [date(2026, 1, 1), date(2026, 4, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "latest_lag",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)],
            "0" * 64,
            None,
            "catalogue_semantic_mismatch",
        ),
        (
            _SMALL_OFR_SPARSE_MONTHLY_SPEC,
            [date(2026, 1, 31), date(2026, 3, 30)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "calendar_month_end",
        ),
        (
            _SMALL_OFR_COMPLETE_MONTHLY_SPEC,
            [date(2026, 1, 31), date(2026, 3, 31)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "monthly_cadence",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 5)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "business_day_cadence",
        ),
        (
            replace(_SMALL_OFR_DAILY_SPEC, max_internal_gap_days=3),
            [date(2026, 8, 31), date(2026, 9, 4)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "a" * 24,
            "maximum_internal_gap",
        ),
        (
            replace(_SMALL_OFR_DAILY_SPEC, minimum_weekday_coverage_ratio=0.9),
            [date(2026, 8, 31), date(2026, 9, 2), date(2026, 9, 4)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "a" * 24,
            "minimum_weekday_coverage",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 2)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "z" * 24,
            "payload_provenance_unrecorded",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 2)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "payload_provenance_unrecorded",
        ),
    ],
)
def test_shadow_liquidity_inventory_rejects_guard_or_hash_failures(
    tmp_path,
    monkeypatch,
    spec,
    dates,
    digest,
    payload_tag,
    failure,
):
    is_bis = spec.source_family == "BIS_GLI"
    monkeypatch.setattr(
        inventory_module,
        "BIS_GLOBAL_LIQUIDITY_SERIES",
        (spec,) if is_bis else (),
    )
    monkeypatch.setattr(
        inventory_module,
        "OFR_SHADOW_LIQUIDITY_SERIES",
        () if is_bis else (spec,),
    )
    prefix = BIS_GLI_CATALOGUE_VINTAGE_PREFIX if is_bis else OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX
    engine = make_engine(tmp_path / f"shadow-{failure}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_shadow_release(
                    spec,
                    prefix,
                    digest,
                    len(dates),
                    payload_tag=payload_tag,
                )
            )
        )
        connection.execute(insert(Observation), _shadow_observations(spec, dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    shadow = inventory["market_history"]["shadow_liquidity"]
    item = shadow["series"][0]

    assert item["storage_status"] == "stored"
    assert item["status"] == "guard_failed"
    assert failure in item["guard_failures"]
    assert shadow["stored_series"] == 1
    assert shadow["ready_series"] == 0
    assert shadow["guard_failed_series"] == 1
    assert shadow["coverage_pct"] == 0.0
    assert inventory["readiness"]["shadow_liquidity"] == "empty"


def test_inventory_handles_a_database_with_only_an_older_table(tmp_path):
    engine = make_engine(tmp_path / "old.db")
    Observation.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Observation).values(
                country="US",
                indicator="policy_rate",
                date=date(2026, 1, 1),
                value=3.0,
                source="FRED",
                series_id="DFF",
            )
        )

    inventory = build_observatory_inventory(engine)

    assert inventory["observations"]["row_count"] == 1
    assert inventory["releases"]["release_count"] is None
    assert inventory["debt_holders"]["table_present"] is False
    assert inventory["reports"]["document_count"] is None
    assert inventory["cross_border_positions"]["table_present"] is False
    assert inventory["cross_border_positions"]["row_count"] is None
