"""Checked denominator tests for the bounded sovereign-refinancing package."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from dalio.data_sources.ecb_refinancing import ECB_REFINANCING_SERIES
from dalio.data_sources.eurostat_refinancing import (
    EUROSTAT_REFINANCING_SERIES,
    build_eurostat_refinancing_url,
)
from dalio.data_sources.sovereign_refinancing import (
    DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH,
    SOVEREIGN_REFINANCING_MANIFEST_SHA256,
    load_checked_sovereign_refinancing_manifest,
    load_sovereign_refinancing_manifest,
)


def _payload() -> dict:
    return json.loads(DEFAULT_SOVEREIGN_REFINANCING_MANIFEST_PATH.read_text())


def _write(tmp_path, payload: dict):
    path = tmp_path / "sovereign_refinancing_v1.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def test_checked_manifest_freezes_the_48_partition_denominator():
    manifest = load_checked_sovereign_refinancing_manifest()

    assert manifest.semantic_sha256 == SOVEREIGN_REFINANCING_MANIFEST_SHA256
    assert len(manifest.partitions) == 48
    assert sum(item.phase == "harmonized_scalar" for item in manifest.partitions) == 31
    assert sum(item.phase == "national_native" for item in manifest.partitions) == 17
    assert manifest.core_sovereign_issuers == (
        "BR",
        "CN",
        "DE",
        "FR",
        "IN",
        "IT",
        "JP",
        "SE",
        "UK",
        "US",
    )
    assert manifest.comparison_entities == ("ES", "EA21_FIXED")
    assert len({item.partition_id for item in manifest.partitions}) == 48
    assert len({(item.publisher, item.native_identity) for item in manifest.partitions}) == 48


def test_harmonized_scalar_denominator_is_exact_and_excludes_se_variable_rate():
    manifest = load_checked_sovereign_refinancing_manifest()
    scalar = {
        item.partition_id for item in manifest.partitions if item.phase == "harmonized_scalar"
    }

    expected = {
        f"eurostat_{geo}_{measure}"
        for geo in ("de", "fr", "it", "es", "se")
        for measure in (
            "avg_residual_maturity",
            "total_debt_rmd_scope",
            "due_le1y",
            "foreign_currency",
            "apparent_cost",
        )
    }
    expected |= {f"eurostat_{geo}_lt_variable_rate" for geo in ("de", "fr", "it", "es")}
    expected |= {
        "ecb_ea21_avg_residual_maturity",
        "ecb_ea21_redemptions_1_12m_pct_gdp",
    }
    assert scalar == expected
    assert "eurostat_se_lt_variable_rate" not in scalar


def test_harmonized_manifest_rows_match_adapter_urls_and_native_identities():
    manifest = load_checked_sovereign_refinancing_manifest()
    scalar = [item for item in manifest.partitions if item.phase == "harmonized_scalar"]

    expected_eurostat = {
        (
            spec.country,
            f"{spec.dataset}|{'|'.join(code for _dimension, code in spec.dimension_codes)}",
            build_eurostat_refinancing_url(spec),
        )
        for spec in EUROSTAT_REFINANCING_SERIES
    }
    actual_eurostat = {
        (item.entity, item.native_identity, item.source_url)
        for item in scalar
        if item.publisher == "Eurostat"
    }
    assert actual_eurostat == expected_eurostat

    expected_ecb = {
        ("EA21_FIXED", spec.native_series_id, spec.url) for spec in ECB_REFINANCING_SERIES
    }
    actual_ecb = {
        (item.entity, item.native_identity, item.source_url)
        for item in scalar
        if item.publisher == "European Central Bank"
    }
    assert actual_ecb == expected_ecb
    assert {spec.country for spec in ECB_REFINANCING_SERIES} == {"EA21"}


def test_manifest_keeps_aggregate_comparator_out_of_sovereign_issuers():
    manifest = load_checked_sovereign_refinancing_manifest()
    rows = {item.partition_id: asdict(item) for item in manifest.partitions}

    assert rows["ecb_ea21_avg_residual_maturity"]["entity_kind"] == "comparison_aggregate"
    assert rows["eurostat_es_due_le1y"]["entity_kind"] == "comparison_sovereign"
    assert "EA21_FIXED" not in manifest.core_sovereign_issuers
    assert "ES" not in manifest.core_sovereign_issuers


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda payload: payload.update(partition_count=47), "partition_count"),
        (
            lambda payload: payload["partitions"].__setitem__(
                1,
                {
                    **payload["partitions"][1],
                    "partition_id": payload["partitions"][0]["partition_id"],
                },
            ),
            "partition IDs",
        ),
        (
            lambda payload: payload["partitions"][0].update(
                source_url="https://example.com/not-official"
            ),
            "official source_url",
        ),
        (
            lambda payload: payload["partitions"][0].update(family="headline_debt"),
            "unsupported family",
        ),
    ],
)
def test_manifest_rejects_denominator_and_source_boundary_drift(tmp_path, mutator, message):
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        load_sovereign_refinancing_manifest(_write(tmp_path, payload))


def test_checked_loader_rejects_semantic_change_even_when_shape_remains_valid(tmp_path):
    payload = _payload()
    payload["scope_rules"][0] += " Changed."
    path = _write(tmp_path, payload)

    changed = load_sovereign_refinancing_manifest(path)
    assert changed.semantic_sha256 != SOVEREIGN_REFINANCING_MANIFEST_SHA256
    with pytest.raises(ValueError, match="semantic SHA-256"):
        load_checked_sovereign_refinancing_manifest(path)
