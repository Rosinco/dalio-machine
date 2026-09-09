"""Offline contract tests for the Eurostat refinancing-risk scalar adapter."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dalio.data_sources.eurostat_refinancing import (
    APPARENT_COST_SERIES,
    AVG_RESIDUAL_MATURITY_SERIES,
    DUE_LE1Y_SERIES,
    EUROSTAT_REFINANCING_SERIES,
    FOREIGN_CURRENCY_SERIES,
    INDICATOR_APPARENT_COST,
    INDICATOR_AVG_RESIDUAL_MATURITY,
    INDICATOR_DUE_LE1Y,
    INDICATOR_FOREIGN_CURRENCY,
    INDICATOR_LT_VARIABLE_RATE,
    INDICATOR_RMD_SCOPE,
    LT_VARIABLE_RATE_SERIES,
    RMD_SCOPE_SERIES,
    EurostatRefinancingResponseError,
    EurostatRefinancingSource,
    build_eurostat_refinancing_url,
    parse_eurostat_refinancing_json,
)

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]
_DIMENSIONS = {
    "gov_10dd_rmd": ("freq", "sector", "maturity", "na_item", "unit", "geo", "time"),
    "gov_10dd_dcur": ("freq", "sector", "currency", "na_item", "unit", "geo", "time"),
    "gov_10dd_acd": ("freq", "sector", "unit", "geo", "time"),
    "gov_10dd_ggd": (
        "freq",
        "na_item",
        "sector2",
        "sector",
        "maturity",
        "unit",
        "geo",
        "time",
    ),
}
_TEST_RMD_SPEC = replace(
    AVG_RESIDUAL_MATURITY_SERIES[0],
    verified_start_year=2021,
    verified_through_year=2023,
)
_TEST_DUE_SPEC = replace(
    DUE_LE1Y_SERIES[0],
    verified_start_year=2020,
    verified_through_year=2023,
)


def _payload(
    spec=_TEST_RMD_SPEC,
    *,
    years=(2021, 2022, 2023),
    values=None,
    statuses=None,
    dimension_order=None,
):
    order = tuple(dimension_order or _DIMENSIONS[spec.dataset])
    codes = spec.dimensions | {"time": None}
    dimension = {}
    for dimension_name in reversed(order):
        if dimension_name == "time":
            # Deliberately reverse insertion order: numeric category positions,
            # not JSON-object order, define the native sequence.
            index = {str(year): position for position, year in reversed(tuple(enumerate(years)))}
        else:
            index = {codes[dimension_name]: 0}
        dimension[dimension_name] = {"category": {"index": index}}
    payload = {
        "version": "2.0",
        "class": "dataset",
        "label": spec.title,
        "source": "ESTAT",
        "updated": "2026-06-05T23:00:00+0200",
        "value": values if values is not None else {"0": 1.25, "1": 2.5, "2": 3.75},
        "id": list(order),
        "size": [len(years) if dimension_name == "time" else 1 for dimension_name in order],
        "dimension": dimension,
        "extension": {
            "lang": "EN",
            "id": spec.dataset.upper(),
            "agencyId": "ESTAT",
            "version": "1.0",
            "datastructure": {
                "id": spec.dataset.upper(),
                "agencyId": "ESTAT",
                "version": "48.0",
            },
        },
    }
    if statuses is not None:
        payload["status"] = statuses
    return payload


def _body(payload, *, pretty=False):
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")


def _response(body: bytes, status_code: int = 200):
    response = MagicMock()
    response.content = body
    response.text = body.decode("utf-8", errors="replace")
    response.status_code = status_code
    return response


def test_catalogue_has_exact_29_partitions_and_explicit_denominators():
    assert (
        *AVG_RESIDUAL_MATURITY_SERIES,
        *RMD_SCOPE_SERIES,
        *DUE_LE1Y_SERIES,
        *FOREIGN_CURRENCY_SERIES,
        *APPARENT_COST_SERIES,
        *LT_VARIABLE_RATE_SERIES,
    ) == EUROSTAT_REFINANCING_SERIES
    assert len(EUROSTAT_REFINANCING_SERIES) == 29
    assert len({spec.series_id for spec in EUROSTAT_REFINANCING_SERIES}) == 29
    assert len({(spec.country, spec.indicator) for spec in EUROSTAT_REFINANCING_SERIES}) == 29
    assert {
        indicator: sum(spec.indicator == indicator for spec in EUROSTAT_REFINANCING_SERIES)
        for indicator in {
            INDICATOR_AVG_RESIDUAL_MATURITY,
            INDICATOR_RMD_SCOPE,
            INDICATOR_DUE_LE1Y,
            INDICATOR_FOREIGN_CURRENCY,
            INDICATOR_APPARENT_COST,
            INDICATOR_LT_VARIABLE_RATE,
        }
    } == {
        INDICATOR_AVG_RESIDUAL_MATURITY: 5,
        INDICATOR_RMD_SCOPE: 5,
        INDICATOR_DUE_LE1Y: 5,
        INDICATOR_FOREIGN_CURRENCY: 5,
        INDICATOR_APPARENT_COST: 5,
        INDICATOR_LT_VARIABLE_RATE: 4,
    }
    assert {spec.country for spec in RMD_SCOPE_SERIES} == {"DE", "FR", "IT", "ES", "SE"}
    assert {spec.country for spec in LT_VARIABLE_RATE_SERIES} == {"DE", "FR", "IT", "ES"}
    assert all(spec.dimensions["freq"] == "A" for spec in EUROSTAT_REFINANCING_SERIES)
    assert all(spec.dimensions["sector"] == "S13" for spec in EUROSTAT_REFINANCING_SERIES)


def test_url_builder_pins_exact_codes_and_never_filters_history():
    spec = AVG_RESIDUAL_MATURITY_SERIES[0]

    assert build_eurostat_refinancing_url(spec) == (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        "gov_10dd_rmd?lang=en&freq=A&sector=S13&maturity=TOTAL&na_item=GD&unit=YR&geo=DE"
    )
    for candidate in EUROSTAT_REFINANCING_SERIES:
        url = build_eurostat_refinancing_url(candidate)
        assert "time=" not in url
        assert "sinceTimePeriod" not in url
        assert "untilTimePeriod" not in url
        assert "lastTimePeriod" not in url


def test_parser_handles_dimension_and_category_order_permutations():
    spec = _TEST_RMD_SPEC
    native_order = _DIMENSIONS[spec.dataset]
    first = parse_eurostat_refinancing_json(
        _body(_payload(spec, statuses={"1": "p"})),
        spec,
    )
    second = parse_eurostat_refinancing_json(
        _body(
            _payload(
                spec,
                statuses={"1": "p"},
                dimension_order=("time", *reversed(native_order[:-1])),
            )
        ),
        spec,
    )

    assert list(first.columns) == _COLUMNS
    assert first.to_dict("records") == second.to_dict("records")
    assert list(first[["date", "value", "status"]].itertuples(index=False, name=None)) == [
        (date(2021, 1, 1), 1.25, "observed"),
        (date(2022, 1, 1), 2.5, "p"),
        (date(2023, 1, 1), 3.75, "observed"),
    ]
    assert set(first["series_id"]) == {"GOV_10DD_RMD.A.S13.TOTAL.GD.YR.DE"}
    assert first.attrs["source_updated_at"] == datetime(2026, 6, 5, 21, tzinfo=UTC)


def test_parser_preserves_zero_flags_explicit_null_and_sparse_absence():
    spec = _TEST_DUE_SPEC
    payload = _payload(
        spec,
        years=(2020, 2021, 2022, 2023),
        values={"0": 0, "1": None, "3": 4.5},
        statuses={"0": "", "1": "c", "2": "u", "3": "P"},
    )

    frame = parse_eurostat_refinancing_json(_body(payload), spec)

    assert list(frame[["date", "value", "status"]].itertuples(index=False, name=None)) == [
        (date(2020, 1, 1), 0.0, "observed"),
        (date(2023, 1, 1), 4.5, "p"),
    ]
    assert frame.attrs["missing_period_records"] == (
        {
            "native_period": "2021",
            "native_position": 1,
            "missing_kind": "json_null",
            "native_token": None,
            "reason": "publisher_null_not_zero",
            "evidence": "eurostat_jsonstat_value_cell",
            "native_status": "c",
            "status_presence": "explicit",
        },
        {
            "native_period": "2022",
            "native_position": 2,
            "missing_kind": "sparse_absent",
            "native_token": None,
            "reason": "publisher_sparse_absence_not_zero",
            "evidence": "eurostat_jsonstat_value_cell",
            "native_status": "u",
            "status_presence": "explicit",
        },
    )
    ledger = json.loads(frame.attrs["missing_provenance_json"])
    assert ledger["records"] == list(frame.attrs["missing_period_records"])
    assert [record["native_status"] for record in ledger["status_records"]] == [
        "",
        "c",
        "u",
        "P",
    ]
    assert ledger["missing_value_policy"].endswith("never zero-fill or impute them")


def test_missing_endpoint_cells_satisfy_axis_floor_but_are_not_fabricated():
    payload = _payload(
        _TEST_RMD_SPEC,
        values={"0": None, "1": 2.5},
        statuses={"0": "c", "2": "u"},
    )

    frame = parse_eurostat_refinancing_json(_body(payload), _TEST_RMD_SPEC)

    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2022, 1, 1), 2.5)
    ]
    assert [record["native_period"] for record in frame.attrs["missing_period_records"]] == [
        "2021",
        "2023",
    ]
    assert [record["missing_kind"] for record in frame.attrs["missing_period_records"]] == [
        "json_null",
        "sparse_absent",
    ]


@pytest.mark.parametrize(
    ("years", "values", "message"),
    [
        ((2022, 2023), {"0": 2.0, "1": 3.0}, "starts in 2022"),
        ((2021, 2022), {"0": 1.0, "1": 2.0}, "ends in 2022"),
        ((2021, 2023), {"0": 1.0, "1": 3.0}, "time axis has a gap"),
    ],
)
def test_parser_rejects_truncated_or_gapped_native_history(years, values, message):
    with pytest.raises(EurostatRefinancingResponseError, match=message):
        parse_eurostat_refinancing_json(
            _body(_payload(_TEST_RMD_SPEC, years=years, values=values)),
            _TEST_RMD_SPEC,
        )


def test_parser_rejects_complete_axis_with_no_finite_observations():
    with pytest.raises(EurostatRefinancingResponseError, match="has no finite observations"):
        parse_eurostat_refinancing_json(
            _body(_payload(_TEST_RMD_SPEC, values={})),
            _TEST_RMD_SPEC,
        )


def _wrong_dataset(payload):
    payload["extension"]["id"] = "GOV_10DD_DCUR"


def _extra_dimension(payload):
    payload["id"].append("unexpected")
    payload["size"].append(1)
    payload["dimension"]["unexpected"] = {"category": {"index": {"X": 0}}}


def _wrong_frequency(payload):
    payload["dimension"]["freq"]["category"]["index"] = {"Q": 0}


def _wrong_sector(payload):
    payload["dimension"]["sector"]["category"]["index"] = {"S1311": 0}


def _wrong_unit(payload):
    payload["dimension"]["unit"]["category"]["index"] = {"PC_GDP": 0}


def _invalid_year(payload):
    payload["dimension"]["time"]["category"]["index"] = {
        "2021": 0,
        "2022-Q1": 1,
        "2023": 2,
    }


def _size_drift(payload):
    payload["size"][payload["id"].index("time")] = 4


def _non_numeric(payload):
    payload["value"]["0"] = "1.25"


def _non_string_status(payload):
    payload["status"] = {"0": 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_wrong_dataset, "dataset identity mismatch"),
        (_extra_dimension, "dimension schema mismatch"),
        (_wrong_frequency, "dimension 'freq' code mismatch"),
        (_wrong_sector, "dimension 'sector' code mismatch"),
        (_wrong_unit, "dimension 'unit' code mismatch"),
        (_invalid_year, "annual period is invalid"),
        (_size_drift, "size does not match"),
        (_non_numeric, "non-numeric value"),
        (_non_string_status, "non-string status"),
    ],
)
def test_parser_rejects_exact_code_and_schema_drift(mutation, message):
    payload = _payload()
    mutation(payload)

    with pytest.raises(EurostatRefinancingResponseError, match=message):
        parse_eurostat_refinancing_json(_body(payload), _TEST_RMD_SPEC)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_non_standard_non_finite_json_numbers(token):
    body = _body(_payload()).replace(b"1.25", token.encode("ascii"), 1)

    with pytest.raises(EurostatRefinancingResponseError, match="non-standard numeric token"):
        parse_eurostat_refinancing_json(body, _TEST_RMD_SPEC)


def test_fetch_archives_exact_canonical_and_missingness_payloads_idempotently(tmp_path):
    spec = _TEST_RMD_SPEC
    source_body = _body(_payload(spec, values={"0": 1.0, "1": None, "2": 3.0}), pretty=True)
    client = MagicMock()
    client.get.return_value = _response(source_body)
    artifact_dir = tmp_path / "artifacts"
    source = EurostatRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    first = source.fetch(spec, use_cache=False)
    second = source.fetch(spec, use_cache=True)

    url = build_eurostat_refinancing_url(spec)
    client.get.assert_called_once_with(url, timeout=30.0)
    required_attrs = {
        "source_url",
        "source_artifact_path",
        "source_artifact_sha256",
        "native_payload_artifact_path",
        "native_payload_sha256",
        "missing_provenance_artifact_path",
        "missing_provenance_sha256",
        "missing_provenance_json",
        "missing_period_records",
        "source_updated_at",
    }
    assert set(first.attrs) == required_attrs
    assert first.attrs == second.attrs
    source_path = Path(first.attrs["source_artifact_path"])
    native_path = Path(first.attrs["native_payload_artifact_path"])
    missing_path = Path(first.attrs["missing_provenance_artifact_path"])
    assert source_path.read_bytes() == source_body
    assert hashlib.sha256(source_body).hexdigest() == first.attrs["source_artifact_sha256"]
    expected_native = json.dumps(
        _payload(spec, values={"0": 1.0, "1": None, "2": 3.0}),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert native_path.read_bytes() == expected_native
    assert (
        hashlib.sha256(native_path.read_bytes()).hexdigest() == first.attrs["native_payload_sha256"]
    )
    assert missing_path.read_text(encoding="utf-8") == first.attrs["missing_provenance_json"]
    assert (
        hashlib.sha256(missing_path.read_bytes()).hexdigest()
        == first.attrs["missing_provenance_sha256"]
    )
    assert len(list((artifact_dir / "source-responses").rglob("*.json"))) == 1
    assert len(list((artifact_dir / "native-payloads").rglob("*.json"))) == 1
    assert len(list((artifact_dir / "missingness-ledgers").rglob("*.json"))) == 1


def test_equivalent_json_has_distinct_source_but_same_native_artifact(tmp_path):
    spec = _TEST_RMD_SPEC
    compact = _body(_payload(spec))
    pretty = _body(_payload(spec), pretty=True)
    client = MagicMock()
    client.get.side_effect = [_response(compact), _response(pretty)]
    source = EurostatRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )

    first = source.fetch(spec, use_cache=False)
    second = source.fetch(spec, use_cache=False)

    assert first.attrs["source_artifact_sha256"] != second.attrs["source_artifact_sha256"]
    assert first.attrs["native_payload_sha256"] == second.attrs["native_payload_sha256"]
    assert (
        first.attrs["native_payload_artifact_path"] == second.attrs["native_payload_artifact_path"]
    )


def test_fetch_rejects_corrupt_content_addressed_artifact_without_overwrite(tmp_path):
    spec = _TEST_RMD_SPEC
    source_body = _body(_payload(spec))
    digest = hashlib.sha256(source_body).hexdigest()
    artifact_dir = tmp_path / "artifacts"
    destination = artifact_dir / "source-responses" / digest[:2] / f"{digest}.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    client = MagicMock()
    client.get.return_value = _response(source_body)
    source = EurostatRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
        attempts=1,
    )

    with pytest.raises(ValueError, match="corrupt archive"):
        source.fetch(spec, use_cache=False)

    assert destination.read_bytes() == b"corrupt"
    assert not (artifact_dir / "native-payloads").exists()
    assert not (artifact_dir / "missingness-ledgers").exists()


def test_invalid_response_is_neither_archived_nor_cached(tmp_path):
    payload = deepcopy(_payload())
    payload["dimension"]["sector"]["category"]["index"] = {"S1311": 0}
    client = MagicMock()
    client.get.return_value = _response(_body(payload))
    artifact_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    source = EurostatRefinancingSource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        attempts=1,
    )

    with pytest.raises(EurostatRefinancingResponseError, match="sector.*code mismatch"):
        source.fetch(_TEST_RMD_SPEC, use_cache=False)

    assert not artifact_dir.exists()
    assert not list(cache_dir.glob("*.json"))


def test_truncated_history_is_neither_archived_nor_cached(tmp_path):
    truncated = _body(
        _payload(
            _TEST_RMD_SPEC,
            years=(2021, 2022),
            values={"0": 1.0, "1": 2.0},
        )
    )
    client = MagicMock()
    client.get.return_value = _response(truncated)
    artifact_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    source = EurostatRefinancingSource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
        attempts=1,
    )

    with pytest.raises(EurostatRefinancingResponseError, match="ends in 2022"):
        source.fetch(_TEST_RMD_SPEC, use_cache=False)

    assert not artifact_dir.exists()
    assert not list(cache_dir.glob("*.json"))


def test_retryable_http_failure_retries_but_permanent_404_does_not(tmp_path):
    spec = _TEST_RMD_SPEC
    valid = _response(_body(_payload(spec)))
    retry_client = MagicMock()
    retry_client.get.side_effect = [_response(b"temporary", 503), valid]
    retry_source = EurostatRefinancingSource(
        client=retry_client,
        cache_dir=tmp_path / "retry-cache",
        artifact_dir=tmp_path / "retry-artifacts",
        attempts=2,
        retry_backoff_seconds=0,
    )

    assert len(retry_source.fetch(spec, use_cache=False)) == 3
    assert retry_client.get.call_count == 2

    permanent_client = MagicMock()
    permanent_client.get.return_value = _response(b"not found", 404)
    permanent_source = EurostatRefinancingSource(
        client=permanent_client,
        cache_dir=tmp_path / "permanent-cache",
        artifact_dir=tmp_path / "permanent-artifacts",
        attempts=3,
        retry_backoff_seconds=0,
    )
    with pytest.raises(ValueError, match=r"request failed \(404\)"):
        permanent_source.fetch(spec, use_cache=False)
    assert permanent_client.get.call_count == 1
    assert not (tmp_path / "permanent-artifacts").exists()
