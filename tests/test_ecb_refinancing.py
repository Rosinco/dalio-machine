from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dalio.data_sources.ecb_refinancing import (
    ECB_GFS_CSV_COLUMNS,
    ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY,
    ECB_GOV_REDEMPTIONS_1_12M,
    ECB_REFINANCING_SERIES,
    EcbRefinancingSeries,
    EcbRefinancingSource,
    build_ecb_gfs_csv_url,
    ecb_refinancing_catalogue_sha256,
    parse_ecb_refinancing_csv,
    validate_ecb_refinancing_catalogue,
)

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]


def _response(body: str | bytes, status_code: int = 200) -> MagicMock:
    content = body.encode("utf-8") if isinstance(body, str) else body
    response = MagicMock()
    response.content = content
    response.text = content.decode("utf-8", errors="replace")
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _base_row(
    spec: EcbRefinancingSeries,
    period: str,
    value: str,
    status: str = "A",
) -> dict[str, str]:
    row = dict.fromkeys(ECB_GFS_CSV_COLUMNS, "")
    row.update(
        {
            "KEY": spec.native_series_id,
            "FREQ": "M",
            "ADJUSTMENT": "N",
            "REF_AREA": "I10",
            "COUNTERPART_AREA": "W0",
            "REF_SECTOR": "S13",
            "COUNTERPART_SECTOR": "S1",
            "CONSOLIDATION": "N",
            "ACCOUNTING_ENTRY": spec.accounting_entry,
            "STO": spec.stock_flow,
            "INSTR_ASSET": "F3",
            "MATURITY": spec.maturity,
            "EXPENDITURE": "_Z",
            "UNIT_MEASURE": spec.native_unit_measure,
            "CURRENCY_DENOM": "_T",
            "VALUATION": "F",
            "PRICES": "V",
            "TRANSFORMATION": spec.transformation,
            "CUST_BREAKDOWN": "_T",
            "TIME_PERIOD": period,
            "OBS_VALUE": value,
            "OBS_STATUS": status,
            "CONF_STATUS": "F",
            "TIME_FORMAT": "P1M",
            "COMMENT_TS": f"{spec.reference_area_label} - pinned test series semantics",
            "COMPILING_ORG": "4F0",
            "DECIMALS": "4",
            "TIME_PER_COLLECT": spec.collection_period,
            "TITLE": spec.title,
            "UNIT_MULT": "0",
        }
    )
    return row


def _csv_text(
    spec: EcbRefinancingSeries,
    observations: list[tuple[str, str, str]],
    *,
    mutate: tuple[str, str] | None = None,
    columns: tuple[str, ...] = ECB_GFS_CSV_COLUMNS,
) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(columns)
    for period, value, status in observations:
        row = _base_row(spec, period, value, status)
        if mutate is not None:
            row[mutate[0]] = mutate[1]
        writer.writerow([row[column] for column in columns])
    return stream.getvalue()


def _complete_observations(
    spec: EcbRefinancingSeries,
    *,
    default_value: str = "1.0",
    default_status: str = "A",
    overrides: dict[str, tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    replacements = overrides or {}
    observations: list[tuple[str, str, str]] = []
    current = spec.expected_start
    while current <= spec.verified_through:
        period = current.strftime("%Y-%m")
        value, status = replacements.get(period, (default_value, default_status))
        observations.append((period, value, status))
        current = date(
            current.year + (1 if current.month == 12 else 0),
            1 if current.month == 12 else current.month + 1,
            1,
        )
    return observations


def test_exact_two_series_denominator_and_full_history_urls():
    assert len(ECB_REFINANCING_SERIES) == 2
    assert {spec.indicator for spec in ECB_REFINANCING_SERIES} == {
        "euro_area_gov_debt_avg_residual_maturity_years",
        "euro_area_gov_redemptions_1_12m_pct_gdp",
    }
    assert {spec.native_series_id for spec in ECB_REFINANCING_SERIES} == {
        "GFS.M.N.I10.W0.S13.S1.N.L.LE.F3.TT._Z.YR._T.F.V.A1._T",
        ("GFS.M.N.I10.W0.S13.S1.N.LD.F.F3.TS._Z.XDC_R_B1GQ_CY._T.F.V.C12._T"),
    }
    assert ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY.url == (
        "https://data-api.ecb.europa.eu/service/data/GFS/"
        "M.N.I10.W0.S13.S1.N.L.LE.F3.TT._Z.YR._T.F.V.A1._T?format=csvdata"
    )
    assert ECB_GOV_REDEMPTIONS_1_12M.url == (
        "https://data-api.ecb.europa.eu/service/data/GFS/"
        "M.N.I10.W0.S13.S1.N.LD.F.F3.TS._Z."
        "XDC_R_B1GQ_CY._T.F.V.C12._T?format=csvdata"
    )
    assert all(
        "startPeriod" not in spec.url and "endPeriod" not in spec.url
        for spec in ECB_REFINANCING_SERIES
    )
    assert all(spec.expected_start == date(2009, 12, 1) for spec in ECB_REFINANCING_SERIES)
    assert all(spec.verified_through == date(2026, 7, 1) for spec in ECB_REFINANCING_SERIES)
    validate_ecb_refinancing_catalogue()
    assert len(ecb_refinancing_catalogue_sha256()) == 64


def test_catalogue_and_url_builder_reject_unpinned_series():
    changed = replace(ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY, reference_area="I9")
    with pytest.raises(ValueError, match="metadata changed"):
        validate_ecb_refinancing_catalogue((changed, ECB_GOV_REDEMPTIONS_1_12M))
    stale_floor = replace(
        ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY,
        verified_through=date(2026, 6, 1),
    )
    with pytest.raises(ValueError, match="metadata changed"):
        validate_ecb_refinancing_catalogue((stale_floor, ECB_GOV_REDEMPTIONS_1_12M))
    with pytest.raises(ValueError, match="unsupported ECB GFS"):
        build_ecb_gfs_csv_url("GFS.M.N.I9.changed")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY,
            [
                (date(2009, 12, 1), 6.3947, "A"),
                (date(2010, 1, 1), 6.4428, "P"),
            ],
        ),
        (
            ECB_GOV_REDEMPTIONS_1_12M,
            [
                (date(2009, 12, 1), 13.9109, "A"),
                (date(2010, 1, 1), 13.5549, "P"),
            ],
        ),
    ],
)
def test_parser_emits_canonical_release_long_frame(
    spec: EcbRefinancingSeries,
    expected: list[tuple[date, float, str]],
):
    text = _csv_text(
        spec,
        _complete_observations(
            spec,
            overrides={
                "2009-12": (str(expected[0][1]), "A"),
                "2010-01": (str(expected[1][1]), "P"),
            },
        ),
    )

    frame = parse_ecb_refinancing_csv(text, spec)

    assert list(frame.columns) == _COLUMNS
    assert (
        list(frame[["date", "value", "status"]].head(2).itertuples(index=False, name=None))
        == expected
    )
    assert len(frame) == 200
    assert frame["date"].iloc[-1] == spec.verified_through
    assert set(frame["country"]) == {"EA21"}
    assert set(frame["indicator"]) == {spec.indicator}
    assert set(frame["source"]) == {"ECB_GFS"}
    assert set(frame["series_id"]) == {spec.native_series_id}
    assert frame.attrs["reference_area"] == "I10"
    assert frame.attrs["reference_sector"] == "S13"
    assert frame.attrs["consolidation"] == "N"
    assert frame.attrs["valuation"] == "F"
    assert frame.attrs["native_periods"][:2] == ("2009-12", "2010-01")
    assert frame.attrs["native_periods"][-1] == "2026-07"
    assert frame.attrs["native_observation_statuses"][:2] == ("A", "P")
    assert frame.attrs["source_updated_at"] is None


@pytest.mark.parametrize(
    ("column", "wrong", "message"),
    [
        ("KEY", "GFS.M.N.I9.WRONG", "KEY mismatch"),
        ("FREQ", "Q", "FREQ mismatch"),
        ("TIME_FORMAT", "P3M", "TIME_FORMAT mismatch"),
        ("UNIT_MEASURE", "EUR", "UNIT_MEASURE mismatch"),
        ("UNIT_MULT", "6", "UNIT_MULT mismatch"),
        ("REF_AREA", "I9", "REF_AREA mismatch"),
        ("REF_SECTOR", "S1311", "REF_SECTOR mismatch"),
        ("COUNTERPART_SECTOR", "S13", "COUNTERPART_SECTOR mismatch"),
        ("CONSOLIDATION", "C", "CONSOLIDATION mismatch"),
        ("VALUATION", "M", "VALUATION mismatch"),
        ("INSTR_ASSET", "GD", "INSTR_ASSET mismatch"),
        ("MATURITY", "T", "MATURITY mismatch"),
        ("TRANSFORMATION", "N", "TRANSFORMATION mismatch"),
        ("ACCOUNTING_ENTRY", "L", "ACCOUNTING_ENTRY mismatch"),
        ("STO", "LE", "STO mismatch"),
        ("TIME_PER_COLLECT", "A", "TIME_PER_COLLECT mismatch"),
        ("COMMENT_TS", "Euro area 20 - wrong composition", "composition label mismatch"),
    ],
)
def test_parser_rejects_dimension_drift(column: str, wrong: str, message: str):
    spec = ECB_GOV_REDEMPTIONS_1_12M
    text = _csv_text(spec, [("2009-12", "13.9", "A")], mutate=(column, wrong))

    with pytest.raises(ValueError, match=message):
        parse_ecb_refinancing_csv(text, spec)


def test_parser_rejects_unexpected_csv_schema_and_field_count():
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    missing_column = tuple(column for column in ECB_GFS_CSV_COLUMNS if column != "COMMENT_DSET")
    with pytest.raises(ValueError, match="schema"):
        parse_ecb_refinancing_csv(
            _csv_text(spec, [("2009-12", "6.4", "A")], columns=missing_column),
            spec,
        )

    text = _csv_text(spec, [("2009-12", "6.4", "A")])
    malformed = text.rstrip("\r\n") + ",unexpected\r\n"
    with pytest.raises(ValueError, match="field count"):
        parse_ecb_refinancing_csv(malformed, spec)


def test_native_statuses_and_nulls_are_preserved_without_zero_fill():
    spec = ECB_GOV_REDEMPTIONS_1_12M
    text = _csv_text(
        spec,
        _complete_observations(
            spec,
            overrides={
                "2009-12": ("13.9", "A"),
                "2010-01": ("", "P"),
                "2010-02": ("..", "E"),
                "2010-03": ("12.2", "P"),
            },
        ),
    )

    frame = parse_ecb_refinancing_csv(text, spec)

    selected = frame[frame["date"].isin({date(2009, 12, 1), date(2010, 3, 1)})]
    assert list(selected[["date", "value", "status"]].itertuples(index=False, name=None)) == [
        (date(2009, 12, 1), 13.9, "A"),
        (date(2010, 3, 1), 12.2, "P"),
    ]
    assert len(frame) == 198
    assert frame.attrs["native_observation_statuses"][:4] == ("A", "P", "E", "P")
    assert frame.attrs["missing_period_records"] == (
        {
            "native_period": "2010-01",
            "native_position": 1,
            "missing_kind": "blank",
            "native_token": "",
            "native_obs_status": "P",
            "reason": "publisher_missing_not_zero",
            "evidence": "ecb_csv_obs_value_cell",
        },
        {
            "native_period": "2010-02",
            "native_position": 2,
            "missing_kind": "null_token",
            "native_token": "..",
            "native_obs_status": "E",
            "reason": "publisher_missing_not_zero",
            "evidence": "ecb_csv_obs_value_cell",
        },
    )
    native = json.loads(frame.attrs["native_payload_json"])
    assert native["rows"][1]["OBS_VALUE"] == ""
    assert native["rows"][1]["OBS_STATUS"] == "P"
    missing = json.loads(frame.attrs["missing_provenance_json"])
    assert len(missing["records"]) == 2


@pytest.mark.parametrize(
    ("periods", "message"),
    [
        (["2009-12", "2009-12"], "duplicate observation periods"),
        (["2010-01", "2009-12"], "not strictly increasing"),
        (["2010-01"], "history start mismatch"),
        (["2009-12", "2010-02"], "cadence gap"),
        (["2009-Q4"], "invalid monthly period"),
        (["2009-13"], "invalid monthly period"),
    ],
)
def test_parser_rejects_duplicate_out_of_order_or_invalid_periods(
    periods: list[str],
    message: str,
):
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    observations = [(period, "6.4", "A") for period in periods]
    with pytest.raises(ValueError, match=message):
        parse_ecb_refinancing_csv(_csv_text(spec, observations), spec)


def test_parser_requires_verified_through_floor_and_a_finite_observation():
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    truncated = _complete_observations(spec)[:-1]
    with pytest.raises(ValueError, match="verified-through floor"):
        parse_ecb_refinancing_csv(_csv_text(spec, truncated), spec)

    all_missing = _complete_observations(spec, default_value="..")
    with pytest.raises(ValueError, match="no finite observations"):
        parse_ecb_refinancing_csv(_csv_text(spec, all_missing), spec)


def test_source_updated_at_is_utc_aware_and_zone_less_value_is_rejected():
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    observations = _complete_observations(spec)
    aware = parse_ecb_refinancing_csv(
        _csv_text(
            spec,
            observations,
            mutate=("LAST_UPDATE", "2026-08-03T12:30:00+02:00"),
        ),
        spec,
    )
    assert aware.attrs["source_updated_at"] == datetime(2026, 8, 3, 10, 30, tzinfo=UTC)
    assert aware.attrs["source_updated_at"].tzinfo is not None

    with pytest.raises(ValueError, match="lacks a timezone"):
        parse_ecb_refinancing_csv(
            _csv_text(
                spec,
                observations,
                mutate=("LAST_UPDATE", "2026-08-03T12:30:00"),
            ),
            spec,
        )


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "not-a-number"])
def test_parser_rejects_nonfinite_negative_or_nonnumeric_values(value: str):
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    with pytest.raises(ValueError, match="non-finite|negative|non-numeric"):
        parse_ecb_refinancing_csv(
            _csv_text(spec, [("2009-12", value, "A")]),
            spec,
        )


def test_parser_rejects_missing_or_non_native_status():
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    for status in ("", "provisional", "p"):
        with pytest.raises(ValueError, match="invalid OBS_STATUS"):
            parse_ecb_refinancing_csv(
                _csv_text(spec, [("2009-12", "6.4", status)]),
                spec,
            )


def test_source_archives_exact_response_native_payload_and_missingness_idempotently(tmp_path):
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    text = _csv_text(
        spec,
        _complete_observations(
            spec,
            overrides={"2009-12": ("6.3947", "A"), "2010-01": ("", "P")},
        ),
    )
    client = MagicMock()
    client.get.return_value = _response(text)
    source = EcbRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )

    first = source.fetch(spec, use_cache=False)
    second = source.fetch(spec, use_cache=True)

    client.get.assert_called_once_with(spec.url, timeout=60.0)
    for attr in (
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
    ):
        assert attr in first.attrs
    assert first.attrs["source_artifact_path"] == second.attrs["source_artifact_path"]
    assert (
        first.attrs["native_payload_artifact_path"] == second.attrs["native_payload_artifact_path"]
    )
    assert (
        first.attrs["missing_provenance_artifact_path"]
        == second.attrs["missing_provenance_artifact_path"]
    )

    source_path = Path(first.attrs["source_artifact_path"])
    native_path = Path(first.attrs["native_payload_artifact_path"])
    missing_path = Path(first.attrs["missing_provenance_artifact_path"])
    assert source_path.read_bytes() == text.encode("utf-8")
    assert first.attrs["source_artifact_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert (
        hashlib.sha256(native_path.read_bytes()).hexdigest() == first.attrs["native_payload_sha256"]
    )
    assert (
        hashlib.sha256(missing_path.read_bytes()).hexdigest()
        == first.attrs["missing_provenance_sha256"]
    )
    assert missing_path.read_text() == first.attrs["missing_provenance_json"]
    assert len(list((tmp_path / "artifacts").rglob("*.csv"))) == 1
    assert len(list((tmp_path / "artifacts").rglob("*.json"))) == 2


@pytest.mark.parametrize(
    "artifact_attr",
    ["source_artifact_path", "native_payload_artifact_path", "missing_provenance_artifact_path"],
)
def test_source_rejects_corrupt_content_addressed_artifact(tmp_path, artifact_attr: str):
    spec = ECB_GOV_DEBT_AVG_RESIDUAL_MATURITY
    text = _csv_text(spec, _complete_observations(spec))
    client = MagicMock()
    client.get.return_value = _response(text)
    source = EcbRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    first = source.fetch(spec, use_cache=False)
    corrupt_path = Path(first.attrs[artifact_attr])
    corrupt_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="corrupt archive"):
        source.fetch(spec, use_cache=True)

    assert corrupt_path.read_bytes() == b"corrupt"


def test_invalid_response_is_rejected_before_artifact_admission(tmp_path):
    spec = ECB_GOV_REDEMPTIONS_1_12M
    invalid = _csv_text(
        spec,
        [("2009-12", "13.9", "A")],
        mutate=("REF_AREA", "I9"),
    )
    client = MagicMock()
    client.get.return_value = _response(invalid)
    artifact_dir = tmp_path / "artifacts"
    source = EcbRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="REF_AREA mismatch"):
        source.fetch(spec, use_cache=False)

    assert not artifact_dir.exists()
    assert not list((tmp_path / "cache").glob("*.csv"))


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (
            _complete_observations(ECB_GOV_REDEMPTIONS_1_12M)[:-1],
            "verified-through floor",
        ),
        (
            _complete_observations(
                ECB_GOV_REDEMPTIONS_1_12M,
                default_value="",
            ),
            "no finite observations",
        ),
    ],
)
def test_truncated_or_all_missing_response_is_not_admitted(
    tmp_path,
    observations: list[tuple[str, str, str]],
    message: str,
):
    spec = ECB_GOV_REDEMPTIONS_1_12M
    client = MagicMock()
    client.get.return_value = _response(_csv_text(spec, observations))
    artifact_dir = tmp_path / "artifacts"
    cache_dir = tmp_path / "cache"
    source = EcbRefinancingSource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match=message):
        source.fetch(spec, use_cache=False)

    assert not artifact_dir.exists()
    assert not list(cache_dir.glob("*.csv"))


def test_non_utf8_response_is_rejected_before_cache_or_artifact_admission(tmp_path):
    spec = ECB_GOV_REDEMPTIONS_1_12M
    client = MagicMock()
    client.get.return_value = _response(b"\xff\xfe\x00not-utf8")
    artifact_dir = tmp_path / "artifacts"
    source = EcbRefinancingSource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="not valid UTF-8"):
        source.fetch(spec, use_cache=False)

    assert not artifact_dir.exists()
    assert not list((tmp_path / "cache").glob("*.csv"))
