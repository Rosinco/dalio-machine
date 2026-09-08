"""World Bank QPSD source and release pipeline tests (mocked HTTP only)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.worldbank_qpsd import (
    PER_PAGE,
    QPSD_SERIES,
    QPSD_SOURCE,
    QpsdApiError,
    QpsdResponseError,
    QpsdSeriesSpec,
    WorldBankQpsdSource,
    parse_qpsd_period,
)
from dalio.pipelines.fetch_sovereign_debt import run_pipeline
from dalio.storage.db import DataRelease, Observation, make_engine
from dalio.storage.releases import make_partition_key, release_history

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _page(page: int, pages: int, total: int, rows: list[dict]) -> str:
    return json.dumps(
        [
            {
                "page": page,
                "pages": pages,
                "per_page": PER_PAGE,
                "total": total,
                "sourceid": "20",
                "lastupdated": "2026-08-28",
            },
            rows,
        ]
    )


def _observation(
    iso3: str,
    period: str,
    value: object,
    code: str = "DP.DOD.DECT.CR.CG.Z1",
) -> dict:
    return {
        "indicator": {"id": code, "value": "Central government debt"},
        "country": {"id": iso3[:2], "value": iso3},
        "countryiso3code": iso3,
        "date": period,
        "value": value,
        "unit": "",
        "obs_status": "",
        "decimal": 1,
    }


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    return response


def _basket():
    return (get_country("US"), get_country("SE"), get_country("EU"))


def _frame(
    spec: QpsdSeriesSpec,
    values: dict[str, list[tuple[date, float]]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": country,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": QPSD_SOURCE,
                "series_id": spec.series_id,
            }
            for country, observations in values.items()
            for observed_on, value in observations
        ],
        columns=_COLUMNS,
    )


class FakeQpsd:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, tuple[str, ...], bool]] = []

    def fetch(self, spec, countries, *, use_cache=True):
        self.calls.append((spec.series_id, tuple(country.iso2 for country in countries), use_cache))
        return self.frames[spec.indicator].copy()


def _at(month: int) -> datetime:
    return datetime(2026, month, 8, 12, tzinfo=UTC)


def test_catalogue_is_exactly_the_twelve_central_government_pct_gdp_codes():
    expected = {
        "central_gov_debt_total_pct_gdp": "DP.DOD.DECT.CR.CG.Z1",
        "central_gov_debt_short_term_pct_gdp": "DP.DOD.DSTC.CR.CG.Z1",
        "central_gov_debt_lt_due_1y_pct_gdp": "DP.DOD.DLTC.CR.L1.CG.Z1",
        "central_gov_debt_lt_due_over_1y_pct_gdp": "DP.DOD.DLTC.CR.M1.CG.Z1",
        "central_gov_debt_securities_pct_gdp": "DP.DOD.DLDS.CR.CG.Z1",
        "central_gov_debt_loans_pct_gdp": "DP.DOD.DLLO.CR.CG.Z1",
        "central_gov_debt_domestic_currency_pct_gdp": "DP.DOD.DECN.CR.CG.Z1",
        "central_gov_debt_foreign_currency_pct_gdp": "DP.DOD.DECF.CR.CG.Z1",
        "central_gov_debt_domestic_creditors_pct_gdp": "DP.DOD.DECD.CR.CG.Z1",
        "central_gov_debt_external_creditors_pct_gdp": "DP.DOD.DECX.CR.CG.Z1",
        "central_gov_debt_d1_pct_gdp": "DP.DOD.DLD1.CR.CG.Z1",
        "central_gov_debt_d2a_pct_gdp": "DP.DOD.DLD2A.CR.CG.Z1",
    }

    assert {spec.indicator: spec.wb_code for spec in QPSD_SERIES} == expected
    assert all(spec.unit == "percent_of_gdp" for spec in QPSD_SERIES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1999Q1", date(1999, 1, 1)),
        ("2026-Q2", date(2026, 4, 1)),
        ("2026Q3", date(2026, 7, 1)),
        ("2026-Q4", date(2026, 10, 1)),
    ],
)
def test_parse_qpsd_period_returns_quarter_start(raw, expected):
    assert parse_qpsd_period(raw) == expected


@pytest.mark.parametrize("raw", ["2026", "2026Q0", "2026Q5", "Q1-2026", None])
def test_parse_qpsd_period_rejects_non_quarters(raw):
    with pytest.raises(ValueError, match="quarterly period"):
        parse_qpsd_period(raw)


def test_fetch_uses_source_20_paginates_maps_iso3_and_drops_nulls(tmp_path):
    spec = QPSD_SERIES[0]
    client = MagicMock()
    client.get.side_effect = [
        _response(
            _page(
                1,
                2,
                4,
                [
                    _observation("USA", "2025Q4", "121.5"),
                    _observation("SWE", "2025Q4", None),
                ],
            )
        ),
        _response(
            _page(
                2,
                2,
                4,
                [
                    _observation("EMU", "2026-Q1", 88.0),
                    _observation("SWE", "2026Q1", 34.25),
                ],
            )
        ),
    ]
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        page_pause_seconds=0,
        retry_backoff_seconds=0,
    )

    frame = source.fetch(spec, _basket(), use_cache=False)

    assert client.get.call_count == 2
    first_url = client.get.call_args_list[0].args[0]
    second_url = client.get.call_args_list[1].args[0]
    assert f"/country/USA;SWE;EMU/indicator/{spec.wb_code}" in first_url
    assert "source=20" in first_url
    assert f"per_page={PER_PAGE}" in first_url
    assert "page=1" in first_url and "page=2" in second_url
    assert "date=" not in first_url and "mrv=" not in first_url
    assert list(frame.columns) == _COLUMNS
    assert frame[["country", "date", "value"]].to_dict("records") == [
        {"country": "EU", "date": date(2026, 1, 1), "value": 88.0},
        {"country": "SE", "date": date(2026, 1, 1), "value": 34.25},
        {"country": "US", "date": date(2025, 10, 1), "value": 121.5},
    ]
    assert frame["indicator"].unique().tolist() == [spec.indicator]
    assert frame["source"].unique().tolist() == [QPSD_SOURCE]
    assert frame["series_id"].unique().tolist() == [spec.wb_code]


def test_valid_empty_response_preserves_canonical_long_shape(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_page(1, 0, 0, []))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path)

    frame = source.fetch(QPSD_SERIES[0], _basket(), use_cache=False)

    assert frame.empty
    assert list(frame.columns) == _COLUMNS


def test_valid_null_empty_response_is_treated_as_not_reported(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(
        json.dumps(
            [
                {"page": 1, "pages": 0, "per_page": PER_PAGE, "total": 0, "sourceid": "20"},
                None,
            ]
        )
    )
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path)

    frame = source.fetch(QPSD_SERIES[0], _basket(), use_cache=False)

    assert frame.empty
    assert list(frame.columns) == _COLUMNS


def test_error_envelope_and_404_fail_closed_without_retry(tmp_path):
    error = json.dumps([{"message": [{"id": "120", "key": "Invalid value", "value": "bad code"}]}])
    client = MagicMock()
    client.get.return_value = _response(error)
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        attempts=3,
        retry_backoff_seconds=0,
    )
    with pytest.raises(QpsdApiError, match="bad code"):
        source.fetch(QPSD_SERIES[0], _basket(), use_cache=False)
    assert client.get.call_count == 1

    client.reset_mock()
    client.get.return_value = _response("not found", 404)
    with pytest.raises(ValueError, match="404"):
        source.fetch(QPSD_SERIES[0], _basket(), use_cache=False)
    assert client.get.call_count == 1


def test_server_error_retries_then_succeeds(tmp_path):
    client = MagicMock()
    client.get.side_effect = [
        _response("temporary", 503),
        _response(_page(1, 1, 1, [_observation("USA", "2025Q4", 120.0)])),
    ]
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        attempts=2,
        retry_backoff_seconds=0,
        page_pause_seconds=0,
    )

    frame = source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)

    assert client.get.call_count == 2
    assert frame["value"].tolist() == [120.0]


def test_cache_hit_avoids_second_http_call(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_page(1, 1, 1, [_observation("USA", "2025Q4", 120.0)]))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path)
    countries = (get_country("US"),)

    first = source.fetch(QPSD_SERIES[0], countries, use_cache=True)
    second = source.fetch(QPSD_SERIES[0], countries, use_cache=True)

    assert client.get.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_semantically_invalid_page_never_poison_caches(tmp_path):
    client = MagicMock()
    client.get.side_effect = [
        _response(
            _page(
                1,
                1,
                1,
                [_observation("USA", "2025Q4", 120.0, code="WRONG")],
            )
        ),
        _response(_page(1, 1, 1, [_observation("USA", "2025Q4", 120.0)])),
    ]
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        attempts=1,
        retry_backoff_seconds=0,
    )
    countries = (get_country("US"),)

    with pytest.raises(QpsdResponseError, match="native series"):
        source.fetch(QPSD_SERIES[0], countries, use_cache=True)
    frame = source.fetch(QPSD_SERIES[0], countries, use_cache=True)

    assert client.get.call_count == 2
    assert frame["value"].tolist() == [120.0]


def test_no_page_is_cached_until_the_complete_response_is_semantically_valid(tmp_path):
    spec = QPSD_SERIES[0]
    first = _page(1, 2, 2, [_observation("USA", "2025Q4", 120.0)])
    bad_second = _page(
        2,
        2,
        2,
        [_observation("USA", "2026Q1", 121.0, code="WRONG")],
    )
    good_second = _page(2, 2, 2, [_observation("USA", "2026Q1", 121.0)])
    client = MagicMock()
    client.get.side_effect = [
        _response(first),
        _response(bad_second),
        _response(first),
        _response(good_second),
    ]
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        attempts=1,
        retry_backoff_seconds=0,
        page_pause_seconds=0,
    )
    countries = (get_country("US"),)

    with pytest.raises(QpsdResponseError, match="native series"):
        source.fetch(spec, countries, use_cache=True)
    assert list(tmp_path.glob("*.json")) == []

    frame = source.fetch(spec, countries, use_cache=True)

    assert client.get.call_count == 4
    assert frame["value"].tolist() == [120.0, 121.0]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_response_source_id_must_be_qpsd_source_20(tmp_path):
    payload = json.loads(_page(1, 1, 1, [_observation("USA", "2025Q4", 1.0)]))
    payload[0]["sourceid"] = "2"
    client = MagicMock()
    client.get.return_value = _response(json.dumps(payload))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match="source 2, expected 20"):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


def test_response_without_optional_source_id_is_still_accepted(tmp_path):
    payload = json.loads(_page(1, 1, 1, [_observation("USA", "2025Q4", 1.0)]))
    payload[0].pop("sourceid")
    client = MagicMock()
    client.get.return_value = _response(json.dumps(payload))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    frame = source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)

    assert frame["value"].tolist() == [1.0]


@pytest.mark.parametrize("source_id", [True, 20.5, "20.5", "not-a-source"])
def test_response_rejects_malformed_source_provenance(source_id, tmp_path):
    payload = json.loads(_page(1, 1, 1, [_observation("USA", "2025Q4", 1.0)]))
    payload[0]["sourceid"] = source_id
    client = MagicMock()
    client.get.return_value = _response(json.dumps(payload))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match="sourceid is invalid"):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("page", True),
        ("page", 1.5),
        ("pages", True),
        ("pages", 1.5),
        ("per_page", True),
        ("per_page", 1.5),
        ("total", False),
        ("total", 0.5),
    ],
)
def test_response_rejects_boolean_and_fractional_pagination_metadata(
    field, invalid_value, tmp_path
):
    payload = json.loads(_page(1, 1, 1, [_observation("USA", "2025Q4", 1.0)]))
    payload[0][field] = invalid_value
    client = MagicMock()
    client.get.return_value = _response(json.dumps(payload))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match=field):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


def test_response_rejects_a_different_page_size_than_requested(tmp_path):
    payload = json.loads(_page(1, 1, 1, [_observation("USA", "2025Q4", 1.0)]))
    payload[0]["per_page"] = PER_PAGE - 1
    client = MagicMock()
    client.get.return_value = _response(json.dumps(payload))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match="per_page"):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


@pytest.mark.parametrize("indicator", [None, {}])
def test_observation_requires_exact_native_indicator_metadata(indicator, tmp_path):
    row = _observation("USA", "2025Q4", 1.0)
    row["indicator"] = indicator
    client = MagicMock()
    client.get.return_value = _response(_page(1, 1, 1, [row]))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match="indicator metadata|native series"):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_observation("ZZZ", "2025Q4", 1.0)], "unrequested country"),
        ([_observation("USA", "2025", 1.0)], "invalid quarter"),
        (
            [
                _observation("USA", "2025Q4", 1.0),
                _observation("USA", "2025-Q4", 2.0),
            ],
            "duplicate",
        ),
    ],
)
def test_malformed_observations_fail_closed(rows, message, tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_page(1, 1, len(rows), rows))
    source = WorldBankQpsdSource(
        client=client,
        cache_dir=tmp_path,
        attempts=1,
        retry_backoff_seconds=0,
    )

    with pytest.raises(QpsdResponseError, match=message):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


def test_incomplete_pagination_is_rejected(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_page(1, 1, 2, [_observation("USA", "2025Q4", 1.0)]))
    source = WorldBankQpsdSource(client=client, cache_dir=tmp_path, attempts=1)

    with pytest.raises(QpsdResponseError, match="received 1 of 2"):
        source.fetch(QPSD_SERIES[0], (get_country("US"),), use_cache=False)


def test_pipeline_splits_basket_into_country_native_series_releases(tmp_path):
    total, foreign = QPSD_SERIES[0], QPSD_SERIES[7]
    countries = (get_country("US"), get_country("SE"))
    source = FakeQpsd(
        {
            total.indicator: _frame(
                total,
                {
                    "US": [(date(2025, 7, 1), 120.0), (date(2025, 10, 1), 121.0)],
                    "SE": [(date(2025, 10, 1), 34.0)],
                },
            ),
            foreign.indicator: _frame(
                foreign,
                {"US": [(date(2025, 10, 1), 30.0)]},
            ),
        }
    )
    engine = make_engine(tmp_path / "qpsd.db")

    summary = run_pipeline(
        (total, foreign),
        countries,
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=_at(1),
    )

    assert source.calls == [
        (total.series_id, ("US", "SE"), False),
        (foreign.series_id, ("US", "SE"), False),
    ]
    assert summary[f"US/{total.indicator}"]["rows"] == 2
    assert summary[f"SE/{total.indicator}"]["status"] == "stored"
    missing = summary[f"SE/{foreign.indicator}"]
    assert missing["status"] == "not_reported"
    assert missing["not_reported"] is True
    assert missing["release_id"] is None

    with Session(engine) as session:
        releases = session.scalars(select(DataRelease).order_by(DataRelease.id)).all()
        current = session.execute(
            select(Observation.country, Observation.indicator, Observation.value).order_by(
                Observation.country, Observation.indicator, Observation.date
            )
        ).all()

    assert len(releases) == 3
    assert {release.source_family for release in releases} == {QPSD_SOURCE}
    assert all("source=20" in (release.source_url or "") for release in releases)
    assert all("/country/USA;SWE/" in (release.source_url or "") for release in releases)
    assert {release.partition_key for release in releases} == {
        make_partition_key(QPSD_SOURCE, total.series_id, "US", total.indicator),
        make_partition_key(QPSD_SOURCE, total.series_id, "SE", total.indicator),
        make_partition_key(QPSD_SOURCE, foreign.series_id, "US", foreign.indicator),
    }
    assert current == [
        ("SE", total.indicator, 34.0),
        ("US", foreign.indicator, 30.0),
        ("US", total.indicator, 120.0),
        ("US", total.indicator, 121.0),
    ]


def test_revisions_are_immutable_and_absence_does_not_erase_prior_data(tmp_path):
    spec = QPSD_SERIES[0]
    countries = (get_country("US"), get_country("SE"))
    source = FakeQpsd(
        {
            spec.indicator: _frame(
                spec,
                {
                    "US": [(date(2025, 7, 1), 120.0), (date(2025, 10, 1), 121.0)],
                    "SE": [(date(2025, 10, 1), 34.0)],
                },
            )
        }
    )
    engine = make_engine(tmp_path / "qpsd.db")
    run_pipeline((spec,), countries, source=source, engine=engine, retrieved_at=_at(1))

    source.frames[spec.indicator] = _frame(
        spec,
        {"US": [(date(2025, 7, 1), 122.5)]},
    )
    second = run_pipeline((spec,), countries, source=source, engine=engine, retrieved_at=_at(2))

    assert second[f"US/{spec.indicator}"]["inserted"] == 1
    assert second[f"US/{spec.indicator}"]["removed"] == 1
    assert second[f"SE/{spec.indicator}"]["status"] == "not_reported"
    us_partition = make_partition_key(QPSD_SOURCE, spec.series_id, "US", spec.indicator)
    se_partition = make_partition_key(QPSD_SOURCE, spec.series_id, "SE", spec.indicator)
    with Session(engine) as session:
        assert len(release_history(session, us_partition)) == 2
        assert len(release_history(session, se_partition)) == 1
        current = session.execute(
            select(Observation.country, Observation.date, Observation.value).order_by(
                Observation.country, Observation.date
            )
        ).all()
    assert current == [
        ("SE", date(2025, 10, 1), 34.0),
        ("US", date(2025, 7, 1), 122.5),
    ]


def test_malformed_basket_fails_before_any_partition_is_written(tmp_path):
    spec = QPSD_SERIES[0]
    countries = (get_country("US"), get_country("SE"))
    malformed = _frame(
        spec,
        {
            "US": [(date(2025, 10, 1), 121.0)],
            "SE": [(date(2025, 10, 1), 34.0)],
        },
    )
    malformed.loc[malformed["country"] == "SE", "series_id"] = "WRONG"
    source = FakeQpsd({spec.indicator: malformed})
    engine = make_engine(tmp_path / "qpsd.db")

    summary = run_pipeline((spec,), countries, source=source, engine=engine, retrieved_at=_at(1))

    assert all(stats["status"] == "error" for stats in summary.values())
    assert all("native series" in stats["error"] for stats in summary.values())
    with Session(engine) as session:
        assert session.scalars(select(DataRelease)).all() == []
        assert session.scalars(select(Observation)).all() == []
