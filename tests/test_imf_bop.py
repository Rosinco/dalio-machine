"""IMF BOP adapter and immutable transaction-release pipeline.

HTTP is always mocked.  The fixtures intentionally keep the IMF accounting
entry in the native key so an asset acquisition can never be mistaken for a
liability incurrence (an inward flow).
"""

from dataclasses import replace
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.imf_bop import (
    BOP_COUNTRIES,
    BOP_HISTORY_START_YEAR,
    BOP_SERIES,
    BOP_SOURCE,
    BopSeriesSpec,
    ImfBopSource,
)
from dalio.pipelines.fetch_flows import run_pipeline
from dalio.storage.db import DataRelease, Observation, ReleaseObservation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

SAMPLE = (
    "DATAFLOW,COUNTRY,BOP_ACCOUNTING_ENTRY,INDICATOR,UNIT,FREQUENCY,"
    "TIME_PERIOD,OBS_VALUE,SCALE,STATUS,FULL_DESCRIPTION\n"
    'IMF.STA:BOP(21.0.0),SWE,A_NFA_T,D_F,USD,Q,2025-Q4,1200000000,6,A,"A\n'
    'multiline description"\n'
    'IMF.STA:BOP(21.0.0),SWE,L_NIL_T,D_F,USD,Q,2025-Q4,800000000,6,,"x"\n'
    'IMF.STA:BOP(21.0.0),SWE,NNAFANIL_T,FAB,USD,Q,2025-Q4,400000000,6,E,"x"\n'
    'IMF.STA:BOP(21.0.0),GBR,A_NFA_T,D_F,USD,Q,2025-Q4,500000000,6,,"x"\n'
    'IMF.STA:BOP(21.0.0),USA,A_NFA_T,D_F,USD,Q,2025-Q4,900000000,6,,"not requested"\n'
    'IMF.STA:BOP(21.0.0),SWE,A_NFA_T,D_F,USD,A,2025,999,6,,"annual ignored"\n'
    'IMF.STA:BOP(21.0.0),SWE,A_NFA_T,D_F,XDC,Q,2025-Q4,111,0,,"local ignored"\n'
    'IMF.STA:BOP(21.0.0),SWE,A_NFA_T,D_F,USD,Q,2026-Q1,,6,,"missing ignored"\n'
)


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _spec(indicator: str) -> BopSeriesSpec:
    return next(spec for spec in BOP_SERIES if spec.indicator == indicator)


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def test_catalogue_uses_explicit_asset_liability_and_net_names():
    names = {spec.indicator for spec in BOP_SERIES}
    assert {
        "direct_investment_assets_flow_usd",
        "direct_investment_liabilities_flow_usd",
        "direct_investment_net_flow_usd",
        "portfolio_equity_assets_flow_usd",
        "portfolio_equity_liabilities_flow_usd",
        "other_loans_net_flow_usd",
        "financial_derivatives_net_flow_usd",
        "financial_account_net_flow_usd",
    } <= names
    assert len(names) == len(BOP_SERIES) == 25
    assert all(len(spec.series_id) <= 64 for spec in BOP_SERIES)
    assert _spec("direct_investment_assets_flow_usd").series_id == (
        "BOP/A_NFA_T/D_F/USD/Q"
    )
    assert _spec("direct_investment_liabilities_flow_usd").series_id == (
        "BOP/L_NIL_T/D_F/USD/Q"
    )
    assert "EU" not in {country.iso2 for country in BOP_COUNTRIES}


def test_fetch_builds_one_compact_query_and_preserves_native_status(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(SAMPLE)
    specs = (
        _spec("direct_investment_assets_flow_usd"),
        _spec("direct_investment_liabilities_flow_usd"),
        _spec("financial_account_net_flow_usd"),
    )
    no_imf = replace(get_country("JP"), imf_id=None)

    frame = ImfBopSource(client=client, cache_dir=tmp_path).fetch(
        (get_country("SE"), get_country("UK"), no_imf),
        specs=specs,
        start_year=2000,
        use_cache=False,
    )

    url = client.get.call_args.args[0]
    assert url == (
        "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,BOP/"
        "SWE+GBR.A_NFA_T+L_NIL_T+NNAFANIL_T.D_F+FAB.USD.Q?"
        "startPeriod=2000&detail=dataonly"
    )
    assert list(frame[["country", "indicator", "date", "value"]].itertuples(index=False, name=None)) == [
        ("SE", "direct_investment_assets_flow_usd", date(2025, 10, 1), 1_200_000_000.0),
        ("SE", "direct_investment_liabilities_flow_usd", date(2025, 10, 1), 800_000_000.0),
        ("SE", "financial_account_net_flow_usd", date(2025, 10, 1), 400_000_000.0),
        ("UK", "direct_investment_assets_flow_usd", date(2025, 10, 1), 500_000_000.0),
    ]
    assert list(frame["series_id"]) == [
        "BOP/A_NFA_T/D_F/USD/Q",
        "BOP/L_NIL_T/D_F/USD/Q",
        "BOP/NNAFANIL_T/FAB/USD/Q",
        "BOP/A_NFA_T/D_F/USD/Q",
    ]
    assert list(frame["status"]) == ["A", "observed", "E", "observed"]
    assert set(frame["source"]) == {BOP_SOURCE}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("COUNTRY,TIME_PERIOD,OBS_VALUE\n", "missing columns"),
        (
            "DATAFLOW,COUNTRY,BOP_ACCOUNTING_ENTRY,INDICATOR,UNIT,FREQUENCY,"
            "TIME_PERIOD,OBS_VALUE,STATUS\n"
            "IMF.STA:IIP(13.0.0),SWE,A_NFA_T,D_F,USD,Q,2025-Q4,1,A\n",
            "unexpected dataflow",
        ),
        (
            "DATAFLOW,COUNTRY,BOP_ACCOUNTING_ENTRY,INDICATOR,UNIT,FREQUENCY,"
            "TIME_PERIOD,OBS_VALUE,STATUS\n"
            "IMF.STA:BOP(21.0.0),SWE,A_NFA_T,D_F,USD,Q,2025-Q5,1,A\n",
            "invalid quarterly period",
        ),
    ],
)
def test_fetch_rejects_malformed_selected_snapshots(tmp_path, text, message):
    client = MagicMock()
    client.get.return_value = _response(text)
    with pytest.raises(ValueError, match=message):
        ImfBopSource(client=client, cache_dir=tmp_path).fetch(
            (get_country("SE"),),
            specs=(_spec("direct_investment_assets_flow_usd"),),
            use_cache=False,
        )


def test_http_errors_and_no_mappable_countries(tmp_path):
    client = MagicMock()
    source = ImfBopSource(client=client, cache_dir=tmp_path)
    client.get.return_value = _response("denied", 403)
    with pytest.raises(ValueError, match="403"):
        source.fetch(
            (get_country("SE"),),
            specs=(_spec("direct_investment_assets_flow_usd"),),
            use_cache=False,
        )

    no_imf = replace(get_country("SE"), imf_id=None)
    assert source.fetch((no_imf,), use_cache=False).empty
    assert client.get.call_count == 1


_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]


def _frame(
    country: str,
    spec: BopSeriesSpec,
    values: list[tuple[date, float, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": country,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": BOP_SOURCE,
                "series_id": spec.series_id,
                "status": status,
            }
            for observed_on, value, status in values
        ],
        columns=_COLUMNS,
    )


class FakeBop:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = []

    def fetch(self, countries, *, specs, start_year, use_cache=True):
        self.calls.append((tuple(c.iso2 for c in countries), tuple(specs), start_year, use_cache))
        return self.frame.copy()

    @staticmethod
    def url_for(countries, *, specs, start_year):
        return f"https://example.test/bop?startPeriod={start_year}"


def test_pipeline_ingests_complete_partitions_and_isolates_missing_one(tmp_path):
    engine = make_engine(tmp_path / "flows.db")
    assets = _spec("direct_investment_assets_flow_usd")
    liabilities = _spec("direct_investment_liabilities_flow_usd")
    source = FakeBop(
        pd.concat(
            [
                _frame("SE", assets, [(date(2025, 1, 1), 10.0, "A")]),
                _frame("SE", liabilities, [(date(2025, 1, 1), 8.0, "observed")]),
                _frame("UK", assets, [(date(2025, 1, 1), 5.0, "E")]),
            ],
            ignore_index=True,
        )
    )
    run_at = _at(2026, 9, 8)

    summary = run_pipeline(
        countries=(get_country("SE"), get_country("UK")),
        specs=(assets, liabilities),
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=run_at,
    )

    assert source.calls == [
        (("SE", "UK"), (assets, liabilities), BOP_HISTORY_START_YEAR, False)
    ]
    assert summary["SE/direct_investment_assets_flow_usd"]["inserted"] == 1
    assert summary["SE/direct_investment_liabilities_flow_usd"]["inserted"] == 1
    assert summary["UK/direct_investment_assets_flow_usd"]["inserted"] == 1
    absent = summary["UK/direct_investment_liabilities_flow_usd"]
    assert absent["status"] == "not_reported"
    assert absent["not_reported"] is True
    assert absent["rows"] == 0

    with Session(engine) as session:
        releases = session.scalars(select(DataRelease).order_by(DataRelease.id)).all()
        release_rows = session.scalars(
            select(ReleaseObservation).order_by(ReleaseObservation.id)
        ).all()
        current = session.execute(
            select(Observation.country, Observation.indicator, Observation.series_id).order_by(
                Observation.country, Observation.indicator
            )
        ).all()

    assert len(releases) == 3
    assert releases[0].partition_key == make_partition_key(
        BOP_SOURCE, assets.series_id, "SE", assets.indicator
    )
    assert all(
        release.source_family == BOP_SOURCE
        and release.available_at == run_at.replace(tzinfo=None)
        and release.retrieved_at == run_at.replace(tzinfo=None)
        and release.source_url == f"https://example.test/bop?startPeriod={BOP_HISTORY_START_YEAR}"
        for release in releases
    )
    assert [row.status for row in release_rows] == ["a", "observed", "e"]
    assert current == [
        ("SE", "direct_investment_assets_flow_usd", assets.series_id),
        ("SE", "direct_investment_liabilities_flow_usd", liabilities.series_id),
        ("UK", "direct_investment_assets_flow_usd", assets.series_id),
    ]


def test_empty_partition_fails_closed_and_old_vintage_remains(tmp_path):
    engine = make_engine(tmp_path / "flows.db")
    assets = _spec("portfolio_investment_assets_flow_usd")
    source = FakeBop(
        _frame(
            "SE",
            assets,
            [(date(2024, 10, 1), 4.0, "observed"), (date(2025, 1, 1), 5.0, "observed")],
        )
    )
    first_at = _at(2026, 8, 1)
    second_at = _at(2026, 9, 8)
    run_pipeline(
        countries=(get_country("SE"),),
        specs=(assets,),
        source=source,
        engine=engine,
        retrieved_at=first_at,
    )
    source.frame = pd.DataFrame(columns=_COLUMNS)

    summary = run_pipeline(
        countries=(get_country("SE"),),
        specs=(assets,),
        source=source,
        engine=engine,
        retrieved_at=second_at,
    )

    partition = make_partition_key(BOP_SOURCE, assets.series_id, "SE", assets.indicator)
    absent = summary[f"SE/{assets.indicator}"]
    assert absent["status"] == "not_reported"
    assert absent["not_reported"] is True
    with Session(engine) as session:
        assert len(release_history(session, partition)) == 1
        old = load_vintage_panel(session, first_at, partition_keys=(partition,))
        current = session.execute(
            select(Observation.date, Observation.value).order_by(Observation.date)
        ).all()
    assert list(old["value"]) == [4.0, 5.0]
    assert current == [(date(2024, 10, 1), 4.0), (date(2025, 1, 1), 5.0)]


def test_pipeline_rejects_mixed_native_series_without_touching_prior(tmp_path):
    engine = make_engine(tmp_path / "flows.db")
    assets = _spec("other_investment_assets_flow_usd")
    source = FakeBop(_frame("SE", assets, [(date(2025, 1, 1), 2.0, "observed")]))
    run_pipeline(
        countries=(get_country("SE"),),
        specs=(assets,),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 8, 1),
    )
    invalid = _frame("SE", assets, [(date(2025, 1, 1), 3.0, "observed")])
    invalid["series_id"] = "BOP/L_NIL_T/O_F/USD/Q"
    source.frame = invalid

    result = run_pipeline(
        countries=(get_country("SE"),),
        specs=(assets,),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 9, 8),
    )

    assert "native series" in result[f"SE/{assets.indicator}"]["error"]
    partition = make_partition_key(BOP_SOURCE, assets.series_id, "SE", assets.indicator)
    with Session(engine) as session:
        assert len(release_history(session, partition)) == 1
        assert session.execute(select(Observation.value)).scalar_one() == 2.0
