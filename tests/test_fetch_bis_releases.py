from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.data_sources.bis import DsrSpec, Sector, TotalCreditSpec
from dalio.pipelines.fetch_bis import run_pipeline
from dalio.storage.db import DataRelease, Observation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _frame(
    country: str,
    indicator: str,
    source: str,
    series_id: str,
    values: list[tuple[date, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": country,
                "indicator": indicator,
                "date": observed_on,
                "value": value,
                "source": source,
                "series_id": series_id,
            }
            for observed_on, value in values
        ],
        columns=_COLUMNS,
    )


class FakeBis:
    def __init__(
        self,
        tc_frames: dict[str, pd.DataFrame] | None = None,
        dsr_frames: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.tc_frames = tc_frames or {}
        self.dsr_frames = dsr_frames or {}
        self.calls: list[tuple[str, str, bool]] = []

    def fetch_total_credit(
        self,
        spec: TotalCreditSpec,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        self.calls.append(("tc", spec.country, use_cache))
        return self.tc_frames[spec.country].copy()

    def fetch_dsr(self, spec: DsrSpec, use_cache: bool = True) -> pd.DataFrame:
        self.calls.append(("dsr", spec.country, use_cache))
        return self.dsr_frames[spec.country].copy()


def test_pipeline_records_tc_and_dsr_as_native_series_releases(tmp_path):
    engine = make_engine(tmp_path / "bis.db")
    retrieved_at = _at(2026, 9, 8)
    tc_spec = TotalCreditSpec("private_nonfin_pct_gdp", "US", Sector.PRIVATE_NON_FIN)
    dsr_spec = DsrSpec("debt_service_ratio", "US")
    tc_series = "Q.US.P.A.M.770.A"
    dsr_series = "Q.US.P"
    source = FakeBis(
        tc_frames={
            "US": _frame(
                "US",
                tc_spec.indicator,
                "BIS_TC",
                tc_series,
                [(date(2025, 7, 1), 140.4), (date(2025, 10, 1), 139.8)],
            )
        },
        dsr_frames={
            "US": _frame(
                "US",
                dsr_spec.indicator,
                "BIS_DSR",
                dsr_series,
                [(date(2025, 7, 1), 14.1)],
            )
        },
    )

    summary = run_pipeline(
        (tc_spec,),
        (dsr_spec,),
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=retrieved_at,
    )

    assert source.calls == [("tc", "US", False), ("dsr", "US", False)]
    assert summary["US/private_nonfin_pct_gdp"] == {
        "country": "US",
        "indicator": "private_nonfin_pct_gdp",
        "rows": 2,
        "inserted": 2,
        "skipped": 0,
        "removed": 0,
        "release_id": 1,
        "release_created": True,
        "series_id": tc_series,
    }
    assert summary["US/debt_service_ratio"]["inserted"] == 1
    assert summary["US/debt_service_ratio"]["series_id"] == dsr_series

    tc_partition = make_partition_key(
        "BIS_TC",
        tc_series,
        "US",
        "private_nonfin_pct_gdp",
    )
    dsr_partition = make_partition_key(
        "BIS_DSR",
        dsr_series,
        "US",
        "debt_service_ratio",
    )
    with Session(engine) as session:
        releases = session.execute(select(DataRelease).order_by(DataRelease.id)).scalars().all()
        current = session.execute(
            select(Observation.source, Observation.series_id, Observation.value).order_by(
                Observation.source, Observation.date
            )
        ).all()

    assert [release.partition_key for release in releases] == [tc_partition, dsr_partition]
    assert [release.source_family for release in releases] == ["BIS_TC", "BIS_DSR"]
    assert [release.row_count for release in releases] == [2, 1]
    assert all(
        release.available_at == retrieved_at.replace(tzinfo=None)
        and release.retrieved_at == retrieved_at.replace(tzinfo=None)
        for release in releases
    )
    assert current == [
        ("BIS_DSR", dsr_series, 14.1),
        ("BIS_TC", tc_series, 140.4),
        ("BIS_TC", tc_series, 139.8),
    ]


def test_complete_release_revises_one_country_without_touching_another(tmp_path):
    engine = make_engine(tmp_path / "bis.db")
    indicator = "private_nonfin_pct_gdp"
    us_spec = TotalCreditSpec(indicator, "US", Sector.PRIVATE_NON_FIN)
    se_spec = TotalCreditSpec(indicator, "SE", Sector.PRIVATE_NON_FIN)
    us_series = "Q.US.P.A.M.770.A"
    se_series = "Q.SE.P.A.M.770.A"
    source = FakeBis(
        tc_frames={
            "US": _frame(
                "US",
                indicator,
                "BIS_TC",
                us_series,
                [(date(2025, 7, 1), 140.0), (date(2025, 10, 1), 141.0)],
            ),
            "SE": _frame(
                "SE",
                indicator,
                "BIS_TC",
                se_series,
                [(date(2025, 7, 1), 160.0), (date(2025, 10, 1), 161.0)],
            ),
        }
    )
    first_at = _at(2026, 1, 10)
    second_at = _at(2026, 2, 10)

    run_pipeline(
        (us_spec, se_spec),
        (),
        source=source,
        engine=engine,
        retrieved_at=first_at,
    )
    source.tc_frames["US"] = _frame(
        "US",
        indicator,
        "BIS_TC",
        us_series,
        [(date(2025, 7, 1), 142.5)],
    )
    second = run_pipeline(
        (us_spec,),
        (),
        source=source,
        engine=engine,
        retrieved_at=second_at,
    )

    us_partition = make_partition_key("BIS_TC", us_series, "US", indicator)
    se_partition = make_partition_key("BIS_TC", se_series, "SE", indicator)
    with Session(engine) as session:
        old = load_vintage_panel(session, first_at)
        new = load_vintage_panel(session, second_at)
        us_history = release_history(session, us_partition)
        se_history = release_history(session, se_partition)
        current = session.execute(
            select(Observation.country, Observation.date, Observation.value).order_by(
                Observation.country, Observation.date
            )
        ).all()

    assert second["US/private_nonfin_pct_gdp"]["inserted"] == 1
    assert second["US/private_nonfin_pct_gdp"]["removed"] == 1
    assert len(us_history) == 2
    assert len(se_history) == 1
    assert len(old) == 4
    assert list(new.loc[new["country"] == "US", "value"]) == [142.5]
    assert list(new.loc[new["country"] == "SE", "value"]) == [160.0, 161.0]
    assert current == [
        ("SE", date(2025, 7, 1), 160.0),
        ("SE", date(2025, 10, 1), 161.0),
        ("US", date(2025, 7, 1), 142.5),
    ]


@pytest.mark.parametrize("invalid_kind", ["empty", "mixed-series"])
def test_invalid_complete_snapshot_fails_closed(tmp_path, invalid_kind):
    engine = make_engine(tmp_path / f"bis-{invalid_kind}.db")
    indicator = "private_nonfin_pct_gdp"
    spec = TotalCreditSpec(indicator, "US", Sector.PRIVATE_NON_FIN)
    series_id = "Q.US.P.A.M.770.A"
    source = FakeBis(
        tc_frames={
            "US": _frame(
                "US",
                indicator,
                "BIS_TC",
                series_id,
                [(date(2025, 7, 1), 140.0)],
            )
        }
    )
    run_pipeline((spec,), (), source=source, engine=engine, retrieved_at=_at(2026, 1, 10))

    if invalid_kind == "empty":
        source.tc_frames["US"] = pd.DataFrame(columns=_COLUMNS)
        expected_error = "empty snapshot"
    else:
        source.tc_frames["US"] = pd.concat(
            [
                _frame(
                    "US",
                    indicator,
                    "BIS_TC",
                    series_id,
                    [(date(2025, 7, 1), 141.0)],
                ),
                _frame(
                    "US",
                    indicator,
                    "BIS_TC",
                    "Q.US.H.A.M.770.A",
                    [(date(2025, 10, 1), 90.0)],
                ),
            ],
            ignore_index=True,
        )
        expected_error = "exactly one"

    failed = run_pipeline(
        (spec,),
        (),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 2, 10),
    )

    assert expected_error in failed["US/private_nonfin_pct_gdp"]["error"]
    partition = make_partition_key("BIS_TC", series_id, "US", indicator)
    with Session(engine) as session:
        assert len(release_history(session, partition)) == 1
        assert session.execute(select(Observation.date, Observation.value)).all() == [
            (date(2025, 7, 1), 140.0)
        ]
