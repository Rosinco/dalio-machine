"""IMF/WEO pipeline integration with the immutable release ledger."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.imf_datamapper import (
    SOURCE_FORECAST,
    SOURCE_HISTORY,
    ImfSpec,
)
from dalio.pipelines.fetch_fundamentals import run_pipeline
from dalio.storage.db import Observation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _frame(
    indicator: str,
    series_id: str,
    rows: list[tuple[str, int, float, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": country,
                "indicator": indicator,
                "date": date(year, 12, 31),
                "value": value,
                "source": source,
                "series_id": series_id,
            }
            for country, year, value, source in rows
        ]
    )


class _FakeImf:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch(self, spec, countries, use_cache=True):
        return self.frames[spec.indicator].copy()


def test_weo_omitted_forecast_leaves_current_but_remains_in_old_vintage(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "imf-releases.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db_path))
    spec = ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP")
    source = _FakeImf(
        {
            spec.indicator: _frame(
                spec.indicator,
                spec.imf_code,
                [
                    ("US", 2024, 120.0, SOURCE_HISTORY),
                    ("US", 2026, 126.0, SOURCE_FORECAST),
                    ("US", 2027, 130.0, SOURCE_FORECAST),
                    ("SE", 2024, 31.0, SOURCE_HISTORY),
                    ("SE", 2026, 33.0, SOURCE_FORECAST),
                ],
            ),
        }
    )
    basket = [get_country("US"), get_country("SE")]

    first = run_pipeline(
        ("imf",),
        countries=basket,
        use_cache=False,
        imf_source=source,
        imf_specs=(spec,),
        retrieved_at=_at(2026, 1, 10),
    )
    source.frames[spec.indicator] = _frame(
        spec.indicator,
        spec.imf_code,
        [
            ("US", 2024, 121.0, SOURCE_HISTORY),
            ("US", 2026, 127.0, SOURCE_FORECAST),
            ("SE", 2024, 31.0, SOURCE_HISTORY),
            ("SE", 2026, 33.0, SOURCE_FORECAST),
        ],
    )
    second = run_pipeline(
        ("imf",),
        countries=basket,
        use_cache=False,
        imf_source=source,
        imf_specs=(spec,),
        retrieved_at=_at(2026, 2, 10),
    )

    us_partition = make_partition_key(
        SOURCE_HISTORY,
        spec.imf_code,
        "US",
        spec.indicator,
    )
    se_partition = make_partition_key(
        SOURCE_HISTORY,
        spec.imf_code,
        "SE",
        spec.indicator,
    )
    with Session(make_engine(db_path)) as session:
        before = load_vintage_panel(
            session,
            _at(2026, 2, 9),
            partition_keys=(us_partition,),
        )
        after = load_vintage_panel(
            session,
            _at(2026, 2, 10),
            partition_keys=(us_partition,),
        )
        current = session.execute(
            select(Observation.country, Observation.date, Observation.value)
            .where(Observation.indicator == spec.indicator)
            .order_by(Observation.country, Observation.date)
        ).all()
        us_releases = release_history(session, us_partition)
        se_releases = release_history(session, se_partition)

    assert first[f"imf/{spec.indicator}"]["inserted"] == 5
    second_stats = second[f"imf/{spec.indicator}"]
    assert {
        key: second_stats[key]
        for key in (
            "source",
            "indicator",
            "series_id",
            "rows",
            "inserted",
            "skipped",
            "countries",
            "removed",
            "releases_created",
        )
    } == {
        "source": "imf",
        "indicator": spec.indicator,
        "series_id": spec.imf_code,
        "rows": 4,
        "inserted": 2,
        "skipped": 2,
        "countries": 2,
        "removed": 1,
        "releases_created": 1,
    }
    assert set(second_stats["release_ids"]) == {us_releases[-1].id, se_releases[-1].id}
    assert before[["date", "value"]].to_records(index=False).tolist() == [
        (date(2024, 12, 31), 120.0),
        (date(2026, 12, 31), 126.0),
        (date(2027, 12, 31), 130.0),
    ]
    assert after[["date", "value"]].to_records(index=False).tolist() == [
        (date(2024, 12, 31), 121.0),
        (date(2026, 12, 31), 127.0),
    ]
    assert current == [
        ("SE", date(2024, 12, 31), 31.0),
        ("SE", date(2026, 12, 31), 33.0),
        ("US", date(2024, 12, 31), 121.0),
        ("US", date(2026, 12, 31), 127.0),
    ]
    assert len(us_releases) == 2
    assert len(se_releases) == 1


def test_derived_interest_burden_is_a_complete_release_partition(tmp_path, monkeypatch):
    db_path = tmp_path / "imf-derived.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db_path))
    overall = ImfSpec("fiscal_balance_pct_gdp", "GGXCNL_NGDP")
    primary = ImfSpec("primary_balance_pct_gdp", "GGXONLB_G01_GDP_PT")
    source = _FakeImf(
        {
            overall.indicator: _frame(
                overall.indicator,
                overall.imf_code,
                [
                    ("US", 2024, -6.0, SOURCE_HISTORY),
                    ("US", 2026, -5.0, SOURCE_FORECAST),
                    ("US", 2027, -4.5, SOURCE_FORECAST),
                ],
            ),
            primary.indicator: _frame(
                primary.indicator,
                primary.imf_code,
                [
                    ("US", 2024, -3.0, SOURCE_HISTORY),
                    ("US", 2026, -2.0, SOURCE_FORECAST),
                    ("US", 2027, -1.8, SOURCE_FORECAST),
                ],
            ),
        }
    )
    basket = [get_country("US")]

    run_pipeline(
        ("imf",),
        countries=basket,
        use_cache=False,
        imf_source=source,
        imf_specs=(overall, primary),
        retrieved_at=_at(2026, 1, 10),
    )
    source.frames = {
        overall.indicator: _frame(
            overall.indicator,
            overall.imf_code,
            [
                ("US", 2024, -6.0, SOURCE_HISTORY),
                ("US", 2026, -5.5, SOURCE_FORECAST),
            ],
        ),
        primary.indicator: _frame(
            primary.indicator,
            primary.imf_code,
            [
                ("US", 2024, -3.0, SOURCE_HISTORY),
                ("US", 2026, -2.2, SOURCE_FORECAST),
            ],
        ),
    }
    summary = run_pipeline(
        ("imf",),
        countries=basket,
        use_cache=False,
        imf_source=source,
        imf_specs=(overall, primary),
        retrieved_at=_at(2026, 2, 10),
    )

    partition = make_partition_key(
        SOURCE_HISTORY,
        "primary-overall",
        "US",
        "interest_burden_pct_gdp",
    )
    with Session(make_engine(db_path)) as session:
        old = load_vintage_panel(
            session,
            _at(2026, 2, 9),
            partition_keys=(partition,),
        )
        current = session.execute(
            select(Observation.date, Observation.value, Observation.source)
            .where(Observation.indicator == "interest_burden_pct_gdp")
            .order_by(Observation.date)
        ).all()

    assert old["date"].tolist() == [
        date(2024, 12, 31),
        date(2026, 12, 31),
        date(2027, 12, 31),
    ]
    assert old["value"].tolist() == pytest.approx([3.0, 3.0, 2.7])
    assert [(row.date, row.source) for row in current] == [
        (date(2024, 12, 31), SOURCE_HISTORY),
        (date(2026, 12, 31), SOURCE_FORECAST),
    ]
    assert [row.value for row in current] == pytest.approx([3.0, 3.3])
    assert summary["imf/interest_burden_pct_gdp"]["removed"] == 1


def test_empty_weo_response_fails_closed_without_erasing_current(tmp_path, monkeypatch):
    db_path = tmp_path / "imf-empty.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db_path))
    spec = ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP")
    source = _FakeImf(
        {
            spec.indicator: _frame(
                spec.indicator,
                spec.imf_code,
                [
                    ("US", 2024, 120.0, SOURCE_HISTORY),
                    ("US", 2026, 126.0, SOURCE_FORECAST),
                ],
            ),
        }
    )

    run_pipeline(
        ("imf",),
        countries=[get_country("US")],
        use_cache=False,
        imf_source=source,
        imf_specs=(spec,),
        retrieved_at=_at(2026, 1, 10),
    )
    source.frames[spec.indicator] = _frame(spec.indicator, spec.imf_code, [])
    summary = run_pipeline(
        ("imf",),
        countries=[get_country("US")],
        use_cache=False,
        imf_source=source,
        imf_specs=(spec,),
        retrieved_at=_at(2026, 2, 10),
    )

    with Session(make_engine(db_path)) as session:
        current = session.execute(
            select(Observation.date, Observation.value).order_by(Observation.date)
        ).all()

    assert "empty" in summary[f"imf/{spec.indicator}"]["error"]
    assert current == [
        (date(2024, 12, 31), 120.0),
        (date(2026, 12, 31), 126.0),
    ]
