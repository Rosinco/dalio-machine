"""Release-ledger coverage for the fresh cycle-input pipeline."""

from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.imf_cpi import SERIES_ID_CPI
from dalio.data_sources.oecd import LFS_UNEMPLOYMENT, QNA_GDP_GROWTH
from dalio.pipelines import fetch_cycle
from dalio.storage.db import DataRelease, Observation, ReleaseObservation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _long(
    country: str,
    indicator: str,
    source: str,
    series_id: str,
    values: list[tuple[date, float]],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "country": country,
            "indicator": indicator,
            "date": period,
            "value": value,
            "source": source,
            "series_id": series_id,
        }
        for period, value in values
    ])


class _MutableCpi:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch(self, countries, use_cache=True, start_year=2010):
        frames = [self.frames[c.iso2] for c in countries if c.iso2 in self.frames]
        if not frames:
            return pd.DataFrame(
                columns=["country", "indicator", "date", "value", "source", "series_id"]
            )
        return pd.concat(frames, ignore_index=True)


class _FixedScb:
    def fetch(self, spec, use_cache=True):
        return _long(
            "SE",
            "cpi_yoy",
            "SCB_CPI",
            "TAB6596/00000804",
            [(date(2026, 6, 1), 1.9)],
        )


def _cpi(country: str, values: list[tuple[date, float]]) -> pd.DataFrame:
    return _long(country, "cpi_yoy", "IMF_CPI", SERIES_ID_CPI, values)


def test_multicountry_frame_becomes_complete_country_series_releases(tmp_path):
    engine = make_engine(tmp_path / "cycle.db")
    countries = [get_country("JP"), get_country("CN")]
    source = _MutableCpi({
        "JP": _cpi(
            "JP",
            [(date(2026, 5, 1), 2.1), (date(2026, 6, 1), 2.3)],
        ),
        "CN": _cpi("CN", [(date(2026, 6, 1), 1.7)]),
    })

    summary = fetch_cycle.run_pipeline(
        ["cpi"],
        countries,
        cpi_source=source,
        engine=engine,
        retrieved_at=_at(7),
    )

    assert summary == {
        "cpi/cpi_yoy": {
            "indicator": "cpi_yoy",
            "source": "IMF_CPI",
            "rows": 3,
            "inserted": 3,
            "skipped": 0,
            "countries": ["CN", "JP"],
        }
    }
    with Session(engine) as session:
        releases = session.execute(
            select(DataRelease).order_by(DataRelease.partition_key)
        ).scalars().all()
        ledger_rows = session.scalar(select(func.count()).select_from(ReleaseObservation))

    assert {release.partition_key for release in releases} == {
        make_partition_key("IMF_CPI", SERIES_ID_CPI, "CN", "cpi_yoy"),
        make_partition_key("IMF_CPI", SERIES_ID_CPI, "JP", "cpi_yoy"),
    }
    assert {release.row_count for release in releases} == {1, 2}
    assert {release.available_at for release in releases} == {
        _at(7).replace(tzinfo=None)
    }
    assert ledger_rows == 3


def test_country_revision_and_omission_do_not_touch_sibling_partition(tmp_path):
    engine = make_engine(tmp_path / "cycle.db")
    us = get_country("US")
    se = get_country("SE")
    source = _MutableCpi({
        "US": _cpi(
            "US",
            [(date(2026, 5, 1), 2.5), (date(2026, 6, 1), 2.6)],
        ),
        "SE": _cpi("SE", [(date(2026, 6, 1), 1.8)]),
    })
    fetch_cycle.run_pipeline(
        ["cpi"],
        [us, se],
        cpi_source=source,
        scb_source=_FixedScb(),
        engine=engine,
        retrieved_at=_at(7),
    )

    source.frames["US"] = _cpi("US", [(date(2026, 5, 1), 2.4)])
    summary = fetch_cycle.run_pipeline(
        ["cpi"], [us], cpi_source=source, engine=engine, retrieved_at=_at(8)
    )

    us_partition = make_partition_key("IMF_CPI", SERIES_ID_CPI, "US", "cpi_yoy")
    se_partition = make_partition_key("IMF_CPI", SERIES_ID_CPI, "SE", "cpi_yoy")
    with Session(engine) as session:
        current = session.execute(
            select(Observation.country, Observation.date, Observation.value)
            .where(Observation.source == "IMF_CPI")
            .order_by(Observation.country, Observation.date)
        ).all()
        partitions = (us_partition, se_partition)
        old = load_vintage_panel(session, _at(7), partition_keys=partitions)
        new = load_vintage_panel(session, _at(8), partition_keys=partitions)
        us_history = release_history(session, us_partition)
        se_history = release_history(session, se_partition)

    assert summary["cpi/cpi_yoy"]["inserted"] == 1
    assert summary["cpi/cpi_yoy"]["skipped"] == 0
    assert current == [
        ("SE", date(2026, 6, 1), 1.8),
        ("US", date(2026, 5, 1), 2.4),
    ]
    assert len(old.loc[old["country"] == "US"]) == 2
    assert new.loc[new["country"] == "US", "value"].tolist() == [2.4]
    assert new.loc[new["country"] == "SE", "value"].tolist() == [1.8]
    assert len(us_history) == 2
    assert len(se_history) == 1


class _FakeBis:
    def fetch_policy_rate(self, spec, use_cache=True):
        return _long(
            spec.country,
            "policy_rate",
            "BIS_CBPOL",
            f"M.{spec.country}",
            [(date(2026, 7, 1), 3.0)],
        )


class _FakeOecd:
    def fetch(self, flow, countries, use_cache=True, start_year=2010):
        when = date(2026, 4, 1) if flow is QNA_GDP_GROWTH else date(2026, 7, 1)
        return pd.concat([
            _long(
                country.iso2,
                flow.indicator,
                flow.source,
                flow.series_id,
                [(when, 1.0)],
            )
            for country in countries
        ], ignore_index=True)


def test_bis_and_oecd_use_native_series_partition_keys(tmp_path):
    engine = make_engine(tmp_path / "cycle.db")
    countries = [get_country("US"), get_country("EU")]

    summary = fetch_cycle.run_pipeline(
        ["cbpol", "qna", "lfs"],
        countries,
        bis_source=_FakeBis(),
        oecd_source=_FakeOecd(),
        engine=engine,
        retrieved_at=_at(8),
    )

    with Session(engine) as session:
        releases = session.execute(select(DataRelease)).scalars().all()

    expected = {
        make_partition_key("BIS_CBPOL", f"M.{country}", country, "policy_rate")
        for country in ("US", "EU")
    }
    expected.update({
        make_partition_key(flow.source, flow.series_id, country, flow.indicator)
        for flow in (QNA_GDP_GROWTH, LFS_UNEMPLOYMENT)
        for country in ("US", "EU")
    })
    assert {release.partition_key for release in releases} == expected
    assert summary["cbpol/policy_rate/US"]["inserted"] == 1
    assert summary["qna/real_gdp_yoy"]["countries"] == ["EU", "US"]
    assert summary["lfs/unemployment_rate"]["rows"] == 2


def test_repeat_snapshot_is_idempotent_and_keeps_summary_contract(tmp_path):
    engine = make_engine(tmp_path / "cycle.db")
    countries = [get_country("JP"), get_country("CN")]
    source = _MutableCpi({
        country.iso2: _cpi(country.iso2, [(date(2026, 6, 1), 1.7)])
        for country in countries
    })

    fetch_cycle.run_pipeline(
        ["cpi"], countries, cpi_source=source, engine=engine, retrieved_at=_at(7)
    )
    again = fetch_cycle.run_pipeline(
        ["cpi"], countries, cpi_source=source, engine=engine, retrieved_at=_at(8)
    )

    with Session(engine) as session:
        release_count = session.scalar(select(func.count()).select_from(DataRelease))

    assert again["cpi/cpi_yoy"]["inserted"] == 0
    assert again["cpi/cpi_yoy"]["skipped"] == 2
    assert release_count == 2
