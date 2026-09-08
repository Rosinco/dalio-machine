"""Release-ledger coverage for non-WEO fundamentals sources."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.bis import DsrSpec
from dalio.data_sources.imf_imts import ImtsSpec
from dalio.data_sources.oec import INDICATOR_ECI, OEC_SERIES_ID, SOURCE_OEC
from dalio.data_sources.worldbank import WbIndicatorSpec
from dalio.pipelines.fetch_fundamentals import run_pipeline
from dalio.storage.db import DataRelease, Observation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _frame(
    indicator: str,
    source: str,
    series_id: str,
    rows: list[tuple[str, date, float]],
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
        for country, period, value in rows
    ], columns=LONG_COLUMNS)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "fundamentals-releases.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(path))
    return path


class _FakeWb:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch(self, spec, countries, use_cache=True, today=None):
        return self.frames[spec.indicator].copy()


def test_world_bank_raw_world_share_and_eu_mean_use_actual_source_tags(db_path):
    exports = WbIndicatorSpec(
        "exports_usd",
        "NE.EXP.GNFS.CD",
        include_world=True,
    )
    rule = WbIndicatorSpec(
        "rule_of_law",
        "GOV_WGI_RL.EST",
        source_id=3,
    )
    source = _FakeWb({
        "exports_usd": _frame(
            "exports_usd",
            "WORLD_BANK",
            exports.wb_code,
            [
                ("US", date(2025, 12, 31), 3_000.0),
                ("DE", date(2025, 12, 31), 1_800.0),
                ("WLD", date(2025, 12, 31), 30_000.0),
            ],
        ),
        "rule_of_law": _frame(
            "rule_of_law",
            "WORLD_BANK_WGI",
            rule.wb_code,
            [
                ("DE", date(2025, 12, 31), 1.6),
                ("FR", date(2025, 12, 31), 1.3),
                ("IT", date(2025, 12, 31), 0.3),
            ],
        ),
    })
    basket = [get_country(code) for code in ("US", "DE", "FR", "IT", "EU")]

    summary = run_pipeline(
        ("wb",),
        countries=basket,
        use_cache=False,
        wb_source=source,
        wb_specs=(exports, rule),
        retrieved_at=_at(7),
    )

    world_share_key = make_partition_key(
        "WORLD_BANK",
        exports.wb_code + "÷WLD",
        "US",
        "exports_share_world",
    )
    eu_mean_key = make_partition_key(
        "WORLD_BANK_WGI",
        rule.wb_code + ":member-mean",
        "EU",
        "rule_of_law",
    )
    with Session(make_engine(db_path)) as session:
        releases = session.execute(select(DataRelease)).scalars().all()
        current = session.execute(
            select(Observation.country, Observation.indicator, Observation.value)
            .where(
                Observation.country.in_(("US", "EU")),
                Observation.indicator.in_(("exports_share_world", "rule_of_law")),
            )
            .order_by(Observation.country, Observation.indicator)
        ).all()

    assert len(releases) == 9
    assert world_share_key in {release.partition_key for release in releases}
    assert eu_mean_key in {release.partition_key for release in releases}
    assert {release.source_family for release in releases} == {
        "WORLD_BANK",
        "WORLD_BANK_WGI",
    }
    assert summary["wb/exports_share_world"]["releases_created"] == 2
    assert summary["wb/rule_of_law:EU"]["releases_created"] == 1
    assert current == [
        ("EU", "rule_of_law", pytest.approx((1.6 + 1.3 + 0.3) / 3)),
        ("US", "exports_share_world", 10.0),
    ]


class _FakeBis:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch_dsr(self, spec, use_cache=True):
        return self.frame.copy()

    def fetch_total_credit(self, spec, use_cache=True):
        raise AssertionError("unexpected total-credit fetch")


def test_tier3_bis_revisions_are_complete_and_empty_refresh_is_noop(db_path):
    spec = DsrSpec("debt_service_ratio", "KR")
    source = _FakeBis(_frame(
        spec.indicator,
        "BIS_DSR",
        "Q.KR.P",
        [
            ("KR", date(2025, 1, 1), 13.0),
            ("KR", date(2025, 4, 1), 13.2),
        ],
    ))
    kwargs = {
        "countries": [get_country("KR")],
        "use_cache": False,
        "bis_source": source,
        "bis_dsr_specs": (spec,),
        "bis_credit_specs": (),
    }
    run_pipeline(("bis",), retrieved_at=_at(7), **kwargs)
    source.frame = _frame(
        spec.indicator,
        "BIS_DSR",
        "Q.KR.P",
        [("KR", date(2025, 1, 1), 13.1)],
    )
    second = run_pipeline(("bis",), retrieved_at=_at(8), **kwargs)
    source.frame = pd.DataFrame(columns=LONG_COLUMNS)
    empty = run_pipeline(("bis",), retrieved_at=_at(9), **kwargs)

    partition = make_partition_key(
        "BIS_DSR", "Q.KR.P", "KR", spec.indicator,
    )
    with Session(make_engine(db_path)) as session:
        history = release_history(session, partition)
        old = load_vintage_panel(
            session, _at(7), partition_keys=(partition,),
        )
        current = session.execute(
            select(Observation.date, Observation.value)
            .where(Observation.indicator == spec.indicator)
        ).all()

    assert second[f"bis/KR/{spec.indicator}"]["removed"] == 1
    assert len(history) == 2
    assert old["value"].tolist() == [13.0, 13.2]
    assert current == [(date(2025, 1, 1), 13.1)]
    assert empty[f"bis/KR/{spec.indicator}"]["rows"] == 0
    assert "error" not in empty[f"bis/KR/{spec.indicator}"]


class _FakeImts:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch(self, spec, countries, use_cache=True, today=None):
        return self.frame.copy()


def _imts_frame(
    ca_rows: list[tuple[date, float]],
    world_rows: list[tuple[date, float]],
) -> pd.DataFrame:
    return pd.concat([
        _frame(
            "exports_to_CA",
            "IMF_IMTS",
            "XG_FOB_USD/CAN",
            [("US", period, value) for period, value in ca_rows],
        ),
        _frame(
            "exports_to_WLD",
            "IMF_IMTS",
            "XG_FOB_USD/G001",
            [("US", period, value) for period, value in world_rows],
        ),
    ], ignore_index=True)


def test_imts_partitions_each_reporter_partner_native_series(db_path):
    spec = ImtsSpec("exports", "XG_FOB_USD", "exports_to")
    source = _FakeImts(_imts_frame(
        [(date(2024, 12, 31), 320.0), (date(2025, 12, 31), 336.0)],
        [(date(2025, 12, 31), 2_185.0)],
    ))
    kwargs = {
        "countries": [get_country("US"), get_country("CA")],
        "use_cache": False,
        "imts_source": source,
        "imts_specs": (spec,),
    }
    first = run_pipeline(("imts",), retrieved_at=_at(7), **kwargs)
    source.frame = _imts_frame(
        [(date(2024, 12, 31), 325.0)],
        [(date(2025, 12, 31), 2_185.0)],
    )
    second = run_pipeline(("imts",), retrieved_at=_at(8), **kwargs)

    ca_partition = make_partition_key(
        "IMF_IMTS", "XG_FOB_USD/CAN", "US", "exports_to_CA",
    )
    world_partition = make_partition_key(
        "IMF_IMTS", "XG_FOB_USD/G001", "US", "exports_to_WLD",
    )
    with Session(make_engine(db_path)) as session:
        ca_history = release_history(session, ca_partition)
        world_history = release_history(session, world_partition)
        current = session.execute(
            select(Observation.indicator, Observation.date, Observation.value)
            .where(Observation.source == "IMF_IMTS")
            .order_by(Observation.indicator, Observation.date)
        ).all()

    assert first["imts/exports_to"]["releases_created"] == 2
    assert second["imts/exports_to"]["inserted"] == 1
    assert second["imts/exports_to"]["skipped"] == 1
    assert second["imts/exports_to"]["removed"] == 1
    assert len(ca_history) == 2
    assert len(world_history) == 1
    assert current == [
        ("exports_to_CA", date(2024, 12, 31), 325.0),
        ("exports_to_WLD", date(2025, 12, 31), 2_185.0),
    ]


class _FakeOec:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch_eci(self, countries, use_cache=True, start_year=1995):
        return self.frame.copy()


def test_oec_raw_and_eu_mean_are_releases_and_empty_refresh_is_noop(db_path):
    source = _FakeOec(_frame(
        INDICATOR_ECI,
        SOURCE_OEC,
        OEC_SERIES_ID,
        [
            ("DE", date(2024, 12, 31), 1.8),
            ("FR", date(2024, 12, 31), 1.4),
            ("IT", date(2024, 12, 31), 0.8),
        ],
    ))
    basket = [get_country(code) for code in ("DE", "FR", "IT", "EU")]
    kwargs = {
        "countries": basket,
        "use_cache": False,
        "oec_source": source,
    }
    first = run_pipeline(("oec",), retrieved_at=_at(7), **kwargs)
    source.frame = pd.DataFrame(columns=LONG_COLUMNS)
    empty = run_pipeline(("oec",), retrieved_at=_at(8), **kwargs)

    eu_partition = make_partition_key(
        SOURCE_OEC,
        OEC_SERIES_ID + ":member-mean",
        "EU",
        INDICATOR_ECI,
    )
    with Session(make_engine(db_path)) as session:
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        eu_history = release_history(session, eu_partition)
        eu_value = session.scalar(
            select(Observation.value).where(
                Observation.country == "EU",
                Observation.indicator == INDICATOR_ECI,
            )
        )

    assert first["oec/economic_complexity"]["releases_created"] == 3
    assert first["oec/economic_complexity:EU"]["releases_created"] == 1
    assert release_count == 4
    assert len(eu_history) == 1
    assert eu_value == pytest.approx((1.8 + 1.4 + 0.8) / 3)
    assert empty["oec/economic_complexity"]["rows"] == 0
    assert empty["oec/economic_complexity:EU"]["rows"] == 0
