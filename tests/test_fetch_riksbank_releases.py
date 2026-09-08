from datetime import UTC, date, datetime
from unittest.mock import patch

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.data_sources.riksbank import (
    DEFAULT_UNAUTHENTICATED_PACING_SECONDS,
    RIKSBANK_API_BASE,
    RIKSBANK_SERIES,
    RiksbankSeriesSpec,
)
from dalio.pipelines.fetch_riksbank import run_pipeline
from dalio.storage.db import DataRelease, Observation, init_db, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _spec(indicator: str) -> RiksbankSeriesSpec:
    return next(spec for spec in RIKSBANK_SERIES if spec.indicator == indicator)


def _frame(spec: RiksbankSeriesSpec, values: list[tuple[date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": spec.country,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": "RIKSBANK_SWEA",
                "series_id": spec.series_id,
            }
            for observed_on, value in values
        ],
        columns=_COLUMNS,
    )


class FakeRiksbank:
    recommended_pacing_seconds = DEFAULT_UNAUTHENTICATED_PACING_SECONDS

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.frames = frames
        self.errors = errors or {}
        self.calls: list[tuple[str, object, object, bool]] = []

    def fetch(
        self,
        spec: RiksbankSeriesSpec,
        *,
        from_date=None,
        to_date=None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        self.calls.append((spec.series_id, from_date, to_date, use_cache))
        if spec.series_id in self.errors:
            raise self.errors[spec.series_id]
        return self.frames[spec.series_id].copy()

    @staticmethod
    def url_for(spec: RiksbankSeriesSpec, *, from_date=None, to_date=None) -> str:
        start = from_date.isoformat() if isinstance(from_date, date) else from_date
        end = to_date.isoformat() if isinstance(to_date, date) else to_date
        return f"{RIKSBANK_API_BASE}/Observations/{spec.series_id}/{start}/{end}"


def test_pipeline_records_each_native_series_as_an_immutable_release(tmp_path):
    engine = make_engine(tmp_path / "riksbank.db")
    retrieved_at = _at(2026, 9, 8)
    policy = _spec("policy_rate")
    yield_10y = _spec("yield_10y")
    source = FakeRiksbank(
        {
            policy.series_id: _frame(policy, [(date(2026, 9, 7), 1.75)]),
            yield_10y.series_id: _frame(yield_10y, [(date(2026, 9, 7), 2.42)]),
        }
    )

    summary = run_pipeline(
        (policy, yield_10y),
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=retrieved_at,
        pacing_seconds=0,
    )

    assert source.calls == [
        (policy.series_id, date(2005, 1, 1), date(2026, 9, 8), False),
        (yield_10y.series_id, date(2005, 1, 1), date(2026, 9, 8), False),
    ]
    assert summary["SE/policy_rate"] == {
        "country": "SE",
        "indicator": "policy_rate",
        "rows": 1,
        "inserted": 1,
        "skipped": 0,
        "removed": 0,
        "release_id": 1,
        "release_created": True,
        "series_id": policy.series_id,
    }
    assert summary["SE/yield_10y"]["series_id"] == yield_10y.series_id

    with Session(engine) as session:
        releases = session.execute(select(DataRelease).order_by(DataRelease.id)).scalars().all()
        current = session.execute(
            select(Observation.indicator, Observation.source, Observation.series_id).order_by(
                Observation.indicator
            )
        ).all()

    assert [release.partition_key for release in releases] == [
        make_partition_key("RIKSBANK_SWEA", policy.series_id, "SE", "policy_rate"),
        make_partition_key("RIKSBANK_SWEA", yield_10y.series_id, "SE", "yield_10y"),
    ]
    assert {release.source_family for release in releases} == {"RIKSBANK_SWEA"}
    assert all(
        release.available_at == retrieved_at.replace(tzinfo=None)
        and release.retrieved_at == retrieved_at.replace(tzinfo=None)
        and release.published_at is None
        for release in releases
    )
    assert releases[0].source_url == (
        f"{RIKSBANK_API_BASE}/Observations/{policy.series_id}/2005-01-01/2026-09-08"
    )
    assert current == [
        ("policy_rate", "RIKSBANK_SWEA", policy.series_id),
        ("yield_10y", "RIKSBANK_SWEA", yield_10y.series_id),
    ]


def test_complete_refresh_keeps_old_vintage_and_does_not_touch_fred(tmp_path):
    engine = make_engine(tmp_path / "riksbank.db")
    policy = _spec("policy_rate")
    source = FakeRiksbank(
        {
            policy.series_id: _frame(
                policy,
                [(date(2026, 8, 31), 2.0), (date(2026, 9, 1), 2.0)],
            )
        }
    )
    first_at = _at(2026, 9, 2)
    second_at = _at(2026, 9, 8)
    run_pipeline(
        (policy,),
        source=source,
        engine=engine,
        retrieved_at=first_at,
        pacing_seconds=0,
    )
    with Session(engine) as session:
        session.add(
            Observation(
                country="SE",
                indicator="policy_rate",
                date=date(2026, 9, 1),
                value=1.95,
                source="FRED",
                series_id="IR3TIB01SEM156N",
            )
        )
        session.commit()

    source.frames[policy.series_id] = _frame(policy, [(date(2026, 8, 31), 1.75)])
    summary = run_pipeline(
        (policy,),
        source=source,
        engine=engine,
        retrieved_at=second_at,
        pacing_seconds=0,
    )

    partition = make_partition_key("RIKSBANK_SWEA", policy.series_id, "SE", "policy_rate")
    with Session(engine) as session:
        old = load_vintage_panel(session, first_at, partition_keys=(partition,))
        new = load_vintage_panel(session, second_at, partition_keys=(partition,))
        history = release_history(session, partition)
        current = session.execute(
            select(Observation.date, Observation.value, Observation.source).order_by(
                Observation.source, Observation.date
            )
        ).all()

    assert summary["SE/policy_rate"]["inserted"] == 1
    assert summary["SE/policy_rate"]["removed"] == 1
    assert len(history) == 2
    assert list(old["value"]) == [2.0, 2.0]
    assert list(new["value"]) == [1.75]
    assert current == [
        (date(2026, 9, 1), 1.95, "FRED"),
        (date(2026, 8, 31), 1.75, "RIKSBANK_SWEA"),
    ]


def test_empty_refresh_fails_closed_and_preserves_current_projection(tmp_path):
    engine = make_engine(tmp_path / "riksbank.db")
    policy = _spec("policy_rate")
    source = FakeRiksbank({policy.series_id: _frame(policy, [(date(2026, 9, 1), 1.75)])})
    run_pipeline(
        (policy,),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 9, 2),
        pacing_seconds=0,
    )
    source.frames[policy.series_id] = pd.DataFrame(columns=_COLUMNS)

    summary = run_pipeline(
        (policy,),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 9, 8),
        pacing_seconds=0,
    )

    assert "empty snapshot" in summary["SE/policy_rate"]["error"]
    with Session(engine) as session:
        releases = session.execute(select(DataRelease)).scalars().all()
        current = session.execute(select(Observation.date, Observation.value)).all()
    assert len(releases) == 1
    assert current == [(date(2026, 9, 1), 1.75)]


def test_one_series_failure_does_not_block_the_next_and_default_pacing_is_safe(tmp_path):
    engine = make_engine(tmp_path / "riksbank.db")
    policy = _spec("policy_rate")
    yield_2y = _spec("yield_2y")
    source = FakeRiksbank(
        {yield_2y.series_id: _frame(yield_2y, [(date(2026, 9, 1), 2.1)])},
        errors={policy.series_id: RuntimeError("temporary upstream failure")},
    )

    with patch("dalio.pipelines.fetch_riksbank.time.sleep") as sleep:
        summary = run_pipeline(
            (policy, yield_2y),
            source=source,
            engine=engine,
            retrieved_at=_at(2026, 9, 8),
        )

    assert "temporary upstream failure" in summary["SE/policy_rate"]["error"]
    assert summary["SE/yield_2y"]["inserted"] == 1
    sleep.assert_called_once_with(DEFAULT_UNAUTHENTICATED_PACING_SECONDS)


def test_pipeline_rejects_a_frame_with_the_wrong_native_series(tmp_path):
    engine = make_engine(tmp_path / "riksbank.db")
    init_db(engine)
    policy = _spec("policy_rate")
    wrong = _frame(policy, [(date(2026, 9, 1), 1.75)])
    wrong["series_id"] = "SEGVB2YC"
    source = FakeRiksbank({policy.series_id: wrong})

    summary = run_pipeline(
        (policy,),
        source=source,
        engine=engine,
        retrieved_at=_at(2026, 9, 8),
        pacing_seconds=0,
    )

    assert "expected native series" in summary["SE/policy_rate"]["error"]
    with Session(engine) as session:
        assert session.execute(select(DataRelease)).scalars().all() == []
