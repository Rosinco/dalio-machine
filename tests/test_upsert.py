"""Equivalence tests: set-based `upsert_observations` vs the original row loop."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from dalio.pipelines.fetch_fred import upsert_observations
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


def _reference_upsert(session, df: pd.DataFrame) -> tuple[int, int]:
    """The pre-slice-18 row-by-row implementation, kept as the oracle."""
    inserted = 0
    skipped = 0
    for row in df.itertuples(index=False):
        existing = session.execute(
            select(Observation).where(
                Observation.country == row.country,
                Observation.indicator == row.indicator,
                Observation.date == row.date,
                Observation.source == row.source,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.value != float(row.value):
                existing.value = float(row.value)
                inserted += 1
            else:
                skipped += 1
            continue
        session.add(Observation(
            country=row.country, indicator=row.indicator, date=row.date,
            value=float(row.value), source=row.source, series_id=row.series_id,
        ))
        inserted += 1
    session.commit()
    return inserted, skipped


def _factory(tmp_path, name):
    engine = make_engine(tmp_path / name)
    init_db(engine)
    return make_session_factory(engine)


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "country": ["US", "SE", "KR"][i % 3],
            "indicator": ["gdp_pc_ppp", "old_age_dependency"][i % 2],
            "date": date(1960, 12, 31) + timedelta(days=365 * (i // 6)),
            "value": float(rng.normal(100, 10)),
            "source": "WORLD_BANK",
            "series_id": "X",
        })
    return pd.DataFrame(rows)


def _dump(session_factory) -> list[tuple]:
    with session_factory() as s:
        rows = s.execute(
            select(Observation.country, Observation.indicator, Observation.date,
                   Observation.source, Observation.value, Observation.series_id)
            .order_by(Observation.country, Observation.indicator, Observation.date, Observation.source)
        ).all()
    return [tuple(r) for r in rows]


def test_set_based_upsert_matches_reference_on_insert_update_noop(tmp_path):
    ref = _factory(tmp_path, "ref.db")
    new = _factory(tmp_path, "new.db")

    initial = _frame(200)
    modified = initial.copy()
    modified.loc[modified.index[::7], "value"] += 1.0          # 29 changed rows
    extra = _frame(30, seed=1)
    extra["date"] = extra["date"] + timedelta(days=36500)       # 30 brand-new keys
    modified = pd.concat([modified, extra], ignore_index=True)

    for frame in (initial, modified, modified):
        with ref() as s:
            r = _reference_upsert(s, frame)
        with new() as s:
            n = upsert_observations(s, frame)
        assert n == r, f"(inserted, skipped) diverged: new={n} ref={r}"

    assert _dump(new) == _dump(ref)


def test_counts_on_fresh_then_noop_then_change(tmp_path):
    sf = _factory(tmp_path, "t.db")
    df = _frame(12)
    with sf() as s:
        assert upsert_observations(s, df) == (12, 0)
    with sf() as s:
        assert upsert_observations(s, df) == (0, 12)
    df2 = df.copy()
    df2.loc[0, "value"] = 999.0
    with sf() as s:
        assert upsert_observations(s, df2) == (1, 11)
    with sf() as s:
        v = s.execute(
            select(Observation.value).where(
                Observation.country == df2.loc[0, "country"],
                Observation.indicator == df2.loc[0, "indicator"],
                Observation.date == df2.loc[0, "date"],
            )
        ).scalar_one()
    assert v == 999.0


def test_empty_frame_is_noop(tmp_path):
    sf = _factory(tmp_path, "t.db")
    empty = pd.DataFrame(columns=["country", "indicator", "date", "value", "source", "series_id"])
    with sf() as s:
        assert upsert_observations(s, empty) == (0, 0)


def test_in_batch_duplicates_collapse_to_last(tmp_path):
    sf = _factory(tmp_path, "t.db")
    df = _frame(3)
    dup = pd.concat([df, df.iloc[[0]].assign(value=[-1.0])], ignore_index=True)
    with sf() as s:
        ins, skp = upsert_observations(s, dup)
    assert (ins, skp) == (3, 0)
    with sf() as s:
        n = s.execute(select(Observation.id)).all()
        v = s.execute(
            select(Observation.value).where(Observation.date == df.loc[0, "date"],
                                            Observation.country == df.loc[0, "country"],
                                            Observation.indicator == df.loc[0, "indicator"])
        ).scalar_one()
    assert len(n) == 3
    assert v == -1.0


def test_accepts_timestamp_dates(tmp_path):
    sf = _factory(tmp_path, "t.db")
    df = _frame(4)
    df["date"] = pd.to_datetime(df["date"])
    with sf() as s:
        assert upsert_observations(s, df) == (4, 0)
    df["date"] = df["date"].dt.date
    with sf() as s:
        assert upsert_observations(s, df) == (0, 4)


def test_failed_batch_leaves_nothing_behind_for_the_next_commit(tmp_path, monkeypatch):
    """A NaN value in chunk 3 must roll back chunks 1–2, so a later successful
    upsert on the same session does not silently commit a partial series."""
    import dalio.pipelines.fetch_fred as ff
    monkeypatch.setattr(ff, "_CHUNK", 2)
    sf = _factory(tmp_path, "t.db")
    good = _frame(6)
    bad = _frame(6, seed=3)
    bad["indicator"] = "broken_series"
    bad["date"] = [date(2000 + i, 12, 31) for i in range(6)]   # 6 distinct keys
    bad.loc[4, "value"] = float("nan")     # NOT NULL constraint → IntegrityError in chunk 3
    later = _frame(6, seed=4)
    later["indicator"] = "later_series"
    later["date"] = [date(2000 + i, 12, 31) for i in range(6)]

    with sf() as s:
        assert upsert_observations(s, good) == (6, 0)
        with pytest.raises(Exception):  # noqa: B017 — any DB error must propagate
            upsert_observations(s, bad)
        assert upsert_observations(s, later) == (6, 0)   # session still usable

    with sf() as s:
        by_ind = {}
        for ind, in s.execute(select(Observation.indicator)).all():
            by_ind[ind] = by_ind.get(ind, 0) + 1
    assert by_ind.get("broken_series", 0) == 0
    assert by_ind["later_series"] == 6
    assert sum(by_ind.values()) == 12


@pytest.mark.parametrize("n", [1, 2])
def test_tiny_frames(tmp_path, n):
    sf = _factory(tmp_path, "t.db")
    with sf() as s:
        assert upsert_observations(s, _frame(n)) == (n, 0)
