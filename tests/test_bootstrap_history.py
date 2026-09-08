"""Explicit migration command for pre-release-ledger databases."""

import argparse
import sys
from datetime import UTC, date

import pytest
from sqlalchemy import func, select

from dalio.pipelines.bootstrap_history import _parse_timestamp, main
from dalio.storage.db import DataRelease, Observation, init_db, make_engine, make_session_factory


def test_parse_timestamp_normalizes_to_utc():
    assert _parse_timestamp("2026-09-08T14:00:00+02:00").hour == 12
    assert _parse_timestamp("2026-09-08T12:00:00Z").tzinfo == UTC
    assert _parse_timestamp("2026-09-08T12:00:00").tzinfo == UTC
    with pytest.raises(argparse.ArgumentTypeError, match="ISO-8601"):
        _parse_timestamp("September sometime")


def test_command_bootstraps_once_without_touching_current(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db_path))
    engine = make_engine(db_path)
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        session.add(Observation(
            country="SE", indicator="policy_rate", date=date(2019, 12, 1),
            value=-0.25, source="FRED", series_id="IR3TIB01SEM156N",
        ))
        session.commit()

    args = ["dalio-bootstrap-history", "--available-at", "2020-01-01T00:00:00Z"]
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    assert main() == 0

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1
        current = session.scalar(select(Observation.value))
    assert current == -0.25
    output = capsys.readouterr().out
    assert "1 new releases covering 1 current rows" in output
    assert "No changes" in output
