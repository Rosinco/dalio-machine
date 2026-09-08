"""SCB CPI adapter and release-pipeline coverage; no live HTTP calls."""

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.scb import (
    INDICATOR_CPI,
    SCB_CPI_YOY,
    SOURCE_SCB_CPI,
    ScbSource,
)
from dalio.pipelines import fetch_cycle
from dalio.storage.db import DataRelease, Observation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

_SAMPLE_JSONSTAT = json.dumps(
    {
        "version": "2.0",
        "class": "dataset",
        "source": "Statistics Sweden",
        "updated": "2026-05-13T06:00:00Z",
        "id": ["ContentsCode", "Tid"],
        "size": [1, 5],
        "dimension": {
            "ContentsCode": {"category": {"index": {"00000804": 0}}},
            # Deliberately not insertion-ordered by period: position is authoritative.
            "Tid": {
                "category": {
                    "index": {
                        "2026M02": 4,
                        "1980M01": 0,
                        "2026M01": 3,
                        "2025M12": 2,
                        "1981M01": 1,
                    }
                }
            },
        },
        "value": [None, 12.3, 0.3, -0.1, 0.2],
    }
)

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _scb_frame(values: list[tuple[date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "SE",
                "indicator": INDICATOR_CPI,
                "date": observed_on,
                "value": value,
                "source": SOURCE_SCB_CPI,
                "series_id": SCB_CPI_YOY.series_id,
            }
            for observed_on, value in values
        ],
        columns=_COLUMNS,
    )


def test_scb_cpi_catalogue_and_jsonstat_parser(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_SAMPLE_JSONSTAT)

    frame = ScbSource(client=client, cache_dir=tmp_path).fetch(use_cache=False)

    assert SCB_CPI_YOY.table_id == "TAB6596"
    assert SCB_CPI_YOY.contents_code == "00000804"
    assert SCB_CPI_YOY.series_id == "TAB6596/00000804"
    assert list(frame.columns) == _COLUMNS
    assert list(frame["date"]) == [
        date(1981, 1, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]
    assert list(frame["value"]) == [12.3, 0.3, -0.1, 0.2]
    assert set(frame["country"]) == {"SE"}
    assert set(frame["indicator"]) == {"cpi_yoy"}
    assert set(frame["source"]) == {"SCB_CPI"}
    assert set(frame["series_id"]) == {"TAB6596/00000804"}
    requested_url = client.get.call_args.args[0]
    assert requested_url == SCB_CPI_YOY.url
    assert "valueCodes[ContentsCode]=00000804" in requested_url
    assert "valueCodes[Tid]=*" in requested_url


def test_scb_cache_empty_shape_and_fail_fast(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(_SAMPLE_JSONSTAT)
    source = ScbSource(client=client, cache_dir=tmp_path)
    source.fetch()
    source.fetch()
    assert client.get.call_count == 1

    empty_client = MagicMock()
    empty_client.get.return_value = _response(json.dumps({"value": []}))
    empty = ScbSource(client=empty_client, cache_dir=tmp_path / "empty").fetch(use_cache=False)
    assert empty.empty and list(empty.columns) == _COLUMNS

    bad_shape = json.loads(_SAMPLE_JSONSTAT)
    bad_shape["value"] = [1.0]
    malformed_client = MagicMock()
    malformed_client.get.return_value = _response(json.dumps(bad_shape))
    with pytest.raises(ValueError, match="size mismatch"):
        ScbSource(client=malformed_client, cache_dir=tmp_path / "malformed").fetch(use_cache=False)

    missing_client = MagicMock()
    missing_client.get.return_value = _response("missing", 404)
    with pytest.raises(ValueError, match="404"):
        ScbSource(client=missing_client, cache_dir=tmp_path / "missing").fetch(use_cache=False)
    assert missing_client.get.call_count == 1


class _MutableScb:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[object, bool]] = []

    def fetch(self, spec=SCB_CPI_YOY, use_cache=True):
        self.calls.append((spec, use_cache))
        return self.frame.copy()


class _EmptyImfCpi:
    def fetch(self, countries, use_cache=True, start_year=2010):
        return pd.DataFrame(columns=_COLUMNS)


def test_scb_cycle_flow_writes_complete_releases_and_fails_closed(tmp_path):
    engine = make_engine(tmp_path / "scb.db")
    sweden = get_country("SE")
    source = _MutableScb(_scb_frame([(date(2026, 1, 1), 1.5), (date(2026, 2, 1), 1.7)]))

    first = fetch_cycle.run_pipeline(
        ["cpi"],
        [sweden],
        cpi_source=_EmptyImfCpi(),
        scb_source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=_at(3),
    )
    source.frame = _scb_frame([(date(2026, 1, 1), 1.6)])
    second = fetch_cycle.run_pipeline(
        ["cpi"],
        [sweden],
        cpi_source=_EmptyImfCpi(),
        scb_source=source,
        engine=engine,
        retrieved_at=_at(4),
    )
    source.frame = pd.DataFrame(columns=_COLUMNS)
    failed = fetch_cycle.run_pipeline(
        ["cpi"],
        [sweden],
        cpi_source=_EmptyImfCpi(),
        scb_source=source,
        engine=engine,
        retrieved_at=_at(5),
    )

    assert source.calls[0] == (SCB_CPI_YOY, False)
    assert first["scb/cpi_yoy/SE"]["inserted"] == 2
    assert second["scb/cpi_yoy/SE"]["inserted"] == 1
    assert "empty" in failed["scb/cpi_yoy/SE"]["error"]

    partition = make_partition_key(
        SOURCE_SCB_CPI,
        SCB_CPI_YOY.series_id,
        "SE",
        INDICATOR_CPI,
    )
    with Session(engine) as session:
        history = release_history(session, partition)
        old = load_vintage_panel(session, _at(3), partition_keys=(partition,))
        new = load_vintage_panel(session, _at(4), partition_keys=(partition,))
        current = session.execute(
            select(Observation.date, Observation.value).where(Observation.source == SOURCE_SCB_CPI)
        ).all()
        release_count = len(session.execute(select(DataRelease)).scalars().all())

    assert len(history) == release_count == 2
    assert history[0].source_url == SCB_CPI_YOY.url
    assert history[0].available_at == _at(3).replace(tzinfo=None)
    assert list(old["value"]) == [1.5, 1.7]
    assert list(new["value"]) == [1.6]
    assert current == [(date(2026, 1, 1), 1.6)]
