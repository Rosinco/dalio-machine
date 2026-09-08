"""Immutable pipeline guards for the non-additive liquidity frontier."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLI_USD,
    SOURCE_BIS_GLI,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_TOTAL_INVESTMENTS,
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    SOURCE_OFR_STFM,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.pipelines.fetch_shadow_liquidity import (
    archive_shadow_catalogues,
    main,
    run_pipeline,
)
from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
    make_engine,
)


def _frame(spec, dates: list[date], values: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": dates,
            "value": values or [float(index) for index in range(1, len(dates) + 1)],
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
    )
    artifact_bytes = json.dumps(
        {
            "native_series_id": spec.native_series_id,
            "dates": [item.isoformat() for item in dates],
            "values": values or [float(index) for index in range(1, len(dates) + 1)],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_dir = Path(tempfile.mkdtemp(prefix="dalio-shadow-test-"))
    artifact_path = artifact_dir / f"{digest}.source"
    artifact_path.write_bytes(artifact_bytes)
    frame.attrs.update(
        {
            "source_artifact_path": str(artifact_path),
            "source_artifact_sha256": digest,
            "native_payload_sha256": digest,
        }
    )
    return frame


def _ofr_frame(
    spec,
    observed_dates: list[date],
    *,
    native_dates: list[date] | None = None,
    disclosure_dates: tuple[date, ...] = (),
    null_dates: tuple[date, ...] = (),
) -> pd.DataFrame:
    frame = _frame(spec, observed_dates)
    missing_records = tuple(
        [
            {
                "date": item.isoformat(),
                "reason": "disclosure_edit",
                "evidence": "ofr_disclosure_edits_subseries",
            }
            for item in disclosure_dates
        ]
        + [
            {
                "date": item.isoformat(),
                "reason": "publisher_null_not_zero",
                "evidence": "ofr_aggregation_null",
            }
            for item in null_dates
            if item not in disclosure_dates
        ]
    )
    missing_payload = {
        "native_series_id": spec.native_series_id,
        "cadence_policy": spec.cadence_policy,
        "missing_value_policy": spec.missing_value_policy,
        "records": missing_records,
    }
    artifact_dir = Path(tempfile.mkdtemp(prefix="dalio-ofr-test-"))
    source_bytes = json.dumps(
        {"response_for": spec.native_series_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    native_bytes = json.dumps(
        {"native_series_id": spec.native_series_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    missing_bytes = json.dumps(
        missing_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source_path = artifact_dir / "source.json"
    native_path = artifact_dir / "native.json"
    missing_path = artifact_dir / "missing.json"
    source_path.write_bytes(source_bytes)
    native_path.write_bytes(native_bytes)
    missing_path.write_bytes(missing_bytes)
    frame.attrs.update(
        {
            "native_periods": tuple(item.isoformat() for item in (native_dates or observed_dates)),
            "disclosure_edit_dates": disclosure_dates,
            "unclassified_null_dates": null_dates,
            "source_url": "https://data.financialresearch.gov/v1/series/multifull?test",
            "native_payload_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "source_artifact_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_artifact_path": str(source_path),
            "native_payload_artifact_path": str(native_path),
            "missing_provenance_artifact_path": str(missing_path),
            "missing_period_records": missing_records,
            "missing_provenance_sha256": hashlib.sha256(
                json.dumps(
                    missing_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    )
    return frame


class _SingleSource:
    def __init__(self, frames: list[pd.DataFrame] | pd.DataFrame):
        self.frames = list(frames) if isinstance(frames, list) else [frames]
        self.calls = 0

    def fetch(self, spec, use_cache=True):
        del spec, use_cache
        frame = self.frames[min(self.calls, len(self.frames) - 1)].copy()
        frame.attrs = self.frames[min(self.calls, len(self.frames) - 1)].attrs.copy()
        self.calls += 1
        return frame


class _BulkOfrSource:
    def __init__(self, frames):
        self.frames = frames
        self.bulk_calls = 0
        self.single_calls = 0

    def fetch_many(self, specs, use_cache=True):
        del use_cache
        self.bulk_calls += 1
        return {spec.native_series_id: self.frames[spec.native_series_id] for spec in specs}

    def fetch(self, spec, use_cache=True):
        del use_cache
        self.single_calls += 1
        return self.frames[spec.native_series_id]


def _small_bis(*, maximum_lag: int = 500):
    return replace(
        BIS_GLI_USD,
        expected_start=date(2025, 1, 1),
        minimum_observations=1,
        max_latest_lag_days=maximum_lag,
    )


def _small_ofr(spec=OFR_MMF_TOTAL_INVESTMENTS, *, maximum_lag: int = 500):
    return replace(
        spec,
        expected_start=date(2025, 1, 31),
        minimum_observations=1,
        max_latest_lag_days=maximum_lag,
    )


def test_combined_pipeline_records_provider_specific_catalogue_identity(tmp_path):
    engine = make_engine(tmp_path / "frontier.db")
    bis_spec = _small_bis()
    ofr_spec = _small_ofr()
    bis_frame = _frame(
        bis_spec,
        [date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1)],
    )
    ofr_frame = _ofr_frame(
        ofr_spec,
        [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)],
    )
    ofr_source = _BulkOfrSource({ofr_spec.native_series_id: ofr_frame})

    summary = run_pipeline(
        (bis_spec, ofr_spec),
        sources={
            SOURCE_BIS_GLI: _SingleSource(bis_frame),
            SOURCE_OFR_STFM: ofr_source,
        },
        engine=engine,
        retrieved_at=datetime(2025, 8, 1, tzinfo=UTC),
    )

    assert not any("error" in item for item in summary.values())
    assert ofr_source.bulk_calls == 1
    assert ofr_source.single_calls == 0
    bis_result = next(item for item in summary.values() if item["source_family"] == SOURCE_BIS_GLI)
    ofr_result = next(item for item in summary.values() if item["source_family"] == SOURCE_OFR_STFM)
    assert bis_result["catalogue_semantic_sha256"] == bis_global_liquidity_catalogue_sha256()
    assert bis_result["non_additive_groups"] == ["bis_gli_offshore_credit_by_currency"]
    assert ofr_result["catalogue_semantic_sha256"] == ofr_shadow_liquidity_catalogue_sha256()
    assert ofr_result["parent_series_id"] is None

    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Observation)) == 6
        assert connection.scalar(select(func.count()).select_from(DataReleaseArtifact)) == 4
        releases = connection.execute(
            select(DataRelease.source_family, DataRelease.vintage_label).order_by(
                DataRelease.source_family
            )
        )
        labels = {row.source_family: row.vintage_label for row in releases}
    assert labels[SOURCE_BIS_GLI] == (
        f"{BIS_GLI_CATALOGUE_VINTAGE_PREFIX}{bis_global_liquidity_catalogue_sha256()}"
    )
    assert labels[SOURCE_OFR_STFM].startswith(
        f"{OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX}{ofr_shadow_liquidity_catalogue_sha256()};p:"
    )
    assert len(labels[SOURCE_OFR_STFM].rsplit(";p:", 1)[1]) == 24


def test_semantic_catalogues_are_content_addressed_and_corruption_fails_closed(tmp_path):
    archived = archive_shadow_catalogues(tmp_path)

    assert set(archived) == {SOURCE_BIS_GLI, SOURCE_OFR_STFM}
    for source_family, path in archived.items():
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 1
        assert payload["source_family"] == source_family
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.name.startswith(content_hash)
        assert content_hash != payload["catalogue_semantic_sha256"]
        assert payload["series"]

    bis_path = archived[SOURCE_BIS_GLI]
    bis_path.write_text("corrupt")
    with pytest.raises(ValueError, match="catalogue archive is corrupt"):
        archive_shadow_catalogues(tmp_path)


def test_ofr_explicit_null_month_stays_missing_and_does_not_break_cadence(tmp_path):
    engine = make_engine(tmp_path / "null.db")
    spec = replace(_small_ofr(), cadence_policy="sparse_monthly")
    native = [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]
    frame = _ofr_frame(
        spec,
        [native[0], native[2]],
        native_dates=native,
        null_dates=(native[1],),
    )
    source = _BulkOfrSource({spec.native_series_id: frame})

    summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: source},
        engine=engine,
        retrieved_at=datetime(2025, 4, 10, tzinfo=UTC),
    )

    result = next(iter(summary.values()))
    assert result["rows"] == 2
    assert result["native_period_count"] == 3
    assert result["missing_native_periods"] == 1
    with engine.connect() as connection:
        values = (
            connection.execute(select(Observation.value).order_by(Observation.date)).scalars().all()
        )
    assert values == [1.0, 2.0]


def test_complete_monthly_series_rejects_an_explicit_null(tmp_path):
    engine = make_engine(tmp_path / "complete-null.db")
    spec = _small_ofr()
    native = [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]
    frame = _ofr_frame(
        spec,
        [native[0], native[2]],
        native_dates=native,
        null_dates=(native[1],),
    )

    summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(frame)},
        engine=engine,
        retrieved_at=datetime(2025, 4, 10, tzinfo=UTC),
    )

    assert "missing numeric observation" in next(iter(summary.values()))["error"]


def test_ofr_preliminary_status_and_official_update_clock_survive_release(tmp_path):
    engine = make_engine(tmp_path / "preliminary.db")
    spec = replace(
        OFR_REPO_DVP_AVERAGE_RATE,
        expected_start=date(2025, 1, 6),
        minimum_observations=1,
        max_latest_lag_days=30,
    )
    frame = _ofr_frame(spec, [date(2025, 1, 6), date(2025, 1, 7)])
    frame["status"] = "preliminary"
    frame.attrs["publisher_last_updated_at"] = datetime(2025, 1, 8, tzinfo=UTC)

    summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(frame)},
        engine=engine,
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )

    assert "error" not in next(iter(summary.values()))
    with engine.connect() as connection:
        statuses = connection.execute(select(ReleaseObservation.status)).scalars().all()
        published_at = connection.scalar(select(DataRelease.published_at))
    assert statuses == ["preliminary", "preliminary"]
    assert published_at == datetime(2025, 1, 8)


@pytest.mark.parametrize(
    ("dates", "error"),
    [
        ([date(2025, 1, 1), date(2025, 7, 1)], "gap in quarterly cadence"),
        ([date(2025, 1, 1), date(2025, 5, 1)], "quarter start"),
    ],
)
def test_bis_quarterly_cadence_fails_closed(tmp_path, dates, error):
    spec = _small_bis()
    summary = run_pipeline(
        (spec,),
        sources={SOURCE_BIS_GLI: _SingleSource(_frame(spec, dates))},
        engine=make_engine(tmp_path / f"bad-{error[:3]}.db"),
        retrieved_at=datetime(2025, 8, 1, tzinfo=UTC),
    )
    assert error in next(iter(summary.values()))["error"]


def test_ofr_requires_native_missing_value_provenance(tmp_path):
    spec = _small_ofr()
    frame = _frame(spec, [date(2025, 1, 31), date(2025, 2, 28)])
    summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(frame)},
        engine=make_engine(tmp_path / "missing-provenance.db"),
        retrieved_at=datetime(2025, 3, 10, tzinfo=UTC),
    )
    assert "native-period provenance" in next(iter(summary.values()))["error"]


def test_snapshot_contraction_requires_explicit_override(tmp_path):
    engine = make_engine(tmp_path / "contraction.db")
    spec = _small_bis()
    full = _frame(
        spec,
        [date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1)],
    )
    contracted = _frame(spec, [date(2025, 1, 1), date(2025, 4, 1)])
    source = _SingleSource([full, contracted, contracted])

    first = run_pipeline(
        (spec,),
        sources={SOURCE_BIS_GLI: source},
        engine=engine,
        retrieved_at=datetime(2025, 8, 1, tzinfo=UTC),
    )
    rejected = run_pipeline(
        (spec,),
        sources={SOURCE_BIS_GLI: source},
        engine=engine,
        retrieved_at=datetime(2025, 9, 1, tzinfo=UTC),
    )
    accepted = run_pipeline(
        (spec,),
        sources={SOURCE_BIS_GLI: source},
        engine=engine,
        retrieved_at=datetime(2025, 9, 2, tzinfo=UTC),
        allow_contraction=True,
    )

    assert "error" not in next(iter(first.values()))
    assert "contracts complete observed history" in next(iter(rejected.values()))["error"]
    assert next(iter(accepted.values()))["removed"] == 1
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Observation)) == 2


def test_sparse_history_rejects_removed_date_even_when_row_count_does_not_fall(tmp_path):
    engine = make_engine(tmp_path / "same-count-contraction.db")
    spec = replace(_small_ofr(), cadence_policy="sparse_monthly")
    first = _ofr_frame(spec, [date(2025, 1, 31), date(2025, 3, 31)])
    replacement = _ofr_frame(spec, [date(2025, 1, 31), date(2025, 4, 30)])
    source = _SingleSource([first, replacement])

    initial = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: source},
        engine=engine,
        retrieved_at=datetime(2025, 4, 10, tzinfo=UTC),
    )
    rejected = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: source},
        engine=engine,
        retrieved_at=datetime(2025, 5, 10, tzinfo=UTC),
    )

    assert "error" not in next(iter(initial.values()))
    assert "first removed date 2025-03-31" in next(iter(rejected.values()))["error"]
    with engine.connect() as connection:
        stored_dates = (
            connection.execute(select(Observation.date).order_by(Observation.date)).scalars().all()
        )
    assert stored_dates == [date(2025, 1, 31), date(2025, 3, 31)]


def test_fresh_null_period_cannot_hide_stale_numeric_history(tmp_path):
    engine = make_engine(tmp_path / "stale-numeric.db")
    spec = replace(_small_ofr(maximum_lag=30), cadence_policy="sparse_monthly")
    frame = _ofr_frame(
        spec,
        [date(2025, 1, 31)],
        native_dates=[date(2025, 1, 31), date(2025, 3, 31)],
        null_dates=(date(2025, 3, 31),),
    )

    summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(frame)},
        engine=engine,
        retrieved_at=datetime(2025, 4, 10, tzinfo=UTC),
    )

    assert "latest numeric observation is stale" in next(iter(summary.values()))["error"]


def test_failed_ofr_bulk_request_falls_back_and_isolates_native_series(tmp_path):
    engine = make_engine(tmp_path / "fallback.db")
    first = _small_ofr()
    second = _small_ofr(OFR_MMF_REPO_INVESTMENTS)
    frames = {
        first.native_series_id: _ofr_frame(first, [date(2025, 1, 31)]),
        second.native_series_id: RuntimeError("publisher rejected this series"),
    }

    class _FailingBulk:
        def fetch_many(self, specs, use_cache=True):
            del specs, use_cache
            raise ValueError("malformed peer in bulk response")

        def fetch(self, spec, use_cache=True):
            del use_cache
            result = frames[spec.native_series_id]
            if isinstance(result, Exception):
                raise result
            return result

    summary = run_pipeline(
        (first, second),
        sources={SOURCE_OFR_STFM: _FailingBulk()},
        engine=engine,
        retrieved_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    results = {item["series_id"]: item for item in summary.values()}
    assert "error" not in results[first.native_series_id]
    assert "publisher rejected" in results[second.native_series_id]["error"]
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Observation)) == 1


def test_future_native_period_and_identity_drift_are_rejected(tmp_path):
    spec = _small_ofr()
    future = _ofr_frame(
        spec,
        [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)],
    )
    future_summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(future)},
        engine=make_engine(tmp_path / "future.db"),
        retrieved_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    assert "after retrieval date" in next(iter(future_summary.values()))["error"]

    bad = _ofr_frame(spec, [date(2025, 1, 31)])
    bad["indicator"] = "wrong"
    bad_summary = run_pipeline(
        (spec,),
        sources={SOURCE_OFR_STFM: _SingleSource(bad)},
        engine=make_engine(tmp_path / "identity.db"),
        retrieved_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    assert "indicator mismatch" in next(iter(bad_summary.values()))["error"]


def test_cli_rejects_unknown_native_series_id():
    with pytest.raises(SystemExit, match="2"):
        main(["NOT-A-NATIVE-SERIES"])
