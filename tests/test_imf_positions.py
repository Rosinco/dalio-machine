"""Official IMF PIP/DIP adapters and complete-partition ETL."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalio.countries import get_country
from dalio.data_sources.imf_positions import (
    DIP_HISTORY_START_YEAR,
    DIP_SERIES,
    IMF_POSITION_COUNTERPARTS,
    IMF_POSITION_COUNTRIES,
    PIP_FREQUENCIES,
    PIP_HISTORY_START_YEAR,
    PIP_SERIES,
    ImfDipSource,
    ImfPipSource,
    PositionSeriesSpec,
)
from dalio.pipelines.fetch_positions import run_pipeline
from dalio.storage.db import CrossBorderPosition, DataRelease, make_engine
from dalio.storage.positions import load_cross_border_vintage

PIP_SAMPLE = (
    "DATAFLOW,COUNTRY,ACCOUNTING_ENTRY,INDICATOR,SECTOR,COUNTERPART_SECTOR,"
    "COUNTERPART_COUNTRY,FREQUENCY,TIME_PERIOD,OBS_VALUE,SCALE,STATUS\n"
    "IMF.STA:PIP(5.0.0),SWE,A,P_TOTINV_P_USD,S1,S1,USA,S,2025-S1,"
    "123456789.5,6,A\n"
    "IMF.STA:PIP(5.0.0),SWE,A,P_TOTINV_P_USD,S1,S1,GBR,S,2025-S1,"
    "234.0,6,\n"
    "IMF.STA:PIP(5.0.0),SWE,A,P_TOTINV_P_USD,S1,S1,G001,S,2025-S1,,6,\n"
)

DIP_SAMPLE = (
    "DATAFLOW,COUNTRY,DV_TYPE,INDICATOR,COUNTERPART_COUNTRY,FREQUENCY,"
    "TIME_PERIOD,OBS_VALUE,SCALE,STATUS\n"
    "IMF.STA:DIP(12.0.1),SWE,O,INWD_D_NETLA_FALL_ALL,USA,A,2024,"
    "345678901.25,6,E\n"
    "IMF.STA:DIP(12.0.1),SWE,O,INWD_D_NETLA_FALL_ALL,G001,A,2024,"
    "456789012.5,6,\n"
)


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _spec(specs: tuple[PositionSeriesSpec, ...], indicator: str) -> PositionSeriesSpec:
    return next(spec for spec in specs if spec.native_indicator == indicator)


def test_catalogues_are_explicit_and_exclude_the_euro_area_aggregate():
    assert [spec.native_indicator for spec in PIP_SERIES] == [
        "P_TOTINV_P_USD",
        "P_F51_P_USD",
        "P_F3_L_P_USD",
        "P_F3_S_P_USD",
    ]
    assert [spec.native_indicator for spec in DIP_SERIES] == [
        "INWD_D_NETLA_FALL_ALL",
        "INWD_D_NETLA_F51_ALL",
        "INWD_D_NETLA_FL_ALL",
        "OTWD_D_NETAL_FALL_ALL",
        "OTWD_D_NETAL_F51_ALL",
        "OTWD_D_NETAL_FL_ALL",
    ]
    assert PIP_FREQUENCIES == ("A", "S")
    assert len(IMF_POSITION_COUNTRIES) == 21
    assert "EU" not in {country.iso2 for country in IMF_POSITION_COUNTRIES}
    assert len(IMF_POSITION_COUNTERPARTS) == 22
    assert IMF_POSITION_COUNTERPARTS[-1].internal_code == "WLD"
    assert IMF_POSITION_COUNTERPARTS[-1].imf_code == "G001"
    assert all(spec.unit == "USD" for spec in (*PIP_SERIES, *DIP_SERIES))


def test_pip_builds_exact_native_key_and_does_not_apply_scale(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(PIP_SAMPLE)
    source = ImfPipSource(client=client, cache_dir=tmp_path)
    spec = _spec(PIP_SERIES, "P_TOTINV_P_USD")

    frame = source.fetch(
        get_country("SE"),
        spec,
        "S",
        counterparts=(
            IMF_POSITION_COUNTERPARTS[0],
            IMF_POSITION_COUNTERPARTS[2],
            IMF_POSITION_COUNTERPARTS[-1],
        ),
        start_year=1997,
        use_cache=False,
    )

    assert client.get.call_args.args[0] == (
        "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,PIP/"
        "SWE.A.P_TOTINV_P_USD.S1.S1.USA+GBR+G001.S?"
        "startPeriod=1997&detail=dataonly"
    )
    assert list(
        frame[
            [
                "reporter_country",
                "counterpart_country",
                "date",
                "frequency",
                "value",
                "status",
            ]
        ].itertuples(index=False, name=None)
    ) == [
        ("SE", "UK", date(2025, 1, 1), "S", 234.0, "observed"),
        ("SE", "US", date(2025, 1, 1), "S", 123456789.5, "A"),
    ]
    assert frame.iloc[1]["series_id"] == ("PIP/SWE.A.P_TOTINV_P_USD.S1.S1.USA.S")
    assert frame.iloc[0]["direction"] == "outward_assets"
    assert frame.iloc[0]["accounting_basis"] == "assets"


def test_dip_uses_reported_not_mirror_data_and_preserves_net_basis(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(DIP_SAMPLE)
    source = ImfDipSource(client=client, cache_dir=tmp_path)
    spec = _spec(DIP_SERIES, "INWD_D_NETLA_FALL_ALL")
    counterparts = (IMF_POSITION_COUNTERPARTS[0], IMF_POSITION_COUNTERPARTS[-1])

    frame = source.fetch(
        get_country("SE"),
        spec,
        "A",
        counterparts=counterparts,
        start_year=2009,
        use_cache=False,
    )

    assert client.get.call_args.args[0] == (
        "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,DIP/"
        "SWE.O.INWD_D_NETLA_FALL_ALL.USA+G001.A?"
        "startPeriod=2009&detail=dataonly"
    )
    assert list(frame["counterpart_country"]) == ["US", "WLD"]
    assert list(frame["value"]) == [345678901.25, 456789012.5]
    assert set(frame["derivation_type"]) == {"O"}
    assert set(frame["direction"]) == {"inward"}
    assert set(frame["accounting_basis"]) == {"net_liabilities_less_assets"}
    assert list(frame["status"]) == ["E", "observed"]


@pytest.mark.parametrize(
    ("source_cls", "text", "message"),
    [
        (
            ImfPipSource,
            "COUNTRY,TIME_PERIOD,OBS_VALUE\n",
            "missing columns",
        ),
        (
            ImfPipSource,
            PIP_SAMPLE.replace("IMF.STA:PIP", "IMF.STA:IIP"),
            "unexpected dataflow",
        ),
        (
            ImfPipSource,
            PIP_SAMPLE.replace(",S1,S1,USA,S,", ",S121,S1,USA,S,"),
            "native key",
        ),
        (
            ImfDipSource,
            DIP_SAMPLE.replace(",SWE,O,", ",SWE,SCC,"),
            "native key",
        ),
        (
            ImfDipSource,
            DIP_SAMPLE.replace("345678901.25", "not-a-number"),
            "non-numeric",
        ),
    ],
)
def test_adapters_reject_malformed_or_semantically_mixed_snapshots(
    tmp_path, source_cls, text, message
):
    client = MagicMock()
    client.get.return_value = _response(text)
    source = source_cls(client=client, cache_dir=tmp_path)
    if source_cls is ImfPipSource:
        spec = _spec(PIP_SERIES, "P_TOTINV_P_USD")
        frequency = "S"
    else:
        spec = _spec(DIP_SERIES, "INWD_D_NETLA_FALL_ALL")
        frequency = "A"

    with pytest.raises(ValueError, match=message):
        source.fetch(
            get_country("SE"),
            spec,
            frequency,
            counterparts=(
                IMF_POSITION_COUNTERPARTS[0],
                IMF_POSITION_COUNTERPARTS[2],
                IMF_POSITION_COUNTERPARTS[-1],
            ),
            use_cache=False,
        )


def test_duplicate_nonfinite_bad_period_and_http_errors_fail_closed(tmp_path):
    source = ImfPipSource(client=MagicMock(), cache_dir=tmp_path)
    spec = _spec(PIP_SERIES, "P_TOTINV_P_USD")
    duplicate = PIP_SAMPLE.replace(
        "IMF.STA:PIP(5.0.0),SWE,A,P_TOTINV_P_USD,S1,S1,G001,S,2025-S1,,6,\n",
        "IMF.STA:PIP(5.0.0),SWE,A,P_TOTINV_P_USD,S1,S1,USA,S,2025-S1,1,6,A\n",
    )
    source._client.get.return_value = _response(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        source.fetch(
            get_country("SE"),
            spec,
            "S",
            counterparts=IMF_POSITION_COUNTERPARTS[:3] + IMF_POSITION_COUNTERPARTS[-1:],
            use_cache=False,
        )

    source._client.get.return_value = _response(PIP_SAMPLE.replace("123456789.5", "inf"))
    with pytest.raises(ValueError, match="finite"):
        source.fetch(
            get_country("SE"),
            spec,
            "S",
            counterparts=IMF_POSITION_COUNTERPARTS[:3] + IMF_POSITION_COUNTERPARTS[-1:],
            use_cache=False,
        )

    source._client.get.return_value = _response(PIP_SAMPLE.replace("2025-S1", "2025-S3"))
    with pytest.raises(ValueError, match="semiannual period"):
        source.fetch(
            get_country("SE"),
            spec,
            "S",
            counterparts=IMF_POSITION_COUNTERPARTS[:3] + IMF_POSITION_COUNTERPARTS[-1:],
            use_cache=False,
        )

    source._client.get.return_value = _response("denied", 403)
    with pytest.raises(ValueError, match="403"):
        source.fetch(
            get_country("SE"),
            spec,
            "S",
            counterparts=IMF_POSITION_COUNTERPARTS[:3] + IMF_POSITION_COUNTERPARTS[-1:],
            use_cache=False,
        )


def _position_frame(
    dataset: str,
    reporter: str,
    spec: PositionSeriesSpec,
    frequency: str,
    counterparts: list[tuple[str, str, float]],
) -> pd.DataFrame:
    reporter_code = get_country(reporter).imf_id
    rows = []
    for counterpart_country, counterpart_code, value in counterparts:
        if dataset == "PIP":
            native_key = (
                f"{reporter_code}.A.{spec.native_indicator}.S1.S1.{counterpart_code}.{frequency}"
            )
            reporter_sector = counterpart_sector = "S1"
            derivation = None
        else:
            native_key = f"{reporter_code}.O.{spec.native_indicator}.{counterpart_code}.{frequency}"
            reporter_sector = counterpart_sector = None
            derivation = "O"
        rows.append(
            {
                "dataset": dataset,
                "reporter_country": reporter,
                "reporter_code": reporter_code,
                "counterpart_country": counterpart_country,
                "counterpart_code": counterpart_code,
                "date": date(2024, 1, 1),
                "direction": spec.direction,
                "accounting_basis": spec.accounting_basis,
                "instrument_code": spec.instrument_code,
                "instrument_label": spec.instrument_label,
                "frequency": frequency,
                "value": value,
                "unit": "USD",
                "source": f"IMF_{dataset}",
                "native_indicator": spec.native_indicator,
                "reporter_sector_code": reporter_sector,
                "counterpart_sector_code": counterpart_sector,
                "derivation_type": derivation,
                "series_id": f"{dataset}/{native_key}",
                "status": "observed",
            }
        )
    return pd.DataFrame(rows)


class FakePositionSource:
    def __init__(self, dataset: str, frames: dict[tuple[str, str, str], pd.DataFrame]):
        self.dataset = dataset
        self.frames = frames
        self.calls = []

    def fetch(
        self,
        reporter,
        spec,
        frequency,
        *,
        counterparts,
        start_year,
        use_cache=True,
    ):
        self.calls.append(
            (
                reporter.iso2,
                spec.native_indicator,
                frequency,
                tuple(c.internal_code for c in counterparts),
                start_year,
                use_cache,
            )
        )
        return self.frames.get(
            (reporter.iso2, spec.native_indicator, frequency), pd.DataFrame()
        ).copy()

    @staticmethod
    def url_for(reporter, spec, frequency, *, counterparts, start_year):
        return (
            f"https://example.test/{reporter.iso2}/{spec.native_indicator}/"
            f"{frequency}?startPeriod={start_year}"
        )


def test_pipeline_ingests_independent_typed_partitions_and_records_absence(tmp_path):
    engine = make_engine(tmp_path / "pipeline.db")
    pip_spec = _spec(PIP_SERIES, "P_TOTINV_P_USD")
    dip_spec = _spec(DIP_SERIES, "INWD_D_NETLA_FALL_ALL")
    pip_source = FakePositionSource(
        "PIP",
        {
            ("SE", pip_spec.native_indicator, "A"): _position_frame(
                "PIP", "SE", pip_spec, "A", [("US", "USA", 100.0)]
            ),
            ("SE", pip_spec.native_indicator, "S"): _position_frame(
                "PIP", "SE", pip_spec, "S", [("US", "USA", 110.0)]
            ),
        },
    )
    dip_source = FakePositionSource(
        "DIP",
        {
            ("SE", dip_spec.native_indicator, "A"): _position_frame(
                "DIP", "SE", dip_spec, "A", [("WLD", "G001", 200.0)]
            )
        },
    )
    run_at = datetime(2026, 9, 8, 12, tzinfo=UTC)

    summary = run_pipeline(
        reporters=(get_country("SE"), get_country("UK")),
        pip_specs=(pip_spec,),
        dip_specs=(dip_spec,),
        pip_source=pip_source,
        dip_source=dip_source,
        use_cache=False,
        engine=engine,
        retrieved_at=run_at,
    )

    assert summary["PIP/SE/portfolio_total/A"]["rows"] == 1
    assert summary["PIP/SE/portfolio_total/S"]["rows"] == 1
    assert summary["DIP/SE/direct_total_inward/A"]["rows"] == 1
    assert summary["PIP/UK/portfolio_total/A"]["not_reported"] is True
    assert summary["DIP/UK/direct_total_inward/A"]["not_reported"] is True
    assert pip_source.calls[0][-2:] == (PIP_HISTORY_START_YEAR, False)
    assert dip_source.calls[0][-2:] == (DIP_HISTORY_START_YEAR, False)

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 3
        assert session.scalar(select(func.count()).select_from(CrossBorderPosition)) == 3
        frame = load_cross_border_vintage(session, run_at)

    assert set(frame["dataset"]) == {"PIP", "DIP"}
    assert set(frame["frequency"]) == {"A", "S"}
    assert set(frame["counterpart_country"]) == {"US", "WLD"}


def test_pipeline_rejects_mixed_native_key_without_replacing_prior(tmp_path):
    engine = make_engine(tmp_path / "pipeline.db")
    spec = _spec(PIP_SERIES, "P_TOTINV_P_USD")
    source = FakePositionSource(
        "PIP",
        {
            ("SE", spec.native_indicator, "A"): _position_frame(
                "PIP", "SE", spec, "A", [("US", "USA", 100.0)]
            )
        },
    )
    run_pipeline(
        reporters=(get_country("SE"),),
        pip_specs=(spec,),
        dip_specs=(),
        pip_frequencies=("A",),
        pip_source=source,
        engine=engine,
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    invalid = _position_frame("PIP", "SE", spec, "A", [("US", "USA", 90.0)])
    invalid["series_id"] = "PIP/SWE.A.P_TOTINV_P_USD.S1.S1.GBR.A"
    source.frames[("SE", spec.native_indicator, "A")] = invalid

    summary = run_pipeline(
        reporters=(get_country("SE"),),
        pip_specs=(spec,),
        dip_specs=(),
        pip_frequencies=("A",),
        pip_source=source,
        engine=engine,
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "native series" in summary["PIP/SE/portfolio_total/A"]["error"]
    with Session(engine) as session:
        frame = load_cross_border_vintage(session, datetime(2026, 9, 1, tzinfo=UTC))
    assert list(frame["value"]) == [100.0]
