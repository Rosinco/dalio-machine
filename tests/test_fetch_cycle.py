"""dalio-fetch-cycle pipeline (slice 26): fake sources, summary, isolation of failures."""
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from dalio.countries import get_country
from dalio.data_sources.oecd import LFS_UNEMPLOYMENT, QNA_GDP_GROWTH
from dalio.pipelines import fetch_cycle
from dalio.storage.db import Observation, make_engine


def _long(country, indicator, source, values: dict[date, float]):
    return pd.DataFrame([{"country": country, "indicator": indicator, "date": d, "value": v,
                          "source": source, "series_id": "x"} for d, v in values.items()])


class _FakeCpi:
    def fetch(self, countries, use_cache=True, start_year=2010):
        return pd.concat([_long(c.iso2, "cpi_yoy", "IMF_CPI", {date(2026, 6, 1): 1.7})
                          for c in countries if c.on_map], ignore_index=True)


class _FakeBis:
    def fetch_policy_rate(self, spec, use_cache=True):
        if spec.country == "CN":
            raise ValueError("BIS series not found (404)")
        return _long(spec.country, "policy_rate", "BIS_CBPOL", {date(2026, 8, 1): 3.0})


class _FakeOecd:
    def fetch(self, flow, countries, use_cache=True, start_year=2010):
        if flow is QNA_GDP_GROWTH:
            return pd.concat([_long(c.iso2, "real_gdp_yoy", "OECD_QNA", {date(2026, 4, 1): 1.0})
                              for c in countries], ignore_index=True)
        assert flow is LFS_UNEMPLOYMENT
        raise RuntimeError("OECD server error 503")


def test_pipeline_stores_flows_and_isolates_failures(tmp_path):
    engine = make_engine(tmp_path / "t.db")
    countries = [get_country(c) for c in ("JP", "CN", "EU")]
    summary = fetch_cycle.run_pipeline(
        cpi_source=_FakeCpi(), bis_source=_FakeBis(), oecd_source=_FakeOecd(),
        countries=countries, engine=engine,
    )
    assert summary["cpi/cpi_yoy"]["rows"] == 2 and summary["cpi/cpi_yoy"]["countries"] == ["CN", "JP"]
    assert summary["cbpol/policy_rate/JP"]["inserted"] == 1
    assert "error" in summary["cbpol/policy_rate/CN"]                        # one country failing…
    assert summary["cbpol/policy_rate/EU"]["rows"] == 1                     # …does not stop the rest
    assert summary["qna/real_gdp_yoy"]["rows"] == 3
    assert "503" in summary["lfs/unemployment_rate"]["error"]
    from sqlalchemy.orm import Session
    with Session(engine) as s:
        rows = s.execute(select(Observation.country, Observation.indicator, Observation.source)).all()
    assert (("JP", "cpi_yoy", "IMF_CPI") in rows and ("EU", "policy_rate", "BIS_CBPOL") in rows
            and ("EU", "real_gdp_yoy", "OECD_QNA") in rows)
    assert not any(r.indicator == "unemployment_rate" for r in rows)


def test_only_subset_and_unknown_flow(tmp_path):
    engine = make_engine(tmp_path / "t.db")
    summary = fetch_cycle.run_pipeline(["qna"], [get_country("US")], oecd_source=_FakeOecd(), engine=engine)
    assert list(summary) == ["qna/real_gdp_yoy"]
    with pytest.raises(ValueError, match="Unknown flow"):
        fetch_cycle.run_pipeline(["nope"], engine=engine)
