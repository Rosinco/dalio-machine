from datetime import date

from sqlalchemy.orm import Session

from dalio.scoring.long_term import extract_features as extract_long_term
from dalio.scoring.short_term import extract_features as extract_short_term
from dalio.scoring.source_priority import SOURCE_PREFERENCES
from dalio.storage.db import Observation, init_db, make_engine


def _add(session, indicator: str, source: str, value: float, observed_on: date) -> None:
    session.add(
        Observation(
            country="SE",
            indicator=indicator,
            date=observed_on,
            value=value,
            source=source,
            series_id=f"{source}/{indicator}",
        )
    )


def test_swedish_official_sources_win_same_date_but_date_remains_primary(tmp_path):
    engine = make_engine(tmp_path / "priority.db")
    init_db(engine)
    same_date = date(2026, 1, 1)
    later = date(2026, 2, 1)

    with Session(engine) as session:
        for source, value in (("FRED", 1.0), ("IMF_CPI", 2.0), ("SCB_CPI", 3.0)):
            _add(session, "cpi_yoy", source, value, same_date)
        for source, value in (
            ("FRED", 0.5),
            ("BIS_CBPOL", 0.75),
            ("RIKSBANK_SWEA", 1.0),
        ):
            _add(session, "policy_rate", source, value, same_date)
        for indicator in ("yield_10y", "yield_2y"):
            _add(session, indicator, "FRED", 2.0, same_date)
            _add(session, indicator, "RIKSBANK_SWEA", 2.5, same_date)
        _add(session, "real_gdp_yoy", "ZZZ", 3.0, same_date)
        _add(session, "real_gdp_yoy", "AAA", 2.0, same_date)
        session.commit()

        short = extract_short_term(session, "SE", as_of=same_date)
        long = extract_long_term(session, "SE", as_of=same_date)

        _add(session, "cpi_yoy", "IMF_CPI", 4.0, later)
        _add(session, "policy_rate", "BIS_CBPOL", 1.25, later)
        _add(session, "yield_10y", "FRED", 3.0, later)
        _add(session, "yield_2y", "FRED", 3.0, later)
        session.commit()
        newer_short = extract_short_term(session, "SE", as_of=later)
        newer_long = extract_long_term(session, "SE", as_of=later)

    assert SOURCE_PREFERENCES[("SE", "cpi_yoy")] == (
        "SCB_CPI",
        "IMF_CPI",
        "FRED",
    )
    assert SOURCE_PREFERENCES[("SE", "policy_rate")] == (
        "RIKSBANK_SWEA",
        "BIS_CBPOL",
        "FRED",
    )
    assert short.cpi_yoy == long.cpi_yoy == 3.0
    assert short.policy_rate == 1.0
    assert short.yield_10y == long.yield_10y == 2.5
    assert short.yield_2y == 2.5
    assert short.real_gdp_yoy == 2.0  # unmapped indicators keep source-ascending ties

    assert newer_short.cpi_yoy == newer_long.cpi_yoy == 4.0
    assert newer_short.policy_rate == 1.25
    assert newer_short.yield_10y == newer_long.yield_10y == 3.0
    assert newer_short.yield_2y == 3.0
