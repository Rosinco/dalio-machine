from datetime import date

import pytest

from dalio.app.views import (
    EUROZONE_ISO3,
    INDICATOR_EXPLANATIONS,
    PHASE_EXPLANATIONS,
    STAGE_EXPLANATIONS,
    compute_country_view,
    compute_world_view,
    expand_iso3_for_map,
    map_iso3_to_country_iso2,
    top_tilts,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

# ─── Static mappings ────────────────────────────────────────────────────────


def test_eurozone_has_at_least_19_members():
    assert len(EUROZONE_ISO3) >= 19
    # Sanity: includes the largest economies
    assert "DEU" in EUROZONE_ISO3
    assert "FRA" in EUROZONE_ISO3
    assert "ITA" in EUROZONE_ISO3


def test_expand_iso3_for_map_expands_eurozone():
    expanded = expand_iso3_for_map("EMU")
    assert len(expanded) == len(EUROZONE_ISO3)
    assert "DEU" in expanded


def test_expand_iso3_passes_through_non_eu():
    assert expand_iso3_for_map("USA") == ("USA",)
    assert expand_iso3_for_map("CHN") == ("CHN",)


def test_map_iso3_to_country_iso2_handles_eurozone_members():
    assert map_iso3_to_country_iso2("DEU") == "EU"
    assert map_iso3_to_country_iso2("FRA") == "EU"


def test_map_iso3_to_country_iso2_handles_basket_members():
    assert map_iso3_to_country_iso2("USA") == "US"
    assert map_iso3_to_country_iso2("CHN") == "CN"
    assert map_iso3_to_country_iso2("GBR") == "UK"
    assert map_iso3_to_country_iso2("JPN") == "JP"
    assert map_iso3_to_country_iso2("SWE") == "SE"


def test_map_iso3_unknown_returns_none():
    assert map_iso3_to_country_iso2("ZZZ") is None


# ─── Explanation coverage ──────────────────────────────────────────────────


def test_phase_explanations_cover_all_phases():
    for phase in (0, 1, 2, 3, 4, 5, 6, 7):
        assert phase in PHASE_EXPLANATIONS
        assert len(PHASE_EXPLANATIONS[phase]) > 20  # at least a sentence


def test_stage_explanations_cover_all_stages():
    for stage in (0, 1, 2, 3, 4):
        assert stage in STAGE_EXPLANATIONS
        assert len(STAGE_EXPLANATIONS[stage]) > 20


def test_indicator_explanations_cover_short_term_indicators():
    expected = {
        "policy_rate", "cpi_yoy", "unemployment_rate",
        "yield_10y", "yield_2y", "real_gdp_yoy",
    }
    assert expected.issubset(INDICATOR_EXPLANATIONS.keys())


def test_indicator_explanations_cover_long_term_indicators():
    expected = {
        "total_credit_pct_gdp", "gov_debt_pct_gdp",
        "hh_debt_pct_gdp", "corp_debt_pct_gdp", "debt_service_ratio",
    }
    assert expected.issubset(INDICATOR_EXPLANATIONS.keys())


# ─── DB-backed view computations ───────────────────────────────────────────


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed_us_minimal(session):
    """Add just enough US data for compute_country_view to return non-empty."""
    today = date(2026, 4, 1)
    obs = [
        ("policy_rate", 4.0),
        ("cpi_yoy", 3.0),
        ("unemployment_rate", 4.0),
        ("yield_10y", 4.5),
        ("yield_2y", 4.0),
        ("real_gdp_yoy", 2.0),
        ("total_credit_pct_gdp", 250.0),
        ("debt_service_ratio", 14.0),
    ]
    for ind, val in obs:
        session.add(Observation(
            country="US", indicator=ind, date=today,
            value=val, source="TEST", series_id="X",
        ))


def test_compute_country_view_returns_full_view(session_factory):
    with session_factory() as s:
        _seed_us_minimal(s)
        s.commit()
    with session_factory() as s:
        view = compute_country_view(s, "US")
    assert view.country.iso2 == "US"
    assert view.short_term.country == "US"
    assert view.long_term.country == "US"
    assert view.allocation.country == "US"
    assert len(view.allocation.tilts) == 8


def test_top_tilts_returns_largest_magnitudes(session_factory):
    with session_factory() as s:
        _seed_us_minimal(s)
        s.commit()
    with session_factory() as s:
        view = compute_country_view(s, "US")
    top = top_tilts(view.allocation, n=3)
    assert len(top) == 3
    # Sorted by absolute magnitude
    mags = [abs(t.tilt) for t in top]
    assert mags == sorted(mags, reverse=True)


def test_compute_world_view_includes_no_data_countries(session_factory):
    with session_factory() as s:
        _seed_us_minimal(s)  # only US has data
        s.commit()
    with session_factory() as s:
        points = compute_world_view(s)
    by_iso2 = {p.iso2: p for p in points}
    assert by_iso2["US"].has_data is True
    assert by_iso2["IN"].has_data is False
    assert by_iso2["BR"].has_data is False
    # All 8 basket countries always present
    assert len(points) == 8


def test_compute_world_view_hover_text_html_safe(session_factory):
    with session_factory() as s:
        _seed_us_minimal(s)
        s.commit()
    with session_factory() as s:
        points = compute_world_view(s)
    us = next(p for p in points if p.iso2 == "US")
    # Plotly hover supports <br> for line breaks
    assert "<br>" in us.hover_text
    assert "United States" in us.hover_text
