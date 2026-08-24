import pytest

from dalio.countries import (
    COUNTRIES,
    CYCLE_COUNTRIES,
    EUROZONE_ISO3,
    ISO2_TO_BIS,
    ISO2_TO_IMTS,
    ISO2_TO_WB,
    ISO3_TO_ISO2,
    RANKING_POPULATION,
    DataQuality,
    Tier,
    get_country,
    get_country_by_iso3,
)


def test_basket_size():
    assert len(COUNTRIES) == 22


def test_iso2_codes_unique():
    codes = [c.iso2 for c in COUNTRIES]
    assert len(set(codes)) == len(codes)


def test_iso3_codes_unique():
    codes = [c.iso3 for c in COUNTRIES]
    assert len(set(codes)) == len(codes)


def test_wb_ids_unique_and_present():
    ids = [c.wb_id for c in COUNTRIES]
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_tier_distribution():
    tiers = [c.tier for c in COUNTRIES]
    assert tiers.count(Tier.TIER_1) == 6
    assert tiers.count(Tier.TIER_2) == 2
    assert tiers.count(Tier.TIER_3) == 14


def test_original_basket_still_first_and_unchanged():
    expected = ["US", "CN", "EU", "UK", "JP", "SE", "IN", "BR"]
    assert [c.iso2 for c in COUNTRIES[:8]] == expected
    assert [c.iso2 for c in CYCLE_COUNTRIES] == expected


def test_cycle_countries_are_exactly_those_with_fred_wiring():
    for c in COUNTRIES:
        assert c.has_cycle_wiring == (c.fred_id is not None)
        assert (c in CYCLE_COUNTRIES) == c.has_cycle_wiring


def test_fundamentals_players_present():
    expected = {
        "US", "CN", "JP", "DE", "IN", "UK", "FR", "IT", "BR", "CA", "RU",
        "KR", "AU", "MX", "ES", "ID", "NL", "SA", "TR", "CH", "SE", "EU",
    }
    assert {c.iso2 for c in COUNTRIES} == expected


def test_eu_aggregate_is_only_off_map_country():
    off_map = [c.iso2 for c in COUNTRIES if not c.on_map]
    assert off_map == ["EU"]


def test_ranking_population_excludes_eu_aggregate():
    assert len(RANKING_POPULATION) == 21
    assert "EU" not in RANKING_POPULATION


def test_eu_members_use_euro_and_currency_union():
    members = [c for c in COUNTRIES if c.eu_member]
    assert {c.iso2 for c in members} == {"DE", "FR", "IT", "ES", "NL"}
    for c in members:
        assert c.currency == "EUR"
        assert c.fx_regime == "currency_union"
        assert c.iso3 in EUROZONE_ISO3


def test_eu_aggregate_members_list_is_eurozone():
    eu = get_country("EU")
    assert eu.members == EUROZONE_ISO3
    assert not eu.eu_member
    assert eu.fx_regime == "reserve_issuer"


def test_static_flags():
    assert get_country("US").fx_regime == "reserve_issuer"
    assert get_country("SA").fx_regime == "peg"
    assert get_country("RU").sanctioned is True
    assert get_country("RU").data_quality == DataQuality.OPAQUE
    assert get_country("CN").data_quality == DataQuality.LOW
    assert get_country("SE").data_quality == DataQuality.HIGH
    assert all(not c.sanctioned for c in COUNTRIES if c.iso2 != "RU")


def test_fx_regime_values_are_known():
    allowed = {"float", "managed", "peg", "currency_union", "reserve_issuer"}
    for c in COUNTRIES:
        assert c.fx_regime in allowed, c.iso2


def test_derived_maps_match_registry():
    assert ISO2_TO_WB["EU"] == "EMU"
    assert ISO2_TO_WB["US"] == "USA"
    assert ISO2_TO_BIS["EU"] == "XM"
    assert ISO2_TO_BIS["UK"] == "GB"
    assert ISO2_TO_BIS["SA"] == "SA"
    assert ISO3_TO_ISO2["DEU"] == "DE"
    assert ISO3_TO_ISO2["EMU"] == "EU"
    assert len(ISO2_TO_WB) == 22
    assert ISO2_TO_IMTS["EU"] == "G163" and ISO2_TO_IMTS["KR"] == "KOR"
    assert all(c.imts_id is None for c in COUNTRIES if c.iso2 != "EU")


def test_get_country_case_insensitive():
    assert get_country("us").iso2 == "US"
    assert get_country("US").iso2 == "US"


def test_get_country_unknown_raises():
    with pytest.raises(KeyError):
        get_country("XX")


def test_get_country_by_iso3():
    assert get_country_by_iso3("KOR").iso2 == "KR"
    assert get_country_by_iso3("emu").iso2 == "EU"
    with pytest.raises(KeyError):
        get_country_by_iso3("ZZZ")


def test_each_country_has_central_bank_and_currency():
    for c in COUNTRIES:
        assert c.central_bank, f"{c.iso2} missing central_bank"
        assert c.currency, f"{c.iso2} missing currency"


def test_oecd_codes_are_iso3_except_euro_area():
    from dalio.countries import ISO2_TO_OECD
    assert ISO2_TO_OECD["EU"] == "EA"
    assert ISO2_TO_OECD["UK"] == "GBR" and ISO2_TO_OECD["US"] == "USA"
    assert all(c.oecd_id is None for c in COUNTRIES if c.iso2 != "EU")
