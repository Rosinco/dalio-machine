import pytest

from dalio.countries import COUNTRIES, Tier, get_country


def test_basket_size():
    assert len(COUNTRIES) == 8


def test_iso2_codes_unique():
    codes = [c.iso2 for c in COUNTRIES]
    assert len(set(codes)) == len(codes)


def test_iso3_codes_unique():
    codes = [c.iso3 for c in COUNTRIES]
    assert len(set(codes)) == len(codes)


def test_tier_distribution():
    tiers = [c.tier for c in COUNTRIES]
    assert tiers.count(Tier.TIER_1) == 6
    assert tiers.count(Tier.TIER_2) == 2


def test_required_codes_present():
    expected = {"US", "CN", "EU", "UK", "JP", "SE", "IN", "BR"}
    assert {c.iso2 for c in COUNTRIES} == expected


def test_get_country_case_insensitive():
    assert get_country("us").iso2 == "US"
    assert get_country("US").iso2 == "US"


def test_get_country_unknown_raises():
    with pytest.raises(KeyError):
        get_country("XX")


def test_each_country_has_central_bank_and_currency():
    for c in COUNTRIES:
        assert c.central_bank, f"{c.iso2} missing central_bank"
        assert c.currency, f"{c.iso2} missing currency"
