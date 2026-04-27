"""8-country basket for the Dalio framework dashboard.

Tier drives dashboard confidence labels. Tier 1 = full Dalio-framework relevance with
complete data coverage across BIS/IMF/WB/OECD. Tier 2 = major EM with strong but
slightly thinner coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class Tier(IntEnum):
    TIER_1 = 1
    TIER_2 = 2


@dataclass(frozen=True)
class Country:
    iso2: str
    iso3: str
    name: str
    tier: Tier
    fred_id: str | None = None
    bis_id: str | None = None
    central_bank: str | None = None
    currency: str | None = None


COUNTRIES: Final[tuple[Country, ...]] = (
    Country("US", "USA", "United States", Tier.TIER_1, fred_id="USA", bis_id="US",
            central_bank="Federal Reserve", currency="USD"),
    Country("CN", "CHN", "China", Tier.TIER_1, fred_id="CHN", bis_id="CN",
            central_bank="People's Bank of China", currency="CNY"),
    Country("EU", "EMU", "Eurozone", Tier.TIER_1, fred_id="EMU", bis_id="XM",
            central_bank="European Central Bank", currency="EUR"),
    Country("UK", "GBR", "United Kingdom", Tier.TIER_1, fred_id="GBR", bis_id="GB",
            central_bank="Bank of England", currency="GBP"),
    Country("JP", "JPN", "Japan", Tier.TIER_1, fred_id="JPN", bis_id="JP",
            central_bank="Bank of Japan", currency="JPY"),
    Country("SE", "SWE", "Sweden", Tier.TIER_1, fred_id="SWE", bis_id="SE",
            central_bank="Riksbank", currency="SEK"),
    Country("IN", "IND", "India", Tier.TIER_2, fred_id="IND", bis_id="IN",
            central_bank="Reserve Bank of India", currency="INR"),
    Country("BR", "BRA", "Brazil", Tier.TIER_2, fred_id="BRA", bis_id="BR",
            central_bank="Banco Central do Brasil", currency="BRL"),
)


def get_country(iso2: str) -> Country:
    code = iso2.upper()
    for c in COUNTRIES:
        if c.iso2 == code:
            return c
    raise KeyError(f"Unknown country: {iso2!r}")
