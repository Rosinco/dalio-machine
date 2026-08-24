"""Country registry — the ONE place to add or describe a player.

Two nested baskets live here:

* **Cycle basket** (Tier 1 + Tier 2, 8 economies): the original Dalio-machine
  countries with FRED/BIS wiring for the short-/long-term cycle classifiers.
  `CYCLE_COUNTRIES` is what the cycle map and classifiers iterate.
* **Fundamentals basket** (all tiers, 22 rows = 21 individual countries + the
  euro-area aggregate): the World Fundamentals Map players. Tier 3 = fundamentals
  only, no cycle classifiers. `RANKING_POPULATION` is the 21 individual
  countries percentile ranks are computed over — the euro-area aggregate is
  excluded so its members are not double-counted.

Tier drives dashboard confidence labels. Tier 1 = full Dalio-framework relevance
with complete BIS/IMF/WB coverage. Tier 2 = major EM with thinner coverage.
Tier 3 = fundamentals-only player (slice 18+).

Static per-country flags (`fx_regime`, `sanctioned`, `data_quality`) are
hand-entered, dated judgments (tier C). They feed the pressure-chain rules and
the data-quality markers on the map; they never enter a percentile.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final


class Tier(IntEnum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3   # fundamentals-only: no cycle classifiers


class DataQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPAQUE = "opaque"


# Euro-area members as ISO-3 — used both to expand the EU aggregate on the map
# and as the aggregate's `members` list. (20 members incl. Croatia, 2023.)
EUROZONE_ISO3: Final[tuple[str, ...]] = (
    "AUT", "BEL", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA",
    "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP", "HRV",
)


@dataclass(frozen=True)
class Country:
    iso2: str
    iso3: str
    name: str
    tier: Tier
    fred_id: str | None = None
    bis_id: str | None = None
    wb_id: str | None = None          # World Bank code: ISO3 or "EMU"
    imf_id: str | None = None         # IMF DataMapper code (ISO3); None until verified
    central_bank: str | None = None
    currency: str | None = None
    eu_member: bool = False           # individual player that is also inside the EU aggregate
    members: tuple[str, ...] = ()     # ISO3 members — non-empty only for aggregates
    fx_regime: str = "float"          # float | managed | peg | currency_union | reserve_issuer
    sanctioned: bool = False
    data_quality: DataQuality = DataQuality.HIGH
    data_quality_note: str | None = None
    on_map: bool = True               # False for aggregates (drawn via `members`)

    @property
    def has_cycle_wiring(self) -> bool:
        """True for the original 8-country basket with FRED/BIS series."""
        return self.fred_id is not None


COUNTRIES: Final[tuple[Country, ...]] = (
    # ─── Cycle basket (unchanged fields; order matters for existing tests) ───
    Country("US", "USA", "United States", Tier.TIER_1, fred_id="USA", bis_id="US",
            wb_id="USA", imf_id="USA", central_bank="Federal Reserve", currency="USD",
            fx_regime="reserve_issuer"),
    Country("CN", "CHN", "China", Tier.TIER_1, fred_id="CHN", bis_id="CN",
            wb_id="CHN", imf_id="CHN", central_bank="People's Bank of China", currency="CNY",
            fx_regime="managed", data_quality=DataQuality.LOW,
            data_quality_note="GDP series smoothed; provincial data revised; opacity on credit."),
    Country("EU", "EMU", "Eurozone", Tier.TIER_1, fred_id="EMU", bis_id="XM",
            wb_id="EMU", imf_id=None, central_bank="European Central Bank", currency="EUR",
            members=EUROZONE_ISO3, fx_regime="reserve_issuer", on_map=False),
    Country("UK", "GBR", "United Kingdom", Tier.TIER_1, fred_id="GBR", bis_id="GB",
            wb_id="GBR", imf_id="GBR", central_bank="Bank of England", currency="GBP"),
    Country("JP", "JPN", "Japan", Tier.TIER_1, fred_id="JPN", bis_id="JP",
            wb_id="JPN", imf_id="JPN", central_bank="Bank of Japan", currency="JPY"),
    Country("SE", "SWE", "Sweden", Tier.TIER_1, fred_id="SWE", bis_id="SE",
            wb_id="SWE", imf_id="SWE", central_bank="Riksbank", currency="SEK"),
    Country("IN", "IND", "India", Tier.TIER_2, fred_id="IND", bis_id="IN",
            wb_id="IND", imf_id="IND", central_bank="Reserve Bank of India", currency="INR",
            fx_regime="managed", data_quality=DataQuality.MEDIUM,
            data_quality_note="GDP deflator methodology contested; frequent revisions."),
    Country("BR", "BRA", "Brazil", Tier.TIER_2, fred_id="BRA", bis_id="BR",
            wb_id="BRA", imf_id="BRA", central_bank="Banco Central do Brasil", currency="BRL",
            data_quality=DataQuality.MEDIUM),
    # ─── Fundamentals-only players (Tier 3) ───
    Country("DE", "DEU", "Germany", Tier.TIER_3, bis_id="DE", wb_id="DEU", imf_id="DEU",
            central_bank="Deutsche Bundesbank", currency="EUR",
            eu_member=True, fx_regime="currency_union"),
    Country("FR", "FRA", "France", Tier.TIER_3, bis_id="FR", wb_id="FRA", imf_id="FRA",
            central_bank="Banque de France", currency="EUR",
            eu_member=True, fx_regime="currency_union"),
    Country("IT", "ITA", "Italy", Tier.TIER_3, bis_id="IT", wb_id="ITA", imf_id="ITA",
            central_bank="Banca d'Italia", currency="EUR",
            eu_member=True, fx_regime="currency_union"),
    Country("ES", "ESP", "Spain", Tier.TIER_3, bis_id="ES", wb_id="ESP", imf_id="ESP",
            central_bank="Banco de España", currency="EUR",
            eu_member=True, fx_regime="currency_union"),
    Country("NL", "NLD", "Netherlands", Tier.TIER_3, bis_id="NL", wb_id="NLD", imf_id="NLD",
            central_bank="De Nederlandsche Bank", currency="EUR",
            eu_member=True, fx_regime="currency_union"),
    Country("CA", "CAN", "Canada", Tier.TIER_3, bis_id="CA", wb_id="CAN", imf_id="CAN",
            central_bank="Bank of Canada", currency="CAD"),
    Country("RU", "RUS", "Russia", Tier.TIER_3, bis_id="RU", wb_id="RUS", imf_id="RUS",
            central_bank="Bank of Russia", currency="RUB",
            fx_regime="managed", sanctioned=True, data_quality=DataQuality.OPAQUE,
            data_quality_note="Statistics withheld since 2022; BIS reporting suspended; "
                              "World Bank series null post-2022."),
    Country("KR", "KOR", "South Korea", Tier.TIER_3, bis_id="KR", wb_id="KOR", imf_id="KOR",
            central_bank="Bank of Korea", currency="KRW"),
    Country("AU", "AUS", "Australia", Tier.TIER_3, bis_id="AU", wb_id="AUS", imf_id="AUS",
            central_bank="Reserve Bank of Australia", currency="AUD"),
    Country("MX", "MEX", "Mexico", Tier.TIER_3, bis_id="MX", wb_id="MEX", imf_id="MEX",
            central_bank="Banco de México", currency="MXN",
            data_quality=DataQuality.MEDIUM),
    Country("ID", "IDN", "Indonesia", Tier.TIER_3, bis_id="ID", wb_id="IDN", imf_id="IDN",
            central_bank="Bank Indonesia", currency="IDR",
            fx_regime="managed", data_quality=DataQuality.MEDIUM),
    Country("SA", "SAU", "Saudi Arabia", Tier.TIER_3, bis_id="SA", wb_id="SAU", imf_id="SAU",
            central_bank="Saudi Central Bank (SAMA)", currency="SAR",
            fx_regime="peg", data_quality=DataQuality.LOW,
            data_quality_note="Fiscal and reserve data partial; no BIS DSR."),
    Country("TR", "TUR", "Türkiye", Tier.TIER_3, bis_id="TR", wb_id="TUR", imf_id="TUR",
            central_bank="Central Bank of the Republic of Türkiye", currency="TRY",
            fx_regime="managed", data_quality=DataQuality.LOW,
            data_quality_note="Official inflation contested (ENAG vs TÜİK); political pressure on CBRT."),
    Country("CH", "CHE", "Switzerland", Tier.TIER_3, bis_id="CH", wb_id="CHE", imf_id="CHE",
            central_bank="Swiss National Bank", currency="CHF"),
)

CYCLE_COUNTRIES: Final[tuple[Country, ...]] = tuple(c for c in COUNTRIES if c.has_cycle_wiring)

# The 21 individual countries percentile ranks are computed over (no aggregates).
RANKING_POPULATION: Final[tuple[str, ...]] = tuple(c.iso2 for c in COUNTRIES if c.on_map)

# Derived lookup maps — consumers import these instead of keeping their own copies.
ISO2_TO_WB: Final[dict[str, str]] = {c.iso2: c.wb_id for c in COUNTRIES if c.wb_id}
ISO2_TO_BIS: Final[dict[str, str]] = {c.iso2: c.bis_id for c in COUNTRIES if c.bis_id}
ISO3_TO_ISO2: Final[dict[str, str]] = {c.iso3: c.iso2 for c in COUNTRIES}
_BY_ISO2: Final[dict[str, Country]] = {c.iso2: c for c in COUNTRIES}


def get_country(iso2: str) -> Country:
    try:
        return _BY_ISO2[iso2.upper()]
    except KeyError as e:
        raise KeyError(f"Unknown country: {iso2!r}") from e


def get_country_by_iso3(iso3: str) -> Country:
    code = iso3.upper()
    try:
        return _BY_ISO2[ISO3_TO_ISO2[code]]
    except KeyError as e:
        raise KeyError(f"Unknown country iso3: {iso3!r}") from e
