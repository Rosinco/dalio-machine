# Project Context

## Overview

dalio-machine is a macro-cycle dashboard built on Ray Dalio's economic machine framework. For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), it pulls macro indicators from FRED, BIS, IMF, OECD, World Bank, WID, and SIPRI, then classifies cycle stage with rule-based logic. Output is a Streamlit dashboard for **asset-allocation decision support** — informing diversified-portfolio tilts based on regime indicators, *not* market-timing signals.

The framework is treated as a descriptive lens, not a predictive oracle. Per Dalio's own writing and academic evidence on macro-overlay strategies, this dashboard surfaces regime state with explicit confidence — it does not output buy/sell signals.

## Tech Stack

- **Runtime:** Python 3.12
- **Framework:** Streamlit (dashboard), SQLAlchemy 2.x (storage), pandas (compute)
- **Test Framework:** pytest with mocked HTTP (pytest-mock + responses)
- **Key Dependencies:** fredapi, requests, pandas, sqlalchemy, streamlit, plotly, python-dotenv

## Architecture

- **Entry Points:**
  - `dalio-fetch-fred` (CLI ETL) — `src/dalio/pipelines/fetch_fred.py:main`
  - `dalio-app` (Streamlit dashboard) — `src/dalio/app/streamlit_app.py:main`
- **Module Structure:**
  - `src/dalio/countries.py` — 8-country tiered registry
  - `src/dalio/data_sources/` — one adapter per provider (FRED first; BIS, IMF, OECD, WB, WID, SIPRI later)
  - `src/dalio/indicators/` — cycle-layer computation (short-term, long-term, big-cycle)
  - `src/dalio/scoring/` — stage classification + asset-allocation map
  - `src/dalio/storage/` — SQLite schema + session helpers
  - `src/dalio/pipelines/` — ETL orchestration (one pipeline per data source)
  - `src/dalio/app/` — Streamlit dashboard
- **Data Flow:** API (FRED/BIS/IMF/...) → DataSource adapter → long-format DataFrame → SQLite via SQLAlchemy → indicator compute → stage classifier → Streamlit chart

## Conventions

- **Naming:** snake_case for files/functions/vars; PascalCase for classes; UPPER_CASE for module-level constants. Indicator names are snake_case strings (`cpi_yoy`, `policy_rate`).
- **File Organization:** Tests mirror src tree (`tests/test_<module>.py`). One module per data source / indicator family.
- **Error Handling:** Pipelines log + collect failures per series; never let one bad series fail the whole batch. Tests use mocked HTTP — never real API calls.
- **Imports:** Absolute imports from `dalio.*`. No relative imports.
- **Time-series invariants:** All observations stored long-format `(country, indicator, date, value, source, series_id)`. UTC-naive `date` type, not `datetime`. No duplicate `(country, indicator, date, source)` rows (enforced by unique constraint).

## Country Basket

| Code | Country | Tier | FRED ID | BIS ID | Currency |
|------|---------|------|---------|--------|----------|
| US | United States | 1 | USA | US | USD |
| CN | China | 1 | CHN | CN | CNY |
| EU | Eurozone (DE/FR/IT) | 1 | EMU | XM | EUR |
| UK | United Kingdom | 1 | GBR | GB | GBP |
| JP | Japan | 1 | JPN | JP | JPY |
| SE | Sweden | 1 | SWE | SE | SEK |
| IN | India | 2 | IND | IN | INR |
| BR | Brazil | 2 | BRA | BR | BRL |

Tier drives dashboard confidence labels — Tier 2 readings are flagged as "data thinner" in the UI.

## Indicator Catalogue

### Short-term debt cycle (slice 1)

| Indicator | Definition | Primary source | Notes |
|-----------|-----------|----------------|-------|
| `policy_rate` | Effective central bank policy rate (%) | FRED `DFF` (US) | Per-country mapping needed for fan-out |
| `cpi_yoy` | CPI year-over-year change (%) | FRED `CPIAUCSL` (US) | YoY transform applied |
| `unemployment_rate` | Headline unemployment (%) | FRED `UNRATE` (US) | |
| `yield_10y` | 10-year sovereign bond yield (%) | FRED `DGS10` (US) | |
| `yield_2y` | 2-year sovereign bond yield (%) | FRED `DGS2` (US) | |
| `real_gdp_yoy` | Real GDP year-over-year change (%) | FRED `GDPC1` (US) | YoY transform applied |

### Long-term debt cycle (slice 4)

| Indicator | Definition | Primary source |
|-----------|-----------|----------------|
| `total_credit_pct_gdp` | Total non-financial credit / GDP | BIS Total Credit |
| `gov_debt_pct_gdp` | Government debt / GDP | BIS / IMF |
| `hh_debt_pct_gdp` | Household debt / GDP | BIS Total Credit |
| `corp_debt_pct_gdp` | Non-financial corporate debt / GDP | BIS Total Credit |
| `debt_service_ratio` | Private non-financial DSR | BIS DSR |
| `real_rate_10y` | 10Y nominal yield − core CPI YoY | computed |
| `credit_impulse` | Δ(total credit) ÷ GDP | computed from BIS |

### Big cycle (slice 5)

| Measure | Source |
|---------|--------|
| Education | OECD PISA, World Bank tertiary attainment |
| Innovation & Tech | WIPO patents, OECD MSTI |
| Competitiveness | WEF GCI / IMD WCY |
| Economic output | IMF WEO, World Bank GDP |
| Share of world trade | UN COMTRADE / WTO |
| Military strength | SIPRI Mil Expenditure DB |
| Financial center | GFCI index (manual or scrape) |
| Reserve currency | IMF COFER |

### Currency lifecycle (slice 6)

| Indicator | Source |
|-----------|--------|
| Reserve currency share | IMF COFER (USD/EUR/JPY/GBP/CNY) |
| FX reserve trend | IMF, BIS |
| Currency vs trade-weighted index | BIS effective exchange rates |

### Wealth & values gaps (slice 6)

| Indicator | Source |
|-----------|--------|
| Top 1% income share | WID.world |
| Gini coefficient | World Bank WDI |
| Populist vote share | manual / V-Dem |

## Cycle Stage Classification

Rule-based, transparent, four stages per Dalio's roadmap (slice 1 — short-term cycle):

| Stage | Conditions (US short-term cycle) |
|-------|----------------------------------|
| 1. Expansion | `real_gdp_yoy > 0`, unemployment falling, `cpi_yoy < 3` |
| 2. Inflationary peak | `cpi_yoy > 3` and rising, central bank tightening, output gap closed |
| 3. Recession | `real_gdp_yoy < 0` OR unemployment rising sharply, credit contracting |
| 4. Reflation | Central bank cutting, `cpi_yoy` moderating, recovery emerging |

Output is the most-fitting stage with confidence (rule match count). Edge cases → "transition" label, never a forced binary.

## Current State

- **Working:** Project scaffold, country registry, SQLite schema, FRED data source, US short-term ETL, basic test suite, minimal Streamlit smoke-test app.
- **In Progress:** Slice 1 (US short-term cycle dashboard).
- **Known Issues:** None yet.

## Roadmap

| Slice | Goal |
|-------|------|
| **1** ✓ in progress | US short-term cycle: ETL + dashboard, end-to-end |
| 2 | Tier-1 fan-out: CN, EU, UK, JP, SE for short-term cycle |
| 3 | Tier-2: IN, BR for short-term cycle |
| 4 | Long-term debt cycle (BIS Total Credit, DSR) for all 8 |
| 5 | Big-cycle power index (8 measures, multi-country) |
| 6 | Currency lifecycle + wealth/values gaps |
| 7 | Allocation-implication module (Dalio All Weather mapping) |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-27 | Frame project as decision-support for allocation tilts, not market-timing | Empirical evidence: macro-overlay strategies underperform passive after costs. Dalio's own All Weather doesn't time. |
| 2026-04-27 | 8-country basket with tiering | Tier 1: full Dalio framework relevance + complete data. Tier 2: major EM with strong but slightly thinner coverage. Drives UI confidence labels. |
| 2026-04-27 | SQLite + SQLAlchemy + long-format observations table | Simple, file-based, easy to reason about. Long format makes adding indicators trivial. |
| 2026-04-27 | Slice-by-slice (US short-term first) | Tracer-bullet vertical slice. Build end-to-end before fanning out. |
| 2026-04-27 | Rule-based stage classification (not ML) | Transparency over sophistication. User must be able to inspect why a stage was assigned. |
| 2026-04-27 | Drop Vietnam from basket | Data coverage gaps in BIS Total Credit, OECD productivity, WID inequality made Tier-3 stretch. Outside Dalio framework's reserve-currency-scale focus. |

## Recent Changes

| Date | Change | Files |
|------|--------|-------|
| 2026-04-27 | Initial scaffold | `pyproject.toml`, `src/dalio/{countries,storage/db,data_sources/fred,pipelines/fetch_fred,app/streamlit_app}.py`, `tests/` |
