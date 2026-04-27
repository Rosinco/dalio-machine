# dalio-machine

Macro-cycle dashboard built on Ray Dalio's economic machine framework — tracks short-term debt cycle, long-term debt cycle, and big-cycle indicators across major economies. Decision-support tool for asset allocation tilts, **not** a market-timing signal generator.

## What it does

For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), pulls macro indicators from FRED, BIS, IMF, OECD, World Bank, WID, and SIPRI, then classifies cycle stage with rule-based logic. Output: a Streamlit dashboard showing current state per indicator family per country, with explicit confidence labels.

## Why

Reading Dalio's framework as a *lens* (not an oracle): the dashboard surfaces where each economy sits on Dalio's eight-stage long-term debt cycle, four-stage short-term debt cycle, and big-cycle power index. Use it to inform diversified-portfolio tilts (more bonds/gold/cash when long-cycle indicators stretch), not to time entry/exit.

See `project_context.md` for full architecture and indicator definitions.

## Quickstart

```bash
# Clone, then:
cd dalio-machine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Add your FRED API key (free, register at https://fred.stlouisfed.org/docs/api/api_key.html)
cp .env.example .env
# edit .env → FRED_API_KEY=your_key_here

# Initialize SQLite schema and pull first US data
dalio-fetch-fred

# Run dashboard
dalio-app
# open http://localhost:8501
```

## Tests

```bash
pytest
```

Tests use mocked HTTP — no real API calls in CI.

## Country basket

| Code | Country | Tier | Coverage |
|------|---------|------|----------|
| US | United States | 1 | Full |
| CN | China | 1 | Full (some series with delay) |
| EU | Eurozone (DE/FR/IT) | 1 | Full |
| UK | United Kingdom | 1 | Full |
| JP | Japan | 1 | Full |
| SE | Sweden | 1 | Full |
| IN | India | 2 | Strong, some BIS series shorter |
| BR | Brazil | 2 | Strong |

Tier drives dashboard confidence labels — Tier 2 readings are flagged as such.

## Status

Pre-alpha. See `project_context.md` for current state and roadmap.
