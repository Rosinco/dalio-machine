# CLAUDE.md — dalio-machine

Project-specific guidance. See `project_context.md` for full architecture and indicator catalogue.

## Workflow

- **Tests first.** Tests use mocked HTTP — never real API calls. Real FRED/BIS/IMF calls only happen in actual ETL runs.
- **Pre-commit:** `pytest` must pass; `ruff check src tests` should be clean.
- **Slice discipline.** Built slice-by-slice (see roadmap in `project_context.md`). Don't broaden scope mid-slice — finish slice N end-to-end before slice N+1.
- **Activate venv:** `source .venv/bin/activate` before any Python work in this repo.

## Stack reminders

- Python 3.12, venv at `.venv/`, install via `pip install -e ".[dev]"`
- SQLAlchemy 2.x style (`select(...)`, not legacy `Query`)
- pandas long-format DataFrames everywhere — wide format only at presentation layer

## Indicator naming

When adding a new indicator: snake_case, semantically meaningful (`cpi_yoy`, not `cpi_change`). Document in `project_context.md` indicator catalogue at the same time as adding the code — don't let docs lag behind code.

## Data source priority

User direction (2026-09-10): use first-hand sources whenever available. For
country-specific collection, prefer the original statistical office, central
bank or debt office. Retain original response bytes, publication/reference and
retrieval clocks, units and definitions. An official distributor is not
necessarily the original producer: keep its underlying source attribution.

Use official harmonized BIS/IMF/World Bank/OECD datasets for a comparable
cross-country baseline, clearly labelled as such. Acquire national originals
alongside that baseline as the collection deepens; never silently substitute
different scopes or rewrite the scoring source order merely because collection
coverage expanded.

The existing cross-country scoring preferences are:

1. **BIS** — for credit/debt series (Total Credit dataset is canonical for cross-country comparability)
2. **IMF** — for cross-country macro
3. **FRED** — for US-native series + their international subset (often slightly delayed)
4. **OECD** — for productivity, R&D, education
5. **WID** — for inequality
6. **National central banks** — native country measures (e.g. Riksbank Swedish-specific series)

These scoring preferences concern comparable definitions for cycle
classification; they do not override the user's preference for original
publishers when collecting national evidence.

## Honest output

Project outputs feed real allocation decisions. Never present a stage classification with more confidence than the data warrants:

- Tier 2 (IN, BR) readings get explicit "data thinner" labels in the UI
- "Transition" / "ambiguous" is a valid classification — don't force binary
- Confidence score derived from rule-match count, displayed alongside stage

## Don't

- Don't add ML-based classifiers in slice 1–4 — rule-based first, transparent always.
- Don't fetch real APIs in tests.
- Don't store wide-format DataFrames in the database.
- Don't import from `dalio` using relative imports — absolute only.
