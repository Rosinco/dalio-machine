# CLAUDE.md — dalio-machine

Project-specific guidance. See `project_context.md` for full architecture and indicator catalogue.

## Workflow

- **Tests first.** Tests use mocked HTTP — never real API calls. Real FRED/BIS/IMF calls only happen in actual ETL runs.
- **Pre-commit:** `pytest` must pass; `ruff check src tests` should be clean.
- **Slice discipline.** Built slice-by-slice (see roadmap in `project_context.md`). Don't broaden scope mid-slice — finish slice N end-to-end before slice N+1.
- **Activate venv:** `source .venv/bin/activate` before any Python work in this repo.

## Company analysis

User direction (2026-09-13, Macro Atlas 0.19; ADR 0042): expand both the KPI catalogue
and list features using already downloaded KPI data. Import through the source-bound,
hash-verified exporter. Preserve exact provider variant IDs, corrected units and
currency boundaries; snapshot dates are not invented report/quote dates. Exclude
known NCAV and split defects. List tools remain descriptive, preserve draft bytes,
and never silently reprice the frozen starter. Migrate v1 list preferences in memory
and write v2 only on explicit user actions; retain old settings for the old app.

User direction (2026-09-13, Macro Atlas 0.18; ADR 0041): Companies → Lists provides
customizable KPI columns, saved views, a personal watchlist and explicit candidate
presets. Use the verified research artifact and preserve actual fiscal periods,
units, dates and missing reasons. Different-currency amounts are not comparable
rankings. Candidate matches are descriptive research leads; personal stars are
separate. List preferences have their own storage key and must never write or
migrate valuation drafts. Do not label annual metrics current/R12 or invent EPS,
P/E or other fields before a separately verified import. Keep unsupported saved
preferences intact until a user action and retain unknown watchlist IDs.

User direction (2026-09-13, Macro Atlas 0.17; ADR 0040): the universe research
screen is read-only descriptive candidate exploration. Keep historical evidence,
coverage, business routes and optional dated starter valuation separate. No
opaque score, automatic buy verdict or formal deep-dive gate follows. Bind the
screen to exact financial/taxonomy/company source identities; mismatches withhold
results. Browsing the screen/card must never mount the valuation workspace or
write/migrate user drafts. Additional raw prices, R12 and KPIs need separate
verified imports; do not silently substitute newer quotes or share bases.

User direction (2026-09-13, Macro Atlas 0.16; ADR 0039): every company has an
editable purchase-price view, defaulting to 30% below Mid equity value. DCF
includes terminal once; NPV is value less proposed price. Keep signed values,
missing inputs, dated prices and the ownership claim explicit. No positive
ceiling for nonpositive value or 100% margin; no artificial minimum purchase
price. Preserve authored purchase settings through history and baseline/reviewed
resets. Per-share values need an explicitly applied dated share basis; generic
reported shares remain unreviewed proxies. No probability or buy verdict follows.

User direction (2026-09-13, Macro Atlas 0.15; ADR 0038): separate sustainable
terminal cash from annual cash uncertainty. Keep the v3 annual benchmark and
calibration. New generic origins add `terminalMethod: historical-median-v1`;
signed historical median and an assumed ±20% seed remain unreviewed until a deep
dive reconciles reinvestment/financing. Resolve first post-horizon cash at r−g;
keep explicit sale as an alternative and crisis sale separate. Preserve edited,
cleared/customized/restored/reviewed studies. Only exact untouched old defaults
upgrade after backup and replacement persistence. Show same-report cash components
and missing definitions without inventing maintenance capex or owner cash.

User direction (2026-09-13, Macro Atlas 0.14): keep the same editable cash-flow,
DCF and NPV workspace for every company. New `empirical-cash-starter-v3` defaults
to five annual periods and latest signed cash held flat. Eligible comparable
operating/property histories use source-bound, model-specific historical-error
ranges for Years 1-4, with an 80% research target and cash-dispersion groups. Keep
sector/branch context; the target is not a company or DCF probability. Years 5-10
carry the Year 4 absolute half-width forward and add an editable 10% of historical
cash scale per later year as an explicit assumption. Weighted trend and weighted
flat mean remain available; unsupported history/model/source/currency settings
use labelled percentage sensitivities. Zero cash does not imply zero uncertainty.
Other financial businesses need manually reviewed equity-cash and capital inputs.

Retain raw COVID and recovery histories and the original all-year calibration.
A separate optional crisis scenario starts disabled: 40% cash reduction, start
Year 1, two shock years, three recovery years, zero extra annual cash cost, 10%
required return and zero final equity sale. It has no assigned probability and
does not overwrite low/mid/high or add liquidation recovery. Deep dives retain
raw facts and document adjustments separately. Only exact untouched v1/v2 defaults
may upgrade after a saved backup succeeds; preserve edited/cleared/customized,
crisis, explicitly restored and reviewed studies. Preserve calibration IDs,
source versions, assumed tails and revisions. See `docs/standard-company-valuation.md` and ADR 0037.

User direction (2026-09-11): use the value-versus-price framework whenever
analyzing companies for Macro Atlas. Read `docs/company-valuation-framework.md`
and start from `docs/company-analysis-template.md` (ADR 0035). Include continuing-
business scenarios, a separate recovery case, dated price, and ordinary/discounted
payback. Tie macro evidence to verified company exposures. Keep source facts,
analyst assumptions and calculated outcomes distinct; missing inputs stay missing.
For completed numerical deep dives, automatically load the reviewed source facts
and explicit scenarios into the Value workspace. Holmen is the first case (see
`docs/holmen-valuation-2026-09-11.md`). Preserve user edits and prior drafts; a
dossier folder alone is not a forecast.

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
