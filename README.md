# dalio-machine — Macro History & Risk Observatory

One auditable, point-in-time account of the world economy, growing from a macro-cycle dashboard built on Ray Dalio's economic-machine framework. It tracks cycles and fundamentals today and is expanding into institutional reports, sovereign debt, large-allocator positions, capital flows, commodity pressure and monetary liquidity for a SEK-based household investor. It is a decision-support tool, **not** a market-timing signal generator.

## What it does

For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), it pulls macro indicators from FRED, BIS, Riksbank, IMF, OECD, SCB and World Bank, then classifies cycle stage with rule-based logic. The World Fundamentals Map extends coverage to 22 players. Numeric pipelines preserve complete source snapshots in an append-only release ledger beside the compatible latest-value table, enabling honest “what was known then?” queries from the ledger cutover onward.

The raw evidence layer now also preserves facts that should not be flattened into one score:

- SCB Financial Accounts positions for Swedish central-government debt securities by holder sector, instrument and quarter (8,349 rows, 1996-Q1 through 2026-Q1).
- IMF BPM6 financial-account transactions: 25 separate asset-acquisition, liability-incurrence and net series for each of 21 individual economies. These are actual reported transactions, not changes inferred from positions.
- World Bank/IMF QPSD central-government debt anatomy: 12 percent-of-GDP measures covering maturity, instrument, currency, creditor residence and D1/D2A scope. The first full run stored 200 country-series histories and recorded 64 voluntary non-reports without filling them.
- AP2, AP3 and AP4 H1 2026 disclosures: 48 page-located allocator facts, with fund-reorganisation transfers kept separate from investment flows and rounded/non-additive publisher values retained as published.
- The complete World Bank monthly Pink Sheet history: 63,179 positive observations across 70 price series and 17 index series, from 1960-01 through 2026-08 (the source worksheets contain 71 price-tab columns plus 16 index-tab columns). Exact workbooks and generator/schema-versioned 87-series catalogues, including a quarantined-cell quality manifest, are content-addressed artifacts; refreshes publish as one atomic release group and 29 series are flagged for a family-balanced research panel, but none is scored yet.
- Ten pinned official-money histories: US M2 and Federal Reserve total assets; euro-area M3 and Eurosystem total assets; Swedish M3; monthly UK M4ex, its non-additive quarterly historical bridge and headline M4; and Japan M3 and broadly-defined liquidity `L`. The validated refresh contains 5,794 native monthly, quarterly and weekly observations. Validated response bodies and canonical missingness ledgers are content-addressed and release-bound, and inventory re-hashes them. Currency, units, adjustment, definition perimeter and research role remain explicit. No synthetic universal M5, FX-summed money level or additive liquidity headline is created.
- A separate 22-series liquidity frontier from first-party BIS and OFR sources: three quarterly offshore USD/EUR/JPY credit stocks, eleven monthly US MMF asset totals/components and eight daily rates/volumes for selected repo venues (20,676 observations in the validated refresh). Exact provider responses, semantic catalogues and OFR native-payload/missingness files are content-addressed; release manifests bind the response and per-series evidence hashes, and inventory rehashes those files while repeating cadence, daily coverage/gap, recency, completeness and catalogue guards. These aggregates show funding segments and conditions, not an end-to-end trace of cash between identified sectors, and they are deliberately non-additive and unscored.

A read-only interpretation layer now turns those money and liquidity releases into
five separate diagnostic families and writes traceable JSON plus Markdown briefs.
It does not create a global liquidity, risk, M5 or investment score and does not
claim that observed relative growth proves a deposit flow or any other causal
“money moved from A to B” path.

The IMF PIP/DIP bilateral-position ledger preserves portfolio assets and direct-investment stocks by reporter, counterpart, instrument, direction, accounting basis and annual/semiannual frequency. The first official pull landed 127,970 rows: 92,529 PIP rows across all 168 requested partitions and 35,441 DIP rows across 123 of 126 partitions. The three non-reports are Saudi Arabian outward direct-investment total, equity and debt; there were no fetch failures.

Institutional prose has its own evidence ledger. Ten official PDFs—the latest and immediately previous eligible issue in each of five allowlisted families—are archived and completely extracted into 852 physical pages: Riksbank Monetary Policy Reports, ECB/Eurosystem staff projections, Federal Reserve Monetary Policy Reports, IMF World Economic Outlooks and BIS Annual Economic Reports. There are currently **zero human-verified report claims**. The PDFs and pages are source material, not yet reviewed conclusions; automated summaries and report-driven risk scoring are not active.

A separate read-only review packet now places 20 tightly bounded, page-cited model drafts—four from each latest report—in front of a human reviewer. Every item is labelled `UNVERIFIED MODEL DRAFT`; this queue does not create database claims, scores, probabilities or portfolio guidance.

## Why

Reading Dalio's framework as a *lens* (not an oracle): the dashboard surfaces where each economy sits in the long-term debt cycle, four-stage short-term debt cycle, and big-cycle power framework. Use it to understand constraints and diversified-portfolio fragilities, not to time entry or exit.

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
dalio-fetch-bis
dalio-fetch-cycle  # includes official SCB CPI for Sweden under the cpi flow
dalio-fetch-riksbank  # Swedish policy rate, sovereign yields and SEK FX

# Existing database created before the point-in-time ledger:
# choose a conservative instant when its legacy rows were known
dalio-bootstrap-history --available-at 2026-09-01T00:00:00Z

# World Fundamentals Map (22 players, keyless World Bank API) → snapshot JSON
dalio-fetch-fundamentals
dalio-score

# Raw debt anatomy, holder positions, reported transactions and position stocks.
dalio-fetch-sovereign-debt
dalio-fetch-debt-holders
dalio-fetch-flows
dalio-fetch-positions  # IMF PIP/DIP position stocks; not flows

# Commodity prices/indices, official money, then the separate liquidity frontier.
dalio-fetch-commodities
dalio-fetch-money-liquidity
dalio-fetch-shadow-liquidity
# A shorter replacement is rejected by default. Use --allow-contraction only
# after independently verifying that the publisher deliberately removed history.

# Read-only, exact inventory of what is actually stored (`--json` for detail)
dalio-audit-observatory --db data/dalio.db

# Read-only liquidity diagnostics → JSON + Markdown in data/snapshots/
dalio-liquidity-brief --db data/dalio.db

# Read-only official-report review queue → JSON + Markdown in data/review/
dalio-report-review-packet --db data/dalio.db

# Local official PDFs are deliberately acquired separately. The input folders
# must contain the deterministic filenames recorded in data/reference/*.json.
python -m dalio.pipelines.ingest_ap_funds --artifact-dir /path/to/verified-ap-pdfs
python -m dalio.pipelines.ingest_reports --artifact-dir /path/to/verified-report-pdfs

# Run dashboard
dalio-app
# open http://localhost:8501
```

The report-ingestion command requires Poppler's `pdftotext`. Both PDF ingestion
commands verify every expected SHA-256 before opening a database transaction.
They do not download documents; the review-packet command only reads
already-ingested evidence.

From Windows Explorer, this checkout is available at `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine`; the local SQLite file is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\dalio.db`, and durable source evidence is under `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts`. The latest generated liquidity brief is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_latest.md`; the latest unverified report review is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_claims_latest.md`. See `data/README.md` before copying, deleting or rebuilding anything under `data/`.

Riksbank SWEA contributes eight daily Swedish series: policy rate; 2-, 5- and 10-year government yields; and SEK per USD, EUR, NOK and GBP. No key is required. Keyless runs use a safe 13-second request interval for the official 5-calls/minute limit; when `RIKSBANK_API_KEY` is set, the adapter sends it in `Ocp-Apim-Subscription-Key` automatically. The NOK feed starts on 2023-11-27 because older observations were quoted per 100 NOK and are not mixed into the current per-1-NOK series.

## Liquidity brief

`dalio-liquidity-brief` opens the database read-only and writes the fixed
`liquidity_latest.json` and `liquidity_latest.md` aliases plus hash-addressed
`liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.{json,md}` exports under
`data/snapshots/`; the prefix is the first 16 characters of the full snapshot
SHA-256 stored in the JSON. The JSON retains methodology/catalogue hashes,
coverage, unavailable states, input dates and release/artifact provenance; the
Markdown is the small-investor reading layer. Complete coverage is required
unless `--allow-partial` is requested explicitly.

The five families remain independent:

- broad-money annual log growth and its three-month acceleration for US M2,
  euro-area M3, Swedish M3, monthly UK M4ex and Japan `L`, with descriptive
  equal-country median/breadth only when all five share one common period, but
  no cross-currency sum;
- US/euro-area broad-money growth versus central-bank-assets growth;
- US MMF total growth versus M2, named asset shares and separately reported
  OFR-published counterparty/clearing-category ratios—FICC is a clearing
  category, and these ratios do not identify an end-to-end cash map;
- selected repo-venue premiums to EFFR, cross-venue fragmentation and separate
  activity context; and
- separate USD/EUR/JPY BIS offshore-credit annual growth and one-quarter
  acceleration.

`--known-at` (alias `--as-known-at`) controls which complete releases were
available; `--as-of` (alias `--through-date`) caps the economic dates used.
`earliest_input_available_at` is the earliest selected non-policy liquidity
release, while `complete_snapshot_available_at` is populated with the latest
selected liquidity-release clock only when every pinned analysis partition,
including EFFR, is present. In the live initial refresh these are respectively
`2026-09-08T20:38:47.888222+00:00` and
`2026-09-08T20:43:28.624056+00:00`; the latter is the first full-snapshot
cutover. Earlier dates shown from those releases are therefore
`current_vintage_history`, not a known-at-the-time backtest. These diagnostics
provide 0–12-month market-plumbing and 1–3-year monetary backdrop; they do not
by themselves assess 3–5-year risks, prove deposit-to-MMF migration or monetary
causality, trace end-to-end money flows, forecast returns or give portfolio
instructions. See [ADR 0009](decisions/0009-derived-liquidity-diagnostics.md)
for the formulas and boundaries.

## Report evidence

The report layer is deliberately review-first:

`official PDF -> content-addressed archive -> complete physical-page extraction -> atomic draft + exact excerpt -> named human review -> point-in-time claim query`

Publisher statements can be reconstructed on the date the source document became public, while their later retrieval and review clocks remain visible. An Observatory inference is different: it needs supporting evidence from at least two independent publishers and never appears before human review. No model can approve its own draft.

The same point-in-time rule applies to numeric and typed data: first choose the newest complete release that was available at the requested UTC instant, then filter its rows. Filtering first could silently resurrect a holder, counterpart or series cell omitted in a newer vintage. Where an official source exposes no trustworthy publication timestamp, `available_at` is conservatively the retrieval time—not the observation date.

### Report-claim review packet

The read-only `dalio-report-review-packet` command combines the checked, versioned
`data/reference/report_claim_candidates.json` catalogue with the latest eligible
issue in each of the five report families, then requires its declared extraction
to be complete: Riksbank MPR, ECB/Eurosystem projections, Federal Reserve MPR,
IMF WEO and BIS Annual Economic Report. The initial packet includes four
candidates per selected issue
(twenty total), and the contract admits no more. Catalogue checks prove identity,
hashes, page locators and exact excerpts—not semantic correctness.

Every candidate in both output formats is labelled exactly
`UNVERIFIED MODEL DRAFT`. Generated files are fixed
`data/review/report_claims_latest.{json,md}` aliases plus content-addressed
`report_claims_YYYY-MM-DD_<first-16-packet-sha256>.{json,md}` copies. Neither a
catalogue entry nor a review packet is a verified conclusion or may feed scores,
scenarios or portfolio guidance.

The packet builder does not write to the database. Approval remains a separate,
human-only gate: a named human may `approve`, `revise` or `reject`; revision
preserves the original draft, rejection creates no verified claim, and no model
may choose a decision or supply a reviewer identity. See
[ADR 0010](decisions/0010-report-claim-review-queue.md) for the implemented
review-queue contract and its deliberately separate approval boundary.

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

**World Fundamentals Map (slice 18+)** adds 14 Tier-3 players — DE, FR, IT, ES, NL, CA, RU, KR, AU, MX, ID, SA, TR, CH — scored by percentile on fundamentals in five categories (real stuff · production · exchange · promises · enforcer). Tier 3 = fundamentals only, no cycle classifiers. See `decisions/0001-fundamentals-map.md`.

## Status

Pre-alpha. The cycle and fundamentals product is working, and the raw-history foundation now includes sovereign-debt anatomy, Swedish debt holders, IMF financial-account transactions, 127,970 bilateral investment-position rows, three Swedish AP-fund disclosures, a ten-document official-report corpus, 63,179 monthly commodity observations, ten official-money histories and 22 separate shadow-liquidity histories. Read-only liquidity diagnostics and a 20-item report review queue are available, but this is not a complete global money-flow map, a universal M5, an additive liquidity total, a causal or deposit-flow model, or an investable commodity return history: all 20 report candidates still need named human review; allocator history has only one H1 2026 release per fund; QPSD and IMF position coverage are voluntary and uneven; and debt cash-flow schedules, broader banking/funding channels and horizon risk scenarios remain to be built. See `project_context.md`, ADRs 0004–0010 and `data/README.md` for current boundaries.
