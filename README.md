# dalio-machine — Macro History & Risk Observatory

One auditable, point-in-time account of the world economy, growing from a macro-cycle dashboard built on Ray Dalio's economic-machine framework. It tracks cycles and fundamentals today and is expanding into institutional reports, sovereign debt, large-allocator positions, capital flows, commodity pressure and monetary liquidity for a SEK-based household investor. It is a decision-support tool, **not** a market-timing signal generator.

## Country assessments

The country-assessment consumer turns verified local evidence into readable
profiles for the 19 saved listing countries: a prior-year IMF estimate/outturn,
the retained current-year plus five-year publisher path, historical structural
context and explicitly conditional scenarios. Each case states assumptions,
transmission mechanisms, signposts, invalidators and company-exposure checks.
Available original Swedish debt-office facts retain their own scope and units.

```bash
python -m dalio.pipelines.build_country_assessments --db data/dalio.db --as-of 2026-09-10
```

Outputs are offline JSON and Markdown under
`data/snapshots/country_assessments/<snapshot-sha256>/`; `LATEST.json` points to
the complete published directory. The source database is read-only. Publisher
forecasts, descriptive arithmetic and Observatory hypotheses remain distinct;
there are no calibrated probabilities, composite risk scores or company
verdicts. These files are not yet bundled into Macro Atlas. See ADR 0030 and
`HANDOFF.md` for the current checkpoint.

## What it does

The **Sweden monitoring pilot** connects the country scenarios to five original
SCB/Riksbank histories: industrial production, orders, business lending rates,
the policy rate and the ten-year government benchmark yield. It adds dated
comparisons, source definitions, freshness and evidence that challenges each
scenario, alongside the existing original debt-office context. Missing data,
forecasts and observed outcomes remain separate.

```bash
python -m dalio.pipelines.fetch_sweden_monitoring --artifact-root data/artifacts/sweden_monitoring
python -m dalio.pipelines.build_sweden_monitoring --db data/dalio.db
```

The collector retains complete original-response supplements; it does not
write to the source database. The offline consumer writes `SE.md` and
`snapshot.json` under `data/snapshots/sweden_monitoring/<hash>/`. Use `--known-at`
for exact historical selection and `--no-latest` to review an export before
publishing its pointer. See ADR 0031 and `HANDOFF.md` for the verified checkpoint.

The **Nordic monitoring comparison** extends this to Norway, Denmark and
Finland. Four common topics retain their national definitions, observation
periods and currencies; Sweden's orders remain supplemental. Each country
report connects the dated signals to its annual assessment and conditional
scenarios. Original loan-rate definitions and monthly/daily yield histories
remain distinct; the comparison does not rank countries.

```bash
python -m dalio.pipelines.fetch_nordic_monitoring --artifact-root data/artifacts/nordic_monitoring
python -m dalio.pipelines.build_nordic_monitoring --db data/dalio.db
```

The complete offline report set is under
`data/snapshots/nordic_monitoring/<hash>/`: open `index.md` for the comparison,
then a country page and its linked annual context. The source database and
desktop bundle are unchanged. See ADR 0032 for capture, scope and replay rules.

For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), it pulls macro indicators from FRED, BIS, Riksbank, IMF, OECD, SCB and World Bank, then classifies cycle stage with rule-based logic. The World Fundamentals Map extends coverage to 22 players. Numeric pipelines preserve complete source snapshots in an append-only release ledger beside the compatible latest-value table, enabling honest “what was known then?” queries from the ledger cutover onward.

The raw evidence layer now also preserves facts that should not be flattened into one score:

- SCB Financial Accounts positions for Swedish central-government debt securities by holder sector, instrument and quarter (8,349 rows, 1996-Q1 through 2026-Q1).
- IMF BPM6 financial-account transactions: 25 separate asset-acquisition, liability-incurrence and net series for each of 21 individual economies. These are actual reported transactions, not changes inferred from positions.
- World Bank/IMF QPSD central-government debt anatomy: 12 percent-of-GDP measures covering maturity, instrument, currency, creditor residence and D1/D2A scope. The first full run stored 200 country-series histories and recorded 64 voluntary non-reports without filling them.
- Macro acquisition now covers all 19 countries in the saved Börsdata listing universe: 361 official source-series histories across 18 distinct metrics (15,616 batch observations), including nine previously absent countries. Original Swedish debt-office evidence adds eight monthly PDFs and funding workbook 2026:1, with 1,595 typed cells and explicit missing/forecast status. First-hand national sources are preferred; WB/IMF harmonized baselines retain producer and dataset metadata. Original bytes and verified offline replay are retained. Scoring populations are unchanged. See ADR 0029.
- Sovereign refinancing now has an atomic 31-partition pipeline and an audit that re-parses retained source evidence. The first collection contains 697 observations: 297 annual Eurostat general-government values across DE/FR/IT/ES/SE and 400 monthly fixed-composition EA21 comparison values. Four artifact roles bind each release. A dated offline brief compares matching reference periods and calculates the due-within-one-year share using matching debt denominators. The full Version 1 denominator remains 48 streams; 16 national-native issuer streams remain after the original Swedish monthly tranche (ADR 0029). EA21 is a comparator; no composite refinancing score is assigned. See ADR 0028.
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

The separate `dalio-report-review` gate can prepare and validate a blank decision document, report progress, and—only from a TTY after showing every candidate/outcome, confirming the canonical full-decision hash and making a verified database backup—atomically record an operator-attributed approve/revise/reject batch. Human-only use is operating policy; the local identity is not cryptographically authenticated. Nobody has used the gate on the live packet: there are still zero human decisions and zero verified report claims.

Institutional press conferences, earnings communications and executive letters
now have a separate rights-gated foundation. The checked catalogue covers 24
first-party source policies for 19 organizations: five central banks, seven
major banks and seven commodity companies spanning six commodity families. The event, artifact,
byte-capture, extraction and speaker-segment schemas preserve publication clocks,
authorship, transcriber/caption provenance, rights state, changed source bytes,
corrections and exact locators.
The first bounded metadata pilot now enumerates all eight 2025 Fed and all eight
2025 ECB regular monetary-policy press conferences and one exact first-party
English text candidate per event. A deterministic offline packet presents those
links and current official rights notices for human review. **No communication
content bytes, database event/artifact rows or transcript segments have been
collected**, and no automated acquisition is authorized. The pilot's 16/16 means
link coverage only—not a rights-cleared or analysis-ready corpus.
Checked communication manifests resolve their catalogue SHA-256 to the complete
frozen source-policy vintage that produced it. Growing the current catalogue can
therefore neither invalidate an old manifest nor reinterpret it through newer
publisher, transcriber or rights metadata; unknown snapshots fail closed.
Communication metadata writes persist that same snapshot's hash, evaluation
clock and policy values rather than stamping current catalogue metadata.
An independent 2025 Bank of England cohort closes all four Monetary Policy
Report press-conference events and measures representations separately: 4/4
exact Bank-hosted transcript links, 4/4 official-page video locators (3/4 exact
external-platform page URLs) and 0/4 verified exact caption-track links. This is
a metadata inventory only; an observed video never counts as a caption, and it
adds no content or database rows.
A separate closed 2025 Sveriges Riksbank cohort covers all eight
monetary-policy-decision press conferences. It records 8/8 exact first-party
Riksbanken Play replay-page locators and 8/8 exact official Swedish slide PDFs,
but 0/8 verified exact first-party transcripts and 0/8 verified exact caption
tracks. Replay pages, slides, transcripts and captions remain four different
representations; this cohort also adds metadata only and writes no database
rows.
The closed 2025 Reserve Bank of Australia cohort covers all eight official
monetary-policy media conferences: 8/8 exact first-party inline-HTML transcript
pages, 8/8 exact external video URLs linked from those official pages, and 0/8
verified exact caption tracks. No slide representation is claimed, and RBA MP3
audio is outside Version 1. These remain locator observations only; no page,
transcript, media or caption content is stored.
Future byte and segment hashes will prove reproducibility, not transcription
fidelity; extracted text still needs trusted execution or named human review.

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

# Blank human decision sheet; preparation/check/status are database-read-only
dalio-report-review prepare --db data/dalio.db
dalio-report-review check --db data/dalio.db --decisions data/review/report_decisions_YYYY-MM-DD_PACKETHASH.json
dalio-report-review status --db data/dalio.db --decisions data/review/report_decisions_YYYY-MM-DD_PACKETHASH.json

# Offline metadata/rights review; no database, network, or source-content reads
dalio-communication-rights-packet

# Offline representation coverage; defaults to the checked 2025 BoE cohort
dalio-communication-metadata-inventory

# Checked 2025 Riksbank cohort; qualified latest aliases, metadata only
dalio-communication-metadata-inventory \
  --cohort riksbank_2025_monetary_policy_press_conferences

# Checked 2025 RBA cohort; qualified latest aliases, metadata only
dalio-communication-metadata-inventory \
  --cohort rba_2025_monetary_policy_media_conferences

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

From Windows Explorer, this checkout is available at `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine`; the local SQLite file is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\dalio.db`, and durable source evidence is under `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts`. The communications source-policy catalogue is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\src\dalio\communications\catalogue.py`; the checked Fed/ECB pilot is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_pilot_events.json`; its latest unverified rights packet is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_rights_latest.md`; and a future rights-cleared archive will live under `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\communications`. The latest generated liquidity brief is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_latest.md`; the latest unverified report review is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_claims_latest.md`; and the untouched human decision sheet is `\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_decisions_2026-09-09_20c687f1cb907c71.json`. See `data/README.md` before copying, deleting or rebuilding anything under `data/`.

The checked BoE cohort has the full Windows path
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_boe_2025_events.json`.
Its generated metadata-only views are
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_latest.md`
and
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_latest.json`.

The checked Riksbank cohort has the full Windows path
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_riksbank_2025_events.json`.
Its cohort-qualified metadata-only views are
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.md`
and
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.json`.

The checked RBA cohort has the full Windows path
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_rba_2025_events.json`.
Its cohort-qualified metadata-only views are
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_rba_2025_monetary_policy_media_conferences_latest.md`
and
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_rba_2025_monetary_policy_media_conferences_latest.json`.

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

The packet builder does not write to the database. The separate
`dalio-report-review` command prepares the unsigned decision sheet, revalidates
the packet and evidence, and reports file/ledger progress without writing. Its
`apply` subcommand refuses non-TTY input, shows the exact candidate/outcome map,
requires the operator to supply a `human:<id>` attribution and type the canonical
full-decision hash, creates a verified SQLite
backup, and records the complete batch in one append-only transaction. Approval
creates a verified successor; revision preserves the model draft and adds a
human replacement plus verified successor; rejection preserves the decision and
creates no verified claim. A local identity is explicit attribution, not
cryptographic authentication. No model may fill decisions, supply the identity,
confirm the hash or invoke `apply`. See [ADR 0010](decisions/0010-report-claim-review-queue.md)
and [ADR 0011](decisions/0011-human-report-claim-decisions.md).

### Communications metadata and rights packet

`dalio-communication-rights-packet` reads only the checked
`data/reference/communication_pilot_events.json` manifest. It validates the
fixed 2025 Fed/ECB 8+8 denominator, manifest-bound source-catalogue snapshot,
exact official domains, selected representation form, provenance and
conservative clocks, then
writes `communication_rights_latest.{json,md}` plus a hash-addressed immutable
pair under `data/review/`. It performs no network, database or source-content
reads.

Every output is labelled `UNVERIFIED RIGHTS REVIEW`, reports zero verified
rights decisions and sets `content_capture_authorized=false`. The storage
contract now maps one artifact version and one byte-capture lineage to an atomic ordered
scope set, so the ECB page can retain separately attributed prepared remarks and
Q&A without duplicate bytes. Extracted segments must resolve back to those
declared scopes. No scope mapping grants acquisition rights: without a recorded
clearance, the 16 candidates remain link metadata only and no source content may
be archived. See [ADR 0012](decisions/0012-institutional-communications-ledger.md),
[ADR 0013](decisions/0013-fed-ecb-communications-metadata-pilot.md) and
[ADR 0014](decisions/0014-communication-section-scope-sets.md).

`dalio-communication-metadata-inventory` is the separate checked-cohort,
offline representation view. With no `--cohort`, it retains the existing BoE
behavior and validates `data/reference/communication_boe_2025_events.json`:
four MPR events, 4/4 exact transcript links, 4/4 official-page video locators
(one ID-only observation) and 0/4 verified exact caption tracks. See
[ADR 0015](decisions/0015-boe-2025-communication-metadata-inventory.md).

Selecting
`--cohort riksbank_2025_monetary_policy_press_conferences` dispatches through
the pinned Riksbank loader and validates
`data/reference/communication_riksbank_2025_events.json`. Against its closed
eight-event first-party denominator, it reports 8/8 exact Riksbanken Play
replay-page locators, 8/8 exact official Swedish slide PDFs, 0/8 verified exact
first-party transcripts and 0/8 verified exact caption tracks. Nondefault
cohorts refresh only cohort-qualified `latest` aliases; generic immutable
filenames remain hash-bound, and the legacy BoE aliases do not move. The
publisher performs no network, database or source-content reads and writes all
views with `content_capture_authorized=false`. See
[ADR 0018](decisions/0018-riksbank-2025-communication-metadata-inventory.md).

Selecting `--cohort rba_2025_monetary_policy_media_conferences` validates the
closed eight-event first-party RBA index and reports 8/8 exact RBA inline-HTML
transcript-page locators, 8/8 exact officially linked external video URLs and
0/8 verified exact caption tracks. No slide representation is claimed, and RBA
MP3 audio remains outside Version 1. It refreshes only the cohort-qualified RBA
aliases; the BoE and Riksbank latest bytes remain unchanged. Like the other
cohorts, this is metadata only: the publisher reads no source content and writes
no database rows. See
[ADR 0019](decisions/0019-rba-2025-communication-metadata-inventory.md).
The immutable source-policy snapshot contract is documented in
[ADR 0016](decisions/0016-communication-catalogue-snapshots.md).

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

Pre-alpha. The cycle and fundamentals product is working, and the raw-history foundation now includes sovereign-debt anatomy, Swedish debt holders, IMF financial-account transactions, 127,970 bilateral investment-position rows, three Swedish AP-fund disclosures, a ten-document official-report corpus, 63,179 monthly commodity observations, ten official-money histories and 22 separate shadow-liquidity histories. Read-only liquidity diagnostics, a 20-item report review queue, an append-only human decision gate and a 24-policy/19-organization institutional-communications schema foundation are available. A closed 2025 Fed/ECB pilot adds 16/16 exact first-party text links and an offline rights-review packet; separate BoE, Riksbank and RBA cohorts add representation-specific metadata. The bounded 2025 Riksbank-plus-RBA central-bank package now closes 16/16 denominator events, while all communication content and database evidence remain absent. Checked manifests bind immutable source-policy snapshots, and the ordered one-artifact/one-byte-lineage, multi-section storage contract is ready. ADR 0017 caps the remaining Version 1 foundation at five bounded packages. The debt package now has 31 audited harmonized refinancing histories (697 observations) inside its checked 48-stream denominator, with an atomic refresh, offline staging promotion and a descriptive comparison brief; 16 national-native streams remain after the original Swedish monthly tranche. This is not a complete global money-flow map, a communications corpus, a universal M5, an additive liquidity total, a causal or deposit-flow model, or an investable commodity return history: all 20 report candidates still need named human review; every communication representation remains rights-gated; allocator history has only one H1 2026 release per fund; QPSD and IMF position coverage are voluntary and uneven; and debt cash-flow schedules, broader banking/funding channels and horizon risk scenarios remain to be built. See `project_context.md`, ADRs 0004–0020 and 0028–0029, and `data/README.md` for current boundaries.
