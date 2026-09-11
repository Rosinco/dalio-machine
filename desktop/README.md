# Macro Atlas desktop

An offline Windows viewer for saved Dalio macro and Börsdata company research. The app
bundles its map, scripts, charts and source data. Starting the packaged executable
requires no terminal, Python environment, WSL, account or network connection.
Windows WebView2 must be installed; it is present on Adam's PC. The Tauri installer
configuration can also bundle its offline installer when distributing to another PC.

## Version 0.9.0


**Macro → Assessments** opens the verified country evidence alongside the older
fundamentals, liquidity and company research. All **19 listing countries** have
annual assessments and native annual histories. **Sweden, Norway, Denmark, Finland,
United States, Germany and Canada** have 29 monitoring topics: 28 source-bound
signals and the documented US corporate new-loan-rate gap. Search includes
countries without an older scored profile, including Finland, Norway and Belgium.
`GB` listing identities resolve to the `UK` macro profile.

- Signal cards show the original latest reference period, value, unit and exact
  comparison window. Open a signal for its saved history, scope, freshness rule,
  scenario checks and source references. Blue/purple identify data series; a rising
  or falling rate is not automatically good or bad and says nothing by itself
  about credit availability. Industry scopes, currencies and rate instruments
  differ by country; the app does not rank these signals.
- Annual charts separate historical IMF estimates/outturns from forecasts using
  the collector's calendar convention. Missing years remain gaps. The baseline
  and 2026–2031 path, dated demographics/energy/R&D, conditional scenarios and
  required company exposure checks are inspectable. There are no scenario
  probabilities, GDP nowcasts or automatic company headwind/tailwind verdicts.
- Sweden's original debt-office context separates observed reference dates and
  monthly means from funding-plan forecasts. Refixing is not principal maturity;
  central-government amounts are separate from IMF general-government ratios.
- This **independent offline country-evidence pack** keeps its own visible UTC
  assessment and known-at dates. Changing a saved fundamentals/research release
  never changes those dates or its source identity. Newly covered countries are
  searchable but do not inherit old score values or map colours.
- Original source URLs, native series and locations, acquisition/availability
  clocks, publisher update precision, native provisional flags, artifact hashes and explicit gaps are
  bundled with the values. Publisher websites require internet; the original
  response binaries remain in the macro project's archive. Loading the app or
  opening sources does not contact a publisher.

Use the **Observatory** selector beside ATLAS to switch between **Macro**,
**Sectors & branches**, and **Companies**. Each has its own Explore views.
Start with **Sectors & branches → Browse → Forest & Wood Products → Holmen**.
Country, branch and company selections carry across observatories and persist
when the app reopens.

- **Sectors & branches → Compare** opens the branch-history workspace. Choose
  up to eight listings, the Y-axis measure, reporting currency, listing country,
  saved-download coverage, fiscal closing month and year range. Click a bubble to
  select its company/year; supporting histories and the financial table follow.
- The grey line is the **median of the whole filtered branch**, with a middle-50%
  band when at least four observations exist. Annual valid/total counts and exact
  reporting periods remain available. Each listing has equal weight; cross-listings
  can repeat an issuer. This is saved-directory history, not a historical universe
  free of survivorship bias. Annual periods outside 330–400 days are excluded.
- **Bubble area** defaults to derived market cap in SEK. Equal-size points, total
  assets and revenue remain selectable. Market cap is also a Y-axis measure, using
  a common SEK scale across reporting currencies; assets/revenue use one reporting
  currency. A missing size omits the bubble but retains a valid Y-axis benchmark input.
  ROIC and CAPEX remain separate future measures; existing proxies are labelled.
- **Save new comparison** keeps the selection, filters, years, notes and exact data
  versions in Atlas's persistent local storage. Reopen it from **Saved in this branch**,
  including after restarting the app. Different data versions cannot silently replace
  the saved basis. Notes/settings must be saved explicitly and are currently separate
  from exported research and financial packs.

- **19,140 company listings** fill all 94 branches across 19 listing countries.
  The newest saved instrument download (2026-08-10) contributes 17,593; another
  1,547 exist only in the 2025-06-21 baseline and keep an **Older download** label.
  Absence from the newest download does not establish whether a listing is active.
  Separate share classes and depositary receipts retain their own Börsdata IDs.
- Search by company name, ticker, ISIN or instrument ID. Global search shows at most
  20 listings; branch and country lists show 50 per page. Filter by listing country,
  all/latest/older downloads, or search within a branch. Each listing opens a
  financial history and an identity panel; badges distinguish report coverage from
  the five saved research profiles.
- The saved Börsdata hierarchy includes **10 sectors and 94 branches**. Search in
  English or Swedish, filter by sector, or show branches with completed studies,
  saved company deep dives, financial histories, or saved research profiles.
- The **2026-09-10** inventory records **17 branches with completed branch studies**
  and **89 saved company deep-dive folders**. Shared bank and beverage studies count
  once in global dossier totals. Document existence does not establish completion,
  freshness or investment quality. Other project studies are inventoried with source
  paths and hashes; the selected five-company slice remains the readable content.
- Börsdata assignments are the default. Company classification details preserve the
  original sector/branch IDs and any separately reviewed correction. Eleven source
  sector IDs conflict with their branch's parent sector; their listings follow the
  supplied branch and show a review flag with both original IDs. Every company
  listing has a mapped country and recognized branch in this export.
  Branches retain their own research coverage; older research packages keep their
  original company coverage without inheriting the full directory.

- **16,895 listings have usable market-cap history**: 159,486 dated valuations.
  All 19,140 listings retain explicit coverage; unavailable or flagged observations
  stay blank. Company histories show the price, reported shares, historical FX,
  valuation date, source snapshot and review reasons. Toggle between SEK and the
  listing’s quote currency. Values are in millions.
- Market cap is derived from reported shares × the first valid close on or within
  30 calendar days after publication. It is a publication-date valuation, not a
  fiscal-year-end value. Shares and prices come from the same saved download.
  Historical FX uses observed direct rates or simultaneous USD cross-rates, at most
  seven days old. No static currency fallback is applied. Source share-basis and
  scale checks withhold suspect rows; preference, receipt and unit instruments
  require a reviewed share basis. Each listing stays separate; these checks do not
  certify every corporate action or consolidate the issuer’s share classes.

- **18,943 listings have usable reports**: 268,884 annual and 603,720 quarterly
  records from the saved 2025-06-21 and 2026-08-10 downloads. Coverage includes the
  first/latest fiscal years and quarters, missing periods within the span, report
  publication dates and source download dates. **197 listings have no reports**.
- **1,174 source rows are withheld** because their dates or period metadata cannot
  be used. Their original metadata and reasons remain inspectable. A newer invalid
  row does not silently fall back to an older value.
- Annual/quarterly charts and **income statement, balance sheet and cash flow**
  tables now work across the directory. Choose a fiscal period, inspect the saved
  currency conversion, and trace every report to its download. The standard source
  balance sheet is available now; investigating individual assets remains later work.
- Amounts are in millions of the report's currency: stored amount divided by
  `currency_ratio`. Missing conversion leaves values unavailable; real zeroes stay
  zero. Currency changes leave gaps in monetary charts, with a currency selector.
  Per-share financial ratios remain omitted because adjustment bases differ.
  Reported share counts appear as explicit inputs in the market-cap audit table.
- Earlier retained periods may come from an older accounting/restatement basis.
  The legacy five-company financial/research document remains available in older
  research packages; its source dates and original content remain unchanged.
- A five-company peer panel uses the latest exactly matching full-year dates
  (**FY 2025** in this package). Ratios are compared without FX conversion;
  absolute SEK and EUR amounts are never added or ranked together. Blue/purple
  identify the selected company and peers, with no good/bad rating.
- May 2026 branch research, the archived Holmen deep dive and source register are
  readable offline. Dates and original financial anchors remain visible. Financial
  refreshes do not refresh old prices, segment descriptions or investment verdicts.
- Dated Dalio observations sit beside qualitative research questions. The older
  fundamentals release has no Finland score profile; the independent country
  evidence pack now includes Finland's assessment and monitoring. Sweden is never substituted.
  No new branch forecast or company headwind/tailwind assessment is calculated.
- The map shows **Börsdata listing-country counts**, with explicit limits. The
  next physical-asset stage will extend the deep-dive funnel with balance-sheet
  analysis and a sourced inventory of owned/leased/joint-venture resources,
  locations and reporting dates (ADR 0022). There are no plant markers yet.

Return on capital is the existing annual pre-tax proxy, EBIT divided by year-end
equity plus net debt, not adjusted ROIC or company-reported ROCE. Forest revaluations
and transactions can distort reported profits. Börsdata FCF retains the provider
definition, excluding lease principal and interest; it is not owner earnings.

The existing macro capabilities remain available:

- **Research library:** import `.atlas.json` files, select earlier releases and
  export a portable copy to Downloads. Refreshing data no longer requires a rebuild.
  Included vintages: 2026-09-08 (fundamentals + liquidity, with company data dated
  separately at 2026-08-10) and 2026-08-24
  (fundamentals only). Each panel follows the active release.
- **Score explanations:** select a category below the radar to see raw observations,
  percentile direction, equal effective weights, contributions and source dates.
  The half-coverage rule is preserved. Comparisons use the nearest earlier distinct
  fundamentals snapshot and require matching ranking populations and indicator
  definitions. Sweden's included category scores are unchanged between these vintages.
- **Liquidity:** national broad money and dated growth history, explicitly labelled
  euro-area context for member countries, and separate global offshore-credit,
  US MMF and US repo panels. Quantities stay blue; no aggregate liquidity/risk score
  or inferred deposit flows are introduced. Source ledger, coverage, formulas and
  known-at timestamps remain accessible.

Assessment colours consistently mean **red = weaker, yellow = mixed/middle,
green = stronger**. Historical maps follow each scored indicator's existing
`higher_is_better` direction: high debt or dependency is red; low debt or dependency
is green. Bands are relative to the covered countries in the selected year, rather
than absolute safe/danger thresholds. Missing observations stay grey. Unscored
quantities and trade use blue; blue/purple chart lines identify countries or series.
The radar's red inner rings represent weak scores and green outer rings strong
scores. None of these presentation changes recalculate the saved fundamentals.

- World map: 177 Natural Earth overview features; research for 21 countries and a
  separately selectable euro-area aggregate.
- Country search and map selection, with an explicit unavailable state elsewhere.
- Five-category fundamentals radar and country comparison. The map uses fixed
  20-point score bands; no default overall score or crisis probability.
- Annual history and forecast lines, with missing years left as gaps. A historical
  map displays raw observations for the selected year; it excludes forecasts and
  does not carry values forward. Its colour breaks adapt to that year's range and
  follow the indicator's saved direction, without calculating historical ranks.
- Goods-trade map, doughnut and paired bar chart. Overlapping euro-area partner
  totals are excluded. The residual preserves the total-export denominator.
- Read-only pressure diagrams for existing triggered rules, labelled as judgment.
- Indicators, observation dates, source names, quality tiers, and snapshot SHA-256.
- CSV history export; Windows writes a new timestamped file to Downloads.

The app reads the selected saved vintage. Historical charts are **not** point-in-time
replays. Source metadata in this snapshot is insufficient for a per-point release
ledger, so the UI does not invent it. The five-year GDP-growth headline and annual
growth history are explicitly distinguished. Geographic asset inventories and new
sector forecasts are later slices; no company sites or forecasts are fabricated.

## Data flow and scale

The canonical Python pipelines remain authoritative. `export_research.py` wraps the
exact UTF-8 fundamentals, optional liquidity, business and taxonomy JSON with their SHA-256 hashes. Its
bundle mode also uses `export_snapshot.py`, which validates
with Dalio's existing parser and creates a compact index, individual country files,
and a history-map file. Startup loads summaries; selecting a country loads only its
history; the full small annual panel loads only in History mode. The source SQLite
database is never opened or modified by the exporter or viewer. Original data and
all Dalio statistical calculations remain unchanged.

`export_business.py` reads selected instruments and projected report columns via
Börsdata's existing `core.data.read_validated` API and shared schemas. Parquet
predicate pushdown selects just five instrument IDs; the screener is never read.
The exporter rejects output paths inside the Börsdata project. Source Parquet and
research files are hashed for provenance, without rewriting them. The business
document projects into a small catalogue, per-company resources and branch prose.
Only the selected company's full history is sent to the view. The cohort and
dated, qualitative segment/driver notes are explicit in the exporter, not a new
statistical classifier or competitor database.

`export_taxonomy.py` reads the maintained bilingual crosswalk, branch-study status,
and deep-dive file inventory without opening financial Parquet or making requests.
`export_listings.py` projects saved instrument/country metadata through the same
validated reader, choosing each ID's latest saved record before filtering company
types 0/1/3/8/9/10. Excluded index, FX, commodity and crypto counts reconcile to the
source snapshots. No issuer deduplication or branch inference by name is performed.
The v2 taxonomy document embeds the identity catalogue and all classifications
(8.26 MB total in this export). It loads on entering the company/sector observatories;
DOM lists stay bounded and the financial companion loads reports only for the selected listing.
Numeric source IDs join records independently of display labels. The document
binds to the exact business hash; its classification clock follows the catalogue's
newest instrument snapshot. Taxonomy v1 and its older classification clock remain
supported.

Native imported packages live in
`%LOCALAPPDATA%\local.research.macro-atlas\research-v1\<package-id>.atlas.json`,
outside the executable's version directory. Version 1 packages keep their original
two-document identity. Version 2 adds the business-document hash; its identity is
SHA-256 of `macro-atlas-research-v2\n<fundamentals-hash>\n<liquidity-hash-or-empty>\n<business-hash-or-empty>`.
Duplicate imports are idempotent. A v1 envelope containing a business document is
rejected, preventing that document from being omitted from identity checks.
Version 3 adds the optional taxonomy document: its identity is SHA-256 of
`macro-atlas-research-v3\n<fundamentals-hash>\n<liquidity-hash-or-empty>\n<business-hash-or-empty>\n<taxonomy-hash-or-empty>`.
V1/v2 IDs are unchanged; their envelopes cannot contain an unhashed taxonomy.
The Rust `research-store` crate checks
format, bounds, required renderable fields and checksums before publishing a complete
file with a non-overwriting hard link. It only exposes fixed resource names and
country codes and numeric company identifiers. A damaged archive entry is reported without hiding usable entries.
Checksums establish file integrity, not publisher authenticity. Browser previews
use IndexedDB; those imports are separate from the installed Windows library.

The local import limit is 32 MiB per package. Native imported resources are projected
from that bounded package on demand; this release is not a large analytical database.
Included assets retain country/company-level lazy loading. The source ledger's referenced
provider artifact files are not copied into the package. Their contents are not
independently verified by the offline viewer.

Large Börsdata Parquet tables are not included in the application. Version 0.8
uses a **109,862,912-byte SQLite companion** (financial format v2), indexed by listing
ID with compressed company records. The original 96 MB v1 pack remains readable.
The v2 companion extends every company with annual valuation records and coverage.
Existing annual/quarterly financial records are unchanged. Its coverage index loads when needed; only the selected company's
reports enter the webview. This supports point lookups, rather than arbitrary
cross-company analytical queries. The 101-million-row screener is never read.

Financial packs are separate from the 32 MiB research JSON and bind to the exact
taxonomy document SHA-256. They live beside the executable in `financial-data/`,
or in `%LOCALAPPDATA%\local.research.macro-atlas\financial-packs-v1\` after import.
**Library → Save financial history pack** exports an identical `.sqlite` file;
**Import financial history pack** validates every record before publishing it under
its whole-file checksum. Transfers use 512 KiB chunks and a 512 MiB cap. Save both
research JSON and the financial pack when moving to another offline computer.
A corrected or otherwise changed directory needs a freshly bound financial pack;
older or unrelated releases cannot inherit it. Importing a pack preserves existing
packs and is idempotent for identical bytes. Checksums do not authenticate a publisher.
The browser development preview needs Node 22 with `node:sqlite`; financial import
is native-only. Neither Node nor Python is needed by the installed app.

## Reviewed classification corrections

`research/classification-corrections.json` starts empty. After reviewing a company,
add a record to its `corrections` array with these fields:

| Field | Meaning |
|---|---|
| `company_id` | Börsdata instrument ID, as a string |
| `expected_sector_id`, `expected_branch_id` | Source assignment that the review examined |
| `branch_id` | Reviewed destination branch; its parent supplies the effective sector |
| `reason`, `source` | Rationale and supporting evidence reference |
| `reviewed_at` | Review date, `YYYY-MM-DD` |

Re-export the directory and research package, then import it into Atlas. There is
no in-app correction editor in 0.6. All corrected listings must be present in the
selected catalogue (or the business export for legacy taxonomy v1). Unknown IDs, duplicate corrections and missing review
evidence are rejected. Source files and archived research are never rewritten.

If a later source assignment differs from the expected original, the saved correction
remains visible as `needs_review` and the source assignment is used. If the source
now equals the reviewed destination, it is `aligned`. Corrections do not rewrite
segment tags or historical peer-study membership. Label edits in the maintained
source crosswalk cannot change numeric identity. Keep the overlay under version
control so its review history remains available.

## Build and refresh

From this `desktop/` folder:

```sh
npm ci
source /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client/.venv-wsl/bin/activate
PYTHONDONTWRITEBYTECODE=1 python scripts/export_business.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client \
  --snapshot 2026-08-10 \
  --output /tmp/atlas-business-2026-08-10.json
PYTHONDONTWRITEBYTECODE=1 python scripts/export_listings.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client \
  --output /tmp/atlas-listings-2026-08-10.json
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
python scripts/export_taxonomy.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client \
  --business /tmp/atlas-business-2026-08-10.json \
  --listings /tmp/atlas-listings-2026-08-10.json \
  --output /tmp/atlas-taxonomy-2026-09-10.json
python scripts/export_research.py \
  --fundamentals /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json \
  --liquidity /home/rosinco/workspace/dalio-machine/data/snapshots/liquidity_latest.json \
  --previous /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_2026-08-24.json \
  --business /tmp/atlas-business-2026-08-10.json \
  --taxonomy /tmp/atlas-taxonomy-2026-09-10.json \
  --bundle public/data
python scripts/verify_pack.py \
  --source /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json
source /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client/.venv-wsl/bin/activate
PYTHONDONTWRITEBYTECODE=1 python scripts/export_financials.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client \
  --taxonomy public/data/taxonomy.json --output financial-data
# Use the v1 pack ID printed by export_financials.py above.
PYTHONDONTWRITEBYTECODE=1 python scripts/export_market_history.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client \
  --financial-pack financial-data/BASE_V1_PACK_ID.sqlite --output financial-data
npm test
npm run build
```

Generated research data is deliberately gitignored. The checked Natural Earth map
has its source, original checksum and processing notes in `public/maps/NOTICE.txt`.
To update the research release in an already installed app, export its JSON and
import it through **Library → Import research file**. Export and import the matching
financial companion separately when financial histories or the directory change:

```sh
python scripts/export_research.py \
  --fundamentals /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json \
  --liquidity /home/rosinco/workspace/dalio-machine/data/snapshots/liquidity_latest.json \
  --business /tmp/atlas-business-2026-08-10.json \
  --taxonomy /tmp/atlas-taxonomy-2026-09-10.json \
  --output /tmp/Macro-Atlas-Research.atlas.json
```

Omit `--taxonomy` for the legacy v2 format, and omit both it and `--business` for v1, and omit `--liquidity` as well for
a fundamentals-only package. Each document retains its own
date; packaging does not make old observations newer. The exporter reads saved
files and never refreshes data from the network. Updated map geometry, new data
schemas or application features still require a new app build.

Native Windows development uses `npm run tauri build`. The optional NSIS installer
configuration includes offline WebView2 setup. For this WSL build,
`scripts/build-windows.sh` uses Rust, cargo-xwin, LLVM and cached MSVC/Windows SDK
libraries to produce `src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe`.
Set `ATLAS_LLVM_BIN`, `ATLAS_LLVM_LIB` and `ATLAS_XWIN_CACHE` to override tool paths.
The build on this PC uses `python3 scripts/build-windows-native.py` after extracting
the same official SDK packages with native `msiextract`. Its tool root defaults to
`/tmp/atlas-tools` (`ATLAS_BUILD_TOOLS` overrides it). The Windows SDK and Universal
CRT are separate packages: `ATLAS_SDK_VERSION` defaults to `10.0.26100.0` and
`ATLAS_UCRT_VERSION` to `10.0.10240.0`. The script validates its include/library
directories and statically links the C runtime. These cached tool paths are build
prerequisites, not a portable toolchain installer.
The cross-compiled executable embeds the whole `dist/` folder. Financial SQLite
packs are external resources, copied beside the executable by the install script
(and included by Tauri's bundle resource mapping). The Linux build uses a Clang
case-insensitive virtual filesystem overlay for Windows SDK headers, allowing
bundled SQLite to compile without changing those headers. Build tools are not
needed to run the installed app.

After verification, run `scripts/install-windows.ps1` in Windows PowerShell. It
copies the app into the current user's LocalAppData and creates **Macro Atlas** on
their actual desktop, including OneDrive desktops. It updates earlier Macro Atlas
shortcuts, keeps prior version folders, and refuses to replace a different
executable in the same version folder or an unrelated desktop shortcut. `-Start`
closes older Atlas viewer windows and opens the installed version.

## Verification

`npm test` checks colour direction (including debt, negative values, missing values,
equal observations and unscored quantities), zero/missing scores, historical gaps, forecast boundaries,
single-vintage trade, overlapping aggregates and trade denominators.
It also checks score arithmetic and comparison eligibility, import checksums and
validation, liquidity geography and gaps, business-document identity and company
chart gaps, taxonomy search, shared-study counts, source binding and corrections.
`python -m pytest tests/test_export_business.py tests/test_export_taxonomy.py tests/test_listing_catalogue.py tests/test_financial_history.py tests/test_market_history.py` (from `desktop/`, with
the Dalio venv activated) checks FX division, zero/missing values, denominator and
quarterly-return rules, period validation, common-year catalogue projection,
taxonomy identity, correction drift, deduplicated document inventories, newest-ID selection,
separate share classes and explicit missing/conflicting metadata.
The standalone Rust stores have filesystem tests runnable with
`cargo test --manifest-path research-store/Cargo.toml` and
`cargo test --manifest-path financial-store/Cargo.toml`.
`cargo run --release --manifest-path financial-store/Cargo.toml --example verify -- financial-data/PACK_ID.sqlite`
validates the real coverage index, every compressed record and selected lookup timing.
`cargo run --manifest-path research-store/Cargo.toml --example verify -- FILE...`
validates real exported packages and checks exact country/liquidity/company/taxonomy round
trips and parity with the Python-generated company catalogue.
`scripts/verify_pack.py` checks every exported country against the original snapshot
and verifies map coverage. `npm run test:browser` runs Playwright against
`npm run preview -- --port 1420`, blocks all external app resources, exercises the
main flows and writes screenshots and timings to `test-results/`.

`scripts/test-windows.ps1` launches the built executable in Windows, checks the
native WebView2 map, charts, comparison and modes, audits startup requests, and
captures a screenshot. It also exports one history CSV to Downloads and verifies
its contents, imports a real older package, rejects damaged input, checks duplicate
imports, exercises all three observatories and the five-company slice, imports a
v2 business and v3 taxonomy packages, exercises all 94 branches and research filters,
and verifies that unrelated or older releases cannot inherit forestry content.
The complete directory reconciles across 94 branches and 19 countries, with bounded
search/pagination, older-only filters, identity-panel persistence and source-conflict
flags tested in both browser and native Windows flows.
It restarts the executable to verify selected-company persistence and exports a
portable research file containing all four source documents. Financial checks
exercise no-report listings, currency conversion, missing/withheld periods and all
three statements. Market checks reconcile Holmen’s shares and price, EUR direct
FX and PLN cross-rates, quality gaps and all-currency SEK branch medians. Native
tests export/import the full 110 MB financial pack,
reject malformed input and check persistence after process restart.
It uses an isolated WebView profile and temporary native archive
(`ATLAS_RESEARCH_DIR` and `ATLAS_FINANCIALS_DIR`, development/test overrides) and closes its own processes.
Pass `-Executable` to check an installed copy. The test never disables the PC's
network connection or closes other applications.
The test copies the companion into local Windows storage. Windows Node and WebView
share that filesystem, so `connectOverCDP({ isLocal: true })` and Playwright's public
file-input API select the full local file without transferring a second buffer or
opening a separate file-selection debugger session.
It needs Playwright's installed JavaScript dependencies and a Windows Node runtime;
the helper can use the Node runtime in an existing VS Code installation.

Use `scripts/test-windows.ps1 -FinancialOnly` for a focused native reproduction.
It uses the included default research release and full 110 MB financial companion,
checks malformed input rejection, byte-identical export/import, reload and process
restart, and skips unrelated directory/branch loops. `-Executable` can select a
particular built or installed binary. The focused run does not replace the full
native suite; both retain the 300-second runner limit.

The native runner uses Playwright's public custom CDP transport with the test-only `ws`
dependency, explicitly declines WebSocket compression, and requires the negotiated
extensions to be empty. Add `-CompressedCdp` to either scope to reproduce the earlier
stock Playwright connection. The focused uncompressed run passed full financial
import, reload and process restart on the unchanged application binary. The full
native suite remains subject to its separate verification receipt. This
compares two debugger clients as well as compression settings; a passing run alone
would not establish the cause of earlier malformed protocol messages.

Native tests write UTC event timelines and diagnostic receipts to
`test-results/windows-native-events-*.jsonl` and
`test-results/windows-native-diagnostics-*.json`. These record runtime versions,
process/WebView lifecycle, observed import progress and available heap readings at
deciles. Displayed 100% is rounded UI progress, not independent proof that validation
finished. Financial file hashes and the success/restart assertions provide that
verification. Failed runs preserve their isolated temporary profile and upload
files for inspection; successful runs remove that temporary profile.

The runner enables only `pw:browser` transport diagnostics. Protocol payloads are
redacted before logging; malformed-message receipts retain bounded envelope
metadata, length, digest and parser position, plus WebSocket close/error reasons.
Earlier failures remain recorded separately from successful runs. Final native
verification and installation status belong in the current handoff.

The browser runner uses `/opt/google/chrome/chrome`; this is a development test
dependency, not a requirement for the Windows application. Measured timings refer
to that test environment and are not a universal performance guarantee.

MapLibre (BSD-3-Clause), React (MIT), React Flow (MIT), ECharts (Apache-2.0), Tauri
(MIT/Apache-2.0), and Lucide (ISC) are the main open-source components. See the
generated release notices for the installed dependency versions and license texts.

The application does not publish the underlying research data or give it a new
license. Its data source terms remain separate from its software dependencies.

## Rebuilding the independent country evidence pack

From `desktop/`, activate the canonical macro project's Python environment, then
export one verified 19-country assessment snapshot plus every monitoring snapshot
to include. `--monitoring` is repeatable and order-independent. Paths below are
examples; use immutable reviewed `snapshot.json` files from the macro archive.

```bash
python scripts/export_country_evidence.py \
  --assessment /path/to/country_assessments/SNAPSHOT/snapshot.json \
  --monitoring /path/to/nordic_monitoring/SNAPSHOT/snapshot.json \
  --monitoring /path/to/national_monitoring/SNAPSHOT/snapshot.json
python scripts/export_country_evidence.py --verify
python -m pytest tests/test_country_evidence.py
npm test
npm run build
npm run test:browser -- --evidence-only
```

The exporter reads no database and makes no network requests. It checks source
snapshot identities and citation references, retains native histories and gap
artifacts, selects the newest **whole country monitoring profile**, and refuses
same-cutoff conflicts. A missing signal in a newer profile cannot fall back to an
older capture. Annual profiles use the newest verified base or embedded assessment
parent and retain that parent's own hash, methodology, citations and cutoff.

`public/data/country-evidence/index.json` points at immutable hashed per-country
files. The index is replaced only after complete files have been written; repeating
an unchanged export preserves bytes and modification times. Frontend checks verify
both index and country hashes before display. Only a small catalogue loads at
startup; full country files load on demand with a five-country memory cache. The
active 19-country/seven-monitoring-country pack is about 13 MB total. Original response binaries are
not duplicated in the executable.

This pack does not modify the research `.atlas.json` contract or the financial
SQLite companion. It is bundled by Vite/Tauri at build time and is not currently a
user-importable pack. Rebuild/install the application to refresh this evidence.
`tests/country-evidence-flows.mjs` is shared with the browser/native test runners;
it covers native values/windows, missing monitoring, countries without scores,
GB/UK identity, independent dates across research-release changes and compact
layout. ADR 0034 records the contract.
