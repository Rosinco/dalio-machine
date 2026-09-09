# Data layout and evidence boundaries

This directory contains local working data for the Macro History & Risk
Observatory. Most generated data and binary source material is intentionally
gitignored; the small JSON catalogues in `reference/` are the reproducible
metadata/transcription inputs.

From Windows Explorer, the directory is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data`

The live local database is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\dalio.db`

The latest human-readable liquidity brief is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_latest.md`

Its machine-readable evidence companion is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_latest.json`

Hash-addressed generated copies use the full Windows path patterns
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.md`
and
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\snapshots\liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.json`.
The prefix is the first 16 lowercase hexadecimal characters of the full
`snapshot_sha256`. These are regenerable, content-addressed exports rather than
source-release artifacts; the two `liquidity_latest` files remain fixed aliases.

The generated report-review packet has these fixed Windows paths:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_claims_latest.md`

and

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_claims_latest.json`.

The content-addressed copies follow
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_claims_YYYY-MM-DD_<packet-hash-prefix>.{json,md}`,
where the prefix is the first 16 characters of the full packet SHA-256. The
current packet is `report_claims_2026-09-09_20c687f1cb907c71.{json,md}`.

The durable liquidity-frontier artifact root is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\liquidity_frontier`

The durable official-money artifact root is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\money_liquidity`

The exact pre-frontier database backup is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-before-liquidity-frontier-2026-09-08.db`

The verified post-frontier database backup is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-after-liquidity-frontier-2026-09-08.db`

## Validated refresh inventory (2026-09-08)

The counts below are from the fully validated live `dalio.db` after promotion.
Run the read-only inventory commands below to reproduce the inventory.

| Evidence | Stored coverage |
|---|---:|
| Current scalar observations | 328,909 rows |
| Immutable releases | 3,054 releases |
| Immutable scalar release observations | 408,487 rows |
| QPSD sovereign-debt anatomy | 200/264 partitions; 64 evidenced non-reports |
| IMF BOP financial-account transactions | 519/525 partitions; 52,700 current rows |
| IMF PIP/DIP bilateral position stocks | 291/294 partitions; 127,970 current rows |
| SCB Swedish government-debt holders | 8,349 current rows |
| AP2/AP3/AP4 allocator disclosures | 48 current facts |
| Official institutional reports | 10 documents; 852 extracted pages; 0 verified claims |
| Report review queue | 5 latest documents; 20 unverified model drafts; 0 promoted claims |
| World Bank monthly commodity history | 63,179 rows; 70 prices + 17 indices; 1960-01–2026-08 |
| Official-money history | 5,794 rows; 10/10 pinned native series |
| Separate liquidity frontier | 20,676 rows; 22/22 series (3 BIS + 19 OFR) |
| Derived liquidity exports | Five independent families; JSON + Markdown; no composite score |

The read-only inventory command derives these counts from the database and also
lists every stored partition and gap:

```bash
dalio-audit-observatory --db data/dalio.db
dalio-audit-observatory --db data/dalio.db --json
```

## Directory roles

| Path | Role | Rebuildability |
|---|---|---|
| `dalio.db` | Local SQLite database used by the pipelines and app | Rebuildable only to the extent that sources and preserved release metadata/artifacts remain available |
| `dalio.db.before-release-clock-migration.sqlite3` | One-time exact backup created beside a legacy database before the release-event uniqueness migration | Recovery evidence; preserve until the migrated database has passed inventory, integrity and foreign-key checks |
| `reference/ap_funds_h1_2026.json` | Versioned metadata and model-checked transcription of 48 AP2/AP3/AP4 facts | Reproducible source input; not a replacement for the PDFs |
| `reference/report_issues.json` | Exact metadata, hashes, filenames and page counts for ten official report issues | Versioned source input; not a replacement for the PDFs |
| `reference/report_claim_candidates.json` | Checked, versioned candidate propositions and exact page locators for the report-review queue | Review input only; structural/excerpt checks do not make a candidate verified |
| `artifacts/allocators/sha256/` | Content-addressed copies of official allocator PDFs | Durable evidence; do not treat as a disposable cache |
| `artifacts/reports/sha256/` | Content-addressed copies of official central-bank/IMF/BIS PDFs | Durable evidence; do not treat as a disposable cache |
| `artifacts/worldbank_commodities/<hash-prefix>/` | Exact World Bank Pink Sheet XLSX vintages and deterministic native-series catalogues, addressed by workbook SHA-256 | Durable evidence; do not treat as a disposable cache |
| `artifacts/money_liquidity/` | Content-addressed validated official-money response bodies and per-series missingness ledgers | Durable release evidence; paths and full hashes are bound in `data_release_artifacts` and rechecked by inventory |
| `artifacts/liquidity_frontier/` | Content-addressed BIS/OFR responses, provider semantic catalogues and OFR per-series payload/missingness ledgers | Durable release evidence; paths and full hashes are bound in `data_release_artifacts` and rechecked by inventory |
| `cache/` | HTTP response cache used to reduce repeated source calls | Disposable, but a fresh rebuild then depends on the upstream source still serving the data |
| `snapshots/` | Generated exports consumed by the dashboard or downstream tools, including fixed `liquidity_latest.{json,md}` aliases and hash-addressed `liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.{json,md}` copies | Regenerable from the database and versioned calculation code; not source evidence |
| `review/` | Generated `report_claims_latest.{json,md}` review aliases and hash-addressed packet copies | Regenerable, unverified review material; never source evidence or database truth |
| `backups/` | Deliberate local database safety copies | Preserve until their replacement has been verified |

## Evidence shapes

Different questions require different storage shapes:

- `observations` is the compatible latest-value projection for ordinary scalar
  time series. `data_releases` and `release_observations` preserve immutable,
  complete provider partitions for point-in-time reconstruction.
- `debt_holder_positions` keeps SCB issuer, instrument, holder sector, balance
  measure and unit dimensions. The landed release has 8,349 quarterly Swedish
  central-government debt-security positions from 1996-Q1 through 2026-Q1.
- QPSD sovereign-debt anatomy is stored as twelve source-native percent-of-GDP
  series. The first full run stored 200 country-series histories; 64 other
  requested country-series combinations were valid voluntary non-reports, not
  zeros and not errors.
- IMF BPM6 BOP data stores 25 reported financial-account transaction series per
  individual country: gross asset acquisition, gross liability incurrence and
  the publisher's net entry stay separate. A position change is never relabelled
  as a flow.
- `cross_border_positions` contains 127,970 first-hand IMF PIP and DIP stock
  observations: 92,529 PIP rows and 35,441 DIP rows. It
  retains reporter/counterpart native codes, direction, accounting basis,
  instrument, annual/semiannual frequency, unscaled USD value, native series
  and status. PIP selects reported portfolio assets at total-economy sectors;
  DIP selects the reporter's own observations rather than counterparty-derived
  mirrors. All 168 requested PIP partitions and 123 of 126 DIP partitions were
  ingested; Saudi Arabia's three outward DIP series were not reported.
- `allocator_facts` contains 48 AP2/AP3/AP4 H1 2026 disclosure facts tied to
  exact PDF artifacts and physical pages. Administrative transfers, allocations
  and exposures remain distinct, and publisher rounding is preserved.
- World Bank Pink Sheet observations retain all 71 `Monthly Prices` columns
  (70 prices plus the publisher's natural-gas index) and all 16 `Monthly Indices`
  columns as separate monthly `WLD` series. The raw benchmark label, stable
  canonical ID, source unit, derived price/index basis, worksheet, family and
  curated-panel flag are in the generator/schema-versioned catalogue beside the
  archived workbook. Missing, spreadsheet-error and nonpositive cells remain
  missing; exact cells stay auditable in the XLSX and each quarantined nonpositive
  cell is listed in the catalogue quality manifest with its coordinate, month,
  benchmark and raw value. A refresh validates catalogue,
  start, cadence, history length and recency before publishing all 87 partitions
  atomically. These are spot/reference prices and indices, not futures returns.
  Inventory calls a commodity group ready only when the expected 87 canonical
  identities, 70/17 semantic split, positive values, history/recency and one
  common versioned workbook release label plus clock pair all qualify; stored
  but unqualified rows remain visible with explicit guard failures.
- The official-money catalogue retains ten source-native histories: US M2 and
  Federal Reserve total assets; euro-area M3 and Eurosystem total assets;
  Swedish M3; monthly UK M4ex, its native quarterly historical bridge and
  headline M4; and Japan M3 and broadly-defined liquidity `L`. Monthly M4ex is
  the primary UK series. The quarterly M4ex bridge has the same perimeter,
  extends that history to 1997-Q4 and overlaps the monthly series at equal
  quarter ends; it is explicitly diagnostic and non-additive. Japan's `L` is
  Japan-specific, not a universal M5. Weekly balance-sheet levels are not
  silently resampled; currencies, M2/M3/M4 perimeters and central-bank assets
  are not treated as directly comparable or additive. Each release's vintage
  label records the deterministic catalogue-semantics SHA-256 exposed by the
  JSON inventory. Each parser-validated response body and canonical ledger of
  publisher-null, blank or absent native cells is content-addressed; an
  immutable manifest binds their full hashes to the release, and inventory
  re-hashes the files before counting the series ready. Source-specific start,
  cadence, minimum-row and latest-lag
  checks reject partial or stale responses before they can replace current
  history. The read-only inventory repeats those checks and requires a matching
  latest release hash before counting a stored series as ready. ECB's native
  weekly key contains an ISO week, not the exact reporting day; stored Fridays
  are consistent end-of-week representatives, with holiday and quarter-end
  exceptions remaining an explicit limitation.
- The separate liquidity frontier has 22 non-additive native histories. BIS
  contributes three quarterly USD/EUR/JPY credit stocks covering bank loans and
  international debt securities owed by non-bank borrowers outside each issuing
  currency area. OFR contributes eleven monthly US MMF asset totals/components
  and eight daily rates, outstanding volumes or transaction volumes in selected
  DVP, GCF and tri-party repo venues. MMF holdings expose only the published
  asset/counterparty categories; OFR's aggregate venue data does not identify
  cash-provider, dealer or ultimate-borrower sectors and cannot prove why funds
  moved. Totals/components, overlapping MMF/venue quantities, stocks, transaction
  volumes and rates must not be summed.
- Liquidity-frontier ingestion archives provider semantic catalogues and exact
  validated source responses by full-file SHA-256. OFR releases additionally
  retain canonical per-series native payloads and missingness ledgers; nulls and
  confidentiality edits remain missing rather than becoming zero. Immutable
  `data_release_artifacts` rows bind the source response and per-series evidence
  files, full hashes and provenance to each release; the catalogue-semantic hash
  separately binds the release to the archived interpretation. Inventory opens
  and rehashes the retained evidence files and repeats
  identity, start, minimum-history, recency, numeric, cadence and contraction
  checks. Complete and sparse monthly policies remain distinct; daily repo
  series also have source-specific internal-gap, weekday-coverage and same-venue
  date-alignment guards. Readiness proves evidence integrity, not additivity,
  causation or investment-signal validity.
- The report ledger holds ten official PDFs and complete, extractor-versioned
  text for all 852 physical pages. It currently contains no human-verified
  claims; raw pages are not approved conclusions.

## Report-claim review packet

ADR 0010 defines the read-only `dalio-report-review-packet` command. It selects
the latest eligible issue from each of the five pinned families—Riksbank MPR,
ECB/Eurosystem projections, Federal Reserve MPR, IMF WEO and BIS Annual Economic
Report—at an explicit known-at cutoff, then requires that issue's declared
extraction to be complete. It does not silently choose an older issue merely
because that issue already has candidate text.

The checked-in `reference/report_claim_candidates.json` catalogue holds four
model-draft candidates per selected issue—twenty in the initial packet—and the
contract permits no more. “Checked” means schema, identity,
artifact/extraction/page hash, physical-page and exact-excerpt validation. It
does not mean human
semantic approval. A catalogue that targets a stale issue, exceeds the limit or
does not match the stored extraction must fail visibly.

Every JSON candidate and every Markdown candidate carries the exact label
`UNVERIFIED MODEL DRAFT`. The builder writes only generated
`review/report_claims_latest.{json,md}` aliases and
`review/report_claims_YYYY-MM-DD_<packet-hash-prefix>.{json,md}` copies, where
the prefix is the first 16 lowercase hexadecimal characters of the full packet
SHA-256. The `review/` directory is ignored as generated output. Packets are
derivatives for human inspection, not preserved source artifacts, verified
claims, scores or scenario inputs.

The packet builder itself is database-read-only. A separate write path
must require a real, named human to choose `approve`, `revise` or `reject`.
Approval creates a verified record only after semantic review; revision retains
the original model draft and creates review lineage; rejection retains the
decision and creates no verified claim. Models cannot choose an outcome, supply
a human identity or approve their own drafts.

## Derived liquidity diagnostics

`dalio-liquidity-brief` is a read-only database consumer. It selects complete
releases at `as_known_at`, caps their observations at `as_of`, verifies the
release-bound evidence manifests and then writes versioned JSON plus a
plain-language Markdown view. The JSON includes coverage and unavailable states,
methodology and catalogue hashes, input series and dates, release IDs/clocks,
`earliest_input_available_at`, `complete_snapshot_available_at` and artifact
hashes. The FRED EFFR input is explicitly marked as a legacy policy benchmark
without a required raw manifest; it is not counted as a validated liquidity
artifact. The command does not write derived observations back to `dalio.db`.

Five calculation families remain independent:

| Family | Calculation | Interpretation boundary |
|---|---|---|
| Broad-money impulse | `100 * ln(x[t] / x[t-12m])`; three-month change in annual growth for US M2, euro-area M3, Swedish M3, monthly UK M4ex and Japan `L` | Equal-country median and breadth are emitted only when all five readings are ready at one common period; levels and currencies are not summed |
| Money / central-bank divergence | Broad-money annual log growth minus central-bank-assets annual log growth for US and euro area; weekly assets use the latest observation on or before month end | A growth gap does not establish monetary transmission or causality |
| US MMF relative expansion and asset allocation | MMF annual log growth minus M2 annual log growth; `100 * component / declared parent` and twelve-month asset-share changes; separately, OFR-published repo counterparty/clearing-category ratios | MMF positions are stocks and relative expansion is not a measured bank-deposit-to-MMF flow. FICC is a clearing category, not an ultimate borrower; the ratios do not identify an end-to-end cash map and are non-additive |
| Repo pricing and activity | Five-aligned-business-day median venue premiums to EFFR and max-minus-min venue rates in basis points; robust z = `0.67448975 * (current - median(previous 252)) / MAD(previous 252)` | Selected venues and transaction mix are not the whole market, a pure credit spread or an end-to-end cash path; volume has no automatic risk direction |
| Offshore reserve-currency credit | `100 * ln(x[t] / x[t-4q])` and one-quarter change in annual growth, separately for USD, EUR and JPY | Currencies remain separate; negative growth is arithmetic contraction and a negative change is deceleration, neither a claim of financial tightness nor an investment signal |

The default run fails closed if any family is incomplete. `--allow-partial` is
available for an explicitly incomplete diagnostic and does not impute missing
values. Nonpositive stocks, non-finite values, missing exact calendar lags,
stale observations, insufficient aligned history or invalid evidence manifests
remain unavailable; publisher gaps are not zero-filled or forward-filled.

The horizon labels are deliberately modest: repo/MMF conditions inform
0–12-month market-plumbing context, while broad money, central-bank assets, MMF
and offshore credit inform a 1–3-year monetary backdrop. The 3–5-year-and-longer
outlook still needs debt, fiscal, pension, demographic, productivity and
human-reviewed report evidence. There is no global liquidity, risk, M5 or
investment score, deposit-flow claim, causal “A to B because X” claim, forecast
or portfolio instruction in these files.

## Point-in-time meaning

`date` or a report period answers *what period does this fact describe?* It does
not answer *when could an investor have known it?* Every immutable release also
has UTC clocks:

- `published_at` when an official, reliable publication timestamp is exposed;
- `available_at` when the release may enter an as-known-at view;
- `retrieved_at` when this system obtained it.

If a source does not expose a trustworthy publication clock, `available_at` is
set conservatively to retrieval time. Point-in-time queries choose the newest
complete release for each partition at the requested `as_known_at` instant
**before** applying holder, counterpart, instrument or other cell filters. This
prevents an omitted cell in a newer vintage from being silently resurrected from
an older release. Data predating the explicit release-ledger cutover is only as
honest as its documented bootstrap availability time; original historical
publication vintages cannot be reconstructed from a latest-value database.
The derived brief exposes two different cutover fields. For its selected release
set, `earliest_input_available_at` is the minimum `available_at` among the
non-policy official-money/frontier inputs; it marks the start of partial raw
coverage, not a complete replay. `complete_snapshot_available_at` is the maximum
of those liquidity clocks only when every exact pinned analysis partition,
including the FRED EFFR benchmark, is present; otherwise it is `null`. In the
validated live initial set the values are
`2026-09-08T20:38:47.888222+00:00` and
`2026-09-08T20:43:28.624056+00:00`, respectively. The second instant is the live
full-snapshot cutover. Older observation dates inside the selected releases are
labelled `current_vintage_history` and must not be presented as an
as-known-at-the-time backtest. Future refreshes enable genuine complete-release
replay only from the full cutover forward.

## Refresh and local-PDF ingestion

Run from the repository root with the virtual environment active:

```bash
python -m dalio.pipelines.fetch_sovereign_debt
python -m dalio.pipelines.fetch_debt_holders
python -m dalio.pipelines.fetch_flows
python -m dalio.pipelines.fetch_positions
python -m dalio.pipelines.fetch_commodities
python -m dalio.pipelines.fetch_money_liquidity
python -m dalio.pipelines.fetch_shadow_liquidity
python scripts/audit_observatory.py --db data/dalio.db
dalio-liquidity-brief --db data/dalio.db
dalio-report-review-packet --db data/dalio.db
```

The liquidity-brief command reads the database in SQLite read-only mode and
refreshes the fixed `data/snapshots/liquidity_latest.{json,md}` aliases plus
content-addressed
`liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.{json,md}` files. For an explicit
reproducible cutoff, supply both clocks, for example:

```bash
dalio-liquidity-brief --db data/dalio.db --as-of 2026-09-08 --known-at 2026-09-08T23:59:59Z
```

`--through-date` is an alias for `--as-of`, and `--as-known-at` is an alias for
`--known-at`. The dated filename includes the first 16 characters of the full
snapshot hash, so different contents for the same economic date do not overwrite
one another. The fixed `latest` aliases do move on each successful run. The
immutable input releases and source artifacts remain the source-evidence audit
record.

The report-review command is also read-only. It validates the checked candidate
catalogue against the latest eligible documents, archived PDF bytes, complete
extractions and exact page excerpts before refreshing the fixed and hash-addressed
files under `data/review/`. Its public-information cutoff is versioned in the
catalogue rather than inferred from the run time.

`fetch_commodities --allow-contraction`,
`fetch_money_liquidity --allow-contraction` and
`fetch_shadow_liquidity --allow-contraction` are exceptional recovery options,
not routine refresh flags. Use one only after verifying that the official
publisher intentionally removed observations or applied a confidentiality edit;
the default rejects a shorter replacement snapshot.

Those network pipelines use first-party official APIs. Optional country and
cache flags are shown by appending `--help` to a command.

The PDF pipelines intentionally do not download anything. For a fresh
ingestion, first place the official files in a staging directory using the
deterministic filenames declared in the matching `reference/*.json`, then run:

```bash
python -m dalio.pipelines.ingest_ap_funds --artifact-dir /path/to/verified-ap-pdfs
python -m dalio.pipelines.ingest_reports --artifact-dir /path/to/verified-report-pdfs
```

Both commands preflight every required file and SHA-256 before creating any
release. Report ingestion also checks the complete physical-page count and
requires Poppler's `pdftotext`. Successful ingestion copies evidence into the
content-addressed archive; that archive is an output layout, not the expected
deterministic input layout.

## Known gaps

- None of the 852 report pages has yet become a named, human-verified atomic
  claim, so central-bank/IMF/BIS conclusions do not yet feed risk analysis. The
  bounded 20-item review packet has shipped, but every item remains an
  `UNVERIFIED MODEL DRAFT` and the human decision path has not shipped.
- AP2/AP3/AP4 currently provide one H1 2026 disclosure release each, not a
  comparable long-run allocator history; other pension and sovereign funds are
  absent.
- QPSD is voluntary and uneven. Missing combinations must remain missing.
- IMF PIP/DIP is voluntary and uneven. The current pull has all requested PIP
  partitions but no Saudi Arabian outward DIP total, equity or debt series;
  those absences are not zeros.
- Debt redemption calendars/cash-flow schedules and cross-border banking claims
  are not yet represented.
- Commodity history currently provides public spot/reference benchmarks, not
  futures curves, roll yields, collateral returns, tradable-index fees or an
  inflation/liquidity score. More columns must not become more voting weight.
- Official money now covers ten pinned native series, while the separate
  frontier covers 22 BIS/OFR histories. A stable machine-readable Swedish
  central-bank total-assets history is not yet verified. FSB non-bank aggregates,
  SEC fund-level filings, commercial paper, dealer financing, collateral reuse,
  FX swaps/cross-currency basis and stablecoins remain outside this layer.
  There is no synthetic universal M5 and no additive money/liquidity headline.
- The derived liquidity brief supplies separate 0–12-month and 1–3-year
  diagnostics, not a causal “money moved from A to B because X” graph. The
  1–5-year scenario, transmission and SEK small-investor portfolio layer remain
  analysis work. Stored transactions, positions and diagnostic co-movements are
  evidence for that work, not causal proof by themselves.
