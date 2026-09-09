# Data layout and evidence boundaries

This directory contains local working data for the Macro History & Risk
Observatory. Most generated data and binary source material is intentionally
gitignored; the small JSON catalogues in `reference/` are the reproducible
metadata/transcription inputs.

From Windows Explorer, the directory is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data`

The live local database is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\dalio.db`

The institutional-communications source-policy catalogue is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\src\dalio\communications\catalogue.py`

Checked manifests bind a full immutable catalogue snapshot by SHA-256. The
current catalogue may grow without repinning or reinterpreting older manifests;
database metadata writes also persist the bound snapshot's own hash, evaluation
clock and policy values. Unknown or future-dated snapshot bindings are rejected.
This compatibility mechanism changes no rights status and authorizes no
source-content collection.

The checked, metadata-only 2025 Fed/ECB pilot manifest is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_pilot_events.json`

Its generated human-readable rights-review packet is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_rights_latest.md`

with machine-readable companion
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_rights_latest.json`.
The immutable pair is
`communication_rights_2026-09-09_f841b5875835f9ef.{json,md}`. The full
canonical semantic manifest SHA-256 is
`91848b6f144e4f2c46e821c3e24cbcf67be7d101e5781f5819dd6bcd10c87670`, and
the canonical semantic packet self-hash is
`f841b5875835f9ef1d469c6484386b88d8f129fee72d1ff62e12cd59768cdd31`.
The checked manifest file SHA-256 is
`303cac84f313040c9bc7012bed82787dfc0e41cd02eddbbf81319a2866afa227`;
the generated JSON and Markdown file SHA-256 values are respectively
`158de96cb06f73b2b8d51b4668a8c15207f5f6f54e7adcc215541858187f0a6d`
and `450030aabe7c87117b32a78f4ec8a055a02137550149db9230b1511a75ed14b0`.
These are link and rights-review metadata only. They contain zero human rights
decisions and authorize no content capture.

The checked, metadata-only 2025 Bank of England MPR cohort is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_boe_2025_events.json`

Its generated representation inventory is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_latest.md`

with machine-readable companion
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_latest.json`.
The immutable pair is
`communication_metadata_2026-09-09_97c6116f1c0b94a7.{json,md}`. The full
semantic manifest SHA-256 is
`4bba6c8415de46718a5ae6906d0d09e9041f7be1903dbeef7be4f40223717eb2`, and
the inventory self-hash is
`97c6116f1c0b94a7d9f6b01c8bebbdff1bc13b84b0e760665b4af24b870cc230`.
The checked manifest file SHA-256 is
`e0da5469e750b4a656130b446f88acc9f204d3ce2d7c7cf8ec3ca2432f3eea2e`;
the generated latest JSON and Markdown file SHA-256 values are respectively
`e17cd292bfba8c509716e158964d5e021bb12d21f654bf84e0788dff42dc6073`
and `644b2fdac5932293fe5ac1fdf75281002d2289360de7b3aa1eb5176733430214`.
The output measures four transcript links, four official-page video locators
(three exact external-platform page URLs and one ID-only observation) and zero
verified exact caption tracks against the same four-event denominator. It
authorizes no content capture. The source pages used for link observation are
not retained, so these hashes protect the recorded metadata rather than a
historical copy of the changing live pages.

The checked, metadata-only 2025 Sveriges Riksbank cohort is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\reference\communication_riksbank_2025_events.json`

Its cohort-qualified representation inventory is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.md`

with machine-readable companion
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.json`.
The immutable pair is
`communication_metadata_2026-09-09_43fd289a37e28f6f.{json,md}`. The full
semantic manifest SHA-256 is
`34f610043e933f74ca71ae56a9d29129fc92e2dd1ea83cc932293581ac4a2468`, and
the inventory self-hash is
`43fd289a37e28f6f8345b43d52c056cb562f87412f5f2294fcf930dda2705f3a`.
The checked manifest file SHA-256 is
`db4ecb23519f7a70e3e58563bf4b9dc895fd0ff78c1ef4ca1f44e6dc5f601b1d`;
the generated JSON and Markdown file SHA-256 values are respectively
`8485f8cd23d740539837921e3699704c8675d1ab908a27a495b7ee966d07f491`
and `35d38c1d7982f603c24f3c3c11fc57c0e2024d4b3212451d37abb612137f9556`.
The output measures eight exact first-party Riksbanken Play replay-page
locators, eight exact official Swedish slide PDFs, zero verified exact
first-party transcripts and zero verified exact caption tracks against the
same closed eight-event denominator. No vendor/player locator is stored. This
metadata authorizes no content capture and causes no database write. Publishing
it does not move the unqualified BoE `communication_metadata_latest` aliases.

The reserved content-addressed root for future, rights-cleared communication
artifacts is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\communications`

That artifact directory is intentionally absent/empty in the first slice. A
catalogued link is not permission to archive or transcribe its content.

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

Blank human decision sheets use the separate Windows path pattern
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_decisions_YYYY-MM-DD_<packet-hash-prefix>.json`.
They are editable local work sheets, not evidence. Reviewer identity and review
time are deliberately absent from the file and are supplied only at the later
interactive write boundary.

The current untouched sheet is
`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\review\report_decisions_2026-09-09_20c687f1cb907c71.json`
(file SHA-256 `cb88e5f1676386fe99edf261b5d0522197efd6664ee7206664df2e7be7ce9780`):
all twenty outcomes and attestations are still null. Its canonical semantic
decision SHA-256 is
`5c15b3d6afa58b3055f4b868a754ba8688e9a867ad1d02c6a49d56f145e7f2b8`;
that fingerprint changes when verdict content changes and is what `apply` asks
the operator to confirm.

The durable liquidity-frontier artifact root is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\liquidity_frontier`

The durable official-money artifact root is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\artifacts\money_liquidity`

The exact pre-frontier database backup is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-before-liquidity-frontier-2026-09-08.db`

The verified post-frontier database backup is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-after-liquidity-frontier-2026-09-08.db`

The SQLite-native backup made immediately before installing the empty human
review ledger is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-before-report-decision-schema-2026-09-09.db`

It preserves the pre-schema source state whose database-file SHA-256 was
`4f9b2110d51a140d98695bd25692a4c7f3240e902496791ad3d67c745df080fb`.
SQLite's native backup can repack pages, so its own bytes differ; it passed
row-by-row schema/content, integrity and foreign-key verification and has
backup-file SHA-256
`04ee92e5b8eae30401e7a9466b28e5c9f82c8fe13434cad0629ace0006c7b0b4`.
The live database after adding only the empty immutable table and its guards is
SHA-256 `e483fe67b35a07d2d3ad031c3c6ad2fe8462a1baf64a48731f61126c255bcc74`.

The SQLite-native backup made immediately before installing the empty
institutional-communications schema is:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\backups\dalio-before-communications-schema-2026-09-09.db`

Its source database SHA-256 was the preceding
`e483fe67b35a07d2d3ad031c3c6ad2fe8462a1baf64a48731f61126c255bcc74`.
The native backup passed row-by-row schema/content, integrity and foreign-key
verification and has backup-file SHA-256
`7857fde8fae443029ec8c6a8427ca717c5d66b0f5f92691e84320f4c6ef71b93`.
After adding only the empty communications tables and guards, the live database
has SHA-256 `8c982d3449c335129d6ffc6906409d40c310959f2c8b016a23b39076cd9e63db`.
That was communications schema v1. Its pinned table/index fingerprint was
`1fe8247a31b767f933b3c6f8646ee8b2536da2655bc8fc087ef99fe5f68dedb3`
and its trigger fingerprint was
`62d2f2c2c9b152aeede8d4d4c203f05c24aba5697b311e534956053f6a315007`.

The controlled empty-ledger upgrade to communications schema v2 made this
adjacent, non-overwriting recovery backup before any schema mutation:

`\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-machine\data\dalio.db.before-communication-schema-v2.sqlite3`

It is a logically exact v1 copy with file SHA-256
`af56d59ba664330b25e0dc5d2337f22c7fc4d7b32271c11872eae5556b48d828`.
The v2 live database has SHA-256
`2fabc5a96d9757d23b249ebd86c8d1f734e38b53c529eaa77d2938e26b16684d`,
table/index fingerprint
`57ea8e7e6de787569aeaaa4e06dd419b4b6ccbbb78458eec2c8b92320d2c3e15`
and trigger fingerprint
`f86163665f9d827d83018423137e8fa38900163735ea8aef5c1587531696504c`.
All 21 unaffected tables matched the v1 backup by row count and bidirectional
SQLite `EXCEPT`; both files passed integrity and foreign-key checks. All
communications evidence/domain tables remain empty.

## Validated refresh inventory (2026-09-09)

The counts below are from the fully validated live `dalio.db` after upgrading
the empty institutional-communications schema to v2 on top of the empty
report-decision ledger. No communication evidence, review or claim rows were
added.
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
| Human report decisions | 0 decisions; 0 verified claims; blank review only |
| Institutional communications | 22 source policies / 19 organizations; 2025 Fed/ECB pilot 16/16 exact text links; 2025 BoE MPR cohort 4/4 events, 4/4 exact transcript links, 4/4 video locators (3 exact page URLs), 0/4 verified exact caption tracks; 2025 Riksbank cohort 8/8 events, 8/8 first-party replay pages, 8/8 official Swedish slide PDFs, 0/8 verified exact transcripts, 0/8 verified exact caption tracks; ordered one-artifact/one-byte-lineage, multi-section schema v2; 0 database events; 0 artifacts/scopes/content captures/extracted segments; 0 sources cleared and acquisition automation false for all |
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
| `reference/communication_pilot_events.json` | Closed 2025 Fed/ECB 8+8 event denominator, one exact first-party English text candidate per event, semantic section order, conservative metadata clocks and pending source-rights evidence | Link metadata only; no bytes, human rights decision or collection authority |
| `reference/communication_boe_2025_events.json` | Closed 2025 BoE four-event MPR press-conference denominator with separate transcript, official-page video and exact-caption observations | Link/platform-ID metadata only; no source-page archive, content bytes, rights decision or collection authority |
| `reference/communication_riksbank_2025_events.json` | Closed 2025 Riksbank eight-event monetary-policy press-conference denominator with separate Swedish transcript, replay-page, slide and exact-caption observations | First-party link metadata only; no source-page/PDF archive, vendor locator, content bytes, rights decision or collection authority |
| `artifacts/allocators/sha256/` | Content-addressed copies of official allocator PDFs | Durable evidence; do not treat as a disposable cache |
| `artifacts/reports/sha256/` | Content-addressed copies of official central-bank/IMF/BIS PDFs | Durable evidence; do not treat as a disposable cache |
| `artifacts/communications/` | Reserved content-addressed home for exact rights-cleared press-conference, earnings-communication, letter, caption and media artifacts | Empty by design until an exact artifact passes a documented rights review |
| `artifacts/worldbank_commodities/<hash-prefix>/` | Exact World Bank Pink Sheet XLSX vintages and deterministic native-series catalogues, addressed by workbook SHA-256 | Durable evidence; do not treat as a disposable cache |
| `artifacts/money_liquidity/` | Content-addressed validated official-money response bodies and per-series missingness ledgers | Durable release evidence; paths and full hashes are bound in `data_release_artifacts` and rechecked by inventory |
| `artifacts/liquidity_frontier/` | Content-addressed BIS/OFR responses, provider semantic catalogues and OFR per-series payload/missingness ledgers | Durable release evidence; paths and full hashes are bound in `data_release_artifacts` and rechecked by inventory |
| `cache/` | HTTP response cache used to reduce repeated source calls | Disposable, but a fresh rebuild then depends on the upstream source still serving the data |
| `snapshots/` | Generated exports consumed by the dashboard or downstream tools, including fixed `liquidity_latest.{json,md}` aliases and hash-addressed `liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.{json,md}` copies | Regenerable from the database and versioned calculation code; not source evidence |
| `review/` | Generated report packets, communication-rights packets, communication representation inventories and blank/editable packet-hash-bound human decision sheets | Regenerable or local review material; never source evidence, rights clearance or database truth before the appropriate gate |
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
- Institutional communications use a sibling ledger rather than the bounded
  report manifest. Stable organizations, effective-dated commodity selection
  mappings, versioned events, representation-specific artifacts, retrieval-bound
  byte captures, atomic ordered section-scope sets, reproducible extractions and
  scope-bound speaker-addressable segments remain separate. One mixed published
  page remains one artifact version and one byte-capture lineage while its prepared
  remarks and Q&A retain different provenance. Host, historical
  publisher, transcriber/caption origin, rights state, publication/availability/
  retrieval clocks and page/paragraph/timecode locators are explicit. The live
  tables are empty: the 22-policy catalogue covers 19 organizations and is
  source-discovery policy, not a downloaded corpus or evidence that an archive
  is complete. The first closed pilot enumerates all eight 2025 Fed and eight
  2025 ECB regular policy press conferences plus one selected official text URL
  per event. Its 16/16 is link coverage, not corpus completeness or clearance.
  The checked BoE and Riksbank institution/year manifests add independent
  representation observations. Riksbank's closed eight-event denominator has
  eight exact first-party replay pages and eight exact official Swedish slide
  PDFs, while exact first-party transcripts and exact caption tracks remain
  unverified for all eight events. Replay pages and slides are not transcript
  or caption substitutes.
  Immutable link metadata can precede a rights-cleared byte capture;
  changed bytes at the same URL append another capture rather than rewriting the
  artifact. Inventory can rehash archived bytes and deterministic segment
  structures, but those checks do not prove transcript fidelity or trusted
  extractor execution; later text must pass a trusted-execution or named-human
  semantic review boundary before it supports analysis.
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

The packet builder itself remains database-read-only. ADR 0011 adds the separate
`dalio-report-review` workflow. `prepare` writes a blank packet-bound decision
sheet; `check` rebuilds and re-hashes every source input; and `status` compares
file progress with the append-only review ledger. Those three operations do not
write the database.

The CLI `apply` path accepts only a complete twenty-item sheet from a TTY. It
displays the outcome counts and every candidate-to-outcome assignment, then
prompts the operator for a `human:<id>` identity and requires the canonical
full-decision SHA-256 to be typed. It then makes an
exact verified SQLite backup under `backups/`, repeats validation and commits
the whole batch atomically. Approval creates a verified successor; revision
retains the original model draft, adds a human-authored replacement and verifies
that replacement; rejection retains the review and creates no verified claim.
The local identity is explicit attribution, not cryptographic authentication.
Models may create the blank template but may not fill decisions, provide the
identity, confirm the hash or run `apply`.

Before making that backup, a database-read-only preflight checks the immutable
review ledger. An exact replay returns the existing receipt without another
backup or database write; a partial or conflicting review also fails before a
backup. The write transaction repeats the same check for race safety.

For each completed entry, replace all four null attestations with explicit
booleans and add a short `review_note`. Use these exact shapes:

- `approve`: all four attestations `true`; `reason_code` and `revision` stay
  null.
- `revise`: all four attestations describe the final wording and are `true`;
  supply a complete non-identical semantic `revision` and one of
  `unsupported`, `misattributed`, `wrong_type`, `wrong_scope`,
  `wrong_period_or_unit`, `missing_condition`, `not_material`, `duplicate` or
  `other` as `reason_code`. Document, extraction and citations cannot change.
- `reject`: at least one attestation is `false`, the same bounded reason-code
  set and a note are required, and `revision` stays null.

If the evidence page or excerpt itself must change, reject the item and create a
new checked candidate instead of editing provenance in the decision sheet.

For `revise`, replace the entry's null value with an object containing exactly
these fields. Copy the unchanged semantics from the paired
`report_claims_...json` packet and edit only what the human reviewer means to
correct; dates are ISO `YYYY-MM-DD` strings or null, and numeric fields are
finite JSON numbers or null:

```json
"revision": {
  "claim_type": "forecast",
  "statement": "Complete corrected statement.",
  "topic_key": "monetary_policy",
  "geographies": ["SE"],
  "claim_series_key": null,
  "reference_start": null,
  "reference_end": null,
  "target_start": "2027-01-01",
  "target_end": "2027-12-31",
  "numeric_value": null,
  "lower_bound": null,
  "upper_bound": null,
  "unit": null,
  "condition_text": "Publisher-stated condition, or null"
}
```

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
dalio-report-review prepare --db data/dalio.db
dalio-communication-rights-packet
dalio-communication-metadata-inventory
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

The `dalio-report-review-packet` command is database-read-only. It validates the
checked candidate catalogue against the latest eligible documents, archived PDF
bytes, complete extractions and exact page excerpts before refreshing the fixed
and hash-addressed files under `data/review/`. Its public-information cutoff is
versioned in the catalogue rather than inferred from the run time.

`dalio-communication-rights-packet` and
`dalio-communication-metadata-inventory` are offline and database-free. The
second command dispatches only to a registered checked cohort. It defaults to
the BoE manifest and preserves
`review/communication_metadata_latest.{json,md}` as BoE aliases. Selecting
`--cohort riksbank_2025_monetary_policy_press_conferences` uses the pinned
Riksbank manifest and refreshes only
`review/communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.{json,md}`.
Both use generic hash-bound immutable filenames. An unknown cohort or a
manifest/cohort mismatch fails before output. Neither command reads
communication source content or grants acquisition authority.

After `prepare`, open the printed Windows path and complete all four
attestations plus one outcome for every candidate. Check and inspect progress
without writes:

```bash
dalio-report-review check --db data/dalio.db --decisions data/review/report_decisions_YYYY-MM-DD_PACKETHASH.json
dalio-report-review status --db data/dalio.db --decisions data/review/report_decisions_YYYY-MM-DD_PACKETHASH.json
```

Only the actual human reviewer should then run `dalio-report-review apply` with
the same `--decisions` path. The CLI refuses non-TTY invocation and absent
candidates. The TTY check is operator confirmation, not identity authentication;
the prohibition against model or automated review is an explicit operating
policy. Do not ask a model to complete or apply the sheet.

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

- Institutional communication history has not yet been collected. The 22-policy,
  19-organization catalogue covers a balanced first-party discovery universe,
  and the first closed denominator now covers the 8+8 regular 2025 Fed/ECB
  policy press conferences with 16 exact official text links. A separate closed
  BoE 2025 cohort adds four MPR events, four exact transcript links, four video
  locators and zero verified exact caption tracks. The closed 2025 Riksbank
  cohort adds eight first-party replay pages and eight official Swedish slide
  PDFs, but zero verified exact first-party transcripts and zero verified exact
  caption tracks. These are only metadata inventories: no source authorizes
  automated content acquisition, no human
  rights decision exists, no transcript/subtitle/letter bytes or segments exist,
  and no communication text can feed scenarios or investor conclusions. The
  one-artifact/one-byte-lineage, multi-section storage mapping now exists, but no named-human
  clearance has been recorded. A hash-bound rights-decision contract and an
  explicit clearance are still required before any ingestion. Until then, safe
  expansion is metadata-only: close additional year denominators and inventory
  exact first-party representation links without downloading their content.
- None of the 852 report pages has yet become a named, human-verified atomic
  claim, so central-bank/IMF/BIS conclusions do not yet feed risk analysis. The
  bounded 20-item review packet has shipped, but every item remains an
  `UNVERIFIED MODEL DRAFT`. The guarded decision path has shipped, but no human
  has completed or applied the sheet.
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
