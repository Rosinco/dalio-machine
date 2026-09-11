# Macro Atlas / dalio-machine handoff — 2026-09-11

## Identity and working locations

- **Macro Atlas 0.9.0** is the new offline Windows desktop integration. The Windows
  build and focused native financial check passed; the complete ordinary native
  suite and installation are pending after the pause. Its working
  tree is `/home/rosinco/workspace/dalio-atlas-desktop`, branch
  `feat/offline-atlas`.
- **dalio-machine**, described as **Macro History & Risk Observatory**, is the
  canonical macro evidence and analysis engine. Its main working tree is
  `/home/rosinco/workspace/dalio-machine`; origin is
  `https://github.com/Rosinco/dalio-machine.git`.
- The Börsdata research repository supplies saved company listings and
  financial histories. Listing geography is distinct from issuer domicile,
  revenue exposure and physical assets.

## Latest checkpoint — US/Germany/Canada and desktop evidence integration

Backend checkpoint **`34a229e`** is committed and pushed on canonical `main`.
The user paused work on September 11 and requested commit, push and handoff.
Desktop checkpoint: **`0152f88`**, pushed on `feat/offline-atlas`.
No further implementation or collection is planned during this pause.
ADR 0033 adds a separate twelve-attempt US/Germany/Canada supplement after a
retained-evidence audit. All **26 HTTP requests succeeded**: **11 original-source
histories** and one source-documented US corporate new-loan-rate gap. The retired
Federal Reserve E.2 survey is not replaced by a prime rate or credit-standards
proxy. Canadian industrial activity retains its native real industrial GDP
volume in millions of chained 2017 CAD, without artificial index rebasing.
Country industry scopes, borrower populations and policy instruments remain
explicit; no sector or company verdict follows from these signals.

Immutable bundle:
`1038458825e26cf7a6768bde5293534a72028234a2b927e068209acc8d9af1a3`.
The exact known-at cutoff is **2026-09-11T06:26:50.807464+00:00**, with UTC
assessment date **2026-09-11**. The canonical eight-file offline snapshot is
`62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46`
under `data/snapshots/national_monitoring/`. Open `index.md`, the `US.md`, `DE.md`
and `CA.md` reports, and their linked `context/<country>.md` annual profiles.
`snapshot.json` retains complete evidence and full-precision values.

Validation: **1,690 backend tests passed**. The full new-source audit reconciled
**5,182 native slots: 4,994 numeric and 188 null**, across all eleven histories.
Independent US/Canada and German decoders checked the original native vectors;
the combined audit checked calculations, citations and report links. Exact
offline replay preserved all **196 checked files**, including source and output
bytes and modification times. The source database is unchanged.

Macro Atlas **0.9.0** has a prepared independent country-evidence pack:
`156bcef284718b80a273ae8d32d03b6c2c6013c1891da5516f2e29430b61fdad`.
It contains **19 annual profiles**, **seven monitored countries** (SE, NO, DK,
FI, US, DE, CA), and **29 monitoring topics: 28 source histories plus the US
loan-rate gap**. Evidence assessment dates span **September 10–11**: twelve annual
profiles retain September 10, while the seven monitored countries use September
11 embedded assessments. These remain separate from the September 8
fundamentals/research release. Annual history, original Swedish
debt context, source references and conditional scenarios are inspectable offline.
**82 frontend tests, all 53 desktop Python tests (including eight new exporter
regressions) and nine focused browser tests passed**; Ruff is clean for the new
exporter and tests. The Windows native build succeeded, producing a 12 MB binary.
The independent desktop audit verified 19 annual profiles/152 annual histories,
seven monitored countries/28 source histories, 29,175 monitoring rows/415 nulls,
and the US structural gap while preserving 23 source/pack files. The complete
browser suite also passed all research, business, directory, financial, comparison
and nine evidence checks, with zero external requests or runtime errors. Windows
binary SHA-256 is `8e9c1dccaa3d55113fe4a097f65cc3668b76a7c07cdab661521797388d386f93`.
The focused native financial rerun passed on that same binary using the revised
uncompressed test transport in **55 seconds**, from
**2026-09-11T07:30:11.128Z** to **07:31:06 UTC**. The complete
**109,862,912-byte** financial pack passed hash verification, byte-identical
import/export, reload and process-restart persistence, with zero external requests
or runtime errors. The **complete ordinary Windows native suite and desktop
installation are pending for the next session**; 0.9.0 is not confirmed installed.
Retained failed native tests captured malformed incoming CDP JSON in
`Network.requestWillBeSent` frames, followed by an automation-client disconnect.
The imported financial file remained intact; an application crash has not been
established. The successful experiment changed both compression and the WebSocket
client path, so it does not isolate compression as the cause. The revised
transport is now the test default; `-CompressedCdp` retains the earlier diagnostic
reproduction option. No production financial-importer change was made.

## Earlier checkpoint — four-country Nordic monitoring

Implementation `1f1d57c` is integrated and pushed on `main` and
`feat/nordic-scenario-monitoring`. ADR 0032 extends Sweden's verified pilot to
Norway, Denmark and Finland. All **17 source signals have complete comparison
windows**: four common topics per country plus Swedish industrial orders.
Each country has a monitoring report linked to its annual IMF reference path,
structural evidence and conditional scenarios. No country ranking, composite
risk score, forecast probability or company verdict is introduced.

The retained-data audit first verified 57 NO/DK/FI annual histories, 2,603
observations and 309 artifact bindings, and inspected 233 caches plus 91 raw
machine-readable artifacts. The fixed monthly/daily panel required new original
inputs. Twelve new inputs were then captured through 24 successful HTTP requests
to SSB, Norges Bank, Danish Statbank, Statistics Finland, Bank of Finland and
ECB. Official distributors retain the underlying producer attribution.

Successful Nordic bundle:
`ac08d5e898d3ba13ab62861fb84d51c396f454de0bf488c752bc1d3a188dc9d2`.
Capture began **2026-09-11T06:00:06.705251+00:00**, outside SSB's update window,
and completed **2026-09-11T06:01:15.044707+00:00**. The report uses that exact
knowledge cutoff and **2026-09-11** as its UTC assessment date. Sweden retains
its independently acquired five-input bundle and original 2026-09-10 receipt
clock. The new supplement contains 12,131 native observation slots.

Canonical ten-file offline snapshot:
`289d3f3eb37afbba9d6f62758d91947d412ac3f7d12a74d72bb809851dbf6804`.
Open `data/snapshots/nordic_monitoring/<hash>/index.md` for the comparison,
`SE.md`, `NO.md`, `DK.md` and `FI.md` for monitoring, and `context/<country>.md`
for the linked annual assessment. `snapshot.json` retains full precision and
source evidence. At that checkpoint the separate Macro Atlas desktop pack was
unchanged; the 0.9.0 export above now includes this verified snapshot.

Source definitions matter. Norway's selected industrial index excludes oil/gas
extraction, related services and electricity but includes mining, quarrying and
petroleum-related manufacturing. Denmark includes oil/gas extraction; Finland
also includes energy. Finnish actual new drawdowns cover all currencies and
exclude housing corporations; the other selected lending rates use their
domestic currencies and different agreement/fee populations. Finnish policy is
the shared ECB deposit-facility rate. Denmark's government yield is monthly;
Norwegian and Finnish reference-yield construction differs. Their original
definitions, historical corrections, date-only updates and actual comparison
windows remain visible.

The observed industrial means rose in SE/NO/DK and fell in FI over May–July
versus February–April. Selected corporate lending rates rose in SE/NO/DK and
fell in FI from April to July. These are within-series historical changes in
one captured vintage. They neither establish a GDP forecast miss nor determine
a company's borrowing cost or performance. Funding and demand scenarios remain
conditional hypotheses requiring further evidence and actual company exposures.

Validation: **1,582 integrated tests passed**; Ruff and diff checks are clean.
The final independent audit reconciled **23,993 native scalar slots** (23,766
numeric and 227 missing), all **17 comparisons**, 76 annual histories/3,478
observations, seven native debt context cells and **217 citations**. All 181
displayed table values and 915 report links passed. The database, 239 protected
source files and ten output files retained their hashes, sizes and modification
times during the audit; SQLite integrity and foreign-key checks passed.
The exact post-publication CLI replay reproduced the same snapshot and preserved
all **251 checked files**, including `LATEST.json`. A separate final comparison
also preserved all 160 pre-collection inventory files, including the database
and desktop taxonomy. Independent Norwegian and Finnish yield decoders separately
verified 2,113 of those native slots and all five associated comparison/citation sets;
the Norwegian audit confirms zero overlap with SSB's maintenance window.

An earlier partial attempt
`82f98d14d713c13e2418109cfeb20b1e0aebbd7bed7769bbab40049058cd82c5`
is retained honestly. Two SSB inputs overlapped its update window; the Finnish
loan structure response exposed missing gzip decoding in the auxiliary curl
transport. The transport was corrected and independently checked, followed by
the complete fresh capture above. The earlier bundle and partial snapshot
`74ce2debd1686975b824a82b63b1b63910b72ff7d6d8ce391133ba6c81fb1fa8`
were never used to fill the final batch's inputs or published through `LATEST`.
No production source contract changed after implementation `1f1d57c`.

## Earlier collection checkpoint

The collection checkpoint and handoff were pushed on `main` at `548aa41`;
the separate desktop branch was pushed at `3d2f8dc`.
The completed country-assessment release and its handoff were pushed on `main`
and `feat/country-assessments` at `22c4b30`. Both branches and the desktop branch
were verified clean and synchronized before this documentation update.
The latest batch collected and verified 361 official country/source-series
histories across all 19 saved listing countries: 19 source series per country,
18 distinct metrics, and 15,616 batch observations. Nine countries previously
had no macro observations. Explicitly map listing GB to engine UK.

Eight original Riksgälden January–August 2026 PDFs and funding workbook 2026:1
add 1,595 typed native cells: 1,400 observed, 49 forecast and 146
missing/structural cells. The selected extract preserves security maturities,
debt composition, interest-rate refixing and financing plans with their native
scope and clocks. Contractual maturity, ATR, duration, monthly means and
reference-date values must remain distinct.

Verified live database totals:

| Evidence | Count |
|---|---:|
| Current scalar observations | 336,949 |
| Immutable scalar observations | 424,800 |
| Releases | 3,455 |
| Artifact bindings | 2,207 |
| Native debt cells | 1,595 |
| Tables | 26 |
| Fixed refinancing-package readiness | 32/48 streams |

The batch added 7,343 net current scalar observations. Every earlier immutable
row and every unrelated current observation was preserved. Combined staging
and live audits passed integrity/foreign-key checks; offline replay created
zero releases. The collection checkpoint passed all **1,307 tests**.

## Country assessment layer

Commit `9a8e1e2`, integrated on `main`, implements ADR 0030 and the user's
authorized analysis pivot. A read-only consumer
now builds 19 country profiles from the retained 361 histories and 15,616
observations. Each profile has five IMF baseline indicators for 2025, the
retained 2026–2031 path, three dated structural indicators, descriptive findings
and conditional scenarios. All 19 have the selected baseline and projection
cells; this is coverage of the defined profile, not completeness of macro
research or predictive confidence.

The release contains **73 conditional scenarios and 748 source-cell citations**,
including seven original Swedish debt-office values. Cases cover weaker demand,
tighter funding, stronger activity and energy-import pressure where supported
by recent historical exposure. Each names assumptions, transmission channels,
monitoring signposts, invalidating evidence and required company checks.
Revenue/customer geography, plants, costs, currencies, debt maturities, interest
terms and hedges must be verified before drawing sector or company conclusions.

IMF baseline rows are estimate/outturn; projection status uses the documented
calendar convention because native per-point actual/estimate cutoffs are not
supplied by this DataMapper collection. Missing years stay missing. WB context
retains its historical year. Swedish central-government currency amounts and
interest-rate refixing measures remain separate from general-government GDP
ratios and principal maturity. No probabilities, numerical stress forecasts,
composite scores or company verdicts were created.

Assessment date: **2026-09-10**. Exact known-at cutoff:
**2026-09-10T21:10:00+00:00**. Final snapshot:
`b2524c09385af9682a462be968e0eb20ccd5e5917e0e73d43b06949f93fd5144`.
Open its `index.md` for the country comparison and links to all 19 profiles.

Validation: **1,365 integrated tests passed**; all **58 assessment tests**
also passed after the final metadata wording correction. Ruff and diff checks
are clean. The final live audit reconciled all 748 citations with retained
source cells and passed SQLite integrity/foreign-key checks. An exact repeat
from canonical `main` preserved database, 775 protected source files and all
21 output files byte-for-byte, including modification times. Validation logs,
receipts and reproducible audit scripts are saved alongside the snapshots.

## Sweden monitoring pilot

The Sweden monitoring pilot in ADR 0031 now extends the country-assessment
layer. Implementation `3579eef` is integrated on `main`. The retained-data audit
checked all 158 Swedish scalar releases and
126 bound scalar artifacts; it distinguished unused existing data from the
specific gaps in industrial production, orders and corporate lending rates.
Legacy caches matched the stored histories but did not establish historical
raw-delivery provenance.

Five original SCB/Riksbank series are now captured with new, honest clocks:
adjusted industrial production and order indices, the SEK business lending rate
on new and renegotiated agreements, the effective policy rate and the ten-year
government benchmark yield. New scalar evidence stays in immutable supplements
and output JSON; this slice does not mutate the source database or scoring.
Existing original Riksgälden context is restored and shown separately.

Successful capture: `3cee14e94e7c3074f392247557b358b0edc9910e246b892dd15ab43dd39e6601`.
Its conservative availability is **2026-09-10T22:10:11.307779+00:00**. The report
uses **2026-09-10 as its UTC assessment date**; collection occurred after
midnight on the local Stockholm calendar. An earlier sandbox DNS-failed batch
is retained as a failed attempt and supplies no monitoring facts.

The first report has five comparable signals. Production and orders compare
May–July with February–April; the business lending rate compares July with
April; daily rates use their actual 90-day anchor dates. Original source
definitions, missingness, freshness and dataset-update clocks are preserved.
Historical same-vintage changes are not forecast errors or revisions since an
earlier capture. Scenarios remain hypotheses, with explicit limits and evidence
that would challenge their assumptions.

Canonical monitoring snapshot:
`a9b80f90572615358ce85c9244ec1b25691be2a3bb2fae3f74aaebf2f9ed7629`.
Open its `SE.md` under `data/snapshots/sweden_monitoring/`.

Validation: **1,449 integrated tests passed**, followed by **85 focused tests**
on the final monitoring code and presentation; Ruff and diff checks are clean.
The canonical source audit checked all **11,862 native scalar slots** (11,788
numeric and 74 missing), all five comparisons, 19 parent histories/875 annual
observations, seven native debt facts and **64 citations**. It also verified
all 52 displayed table values and exact report replay. SQLite integrity and
foreign-key checks pass. The database still matches its pre-collection hash,
size and modification time. Exact export replay preserves all 100 protected
source files and both immutable output files, including modification times.

## Durable local evidence

Database, downloaded source files and generated reports are intentionally
gitignored; pushing code does not transfer these local artifacts. Preserve
them together:

- `data/dalio.db`
- `data/backups/dalio-before-country-native-expansion-2026-09-10.sqlite3`
- `data/artifacts/company_country_macro/`, especially the authoritative v2
  bundle `bundles/81/8106b9fc2be2eea3ec1623927fddd752bd3e4db7c75b0a5eccf55eddf219c605.json`
- `data/artifacts/debt_refinancing/national/riksgalden/`
- `data/artifacts/debt_refinancing/runs/2026-09-10-country-expansion/`: staging,
  live and replay receipts, inventories, full test log and reproducible report
  scripts
- `data/snapshots/company-country-macro-2026-09-10-summary.{md,json}`
- `data/snapshots/sweden-native-debt-2026-09-10.{md,json}`
- `data/snapshots/company-country-first-hand-sources-2026-09-10.md`: discovery
  backlog, not proof that those national feeds have been ingested
- `data/snapshots/refinancing_2026-09-10_497a167a1895.{md,json}`
- `data/snapshots/country_assessments/LATEST.json`: pointer to the complete
  immutable assessment directory with `index.md`, `snapshot.json` and 19
  `countries/<code>.md` reports
- `data/snapshots/country_assessments/validation/`: source-cell audit,
  deterministic read-only rerun receipt, audit scripts and integrated test log
- `data/artifacts/sweden_monitoring/`: immutable five-signal attempt bundles
  and exact original SCB/Riksbank response bytes
- `data/snapshots/sweden_monitoring/LATEST.json`: pointer to the complete
  offline `SE.md` and `snapshot.json` monitoring export
- `data/snapshots/sweden_monitoring/validation/`: retained-data audit,
  source-discovery records, capture receipts, source-cell reconciliation,
  deterministic replay checks and test logs
- `data/artifacts/nordic_monitoring/`: immutable twelve-input attempt bundles
  and original SSB, Norges Bank, Danish Statbank, Statistics Finland, Bank of
  Finland and ECB response entities
- `data/snapshots/nordic_monitoring/`: immutable four-country comparison,
  four monitoring reports and four linked annual profiles; `validation/`
  contains the retained-data inventory, original transport receipts, capture
  summaries, independent source audits and exact-replay verification
- `data/artifacts/national_monitoring/`, especially
  `bundles/1038458825e26cf7a6768bde5293534a72028234a2b927e068209acc8d9af1a3.json`:
  complete US/DE/CA attempt bundle, original responses and discovery receipts
- `data/snapshots/national_monitoring/62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46/`:
  complete offline national comparison and linked annual profiles
- `data/snapshots/national_monitoring/validation/`: preserve the audit scripts,
  full test log, transport/capture receipts and these final verification receipts:
  `source-cell-audit-62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46-20260911T062735.json`,
  `independent-us-ca-62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46.json`,
  `independent-germany-62c94ffb.json`, and
  `reproducibility-62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46.json`
- `/home/rosinco/workspace/dalio-atlas-desktop/desktop/public/data/country-evidence/`:
  small `index.json` plus immutable hashed country files. The pack retains source
  identities and references; original response binaries stay in the backend archive.
- `/home/rosinco/workspace/dalio-atlas-desktop/desktop/test-results/country-evidence-validation/`:
  `desktop-pack-156bcef2-independent.json` and
  `audit_desktop_country_pack.py` retain the independent pack/parent/native-history
  audit and reproducible checks.
- `/home/rosinco/workspace/dalio-atlas-desktop/desktop/test-results/browser-report.json`:
  complete browser regression results, including the nine evidence checks.
- `/home/rosinco/workspace/dalio-atlas-desktop/desktop/test-results/windows-native-financial-report.json`:
  focused native financial import/reload/process-restart check; retain the
  `windows-native-diagnostics-*.json` files beside it, including the successful
  `2026-09-11T07-04-02.490Z` run and retained failed automation sessions.

The older country v1 bundle is superseded by v2; do not promote it. Source
bytes and publication/reference/acquisition dates are retained separately.
Earlier historical observations represent the collected vintage, not evidence
of what this system knew before the ledger cutover.

## Desktop boundary

Macro Atlas retains 19,140 listings across 10 sectors and 94 branches, financial
statements, market-cap histories and branch comparisons. Version 0.9.0 adds
**Macro → Assessments** with the verified country-evidence pack described above;
the export, Windows native build, complete browser suite and focused full-pack
native financial test passed. The complete ordinary Windows native suite and
desktop installation remain pending after the pause. The pack keeps each
assessment and monitoring cutoff visible,
including **2026-09-10–11** assessments, independently of the **2026-09-08** saved
fundamentals/research release. Newly covered countries do not inherit old scores
or map colours. Country files load on demand; the complete active pack is about
13 MB. Publisher websites need internet, but saved profiles and histories do not.
Standard balance-sheet tables work; detailed physical-asset inventories and
map markers are future work.

## Next development step

On resume, first run the complete ordinary Windows native suite with the new
default test transport, then install and verify the exact 0.9.0 executable if
those checks pass. Preserve the focused native financial receipt and earlier
transport diagnostics. The app is not claimed installed at this checkpoint.

Extend verified original-source monitoring to the remaining **twelve listing
countries: BE, CH, EE, ES, FR, GB, IT, LT, LV, NL, PL and PT**. Listing GB maps
explicitly to engine UK. All already have annual WB/IMF profiles. Select bounded
batches according to actual primary-source accessibility and suitable native
definitions, rather than listing counts alone. Audit existing series first, then
select original sources
with explicit native scopes, seasonal adjustment, units and publication clocks.
Do not force non-comparable country loan-rate or industry definitions into one
ranking. Retain named gaps when an original signal is unavailable.

Before the next German industrial refresh, migrate the Destatis short-term CSV
to an appropriate original GENESIS selection: the publisher will stop updating
the current CSV in **October 2026**. Preserve the selected German industrial
scope, adjustment and units, and verify the replacement against retained values.
The existing CSV remains valid evidence of its captured vintage, not a source of
new releases after retirement.

The completed US/Germany/Canada slice is the reference for honest original-source
collection, documented structural gaps and retained native scopes. Listing counts
remain listing records, not deduplicated issuers or operating exposure. Existing
legacy scalar rows without bound original responses cannot be retroactively
promoted as verified delivery evidence; acquire missing original bytes with honest
new receipt clocks. National debt-office evidence remains a separate funding
context with its own scope, units and publication dates.

A later compatible capture can support a separate change-since-last-capture
comparison, distinguishing revised historical values from newly added periods.
The current report only calculates historical changes within one captured
vintage. Energy, lending standards, defaults, actual government funding
execution and company/customer geography remain additional research inputs.

Company implications still require actual revenue, asset, cost and financing
exposures. The desktop integration presents evidence and conditional country
scenarios; sector/branch and company headwind/tailwind verdicts are not yet produced.

The user explicitly prefers **first-hand sources whenever available**. This
is recorded in `CLAUDE.md`; official WB/IMF harmonized baselines retain their
underlying producer and dataset/vintage metadata. National originals should
deepen country-specific evidence without silently replacing definitions.
The ranking population and existing scoring methodology are unchanged.

Twenty institutional report drafts remain unverified; communication content
remains outside the acquired evidence. Numeric analysis can proceed without
inventing report reviews or communications clearance. See ADRs 0004, 0017,
0028–0033, and `project_context.md` for the current architecture.

## Working commands

```bash
source .venv/bin/activate
export PYTHONPATH=src
pytest -q
ruff check src tests
python -m dalio.storage.inventory --db data/dalio.db --json
python -m dalio.pipelines.build_country_assessments --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T21:10:00+00:00
python -m dalio.pipelines.fetch_sweden_monitoring --artifact-root data/artifacts/sweden_monitoring
python -m dalio.pipelines.build_sweden_monitoring --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T22:10:11.307779+00:00
python -m dalio.pipelines.fetch_nordic_monitoring --artifact-root data/artifacts/nordic_monitoring
python -m dalio.pipelines.build_nordic_monitoring --db data/dalio.db --as-of 2026-09-11 --known-at 2026-09-11T06:01:15.044707+00:00
python -m dalio.pipelines.fetch_national_monitoring --artifact-root data/artifacts/national_monitoring
python -m dalio.pipelines.build_national_monitoring --db data/dalio.db --as-of 2026-09-11 --known-at 2026-09-11T06:26:50.807464+00:00 --bundle data/artifacts/national_monitoring/bundles/1038458825e26cf7a6768bde5293534a72028234a2b927e068209acc8d9af1a3.json --no-latest
```

The national collector makes fresh network requests and records a new complete
attempt batch. The exact-cutoff national build above replays retained evidence
offline and preserves `LATEST.json`. Run from canonical `dalio-machine` so bound
source paths retain the published snapshot identity.

To reproduce the desktop pack, run from
`/home/rosinco/workspace/dalio-atlas-desktop/desktop` with the canonical venv:

```bash
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
python scripts/export_country_evidence.py \
  --assessment /home/rosinco/workspace/dalio-machine/data/snapshots/country_assessments/b2524c09385af9682a462be968e0eb20ccd5e5917e0e73d43b06949f93fd5144/snapshot.json \
  --monitoring /home/rosinco/workspace/dalio-machine/data/snapshots/nordic_monitoring/289d3f3eb37afbba9d6f62758d91947d412ac3f7d12a74d72bb809851dbf6804/snapshot.json \
  --monitoring /home/rosinco/workspace/dalio-machine/data/snapshots/national_monitoring/62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46/snapshot.json
python scripts/export_country_evidence.py --verify
```

The exporter reads no database or network. It verifies source identities, selects
the newest complete country profile, and does not fill newer gaps from old batches.
Repeating an unchanged export preserves pack bytes and modification times.

Assessment worktree: `/home/rosinco/workspace/dalio-country-assessments`, branch
`feat/country-assessments`. Use the canonical venv and `PYTHONPATH=src` in a
feature worktree. Tests use mocked sources; actual assessment builds use
retained local bytes and open the source database read-only.

Monitoring worktree: `/home/rosinco/workspace/dalio-sweden-monitoring`, branch
`feat/sweden-scenario-monitoring`. The fetch command makes fresh network
requests and retains new acquisition clocks; the build command is offline.
For exact replay, use the captured cutoff above and preserve the associated
source bundle and response files. Source locations participate in snapshot
identity, so publication is finalized from canonical `main`.

Nordic worktree: `/home/rosinco/workspace/dalio-nordic-monitoring`, branch
`feat/nordic-scenario-monitoring`. Use the canonical venv and `PYTHONPATH=src`.
The Nordic collector records twelve new inputs in a complete attempt bundle;
the offline builder restores Sweden independently at the same knowledge cutoff.
Keep rejected attempts and their original clocks. SSB requests overlapping
05:00–08:00 Europe/Oslo are excluded because the publisher warns that temporary
update placeholders may appear. A fresh complete capture outside that window
is required; successful values are never patched in from an older attempt.

Routine implementation, read-only verification, original-source collection,
commit, push and handoff are authorized in the conversation. The user asked
for autonomous progress overnight without repeated permission questions.
Existing sandbox rules still apply; use already-approved command prefixes
correctly and do not treat elapsed time as approval.

The latest user instruction pauses implementation and collection after this
commit/push/handoff checkpoint. Resume the pending native verification,
installation and country expansion when the user returns and asks to continue.
