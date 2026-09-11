# Macro Atlas / dalio-machine handoff — 2026-09-11

## Identity and working locations

- **Macro Atlas 0.8.0** is the offline Windows desktop application. Its working
  tree is `/home/rosinco/workspace/dalio-atlas-desktop`, branch
  `feat/offline-atlas`.
- **dalio-machine**, described as **Macro History & Risk Observatory**, is the
  canonical macro evidence and analysis engine. Its main working tree is
  `/home/rosinco/workspace/dalio-machine`; origin is
  `https://github.com/Rosinco/dalio-machine.git`.
- The Börsdata research repository supplies saved company listings and
  financial histories. Listing geography is distinct from issuer domicile,
  revenue exposure and physical assets.

## Latest checkpoint — four-country Nordic monitoring

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
source evidence. The separate Macro Atlas desktop pack remains unchanged.

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

The older country v1 bundle is superseded by v2; do not promote it. Source
bytes and publication/reference/acquisition dates are retained separately.
Earlier historical observations represent the collected vintage, not evidence
of what this system knew before the ledger cutover.

## Desktop boundary

The installed Macro Atlas application has 19,140 listings across 10 sectors
and 94 branches, financial statements, market-cap histories and branch
comparisons. Its bundled macro snapshot is still **2026-09-08**. The new country
and native-debt evidence has not yet been exported into the desktop pack.
Standard balance-sheet tables work; detailed physical-asset inventories and
map markers are future work.

## Next development step

Extend the verified Nordic monitoring approach to the remaining listing countries
in bounded batches. Audit existing series first, then select original sources
with explicit native scopes, seasonal adjustment, units and publication clocks.
Do not force non-comparable country loan-rate or industry definitions into one
ranking. Retain named gaps when an original signal is unavailable.

The next candidate batch is **US, Germany and Canada**, in that order; a smaller
first tranche can stop at US and Germany. The saved universe contains 6,345 US,
6,340 German and 2,321 Canadian listings, together 15,006/19,140 (78.4%). These
are listing records, not deduplicated issuers or operating exposure; Germany's
large catalogue especially requires that distinction. Each country already has
19 verified annual WB/IMF histories and a country assessment. Audit those
histories and the substantial existing US/Canadian scalar data before collecting
gaps. Existing US DFF/DGS10 releases have no raw-artifact bindings; they cannot
be promoted as original-response-verified monitoring without honest new capture.
Germany's applicable ECB instrument must remain explicitly shared euro-area
policy. Retained harmonized German debt measures and saved US/German debt-office
entrypoints are separate funding-context leads, not proof that a national
four-signal monitoring contract is ready.

A later compatible capture can support a separate change-since-last-capture
comparison, distinguishing revised historical values from newly added periods.
The current report only calculates historical changes within one captured
vintage. Energy, lending standards, defaults, actual government funding
execution and company/customer geography remain additional research inputs.

Company implications still require actual revenue, asset, cost and financing
exposures. A later desktop export can present the profiles and monitoring in
Macro Atlas's sidebar; the bundled application data remains unchanged.

The user explicitly prefers **first-hand sources whenever available**. This
is recorded in `CLAUDE.md`; official WB/IMF harmonized baselines retain their
underlying producer and dataset/vintage metadata. National originals should
deepen country-specific evidence without silently replacing definitions.
The ranking population and existing scoring methodology are unchanged.

Twenty institutional report drafts remain unverified; communication content
remains outside the acquired evidence. Numeric analysis can proceed without
inventing report reviews or communications clearance. See ADRs 0004, 0017,
0028–0032, and `project_context.md` for the current architecture.

## Working commands

```bash
source .venv/bin/activate
pytest -q
ruff check src tests
python -m dalio.storage.inventory --db data/dalio.db --json
python -m dalio.pipelines.build_country_assessments --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T21:10:00+00:00
python -m dalio.pipelines.fetch_sweden_monitoring --artifact-root data/artifacts/sweden_monitoring
python -m dalio.pipelines.build_sweden_monitoring --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T22:10:11.307779+00:00
python -m dalio.pipelines.fetch_nordic_monitoring --artifact-root data/artifacts/nordic_monitoring
python -m dalio.pipelines.build_nordic_monitoring --db data/dalio.db --as-of 2026-09-11 --known-at 2026-09-11T06:01:15.044707+00:00
```

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
