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

## Completed checkpoint

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

Extend the verified Sweden monitoring approach to the other listing countries
in bounded batches. Audit existing series first, then select original sources
with explicit native scopes, seasonal adjustment, units and publication clocks.
Do not force non-comparable country loan-rate or industry definitions into one
ranking. Retain named gaps when an original signal is unavailable.

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
0028–0031, and `project_context.md` for the current architecture.

## Working commands

```bash
source .venv/bin/activate
pytest -q
ruff check src tests
python -m dalio.storage.inventory --db data/dalio.db --json
python -m dalio.pipelines.build_country_assessments --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T21:10:00+00:00
python -m dalio.pipelines.fetch_sweden_monitoring --artifact-root data/artifacts/sweden_monitoring
python -m dalio.pipelines.build_sweden_monitoring --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T22:10:11.307779+00:00
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
