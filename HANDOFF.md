# Macro Atlas / dalio-machine handoff — 2026-09-10

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

Deepen the specific national evidence needed to test the scenario signposts:
recent activity/orders, credit conditions, debt maturities and refinancing
costs, and sector-relevant energy exposure. Prefer the statistical office,
central bank and debt office for each country, retain their own scopes, and
join actual company exposures when available. A later desktop export can
present the country profiles and scenarios in Macro Atlas's sidebar. The first
assessment layer is complete; broader national inputs, numerical stress models
and full sector/company assessment remain future work.

The user explicitly prefers **first-hand sources whenever available**. This
is recorded in `CLAUDE.md`; official WB/IMF harmonized baselines retain their
underlying producer and dataset/vintage metadata. National originals should
deepen country-specific evidence without silently replacing definitions.
The ranking population and existing scoring methodology are unchanged.

Twenty institutional report drafts remain unverified; communication content
remains outside the acquired evidence. Numeric analysis can proceed without
inventing report reviews or communications clearance. See ADRs 0004, 0017,
0028–0030, and `project_context.md` for the current architecture.

## Working commands

```bash
source .venv/bin/activate
pytest -q
ruff check src tests
python -m dalio.storage.inventory --db data/dalio.db --json
python -m dalio.pipelines.build_country_assessments --db data/dalio.db --as-of 2026-09-10 --known-at 2026-09-10T21:10:00+00:00
```

Assessment worktree: `/home/rosinco/workspace/dalio-country-assessments`, branch
`feat/country-assessments`. Use the canonical venv and `PYTHONPATH=src` in a
feature worktree. Tests use mocked sources; actual assessment builds use
retained local bytes and open the source database read-only.
