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

Engine implementation through commit `32751f5` is integrated on `main`.
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
zero releases. All **1,307 tests passed**, and Ruff is clean.

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

## User-authorized continuation

After committing, pushing and completing this handoff, continue from collection
into **clear country assessments and scenarios that can later support sector
and company analysis**. Start from the verified numeric evidence already
available for the 19 listing countries. Record this analysis pivot in a new ADR
without declaring the remaining collection packages complete.

Each assessment should separate reported data/estimates, publisher forecasts,
derived arithmetic, Observatory interpretation and conditional scenarios. Bind
every numeric conclusion to exact releases, periods and source artifacts.
State 1–5-year horizons, transmission mechanisms, observable signposts,
invalidators, evidence gaps and limits on confidence. Sector/company channels
are conditional research hypotheses until company exposures are verified.
Avoid invented probabilities, composite scores and automatic buy/sell verdicts.

The user explicitly prefers **first-hand sources whenever available**. This
is recorded in `CLAUDE.md`; official WB/IMF harmonized baselines retain their
underlying producer and dataset/vintage metadata. National originals should
deepen country-specific evidence without silently replacing definitions.
The ranking population and existing scoring methodology are unchanged.

Twenty institutional report drafts remain unverified; communication content
remains outside the acquired evidence. Numeric analysis can proceed without
inventing report reviews or communications clearance. See ADRs 0004, 0017,
0028 and 0029, and `project_context.md` for the current architecture.

## Working commands

```bash
source .venv/bin/activate
pytest -q
ruff check src tests
python -m dalio.storage.inventory --db data/dalio.db --json
```

Use an isolated feature worktree for the assessment slice, the canonical venv,
and `PYTHONPATH=src` when running its code. Tests use mocked sources. The new
assessment consumer should read the database without changing raw evidence.
