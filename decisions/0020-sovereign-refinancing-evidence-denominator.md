# ADR 0020 — Sovereign refinancing evidence denominator and source-ready checkpoint

**Date:** 2026-09-09 · **Status:** accepted · **Checkpoint:** source-ready, not ingested

## Context

The bounded Version 1 gate in ADR 0017 names core-economy debt maturity and
refinancing structure as the next evidence package. Existing QPSD ratios describe
useful debt anatomy, but they do not provide contractual redemption calendars,
issuer funding plans or a comparable maturity wall. This package therefore needs
a fixed denominator and explicit scope boundaries before database ingestion or
derived refinancing-risk measures begin.

## Decision

### 1. Fix the Version 1 denominator at 48 logical streams

The checked `data/reference/sovereign_refinancing_v1.json` manifest defines the
complete Version 1 package:

| Phase | Streams | Current state |
|---|---:|---|
| Harmonized scalar | 31 | Eurostat and ECB source adapters implemented and offline-tested |
| National native | 17 | Planned; no adapter, pipeline or ingestion yet |
| **Total** | **48** | Fixed checked denominator |

The core sovereign issuers are BR, CN, DE, FR, IN, IT, JP, SE, UK and US. Spain
is retained as a sovereign comparison and fixed-composition EA21 as an aggregate
comparison. The euro area is not represented as a sovereign issuer.

### 2. Make the first 31 partitions source-ready without calling them collected

The Eurostat adapter defines 29 annual general-government partitions for DE,
FR, IT, ES and SE: average residual maturity, its matching debt denominator,
debt due within one year, foreign-currency debt and apparent cost, plus
long-term variable-rate debt for DE, FR, IT and ES. Sweden's unavailable
variable-rate partition is excluded from the denominator rather than recorded
as zero.

The ECB adapter defines two monthly non-consolidated general-government
securities histories for fixed-composition EA21: average residual maturity and
redemptions due in the next one to twelve months as a share of GDP.

Both adapters enforce exact source identity, dimensions, schema, chronology,
cadence, numeric cells and missingness. Successful validation produces
content-addressed source-response, native-series and missingness evidence.
Saved official-response checks found 31 annual observations for the Eurostat DE
example from 1995 through 2025 and 200 monthly observations in each ECB history
from 2009-12 through 2026-07. These checks establish adapter readiness only;
they are not live-database coverage.

### 3. Keep unlike debt concepts separate

Central-government, general-government, marketable-debt and
non-consolidated-security scopes remain distinct. Original maturity, residual
maturity, duration and time to refixing are not interchangeable. Face value,
market value and uplifted/index-linked values retain their publisher basis.
Calendar-year and fiscal-year observations, and plans, forecasts, estimates,
scheduled cash flows and outturns, remain separately labelled.

A current surviving-security inventory is never backcast as historical
issuance, and broad publisher buckets are never prorated into invented maturity
buckets. A harmonized ratio is not a substitute for a national issuer's exact
redemption schedule.

### 4. Stop this checkpoint before persistence or interpretation

There is no sovereign-refinancing pipeline, database ingestion, live database
mutation, release-bound artifact set, derived indicator, scenario input or risk
conclusion at this checkpoint. In particular, 31/48 means source-ready logical
coverage, not stored coverage and not completion of the debt package.

The next implementation must resolve any open review blocker, verify that the
65-character ECB native series identity is preserved safely in the compatible
`observations` projection, preflight all 31 partitions before an atomic release,
bind their content-addressed evidence, ingest into a non-live database first and
audit the result before any live refresh. The 17 national-native streams follow
with storage shapes suited to issuer schedules and funding plans rather than
being forced into scalar ratios.

## Consequences

The refinancing package now has a bounded and reviewable source contract. The
first scalar tranche can proceed to atomic ingestion without redefining the
question mid-pipeline, while the remaining national evidence and every current
storage gap remain visible.

No investor inference follows yet. Rollover pressure, funding peaks, cost gaps
and scenario signposts can be designed only after scope-matched releases are
stored and audited; they must not combine the EA21 comparator with sovereign
issuers or treat annual harmonized ratios as contractual cash-flow schedules.

## Rejected alternatives

- Treat the euro area as another sovereign issuer.
- Count Sweden's unavailable variable-rate series as zero or as a failed 32nd
  scalar partition.
- Call successful saved-response parsing a live ingestion.
- Derive a five-year maturity wall by prorating a one-year Eurostat bucket.
- Backcast today's surviving securities into historical redemption schedules.
- Force national redemption calendars and funding plans into one scalar schema.
- Generate refinancing scores or investor conclusions before atomic ingestion
  and release/artifact audit.
