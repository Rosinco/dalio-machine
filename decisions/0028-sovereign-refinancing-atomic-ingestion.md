# ADR 0028 — Atomic refinancing history ingestion and descriptive comparison

**Date:** 2026-09-10
**Status:** Accepted
**Extends:** ADR 0020; the 48-stream Version 1 denominator is unchanged.

## Context

The user deferred further Atlas company-comparison UI work and resumed
Macro/Dalio collection and analysis. ADR 0020 already supplies strict source
adapters for 31 harmonized histories. Its next required boundary is an atomic,
auditable ingestion, first in a non-live database.

## Decision

1. Match all 29 Eurostat and two ECB adapters exactly to the checked manifest.
   Preserve the explicit EA21_FIXED denominator entity / EA21 observation-code
   mapping. Store the full 65-character native ECB series ID. The compatible
   observation model now declares VARCHAR(128), matching release observations;
   existing SQLite VARCHAR(64) tables accept the full value without a rebuild.
2. Fetch the complete batch before opening or initializing its target database.
   Independently re-hash retained files and re-parse exact source bytes. Compare
   canonical values, dates, identities and statuses with the prepared frame.
3. Bind four artifact roles to every immutable release: source_response,
   native_payload, missingness_ledger and catalogue_manifest. The catalogue
   binds the full denominator, adapter definitions, dimensions, units, history
   floors and ingestion policies. Existing source files are never substituted
   silently when a hash differs.
4. Use the completed batch's retrieval time as conservative availability.
   Retain the publisher update separately when supplied; unknown ECB updates
   remain null. Neither an old reference year nor a publisher update backdates
   what this project knew. Native flags remain in evidence; the generic ledger
   lowercases status codes.
5. Preflight all replacement date sets against the immutable and current
   stores. A missing formerly finite period or retrograde release blocks the
   batch. Bind each partition's logical commit to one outer transaction, so a
   failure after the final inner commit rolls everything back.
6. Audit through read-only queries: re-hash and re-parse evidence, verify release
   metadata and content hashes, and compare immutable and current observations.
   A table, artifact or source-contract gap never counts as ready. The inventory
   reports 31 harmonized expected and 17 national-native planned within 48 total.
7. Promote the same audited staging evidence offline with `--from-db`. Retain
   exact source files and use the latest stored retrieval time if deduplicated
   partitions have different clocks. Make and verify an exact SQLite backup
   before the first live promotion, then check integrity, foreign keys and that
   every pre-existing row remains unchanged.
8. Generate a dated JSON/Markdown brief only from fully audited inputs. Compare
   the five countries at their latest common annual period. The calculated
   due-within-one-year share divides the matching Eurostat Y_LE1 and TOTAL
   residual-maturity debt/GDP ratios; it does not use QPSD or ECB denominators.
   Three-year changes require exact endpoints. Keep EA21 securities comparisons
   separate. No composite score or five-year funding forecast is assigned.

## First collection

The 2026-09-10 batch contains **697 finite observations** in **31 releases** with
**124 artifact bindings**. Eurostat contributes 297 annual values; ECB contributes
400 monthly values (2009-12 through 2026-07 for both series). The latest annual
comparison is 2025. Germany has selected histories from 1995; most FR/IT/ES
histories begin in 2020, and Swedish histories begin in 2021 or 2022. The full
native annual axis, including unreported earlier periods, is retained.

Collection availability is `2026-09-10T20:11:12.607054+00:00`. The ingestion
catalogue SHA-256 is
`93f38ac76f2bb7b03a88d9f5a7b171a850309ae25a4ff6fee2115f519bd2ea97`.
Staging evidence and audit files live under
`data/artifacts/debt_refinancing/runs/2026-09-10/`.

The working database was then refreshed from the exact audited staging
evidence without a second download. Its full observatory inventory equals the
staging inventory. The dated descriptive export is
`data/snapshots/refinancing_2026-09-10_651235cedddb.{json,md}`.

## Validation

- Full offline suite: **1,224 passed**; `ruff check src tests` is clean.
- Failure tests cover unavailable sources, altered source/native/missingness/
  catalogue files, forged frames and clocks, incomplete batches, historical
  contractions, projection tampering and rollback after the final inner commit.
- Verified offline staging promotion and unchanged-vintage deduplication;
  a valid revision appends one release while retaining the old snapshot.
- Legacy SQLite VARCHAR(64) and the updated model both preserve the full ECB ID.
- Both staging and live pass SQLite integrity and foreign-key checks. Set
  comparisons across **25 tables** confirm that every pre-existing row remains
  unchanged. Only the expected 697 current observations, 697 immutable
  observations, 31 releases and 124 artifact bindings were added.
- The exact pre-refresh backup is
  `data/backups/dalio-before-refinancing-2026-09-10.sqlite3`. Preserve it with the
  retained evidence and `stage/live-append-verification.json` audit files.

## Consequences and remaining boundary

The harmonized tranche is usable for descriptive comparisons. Readiness means
source-contract and evidence integrity, not timeliness for every future date,
sovereign safety, completeness of the whole debt package, or permission to
reuse unreviewed institutional report conclusions.

The next collection work is the **17 national-native streams**, starting with
Swedish debt composition, actual redemption schedules and funding plans to
explain the general-government comparison. Residual maturity, original maturity,
interest refixing/duration, currency hedges, issuer scope and plans versus
outturn must remain distinct. The annual Eurostat bucket and ECB scheduled
redemptions do not establish actual future gross financing needs. No Atlas UI,
institutional communication capture or human report-review state changed.
