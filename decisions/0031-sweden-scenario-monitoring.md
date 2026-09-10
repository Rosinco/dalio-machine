# ADR 0031 — Sweden scenario monitoring from original evidence

Date: 2026-09-11
Status: accepted

## Scope and prior inventory

The user authorized the Sweden pilot recorded in the handoff. Connect a small
fixed set of scenario signposts to dated evidence before expanding monitoring
to the other 18 listing countries. This is an offline analysis consumer; it
does not update the desktop bundle, scoring populations or company verdicts.

The complete retained-Sweden audit found relevant GDP, employment, inflation,
rates, money, debt and funding evidence. An indicator outside ADR 0030's input
set is not necessarily missing from the database. The audit rechecked 158
Swedish scalar releases and 126 bound artifact records, and replayed the nine
original debt-office documents. Legacy SWEA and cycle caches match stored
histories but are not evidence of the original delivery or historical release
binding. Fresh capture must not be attributed to their earlier availability.

The fixed scalar selection is:

| Indicator | Original selection | Interpretation scope |
|---|---|---|
| `industrial_production` | SCB TAB1872 / B+C / NV0402AL | Mining and manufacturing, excluding energy; calendar and seasonally adjusted volume index, 2021=100 |
| `industrial_orders` | SCB TAB1710 / TOTALA / B+C / NV0501BD | Domestic and export industrial orders; adjusted constant-price index, 2021=100 |
| `corporate_new_lending_rate` | SCB TAB5780 / MFI / NFC / new agreements / fixation-period loans / 000004ZT | SEK new and renegotiated agreements, amount-weighted annualised agreed rate; includes floating-rate loans, excludes transaction-account balances |
| `policy_rate` | Riksbank SWEA SECBREPOEFF | Effective policy/repo rate |
| `yield_10y` | Riksbank SWEA SEGVB10YC | Secondary-market government ten-year benchmark yield |

SCB publishes interest statistics collected on behalf of Riksbank; both
attributions remain in metadata. Original SCB MIR instructions inform the
currency, agreement-period and rate definition. The current PDF was reviewed
through official browser extraction, but a direct download returned HTTP 404;
local retention of that PDF is not claimed. Numeric metadata/data responses
and the documentation locator are retained. This limitation is recorded in
the source-discovery receipt.

## Acquisition and evidence contract

A complete immutable supplement records all five attempted series, exact
official request URLs, HTTP outcomes, original response bytes, hashes and
request/response clocks. Successful data require the expected final URL without
redirects. Failed HTTP attempts and invalid source payloads remain named gaps.
SCB native dimensions, adjustment, units, update clocks, complete requested
periods and missing flags are checked against metadata. SWEA has exact native
series URLs, bounded dates and unambiguous finite/null values. Duplicate JSON
keys, inconsistent clocks and changed identities fail validation.

The batch-completion timestamp is conservative availability for every input.
The consumer selects the newest eligible whole batch before filtering facts;
a missing or failed signal never falls back to an older successful capture.
Raw corruption in the selected batch fails the build. Future captures cannot
provide facts to an earlier known-at report. Dataset-update clocks remain
distinct from separately established publication dates and acquisition clocks.

New numeric evidence is retained in the supplement and output JSON; this slice
does not ingest it into the scalar database or rewrite legacy releases. This
keeps provenance honest while providing an auditable monitoring consumer.
The source database is opened read-only. Legacy SWEA releases are rehashed
and exposed as inventory metadata only. Original national debt context and
the Sweden country assessment are independently restored through their
existing source-verification contracts at the same exact known-at cutoff.

## Comparisons and scenario interpretation

Rules are explicit before applying them to the capture:

- Industrial momentum is the percentage difference between two adjacent,
  complete three-calendar-month arithmetic means of positive adjusted volume
  indices. Require all six native months; never interpolate or annualise.
- The business lending-rate change is a percentage-point difference from the
  exact month three months earlier. Retain the new-and-renegotiated scope and
  the possibility of changes in borrower, loan and fixation-period composition.
- Daily rates use the latest observation against the last observation at or
  before the date 90 calendar days earlier, within seven days of that target.
  Display the actual comparison dates; do not silently fill a distant anchor.
- Freshness windows are ten days for daily rates and 75 days from monthly
  period end. These are editorial applicability rules, not publisher expiry
  dates or predictive-confidence estimates. Stale values remain visible, with
  current interpretation suppressed. A missing latest native value is not
  replaced by an older one.

Every directional statement describes the specific measured series and links
to the relevant conditional country scenario and evidence that would challenge
its assumptions. Industry indices do not establish an annual GDP forecast
error; a policy or government benchmark rate is not a company's borrowing
rate. Credit availability, lending standards, company defaults and actual
funding execution require separate evidence. No probability, composite vote,
causal conclusion, synthetic forecast or company outlook is calculated.

Original debt stock and refixing measures remain separate from principal
maturity and from annual funding forecasts. Forecasts are context, not realised
funding outcomes. Same-vintage historical differences are not revisions across
knowledge dates. This first capture has no compatible prior monitoring capture
from which to claim a change-since-last-capture result.

## Offline publication and completion

`fetch_sweden_monitoring` captures the fixed source supplement without opening
the database. `build_sweden_monitoring` verifies it and the upstream Sweden
assessment, then publishes `snapshot.json` and `SE.md` together in an immutable
hash-addressed directory. The hash covers full evidence and methodology,
including core/renderer code. `LATEST.json` changes only after complete
publication. Existing source paths and the artifacts namespace are protected
from report output, and an identical rerun preserves existing bytes and mtimes.

Tests use offline fixtures or mocked HTTP. Integrated tests, actual source-cell
reconciliation, independent interpretation review and a read-only deterministic
rerun are required for the completed checkpoint. The next stage is to extend
this verified monitoring contract to the other countries, with explicit native
definition differences and gaps, then join actual sector/company exposures.

### Verified checkpoint

Implementation `3579eef` is integrated on `main`. The complete original capture
`3cee14e94e7c3074f392247557b358b0edc9910e246b892dd15ab43dd39e6601`
finished at `2026-09-10T22:10:11.307779+00:00` (after local Stockholm midnight).
All five series succeeded. The UTC assessment date is 2026-09-10.

Canonical published snapshot:
`a9b80f90572615358ce85c9244ec1b25691be2a3bb2fae3f74aaebf2f9ed7629`.
The independent raw-cell audit reconciles 11,862 scalar slots (11,788 numeric,
74 explicitly missing), all five comparisons, seven native debt facts, 64
citations and 52 displayed table values. It revalidates the 19 parent source
histories with 875 annual observations and exact renderer output. SQLite
integrity and foreign-key checks pass; the database hash/size/mtime matches the
pre-collection inventory. An exact export rerun preserves 100 protected source
files and both output files in bytes and modification time.

The integrated suite passed 1,449 tests; all 85 monitoring/source/export tests
passed again after the final presentation corrections. Ruff and diff checks
are clean. Audit scripts, receipts, raw discovery evidence, capture outcomes
and logs are retained under `data/snapshots/sweden_monitoring/validation/`.
