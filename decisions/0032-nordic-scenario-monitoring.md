# ADR 0032 — Nordic country monitoring from original sources

Date: 2026-09-11
Status: accepted

## Scope and retained-data audit

Extend the completed Sweden pilot to Norway, Denmark and Finland, then publish
a four-country comparison. The common panel contains industrial production,
the applicable corporate new-lending rate, a named policy instrument and a
ten-year government reference yield. Sweden's order intake remains a fifth,
supplemental signal. Native definitions need not be identical; the comparison
does not rank countries or manufacture harmonized series.

The pre-collection audit verified 57 retained WB/IMF annual histories, 2,603
observations and 309 artifact bindings across the three new countries. It also
inspected 233 caches and 91 machine-readable raw artifacts. Unused annual OEC
economic-complexity history exists, but neither it nor the annual WB/IMF inputs
supplies the four monthly/daily monitoring measures. Existing EU-labelled
FRED/BIS ECB policy histories lack bound raw deliveries and cannot be silently
relabeled as original Finnish policy evidence. The source database was not
changed. The complete reproducible inventory is retained under
`data/snapshots/nordic_monitoring/validation/`.

## Native selections

| Country | Industrial production | Corporate lending | Policy instrument | Government reference yield |
|---|---|---|---|---|
| SE | Existing SCB TAB1872 B+C, excluding energy | Existing SCB TAB5780 new and renegotiated SEK agreements | Riksbank effective policy rate | SWEA ten-year benchmark |
| NO | SSB 07095/P103, excluding extraction and electricity | SSB 10729/02/03, new NOK repayment loans to NFCs | Norges Bank IR/B.KPRA.SD.R | GOVT_GENERIC_RATES/B.10Y.GBON, nearest-maturity security |
| DK | IPOP21/BC, current DB25 classification | DNRNUPI, NFC new DKK lending, excluding repos | DNRENTD/OIBNAA, certificates of deposit | MPK3/5500701004, monthly ten-year redemption yield |
| FI | Statistics Finland 14mh, total B+C+D including energy | Bank of Finland MFI_PUBL, actual new drawdowns to domestic NFCs, all currencies | ECB deposit-facility rate, euro-area scope | Bank of Finland daily ten-year reference report, underlying LSEG data |

Danish IPOP21 is the current DB25 table; the discontinued DB07 table is not
spliced into it. MPK3 documents an August 2026 correction of previously
published January 2021–June 2026 yield figures. The corrected vintage and this
notice are retained; it is not presented as a newly measured economic change.
Norwegian generic yields use a nearest-maturity security's closing mid-yield;
the Finnish reference report describes primary-dealer selling-price inputs at
13:00. Neither is silently treated as an identical instrument or auction cost.
The Finnish lending selection excludes housing corporations, overdrafts, card
credit and non-recourse factoring; it covers all currencies together. The
Swedish, Norwegian and Danish selections use their respective domestic
currencies. Actual drawdowns and new/renegotiated agreements retain different
transaction populations; comparisons describe each series' own history.
Denmark's B+C includes oil/gas extraction, unlike the selected Norwegian P103
aggregate; Finland's B+C+D additionally includes energy. Danish and Finnish
production measures can include activity abroad under their national
statistical definitions. They cannot establish output at domestic plant
locations or identify a company's actual physical footprint.

## Source contract and acquisition

The new `nordic-monitoring-v1` supplement preserves the separate, unchanged
`sweden-monitoring-v1` meaning. Twelve fixed country/signal attempts form a
complete batch. Original GET/POST destinations, selected JSON requests,
response bytes, SHA-256 hashes, HTTP outcomes, redirect evidence and UTC
request/response clocks are retained. Batch completion is conservative
availability, not a claimed publisher first-release timestamp.

Pure national adapters check exact native dimensions, units, frequencies,
periods, missingness and supplied metadata. Each complete source specification
is hashed into the acquisition envelope and reconstructed during replay.
Changes to the fixed interpretation must not silently reinterpret old bundles.
HTTP failures and malformed original payloads become named country/signal gaps;
corrupted retained bytes or altered request identities fail the build.

The latest eligible whole Nordic batch is selected before examining successes.
A failed or omitted signal cannot fall back to an older successful batch.
Future raw captures supply no earlier facts. Sweden is restored independently
from its latest eligible original supplement, preserving its earlier receipt
clock; all four annual country assessments use the same exact known-at cutoff.

SSB warns that tables can temporarily contain zero/dot placeholders during
05:00–08:00 Europe/Oslo updates. Retained SSB request intervals overlapping that
window are ineligible for calculation, including across daylight-saving changes.
Their original bytes and the reason remain visible in the acquisition ledger.

## Interpretation

Industry momentum requires two adjacent complete three-calendar-month means of
positive seasonally adjusted volume indices. Rates use percentage-point changes
from the exact month three months earlier, or a daily observation at/before the
90-day target within seven days. Monthly official yields remain monthly, with
their actual comparison periods. No interpolation, annualisation or frequency
conversion is performed. Latest native nulls stay null; stale values remain
visible with current directional interpretation suppressed. The editorial
freshness limits remain ten days for daily and 75 days for monthly observations.

Policy instruments, currencies, loan populations, fee treatment, industrial
coverage, yield construction and reference periods differ across countries.
Those definitions are displayed with the measured changes. Finland's ECB policy
context is a euro-area instrument, not an independent Finnish policy decision.
Norwegian industry excludes extraction but still includes petroleum-related
manufacturing. A secondary-market yield is not a company's borrowing cost or
government auction execution. A new-business loan rate does not measure credit
access, and a changing loan mix can change the average rate.

Each signal links to the existing conditional scenario and evidence that would
challenge it. These are dated observations and explicit hypotheses, not
probabilities, a composite risk score, a forecast revision or a company verdict.
Current-vintage historical changes do not establish what was known at earlier
reference dates. Listing country is not revenue, asset, cost or financing
exposure. Annual IMF paths retain the existing documented calendar convention;
native debt stocks, refixing measures and funding forecasts remain separate.

## Offline publication

`fetch_nordic_monitoring` captures supplements without opening the database.
`build_nordic_monitoring` opens SQLite read-only and publishes one immutable
directory containing `snapshot.json`, `index.md`, four country monitoring
reports and four linked annual country-assessment reports. Original source
paths and the artifacts namespace are protected from output. `LATEST.json`
changes only after the complete report set exists; identical reruns preserve
bytes and modification times. Source files, downloads and generated reports
remain local gitignored artifacts and must be preserved separately from Git.

Completion requires offline source/contract tests, the integrated test suite,
raw-cell and comparison reconciliation, valid report links, and an exact
read-only replay. This slice does not change the scalar database, scoring
population or Macro Atlas's bundled desktop data.
