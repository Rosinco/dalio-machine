# ADR 0033 — Original-source monitoring for the US, Germany and Canada

Date: 2026-09-11
Status: accepted

## Scope

The user requested the verified assessments and Nordic monitoring in Macro
Atlas, followed by more country coverage. This bounded backend slice adds the
US, Germany and Canada: 15,006 saved listing records combined. Listing counts
guide collection priority, not issuer deduplication or actual company exposure.
The desktop consumes explicit versioned exports; it does not write conclusions
back to the source database. Its independent presentation change is documented
in its own working tree.

Audit retained sources before new acquisition. Each country already has a
verified annual country assessment. Existing US rate observations lack retained
raw-delivery bindings, while original ECB policy evidence exists in the Nordic
supplement. New acquisition preserves honest new receipt clocks without
rewriting those earlier records.

## Separate source contract

`national-monitoring-v1` is a fixed twelve-attempt US/DE/CA supplement with the
same four monitoring topics as the Nordic pilot. Earlier Sweden and Nordic
contracts remain unchanged. Native request destinations and selections, response
entities, HTTP outcomes, SHA-256 hashes, source definitions and request/receipt
clocks are retained. The latest eligible whole attempt bundle supplies facts;
missing or failed inputs cannot borrow successes from an older batch.

A documented structural gap is distinct from a failed download. The US Federal
Reserve's E.2 business-loan survey ceased in 2017. An original documentation
response must verify that fact before the gap is reported. Neither a bank prime
rate nor a narrower small-business survey silently replaces a current broad
new-business loan rate. Structural gaps carry their own source artifacts and
clock, contribute no scalar observations or directional reading, and remain
visible in country reports and the app.

## Native concepts

US industrial output includes manufacturing, mining and utilities. Its effective
federal funds rate is a market outcome rather than a target-range bound. Treasury
constant-maturity par yields are fitted estimates from market inputs, not the
yield on one outstanding bond or actual auction execution.

Germany uses original Destatis adjusted industry, Bundesbank new-business
lending and the current ten-year federal benchmark, and the shared ECB
deposit-facility instrument. Lending-counterparty and currency scope must remain
explicit. The available Destatis CSV currently announces an October 2026
migration; retain that limitation and require a new verified source contract
when the current feed ceases, rather than silently switching sources.

Canada's industrial production aggregate is real industrial GDP at basic
prices, in millions of chained 2017 Canadian dollars, seasonally adjusted at
annual rates. It is real industrial value added rather than an output index.
Keep its native scale, industry classification and `measure_kind`; do not
rebase it into an invented index. The selected Canadian business-loan population
also differs from a pure non-financial-corporation universe. Exact definitions
and limitations accompany all selected measures.

Two adjacent complete three-month averages of positive adjusted industrial
volume measures support a within-series percentage change. Monthly rates use
an exact three-month difference; daily rates use an actual observation at or
before the 90-day target within seven days, in percentage points. Missing latest
periods stay missing. The inherited ten-day daily and 75-day monthly freshness
thresholds are editorial applicability rules, not predictive confidence.

No cross-country ranking, composite score, forecast error, causal conclusion,
scenario probability or company verdict follows from these comparisons.
Scenarios remain conditional and require actual revenue, asset, cost, currency
and financing exposures before company interpretation.

## Publication and verification

The collector does not open the database. The read-only builder restores all
three annual assessments at the same exact cutoff and publishes eight immutable
files: `snapshot.json`, `index.md`, three monitoring reports and three annual
context reports. Only a complete report set can update `LATEST.json`; repeat
exports preserve bytes and modification times. Original response files and the
database are protected from output writes.

Completion requires offline adapter tests, complete capture/replay tests,
integrated tests and Ruff, original-cell and comparison reconciliation, report
link checks, exact read-only CLI replay, and a tested desktop pack built from
the verified exports. Downloaded evidence and generated reports remain local
gitignored artifacts and must be retained separately from Git. The database and
the existing scoring population remain unchanged.

## Verified backend checkpoint

All **1,690 integrated tests** passed; Ruff and diff checks are clean. The live
capture completed **2026-09-11T06:26:50.807464+00:00**, with 26 successful HTTP
requests, eleven source histories and the documentation-bound US lending gap.
Bundle: `1038458825e26cf7a6768bde5293534a72028234a2b927e068209acc8d9af1a3`.
The as-of 2026-09-11 snapshot uses that exact cutoff:
`62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46`.

Independent raw-vector audits reconciled all **5,182 native slots**: 4,994
numeric and 188 missing. The report audit reconciled all eleven comparisons,
57 annual histories/2,581 observations, 151 citations, 127 displayed table values
and 624 report links. All eight report files and 186 protected source files
remained unchanged. Exact post-publication CLI replay preserved all 196 checked
files, including the source database and latest pointer; SQLite integrity and
foreign-key checks passed. Original response, source-audit and replay receipts
are retained under `data/snapshots/national_monitoring/validation/`.

Germany's July lending observation retains its native P/provisional status.
All four Canadian signals retain their actual periods, including June industrial
GDP and business lending. Neither new capture dates nor desktop packaging make
those observations newer. Macro Atlas's application build and installation are
verified separately; see the final `HANDOFF.md` checkpoint for their status.
