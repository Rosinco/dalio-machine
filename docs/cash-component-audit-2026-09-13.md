# Cash components: availability, definitions and source-vintage differences

Date: 2026-09-13. Financial source snapshot: 2026-08-10. This audit reads the
retained files and produces diagnostics; it does not alter company data,
calibration, forecasts or saved studies.

Atlas can expose the five saved cash totals and their arithmetic differences
for every listing with available reports. It cannot build verified maintenance
capex or shareholder cash from these fields alone. Only 3,297 of 18,325 complete,
nonplaceholder latest reports have provider FCF equal to operating plus investing
cash under the stated tolerance. The other 15,028 differ. This is a definition
check, not evidence that either amount is necessarily wrong.

## What actually exists

The retained provider report schema and the desktop pack contain these five
aggregate fields:

| Saved field | Meaning used in Atlas | Latest available native amounts |
| --- | --- | ---: |
| `cash_flow_from_operating_activities` | Operating cash, as supplied | 18,917 |
| `cash_flow_from_investing_activities` | Signed net investing cash, as supplied | 18,914 |
| `cash_flow_from_financing_activities` | Signed financing cash, as supplied | 18,916 |
| `cash_flow_for_the_year` | Net cash for the period, as supplied | 18,917 |
| `free_cash_flow` | Provider FCF proxy, as supplied | 18,917 |

The denominators are 19,140 company **listings**, of which 18,941 have a retained
annual report and 199 have none. The two-directory union contains 17,593 listings
present in the latest directory and 1,547 observed only in the older directory.
Counts are not independent issuers. Latest-report availability includes stale
reports, missing publication dates and possible placeholders; it is broader than
automatic forecast eligibility. No older report fills a missing latest component.

The report fields do **not** separately contain gross capex, maintenance versus
growth capex, acquisition payments versus disposal proceeds, lease principal,
lease interest, cash interest, cash taxes, working-capital cash changes,
restricted cash or regulatory-capital cash requirements. Aggregate balance-sheet
assets/liabilities and a provider Capex percentage cannot supply those missing
cash components without additional definitions and sources.

All audit calculations divide each saved monetary value by that same row's
positive `currency_ratio`, require a valid reporting-currency label, and retain
signed values. Amounts are in millions of each report's native currency; there
are no cross-currency money totals. The provider documents that converted report
values are the default while the `currency` field still names the original
report currency. [Börsdata API report documentation](https://github.com/Borsdata-Sweden/API/wiki/Reports)

## Useful arithmetic without an invented cash definition

| Derived amount | Formula | Interpretation |
| --- | --- | --- |
| Operating + investing cash | CFO + signed CFI | A broad cash subtotal before financing |
| Difference from provider FCF | Provider FCF − (CFO + CFI) | An unexplained definition/reconciliation difference |
| Operating-to-provider-FCF gap | CFO − provider FCF | A gap; not observed capex |
| Three-component cash sum | CFO + CFI + CFF | Signed aggregate cash activity |
| Net cash less component sum | Supplied net cash − (CFO + CFI + CFF) | An unexplained residual, not automatically FX or an error |

Each calculation requires every input from the **same report, source and currency
basis**. A missing input produces a missing derived value. Zero remains an
observed zero; a whole statement consisting only of zeros/missing amounts gets a
separate placeholder flag. Matching zeros do not establish a viable business.

The diagnostic tolerance is
`abs(a-b) <= 0.000001 + 0.00000001 × max(abs(a), abs(b))`, in native millions.
This tests numerical agreement; it does not verify cash distributable to common
equity. A residual can also reflect reported rounding. Do not automatically
subtract aggregate financing cash from FCF: financing can contain borrowing,
repayment, dividends and equity transactions with different economic roles.

IAS 7 distinguishes operating, investing and financing activities. Investing
includes long-term asset purchases/disposals and obtaining or losing control of
businesses; financing includes changes in contributed equity and borrowings.
[IAS 7 overview](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/)

The provider's FCF page offers a broad operating/investing description without
an exact, versioned reconciliation for the API field. Its Capex page explicitly
includes acquisitions and divestments. Neither establishes maintenance capex or
a stable universal FCF bridge from the retained aggregates.
[Provider FCF description](https://borsdata.se/en/info/ratios/fcf-per-share),
[provider Capex description](https://borsdata.se/en/info/ratios/capex)

The source repository's `DATA.md` and ADR 0016 additionally document an
AcadeMedia lease-cash problem. That is a reason to seek company-specific lease
evidence, not proof of identical lease treatment for every issuer or a numerical
universe correction. The report parquet has no separate lease cash lines.

## Measured reconciliation

| Retained annual cohort | Rows | All-statement placeholders | FCF matches CFO + CFI | FCF differs | Comparison unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| All saved annual rows | 268,884 | 8,648 | 37,925 | 230,693 | 266 |
| Latest annual per listing | 18,941 | 589 | 3,886 | 15,028 | 27 |
| Latest, excluding placeholders | 18,352 | 0 | 3,297 | 15,028 | 27 |
| Latest operating/property reports | 16,312 | 131 | 3,061 | 13,239 | 12 |
| Latest manual Financials reports | 2,629 | 458 | 825 | 1,789 | 15 |

Property branches 75/76 remain in operating/property; other Financials require
reviewed equity-cash inputs. The report counts above exclude listings without an
annual report, explaining why manual Financials reports are fewer than manual
Financials listings in the starter audit.

For the latest three-component versus net-cash check, 16,977 rows match, 1,936
differ and 28 lack a complete comparison. Removing the 589 all-zero placeholders
leaves 16,388 matches, 1,936 differences and 28 unavailable. These tolerances are
deliberately explicit; even a one-million reporting-rounding difference can be
flagged and must not automatically be called an accounting error.

Selected FY2025 amounts, in each company's native millions:

| Listing | Currency | Operating | Investing | Provider FCF | CFO + CFI | FCF difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stora Enso, 696 | EUR | 645 | 60 | 705 | 705 | 0 |
| SCA, 197 | SEK | 4,017 | −2,605 | 1,412 | 1,412 | 0 |
| Holmen, 102 | SEK | 3,851 | −2,115 | 1,736 | 1,736 | 0 |
| Atrium Ljungberg, 20 | SEK | 1,190 | −2,600 | −1,410 | −1,410 | 0 |
| ABB, 3 | USD | 5,469 | −2,389 | 3,080 | 3,080 | approximately 0 |

Stora's `CFO − FCF` is −60. Calling that number capex would turn a broad net
investing inflow into invented negative investment needs. The aggregate source
does not identify the individual transactions or recurring investment requirement.
ABB's native USD figures also differ from the quote-currency SEK amounts used by
its current automatic starter; the component view must follow its stated basis.

## What the two vintages can and cannot explain

The original annual downloads overlap on 182,888 fiscal keys. Among 182,100 with
the same currency and comparable native FCF:

- 160,845 change FCF; 19,833 change sign.
- 146,682 change FCF while operating and investing cash stay unchanged.
- **145,747 change FCF while all four other cash totals stay unchanged:**
  operating, investing, financing and net cash for the period.

The current client maps API `free_Cash_Flow` directly to `free_cash_flow`; the
snapshot fetcher stores the report model. The saved schema supplies no missing
adjustment that explains these changes. The reviewed provider pages and saved
Swagger schema do not give a dated definition-change explanation. **The cause
remains unconfirmed.** It could require a provider clarification or paired
original-company-report reconciliation; do not label every change a restatement.

For example, 1-800-Flowers.com (10056), FY2024, USD millions, has operating cash
94.999, investing −42.304, financing −20.065 and net cash 32.630 in both downloads.
Provider FCF changes from 56.367 to 29.018. Neither amount equals the 52.695
operating-plus-investing subtotal. This proves a supplied-measure difference,
without identifying its economic or technical cause. The examples ledger uses
ascending listing ID within FY2024, without selection by forecast performance.

The [earlier vintage note](cash-flow-backtest-data-vintages-2026-09-12.md) reported
57,598 frozen and 48,754 fresh matches of FCF to CFO + CFI. Those matching counts
are not reproduced by the retained generator/receipt or two independent current
calculations. Direct parquet-to-NumPy arithmetic and this row-level audit agree:

| Original raw annual source | Complete rows with valid native currency | Matches at stated tolerance |
| --- | ---: | ---: |
| 2025-06-21 | 205,586 | 43,293 |
| 2026-08-10 | 250,862 | 37,290 |

One denominator difference is identified precisely: frozen listing 87, FY2024,
has currency label `0`. Counting finite amounts alone includes it and gives
205,587; requiring a valid native currency excludes it. That does not explain the
older matching-count discrepancy. Use the reproduced counts above; this is an
audit-documentation correction, not an explanation of the provider's FCF changes.

## Safe runtime evidence contract

Keep `StarterAnnualInput.components` on the existing same-report evidence row:
`{ operating, investing, financing, netCash, providerFcf }`, each number or null.
The parent row carries fiscal year, period boundaries, publication date, source
ID/date/path/hash, native and quoted currencies and saved currency ratio. Keep
the surrounding `amountBasis` and display currency explicit. For quote-currency
starters, every component must pass the same row's quote-currency lineage guard;
do not mix native component amounts with a quoted FCF.

Expose derived `operatingPlusInvesting`, `providerDifference`, `componentSum` and
`netDifference` separately from saved fields, with `matches | differs |
unavailable` comparisons and the stated tolerance. A placeholder flag should be
visible on all-zero source rows. The optional `operatingMinusProviderFcf` remains
a diagnostic gap only. All missing subcomponents stay missing; no balancing
figure becomes capex, an acquisition, lease cash or owner earnings. Components
do not change the selected cash proxy or calibration.

Deep dives should add a separate sourced bridge for material acquisitions,
disposals, lease cash, working capital, taxes and financing/capital needs. Each
adjustment needs a signed amount, period, currency, original-report citation,
rationale and revision identity. Preserve the original provider row so a reader
can follow the bridge. A normalization is an analyst judgment until supported;
no automatic adjustment is justified by a high residual alone.

## Reproduction and verification

[Generator](../desktop/scripts/audit-cash-components.py),
[17 focused tests](../desktop/tests/test_audit_cash_components.py),
[complete audit JSON](../desktop/test-results/cash-component-audit-2026-09-13/cash-component-audit.json),
[source/output receipt](../desktop/test-results/cash-component-audit-2026-09-13/receipt.json).
The artifact directory also retains a 19,140-listing latest-component ledger and
all 182,888 cross-vintage comparisons as compressed JSON lines. JSON nulls remain
nulls. SHA-256 identities bind the pack, taxonomy, unchanged calibration, both
annual source files, source documentation, generator and outputs.

```bash
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
pytest -q desktop/tests/test_audit_cash_components.py
ruff check desktop/scripts/audit-cash-components.py desktop/tests/test_audit_cash_components.py
python desktop/scripts/audit-cash-components.py \
  --borsdata-root /mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client
```

Tests cover signed identities, missing components, valid observed zero, invalid
currency/conversion, tolerance boundaries, unchanged native values under changed
FX, unexplained FCF-only changes, currency-change withholding, latest-period
non-backfilling, no-history cohort retention, source immutability and tampered
company-hash rejection. The full audit additionally checks every compressed
company payload against its index hash and independently reproduces both raw
vintage arithmetic counts with NumPy. No company facts or calibrated factors
were written or refitted.
