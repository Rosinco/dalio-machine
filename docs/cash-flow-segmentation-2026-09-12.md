# Cash-flow uncertainty segmentation — 2026-09-12

The user asked whether sector, branch, tangible book value/EBIT and other
common-sense business characteristics can improve the standardized cash-flow
uncertainty range. This completed offline experiment tests **18 fixed grouping
rules**, then **four additional refinements within branches**. The installed
0.13.0 app and all saved valuation drafts remain unchanged.

The strongest result is **historical cash behavior**. Sector and branch provide
context but add little by themselves to the pooled interval score. Tested asset
intensity and book-equity/EBIT proxies provide little additional predictive
benefit in these fixed comparisons. This does not remove their economic meaning
for reinvestment, operating returns, financing or deep-dive analysis.

## Open the report

Desktop worktree:
`desktop/test-results/cash-flow-segmentation-2026-09-12/report.html`

Windows File Explorer:

```text
\\wsl.localhost\Ubuntu\home\rosinco\workspace\dalio-atlas-desktop\desktop\test-results\cash-flow-segmentation-2026-09-12\report.html
```

The standalone report embeds its data and includes model/horizon/scope/period/
weighting controls, a cash-group coverage and width chart, all candidate
comparisons, group diagnostics, within-branch refinements and FY2024 company
feature search. Those company descriptors are historical inputs, not current
risk ratings or proof that every displayed company has a date-eligible test.
Detailed JSON/CSV and hashes sit alongside the HTML. This generated directory is
gitignored and needs separate retention from Git publication.

## What was tested

Extracted **164,166 strict five-year company/origin histories across 15,496
listings**, with **153,055 distinct financial-history fingerprints**. Extractable
histories are conditional on having an eligible fold in the first audit; these
are not feature-availability counts for every company ever listed.

The source-bound slim ledger contains **1,130,050** existing forecasts from the
weighted linear and last-cash models, five-year histories and horizons one to
four. **878,254** rows survive the positive cash-scale and calendar gates for
calibration or application. Source data and the existing forecasts are unchanged.

The primary scope includes operating and property companies: it excludes the
source's Financials sector **except Real Estate and REITs**. Property appears
inside Financials in the source taxonomy and is identified explicitly in the
report. A separate all-listing sensitivity retains lenders, insurers and other
financial businesses, whose generic cash proxy needs a different economic model.

For the primary one-year weighted-trend comparison, validation has **20,989
forecasts across 11,427 listings** (FY2022–2023 after timing gates); the recent
comparison has **24,440 forecasts across 13,069 listings** (FY2024–2025).

Features use only the origin and previous four annual source rows, converted to
native reporting currency with each row's own saved ratio. They include sector,
branch, cash-sign persistence, raw and fitted-residual cash dispersion, operating
cash dispersion, revenue dispersion, EBIT margin and margin dispersion, tangible
assets/sales, tangible assets/total assets, guarded tangible book equity/EBIT,
net debt/assets, a working-capital proxy and CFO-minus-provider-FCF dispersion.
Only two main intersections were fixed: sector×cash dispersion and branch×cash
dispersion. Bins and all guards are stored in `protocol.json` and source code.
The supplementary four within-branch combinations use those same bins.

## Results: sector helps less than cash behavior

Positive percentages below mean a reduction in the **mean interval score**
relative to a global range calibrated for the exact same scope/model/horizon.
The score penalizes both excessive width and outcomes outside the band. It is
not a percentage improvement in the mid forecast, which remains unchanged.
Listings receive equal total weight within each comparison period.

| Grouping | Validation score improvement | Recent score improvement |
|---|---:|---:|
| Sector | +0.25% | +0.36% |
| Branch | +0.36% | +0.59% |
| Historical cash dispersion | +3.89% | +6.81% |
| Dispersion around fitted historical trend | +2.98% | +4.35% |
| Historical cash-sign pattern | +2.64% | +4.81% |
| Operating-cash dispersion | +1.31% | +2.19% |
| Tangible assets / sales proxy | +0.11% | -0.11% |
| Tangible book equity / EBIT proxy | +0.05% | +0.29% |
| Net debt / assets | -0.00% | -0.01% |

The cash-dispersion result remains positive with the last-cash mid model
(**5.74%** recent one-year improvement) and after reducing the weight of exact
duplicate histories (**6.64%** for the primary weighted-trend case). Its primary
one-year score improvement is positive in each FY2022–2025: **2.20%, 8.80%, 8.43%,
5.62%**. This is more encouraging than a gain confined to one pooled sample,
but is still exploratory evidence from a few calendar regimes.

More detailed taxonomy does not automatically improve the result. Sector×cash
dispersion gives **6.61%** recent improvement, slightly below cash dispersion
alone; branch×cash dispersion gives **3.91%**, with substantial fallback to
broader peers. This hard-bin experiment does not test every possible pooling
method. Thin subdivisions should not be mistaken for precise company-specific
estimates.

## The useful distinction is conditional calibration

The global one-year research range covers **83.04%** of recent primary outcomes
overall. That pooled average hides very different behavior within cash groups:

| Historical cash dispersion | Forecast observations | Inside global range | Inside group range | Group mean full width / training cash scale |
|---|---:|---:|---:|---:|
| Low | 3,744 | 97.4% | 82.6% | 1.09× |
| Medium | 11,486 | 90.8% | 82.3% | 2.30× |
| High | 9,210 | 67.9% | 82.5% | 4.63× |

Here dispersion is population standard deviation of five signed annual cash
values divided by their mean absolute cash: low <0.25, medium 0.25–0.75, high
≥0.75. It includes growth and changes in business scale, not just random volatility.
The denominator is a **historical cash scale**, not future mid cash. The widths
above are full widths; half-width is half the reported amount.

One pooled range is unnecessarily wide for many stable histories and too narrow
for volatile histories. Grouping redistributes width where historical errors
indicate it is needed. Recent group coverage is roughly 82–83%, but the same
cash-group method covers only **75.6% in FY2022**. Calibrating to an 80% target
does not guarantee 80% coverage in a later economic regime or for each company.
Even a correctly calibrated 80% range is expected to miss around one in five
outcomes; review the magnitude, direction and persistence of misses.

## Does a ratio add information inside a branch?

The additional experiment was specified before inspecting its results. It holds
the branch baseline, forecasts, observations, bins and support thresholds fixed.
All improvements here are **relative to the branch range**, not the global range
in the earlier table.

| Additional distinction within branch | Validation improvement | Recent improvement | Recent fallback to parent |
|---|---:|---:|---:|
| Tangible assets / sales | +0.053% | +0.019% | 62.8% |
| Tangible book equity / EBIT | +0.033% | +0.156% | 91.6% |
| EBIT margin | +0.153% | +0.445% | 66.7% |
| Historical cash residual dispersion | +1.492% | +2.032% | 33.3% |

These asset ratios add very little in this fixed experiment. Sparse subdivision
also limits applicability: the TBV/EBIT refinement often cannot support a group
and falls back. A weak result does not prove that a reconciled measure, a different
business population or a future period could never provide useful information.
Historical cash residual dispersion adds more within branches, consistent with
the main experiment.

## Tangible book value / EBIT: meaning and limitations

The experiment interprets tangible book value as the saved **total equity minus
intangibles** proxy. This is not a full reconciliation of common tangible book
equity. With aligned definitions:

```text
TBV / EBIT = (TBV / sales) / (EBIT / sales)
```

The ratio combines equity capital intensity and operating margin. A debt-funded
payout can reduce book equity without changing the operating assets. An earnings
downturn can sharply increase the ratio without adding any physical assets.
Its units resemble years of EBIT relative to book equity, but it is not cash
payback: EBIT precedes interest/tax and does not subtract required reinvestment.

For five-year TBV/EBIT classification, every included year must have positive
tangible book equity, positive revenues and EBIT/revenue ≥1%. Missing, loss,
nonpositive book equity and tiny-denominator cases remain unavailable. Only
**36.9% of all extracted historical windows** satisfy this rule. In the recent
primary experiment, the TBV/EBIT grouping uses global fallback for **73.4% of
weighted observations**. These percentages have different denominators.

Separate the analytical questions:

| Question | Better starting measure | What still needs review |
|---|---|---|
| How much physical capital supports sales? | Average tangible operating assets / sales; PP&E/sales with consistent leases | Vendor `tangible_assets` is only a proxy; age, depreciation and replacement cost matter. |
| How much total operating capital is tied up? | Average tangible operating capital / sales | Reconcile working capital, operating liabilities, surplus cash and nonoperating assets. |
| How productive is that capital? | Normalized NOPAT / average tangible operating capital | Normalize cycle and taxes; separate operating economics from acquisition cost. |
| How uncertain is cash? | Horizon-specific errors, cash patterns and business exposures | Do not infer reliability or investment attractiveness from asset intensity alone. |
| How fragile is equity cash? | Liquidity, gross debt, interest and maturity schedule | A single net-debt/assets ratio misses refinancing timing and restrictions. |

Asset intensity matters for reinvestment and capital productivity; these tested
proxies do not support making it the main uncertainty setting. Internally built
brands, research and capabilities can also be absent from book assets, while
leases and depreciated old assets complicate comparisons. The detailed
[source-backed conceptual note](cash-flow-segmentation-concepts-2026-09-12.md)
explains these accounting issues and includes arithmetic examples. The distinction
between operating tangible returns and acquisition cost is consistent with
[Berkshire's 1983 discussion](https://www.berkshirehathaway.com/letters/1983.html);
[IAS 38](https://www.ifrs.org/issued-standards/list-of-standards/ias-38-intangible-assets/)
explains why some internally generated capabilities are absent from book assets.

## Next common-sense distinctions

The most promising next company profile combines **sector/branch context, cash
variability and sign pattern, capital requirements, and financing commitments**.
Keep business quality and the price paid separate from forecast uncertainty.
Predictable negative cash is not a high-quality investment merely because its
range is narrow.

For deep dives, add dated evidence on recurring/contracted demand, renewal/churn,
customer concentration, commodity and pricing exposure, price resets, fixed versus
variable cost commitments, capacity utilization, maintenance/growth capex,
acquisitions/disposals and debt maturities. These are proposed next inputs, not
features validated by this experiment. Test whether each improves calibration
and interval score on comparable future observations before changing defaults.

Cash-flow decomposition is especially useful after the earlier Stora/SCA miss
reviews: separate operations, working capital, investment and financing rather
than extrapolating an unexplained aggregate. [IAS 7](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/)
separates those activities; net investing cash cannot be treated as universal
maintenance capex. [Barth, Cram and Nelson's original cash-flow prediction study](https://www.gsb.stanford.edu/faculty-research/publications/accruals-prediction-future-cash-flows)
provides prior evidence for studying cash/accrual components, not a guarantee for
this global provider dataset.

## Timing, fairness and evidence limits

Calibration uses target years through FY2020, with the existing nominal
30 June 2021 publication cutoff and valid ordered training dates. Factors are
estimated today from later-vintage saved values. Application requires origin
FY≥2021, origin publication on/after that cutoff, known ordered training
publications, and publication before target fiscal end. No recent residual tunes
the group factors. Year 1 validation has only FY2022–2023; Year 2 only FY2023;
Years 3–4 have no date-eligible validation period. The recent years were already
inspected in the earlier audit and are **exploratory**, not untouched confirmation.

Each supported group needs at least **300 calibration folds, 100 listings and
100 distinct origin-history fingerprints**. These are chosen support guards,
not guaranteed statistical adequacy. Missing or small groups retain the same
observations and receive parent/global factors. The whole scope/model/horizon
pool gives each listing total calibration weight one; those weights are fixed
before grouping. Evaluation similarly gives each listing equal total weight in
the full comparison period. Subgroup tables retain those weights. This differs
from the first report's fold-weighted scoring, so raw score levels need not match.

The sensitivity refits calibration and evaluation after downweighting identical
history/target multiplicities. Conflicting outcomes remain included. Primary
recent Year 1 contains **22,889 unique history/target combinations** and one
conflicting-outcome combination across its 24,440 listing observations. This is
not definitive corporate-family deduplication or independent-issuer counting.

Current taxonomy and the saved universe can omit historical classifications,
former issuers and delisted businesses. Later-vintage financial statements can
contain revisions, and substantial provider FCF measurement changes remain
unresolved. Cash dispersion and forecast errors share a denominator, so the
association is useful for this model without establishing causation or intrinsic
business risk. No extreme miss was silently winsorized or removed; the worst
1% of observations account for about **79% of the cash-group interval score in
FY2022**, which limits precision of mean-score comparisons. The report includes
coverage, widths, counts, annual comparisons and duplicate-history sensitivities
alongside scores.

## Reproduction and verification

Implementation and artifacts reside in the desktop worktree. Use its existing
Python environment and saved first-audit forecast ledger. Run new experiments
in a new output directory to preserve the dated outputs. The feature extractor
checks source identities, the main experiment checks upstream feature hashes,
and the within-branch supplement pins the helper hash declared in its protocol.
The exact executed feature source is retained in the generated directory; a
subsequent import/zip style cleanup is recorded separately without replacing
feature outputs.

```sh
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
pytest -q desktop/tests/test_cash_flow_segmentation_features.py desktop/tests/test_cash_flow_segmentation.py desktop/tests/test_cash_flow_segmentation_within_branch.py
python3 desktop/scripts/cash-flow-segmentation-features.py --pack desktop/financial-data/1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a.sqlite --forecasts desktop/test-results/cash-flow-backtest-2026-09-12/forecasts.csv.gz --output-dir /tmp/atlas-segmentation-reproduction
OPENBLAS_NUM_THREADS=1 python3 desktop/scripts/cash-flow-segmentation.py /tmp/atlas-segmentation-reproduction
cp desktop/research/experiments/cash-flow-segmentation-within-branch-2026-09-12.json /tmp/atlas-segmentation-reproduction/within-branch-protocol.json
OPENBLAS_NUM_THREADS=1 python3 desktop/scripts/cash-flow-segmentation-within-branch.py /tmp/atlas-segmentation-reproduction
python3 desktop/scripts/render-cash-flow-segmentation.py /tmp/atlas-segmentation-reproduction
```

The existing dated artifact receipts are `experiment-receipt.json`,
`within-branch-receipt.json`, `report-receipt.json`, feature provenance/QC receipts,
`independent-validation.json` and `within-branch-independent-validation.json`.
Independent scalar checks reconcile 896 supported/unsupported main factors and
112 summary cells, all 2,304 reported main comparisons, 2,060 within-branch factors,
16 supplemental summary cells, all 170 supplemental comparisons and all 34
branch baselines. The original full-CSV global benchmarks also reconcile.
No production forecast, saved valuation, source pack or installed app was changed.

Final focused verification: **23 Python tests and 19 subtests passed**, and all
new Python research scripts/tests pass Ruff. The retained browser validator
`desktop/tests/cash-flow-segmentation-report.mjs` passed seven check groups on the
final report, including source reconciliation, selectors, missing cohorts,
branch-relative comparisons, company search and 1440px/390px layouts. It recorded
no runtime errors, HTTP(S) requests or page overflow. See
`report-browser-validation.json` and the three retained screenshots.

Final report SHA-256:
`aa81f5752762d07b81659bca2cf51d3d88b97784fd72913b3590220a03d9d429`.
