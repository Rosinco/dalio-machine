# Cash-flow backtest audit — 2026-09-12

The first audit evaluates the standardized historical cash-flow starter across
all **19,140 saved company listings**. It produces **4,139,652 forecast rows**
across four fixed models, five- and ten-year histories, and horizons one to five.
**15,496 listings** have at least one eligible historical fold; **3,644** have
none. Those are historical-test eligibility counts, distinct from current app
starter availability. The installed **0.13.0** app and saved valuations are
unchanged by this research slice.

Open the interactive local report in the desktop worktree:
`desktop/test-results/cash-flow-backtest-2026-09-12/report.html`.
It embeds its data for offline use, supports company/outcome/horizon/pattern
filters, shows actual cash against the historical forecast, exports filtered
rows, and includes a frozen-snapshot cohort ledger. Full-precision compressed
CSVs and input/output hashes sit alongside it. Generated artifacts are gitignored;
keep this directory separately when preserving or handing off the audit.

## Measured findings

Recent comparison: targets **FY2024–2025**, current five-year weighted trend,
native reporting currency, original sensitivity range. Counts are
listing/origin/horizon observations; share classes and targets are dependent.
Each listing has at most two target years at one horizon in this comparison.

| Horizon | Original range around mid | Evaluated forecasts | Inside | Median absolute error / training scale |
|---|---|---:|---:|---:|
| 1 | ±10% | 28,249 | 13.34% | 0.571 |
| 2 | ±20% | 26,265 | 17.54% | 0.933 |
| 3 | ±30% | 24,415 | 20.91% | 1.245 |
| 4 | ±40% | 23,129 | 25.50% | 1.455 |
| 5 | ±50% | 22,017 | 30.88% | 1.610 |

The ±10/20/30…% bands are therefore sensitivity assumptions, not empirically
established predictive intervals. One-source-snapshot folds show nearly identical
Year 1 coverage (13.34% on 28,227 observations); mixing downloads within a fold
does not by itself explain the narrow-band result. One source ID does not prove
consistent cash definitions or restore historical vintages.

On the same 28,249 five-year-eligible Year 1 observations, holding the latest
cash value constant has median normalized absolute error **0.419**, compared
with **0.571** for the full weighted trend, **0.523** for the half-strength trend,
and **0.540** for the weighted flat average. These describe this sample; no new
model is selected or deployed from the test years. Five-versus-ten-year
comparisons require matching both the eligible folds and the normalization
scale; the report uses a common latest-five-year scale for that comparison.

An illustrative 80%-target range learned from older residuals gives **82.11%**
Year 1 coverage on **27,755** eligible recent observations, versus **12.99%** for
the original range on those exact observations. Mean full width increases from
**0.259 to 3.019 times training cash scale** (about 11.6 times wider), while the
mean normalized interval score improves from **14.009 to 10.974**. This is evidence
about a coverage/width tradeoff, not improved accuracy of the unchanged mid line.
Learned Year 5 test ranges are unavailable under the temporal calibration gate.

## Separate snapshot experiment and measurement caveat

A retained **21 June 2025** download permits a frozen-input pilot, with outcomes
from **10 August 2026**. The complete frozen directory has **15,646 listings**.
The main FY2024-history → FY2025-outcome cohort has **11,407** eligible training
histories, **9,741** admissible outcomes and **1,115** hits (**11.45%**). The other
**1,666** training-eligible cases remain unavailable, including **916** absent
from the later company directory. No missing outcome becomes zero, and directory
disappearance is not labelled bankruptcy. The separate FY2025→FY2026 cohort is
not pooled into the main result.

The low pilot agreement cannot isolate forecasting skill: among **182,100**
comparable same-currency historical rows, **146,682** changed native FCF while
native operating and investing cash remained unchanged. **92,375** FCF changes
exceeded 20% of the larger absolute value. This points to a provider measurement
comparability issue. Definition changes, corrections and revisions are possible
causes; the vendor cause has not been established. See
[cash-flow-backtest-data-vintages-2026-09-12.md](cash-flow-backtest-data-vintages-2026-09-12.md).
The historical snapshot controls input vintage from that date; the formula was
applied today, not actually saved as a forecast in June 2025. Outcome values may
also contain revisions, and targets were often partly elapsed at the origin.

## Case reviews

Current-vintage FY2024-origin, Year 1 projections; amounts are native millions.
These three companies' five-year training cash series also match across the two
retained source vintages. Case selection is illustrative, not representative.

| Listing | Currency | Mid forecast | Original low–high | FY2025 actual | Result |
|---|---|---:|---:|---:|---|
| Stora Enso 696 | EUR | −654.09 | −719.49 to −588.68 | 705 | Above |
| SCA 197 | SEK | 1,638.26 | 1,474.43–1,802.08 | 1,412 | Below |
| ABB 3 | USD | 2,996.40 | 2,696.76–3,296.04 | 3,080 | Inside |

**Stora Enso:** the saved FY2025 proxy reconciles to statutory operating cash
plus net investing cash. Disposals and lower capex contributed to the reversal.
The issuer's alternative cash-flow measure differs. Inference: separate recurring
operations, investment spending and asset transactions before extrapolating.
The report links the April 2025 information available before the frozen origin
separately from later realised cash.
[Official FY2025 financial statements release](https://www.storaenso.com/-/media/documents/download-center/documents/interim-reports/2025/storaenso_results_q425_eng.pdf).

**SCA:** operating cash increased, but greater investment outflows and lower
asset-sale proceeds reduced the saved cash proxy. Inference: investment timing
explains this modest downside miss more directly than a deterioration in statutory
operating cash. [Official FY2025 report, page 14](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260130-year-end-report-q4-2025-en-0-5296365.pdf).

**ABB:** its FY2025 outcome is inside the range, while FY2024 is outside. A single
hit is not a durable company classification. These are calculations from saved
provider data; no business cause is inferred for ABB here.

Other misses have descriptive numerical flags and an unreviewed cause. The audit
does not claim that every company has been researched. Pattern flags cover sign
flips, opposing trends, near-zero mid cash, volatile history, financial-sector
proxies, and mixed source vintages. Some are observed only after the outcome;
they are review prompts, not features available to an earlier forecast.

## Reproducible protocol

1. Require a full five- or ten-year consecutive annual training window, valid
   signed cash, valid period dates and one consistent reporting currency through
   training and the target. Reject missing/withheld/placeholder/overlapping rows;
   retain exclusions and companies without a score. Future intermediate fiscal
   periods must also be admissible. This full-window audit is stricter than the
   app's explicitly disclosed partial-history starter.
2. Use each row's own saved currency ratio to recover native reporting-currency
   cash. No price is needed. The app's historical quote-currency fallback is not
   tested here. Provider FCF remains an unreviewed cash proxy, especially for
   financial firms; it is not automatically FCFF or distributable equity cash.
3. Fit the same weighted regression as v2 (30/25/20/15/10 newest first for five
   years; 19/17/15/13/11/9/7/5/3/1 for ten). Competing mids are last cash, weighted
   flat cash, fitted intercept plus half slope × horizon, and full trend. All
   four models share the exact eligible folds within a window.
4. Keep calibration targets through FY2020, validation targets FY2021–2023, and
   fixed test targets FY2024–2025. Fit each mid using only fiscal years before its
   target. This is a retrospective analysis of later-vintage values, not a
   historically executable investment strategy or a prospective experiment.
5. For empirical half-width, multiply the nearest-rank 80th percentile of older
   absolute forecast errors / training mean absolute cash by the new training
   scale. Pool separately by model/window/horizon; require at least 30 calibration
   errors. Zero scale remains missing. No recent test residual tunes these factors.
6. The nominal calibration cutoff is 30 June 2021. Calibration outcomes must have
   known publication by that date and ordered, plausible training publications.
   Applying a factor requires origin FY≥2021, origin publication on/after the
   cutoff, known ordered training publications, and origin publication before the
   target fiscal end. Missing target publication is disclosed but does not erase
   an identified later outcome. The factors are estimated now from saved data,
   not factors actually frozen in 2021. All Year 5 recent-test origins predate the
   admissible cohort, so no learned Year 5 coverage is reported.
7. Measure inside/below/above, normalized absolute error and bias, interval width,
   and proper interval score. At the illustrative 80% target, the score is width
   plus 10 times the distance of an outside outcome; lower is better. Compare
   original and learned bands only on the same eligible observations. Cross-window
   model comparisons use matched folds and a common latest-five-year scale.

[Rolling-origin evaluation](https://otexts.com/fpp3/tscv.html) motivates training
only on preceding observations. The interval score follows
[Gneiting and Raftery (2007), section 6.2](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
There is no calibrated probability guarantee for a particular company, no
independence assumption across listings, and no validation of DCF terminal values,
discount rates, purchase prices or investment returns.

## Next improvement and acceptance criteria

Reconcile the provider cash definition first. Preserve separate raw source facts,
recurring operating cash, maintenance/growth investment, acquisitions/disposals,
and analyst adjustments. Financial businesses require suitable capital models.
Freeze future company membership, source snapshots, model versions and forecasts
at creation; retain the original baseline alongside each deep-dive revision.

Evaluate candidate mids and horizon-specific error ranges on new untouched
periods. Use pooled sector/cash-pattern errors when sample sizes support them;
show sample sizes and missingness, and avoid fitting company-specific confidence
labels from one or two outcomes. Directional residual quantiles can later allow
asymmetric bands. Any change must improve point error or the coverage/width score
on comparable data, preserve negative and near-zero cash, and survive data-quality
and adverse-cycle checks. Cash-path dependence and terminal assumptions need their
own analysis before translating annual ranges into a probabilistic DCF interval.

## Commands and artifacts

From the desktop worktree, with Node 22+ available:

```sh
node desktop/tests/cash-flow-backtest.test.mjs
node desktop/scripts/cash-flow-backtest.mjs --output desktop/test-results/cash-flow-backtest-2026-09-12
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
python3 desktop/scripts/cash-flow-backtest-common-scale.py desktop/test-results/cash-flow-backtest-2026-09-12
# Generate and validate the frozen pilot using the companion vintage-audit commands.
python3 desktop/scripts/render-cash-flow-backtest.py desktop/test-results/cash-flow-backtest-2026-09-12
python3 desktop/scripts/verify-cash-flow-backtest.py --directory desktop/test-results/cash-flow-backtest-2026-09-12
node desktop/tests/cash-flow-backtest-report.mjs --directory desktop/test-results/cash-flow-backtest-2026-09-12
```

The renderer requires both the rolling backtest and the companion frozen-pilot
outputs in the selected directory. The common-scale postprocessor retains the
original forecasts and normalizes paired windows to the same cash denominator.

The source pack is opened read-only and checked against its content-addressed
filename; taxonomy and every company payload are verified. The source hash is
`1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a`.
`receipt.json` records executed-script and output hashes. Source audit and frozen
pilot reproduction are documented in the companion data-vintage note. Focused
verification covers regression parity, negative and zero cash, future-value
poisoning, source hashes, periods, timing gates and compressed-export consistency.
Browser/report validation receipts are retained with the generated report.

Verification completed: **22 Node tests, three Python tests and seven browser
check groups passed**. Independent validation reconciled all 15,646 frozen-pilot
rows, all 4,139,652 forecast-group totals and source/output hashes, 30 recent
ABB/SCA/Stora projections, and 379,768 paired history-window comparisons. The
report loaded offline without external requests or runtime errors and passed
1440px/390px layout checks. Ruff and whitespace checks are clean. This audit did
not modify the app model, installed binary, saved valuations or vendor sources.
