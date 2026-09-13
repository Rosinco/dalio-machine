# Proposed universe-wide cash-flow setup — 2026-09-12

Status: recommendation for the next evaluated model, not an adopted application
default. The installed weighted-cash-starter-v2 and saved drafts remain unchanged.

## Common workflow and automatic eligibility

Every listing retains historical cash, editable forecasts, DCF/NPV, source dates,
assumptions and saved revisions. A reviewed deep dive refines the same workspace.
Full five-year consecutive signed histories with valid currency/source information
and a positive mean-absolute cash scale qualify for the tested research grouping.
Short, missing or all-zero histories need explicitly assumed ranges or unavailable
empirical bounds; pooled factors do not validate untested short histories.
Very small cash scales and sign changes need visible review flags.

Banks and insurers need capital-constrained distributable-equity cash models.
Property needs consistent recurring cash, maintenance investment and financing
treatment. Generic provider FCF is a proxy until reconciled; common charts do not
establish a common economic cash definition. See
[financial-firm valuation characteristics](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/financialsvccompanies.htm).

## Proposed automatic baseline

Use five years for current cash-behavior classification, with a ten-year history
comparison where available. Keep all valid annual source facts. Do not shorten
or skip the window to manufacture a favorable result.

The simplest latest-reported-cash flat model is the provisional benchmark for
the automatic midline. It outperformed the unrestricted weighted trend in the
observed comparisons, but is not established as an optimal ten-year forecast.
Retain the recency-weighted historical trend as an editable alternative. A trend
that gradually fades is a next candidate to test; no arbitrary damping rule or
cap has been selected. A deep dive should reconcile unusual latest cash and
supply evidence for growth, normalization and reinvestment.

For eligible operating/property histories, use low/medium/high five-year signed
cash dispersion as the first uncertainty grouping. Use the tested cutoffs and
support rules from the segmentation protocol; missing/sparse groups fall back
to broader supported pools on the same eligible population. Sector and branch
remain business context. Asset intensity, margins and financing remain separate
analytical dimensions, rather than untested automatic multipliers.

Estimate a separate 80%-target historical-error range for each forecast horizon
and mid model. Half-width is the calibrated group error factor times the mean
absolute historical cash scale. This avoids the automatic zero-width problem
when future mid cash crosses zero. The tested 80% target is not an equal-tail or
future coverage guarantee. Keep user-entered percentage sensitivities available.

Evidence is strongest at Year 1, thinner at Year 2, sparse at Years 3–4, and absent
under this test protocol at Years 5–10. Longer horizons remain explicitly assumed
scenarios until separately evaluated. Annual cash ranges do not establish an
80% interval for the full cash path or DCF: cross-year dependence, discount rates
and terminal values require their own assumptions and validation.

## COVID and other exceptional events

Preserve raw COVID and rebound years. Annotate company-specific shutdowns,
temporary support, exceptional demand, inventory movements and structural changes.
A sourced deep-dive revision may normalize temporary effects, with both benefits
and losses treated consistently. Do not remove a shutdown while extrapolating its
subsequent rebound. Unusual values can contain information and should not be
replaced merely for being unusual; see
[Forecasting: Principles and Practice](https://otexts.com/fpp3/missing-outliers.html).

Show a separate crisis cash path with stated shock size, duration, recovery,
investment and funding assumptions. Do not assign a probability without evidence.
The default historical-error range should retain exposure to difficult years;
an optional ordinary-conditions view needs an explicit conditioning assumption.

For a company with FY2025 available, a five-year window covers FY2021–2025:
FY2020 has already rolled out, while recovery years remain. Longer history is a
useful comparison, not an automatic solution to structural change.

## Additional FY2020 calibration sensitivity

A fixed small sensitivity removed only target FY2020 errors from calibration,
recomputed listing-balanced weights and refitted the same group factors. Both
scenarios evaluated identical admissible FY2022–2025 observations for one-year
linear and latest-cash mids. It tested global, cash-dispersion and residual-
dispersion ranges, preserving the original experiment.

Calibration fell from 83,949 to 74,250 observations and 10,069 to 9,635 listings.
FY2020 represented 17.22% of original calibration weight; 434 listings lost
calibration support altogether. This changes sample composition as well as the
calendar mix, so it is not a causal estimate of COVID.

For the weighted-trend cash-group range:

| Measure | Original calibration | Omit target FY2020 |
|---|---:|---:|
| Recent FY2024–2025 coverage | 82.45% | 81.63% |
| Recent full width / historical cash scale | 3.0057 | 2.9381 |
| Recent interval score | 6.33510 | 6.32820 |
| FY2022 coverage | 75.63% | 75.07% |

Recent width falls only 2.25% and interval score improves only 0.109%, while
validation score worsens 0.101%. Cash dispersion remains better than residual
dispersion and global ranges for both mids and both combined periods. There is
little support here for automatically deleting FY2020.

This removes only FY2020 forecast errors. Pandemic years remain in historical
features/forecast inputs; FY2021 was already outside the calibration period.
The sensitivity does not measure a fully pre-COVID model's pandemic performance.
That requires a genuine pre-shock information cutoff, followed by shock/recovery
evaluation. Future model selection should use rolling chronological tests and
untouched outcomes; see
[time-series cross-validation](https://otexts.com/fpp3/tscv.html).
Provider cash-definition changes and later-vintage source data remain unresolved.

Artifacts in the desktop worktree:
`desktop/test-results/cash-flow-segmentation-2026-09-12/covid-sensitivity/`.
The script is `desktop/scripts/cash-flow-covid-sensitivity.py`.
The sensitivity reproduces all 36 original metric cells before comparing its
72 total cells. A separate scalar root check reconciled eight linear-model
metric cells and their factors without the experiment helper. Three focused
tests cover reweighting/omission, unchanged evaluation rows and exclusion of
future errors from calibration. Original source and experiment hashes remain
unchanged.
