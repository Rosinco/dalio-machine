# ADR 0040: An explainable research screen for every saved listing

Date: 2026-09-13

Status: Accepted by user direction; release evidence is recorded in the handoff.

## Context

The user wants to choose companies for deep dives from the downloaded universe
without mistaking generic DCF scenarios for reviewed valuations. A feasibility
audit found 19,140 saved listings with uneven statement and valuation coverage.
The broader raw archive contains additional prices, R12 reports and KPI histories;
these require separate import and ownership/date reconciliation before use.

## Decision

Add a universe research table, a branch-scoped entry point and a read-only company
summary card. Present historical cash/profit persistence, recent comparable
quarter changes, financing and asset proxies, and data readiness separately.
Search, pagination and declared filters narrow research questions. Alphabetical
order is the default; no overall score, buy verdict or forecast probability is
created. Counts refer to listings, including older-only identities and potentially
dependent share classes or cross-listings.

The first slice uses the current verified financial companion: annual and
quarterly reports and its dated publication-window market values. It does not
silently replace these with newer raw closes. Missing, zero, negative,
incompatible, unpublished and placeholder data retain distinct meanings.
Historical indicators expose the selected periods, eligibility, counts,
currency and source. An invalid latest period is not silently replaced by an
older apparently healthier observation. Keep COVID and recovery observations.

Valuation context is optional. It uses a frozen standard starter, separate from
reviewed studies and local working drafts. Display cash PV, terminal PV, dated
equity price, Low NPV and the required cash factors P/V and P/((1-MOS)*V), where
P and V are strictly positive and the margin is explicit. These factors scale
annual and terminal cash together, holding horizon, rates and growth fixed;
they are not growth forecasts. Signed cash includes funding requirements.
Terminal value is included once; DCF, NPV and terminal value are not independent
votes. A 30% margin is a user policy, not a model-error guarantee.

Keep operating, property, financial and unclassified routes visible. Financial
businesses require reviewed equity-cash/capital inputs. Property histories and
asset ratios carry definition limits. Provider FCF is not verified owner cash;
aggregate liabilities are not renamed debt, and tangible asset intensity does
not establish maintenance capex or competitive advantage. Missing data means
reconciliation is needed, not that the business has failed a research gate.

Generate a compact deterministic asset offline, bound to the exact financial
pack, taxonomy, company source hashes and model/calibration identities. Pin its
hash and byte limits in application source. Verify the asset before displaying
any rows and withhold results when the selected pack or directory differs. The
screen does not load all company histories at runtime or mount the draft-writing
valuation workspace. Explicitly opening Value remains a separate user action.

## Consequences and validation

This is descriptive candidate exploration for the native Börsdata deep-dive
workflow. It does not write formal research gates, strategy filters or queues,
and does not claim an investment edge. Further validation must use chronological
forecast origins, issuer-aware evaluation and separately reported crisis periods;
sector sample sizes alone do not calibrate uncertainty.

Verify exact universe/source coverage, independent arithmetic, signed and missing
cases, source/asset corruption rejection, bounded rendering and filtering,
desktop/mobile behavior, offline operation and unchanged local studies. Preserve
prior release artifacts before replacing shared test reports. Record build,
Windows testing and installation independently in the handoff.
