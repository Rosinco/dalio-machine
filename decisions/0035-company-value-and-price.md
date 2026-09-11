# ADR 0035 — Value and price in company analysis

Status: adopted methodology and interactive workspace, 2026-09-11.

## User direction

The user requested that the value-versus-price framework developed in the
Buffett/tangible-capital discussion become part of Macro Atlas company analysis.
The user then explicitly requested an interactive workspace, high/mid/low NPV
and DCF charts, and years to payback for each scenario.

## Decision

Use the [company valuation framework](../docs/company-valuation-framework.md)
and [reusable analysis template](../docs/company-analysis-template.md) for new
company analyses. Compare a range of continuing-business equity values and
separate, feasible recovery/breakup scenarios with dated equity market prices.
DCF estimates present value; NPV subtracts the purchase price. Recovery values
are estimates, not guaranteed downside floors, and are not added to the full
going-concern valuation.

Connect tangible capital, normalized returns and verified free cash flow to the
forecasts. Reconcile surplus cash, gross debt and other claims without double
counting. Macro/branch observations require sourced company exposures and an
explicit transmission mechanism before they affect a valuation assumption.
Growth, moat, reinvestment, uncertainty, financing risk and the path to shareholder
cash realization remain visible. Preserve dates, currencies and missing evidence.

## Integration and scope

The method and template are documented in the canonical engine and desktop
working trees, with entry points in project guidance, READMEs and handoffs.
Company valuation remains downstream research; it does not change macro facts
or introduce a buy/sell score into the evidence database.

The desktop implements Companies → Value as an equity-distribution DCF workspace.
Analysts enter a dated price, annual payments, required equity returns and optional
final net equity sale proceeds for low/mid/high scenarios. DCF and cumulative NPV
charts retain named paths and shade their range. Year-end cash timing determines
ordinary and discounted payback, with non-recovery and later reversal explicit.
Cash-only payback and payback including sale remain distinct. Net liquidation or
breakup recovery is a separate alternative, never an addition to operating value.

Company capital and financing inputs, scenario explanations and macro/exposure
notes accompany the calculations. Working drafts autosave by company and exact
data versions; saved study revisions preserve old assumptions. CSV exports retain
dated price inputs, annual calculations and source versions. The first workspace
requires analyst cash forecasts; it does not manufacture new company valuations
from the vendor FCF series. Installation/build verification is recorded in the
desktop handoff, not inferred from this methodology decision.

The user subsequently requested automatic population for companies already
researched, with Holmen as the first test. Version 0.11 loads a reviewed numerical
study using the matching listing/ISIN and archived deep-dive source hash. Source
facts, normalization choices and scenarios remain distinguishable. It preserves
earlier work, does not overwrite entered forecasts on reopening, and records
study origin in saved drafts/revisions and CSV. Other dossiers require their own
reviewed numerical study before joining this automatic path. See the
[Holmen worksheet](../docs/holmen-valuation-2026-09-11.md).

## Verification

Review the formulas, example arithmetic, linked files and consistency of the
method/template across the active working trees. Documentation changes do not
alter runtime calculations or require a new API collection.
