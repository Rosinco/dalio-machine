# ADR 0039: Editable purchase-price range from scenario equity value

Date: 2026-09-13. Status: adopted for Macro Atlas 0.16.0. Build, verification
and installation status are recorded separately in the desktop handoff.

## Decision

Every company retains the same purchase-price workspace, including companies
whose incomplete cash inputs cannot yet produce a numerical ceiling. The user
selected a **30% editable margin of safety**. The initial reference is Mid.

```text
V_s = present value of forecast equity cash + present value of final equity sale
maximum purchase equity price_s = V_s * (1 - margin / 100)
NPV_s(P) = V_s - P
```

Terminal sale is resolved from the existing scenario once. NPV is value less
purchase cost; it is not another value to add to DCF. The final-sale checkbox on
the cumulative NPV display does not remove terminal sale from this total-value
purchase calculation. Recovery and crisis cases remain separate alternatives.

Show each named scenario's ceiling without relabelling scenarios when they cross.
The chart uses proposed price on the x-axis and NPV on the y-axis. Its shaded
positive-price region extends to the selected ceiling. There is no economic
minimum purchase price: lower positive prices increase the modeled discount.
The numerical span between all three ceilings compares scenario assumptions;
it is not an instruction to pay at least the lowest ceiling.

Nonpositive values retain signed DCF and NPV but have no positive purchase ceiling.
A 100% margin also admits no positive price. Missing or invalid cash, terminal,
return, margin or reference inputs remain explicit. The cash-only ceiling and
terminal share of value expose dependence on the final sale; cash-only value is
not a liquidation or recovery floor. Thirty percent is a user policy, not an
empirically optimal discount, probability or additional annual required return.

## Price and ownership basis

Cash-based value does not require a market quote or hypothetical stake size.
When no candidate price has been authored, an available positive dated study
market capitalization supplies the initial candidate. Its date/source are shown;
it is not a live quote. An explicitly cleared candidate remains blank. A manually
entered positive candidate can be compared even without any market quote.

The initial display is total common-equity value in valuation-currency millions.
Per-share conversion divides both chart axes and all amounts by shares in
millions. Require a positive share count, valid date no later than valuation,
source/ownership description and matching valuation currency. Do not infer shares
by dividing market capitalization by a price or parse ownership from prose.

Reviewed Holmen/SCA share suggestions are bound to their exact dated studies and
ownership evidence. Generic reported-share suggestions require matching source,
timing, currency, original price and arithmetic. They are explicitly unreviewed
ownership proxies. Applying any reference is an explicit user action. Changed or
incompatible bases must be entered/reviewed by the user; a saved reference never
silently overwrites the study. Treasury shares, classes, dilution, receipts and
the equity claim represented by cash remain company-review questions.

## Persistence and verification

Purchase settings are an optional draft extension. Old drafts keep their exact
shape, and viewing the default does not author a policy. Preserve authored
settings and deliberate blanks through save/reload, revisions, history application
and deliberate baseline/reviewed resets. A purchase-only edit prevents automatic
replacement; it does not relabel unchanged forecast cash as edited.

Provide a separate purchase CSV with each named scenario, cash/terminal values,
margin, candidate, ceiling, NPV, display basis, source versions and missing/error
states. Validate independent arithmetic, terminal timing/counting, ownership
units, missing inputs, crossings, persistence, CSV and browser/native behavior.
Compare all universe starters with the retained 0.15.0 valuation runtime to detect
calculation drift. These checks establish implementation correctness, not future
return accuracy or company investability.

The cash/value distinction follows [NYU's terminal-value framework](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm).
The price/value gap is discussed in [Berkshire's 1992 shareholder letter](https://www.berkshirehathaway.com/letters/1992.html);
neither source supplies Atlas's user-selected 30% policy.
