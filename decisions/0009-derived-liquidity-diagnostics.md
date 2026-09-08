# ADR 0009 — Read-only, unscored derived liquidity diagnostics

**Status:** Accepted

**Date:** 2026-09-08

## Context

ADRs 0007 and 0008 established auditable official-money, central-bank-assets,
money-market-fund, repo and offshore-credit histories without collapsing unlike
measures into a synthetic M5. Those raw histories make calculation possible,
but they do not yet answer the smaller investor's first practical questions:
which monetary quantities are expanding or slowing, where short-term funding
conditions differ from policy, and which observations matter at different time
horizons.

A useful interpretation layer must retain two distinctions. First, an economic
observation date is not the date on which this system could have known the
release. Second, co-movement or relative growth is not evidence that money moved
from one identified sector to another or that one series caused another. The
first archived official-money and liquidity-frontier release clocks begin on
2026-09-08; older dates inside those releases are revised/current-vintage
history, not reconstructed real-time vintages.

## Decision

1. **Generate a separate read-only brief.** `dalio-liquidity-brief` opens
   `data/dalio.db` in SQLite read-only mode and writes generated exports only.
   It does not add derived database rows or modify source releases. Complete
   coverage is required by default; `--allow-partial` is an explicit diagnostic
   override that preserves unavailable states rather than filling data.
2. **Expose both query clocks and both availability cutoffs.** `as_known_at`
   (CLI `--known-at` or `--as-known-at`) selects the newest complete release
   available for each partition before any observation-date filter is applied.
   `as_of` (CLI `--as-of` or `--through-date`) then caps the economic dates used
   in the calculation. `earliest_input_available_at` is the minimum selected
   non-policy official-money/frontier availability clock and therefore marks
   partial raw coverage only. `complete_snapshot_available_at` is the maximum of
   those clocks only when every pinned analysis partition, including the FRED
   EFFR benchmark, is present; otherwise it is `null`. In the validated live
   initial set these are `2026-09-08T20:38:47.888222+00:00` and
   `2026-09-08T20:43:28.624056+00:00`, respectively, so the latter is the first
   full-snapshot replay cutover. A historical chart from one selected release is
   labelled `current_vintage_history`; no pre-cutover vintage is synthesized.
3. **Publish a traceable, versioned contract.** Each run writes machine-readable
   JSON and a plain-language Markdown rendering as fixed
   `data/snapshots/liquidity_latest.{json,md}` aliases plus content-addressed
   `liquidity_YYYY-MM-DD_<snapshot-hash-prefix>.{json,md}` exports. The prefix is
   the first 16 lowercase hexadecimal characters of the full `snapshot_sha256`,
   so distinct contents for one economic date do not overwrite one another.
   The JSON carries the snapshot and methodology versions and hashes,
   source-catalogue hashes, coverage and unavailable states, input
   series/dates/releases, both availability cutoffs, release clocks and artifact
   manifest checks. The legacy FRED EFFR policy benchmark is identified as
   `not_required_for_legacy_benchmark` rather than falsely presented as having a
   raw artifact manifest. The fixed `latest` aliases move on a successful run;
   hash-addressed exports remain regenerable derived files, not source artifacts.
4. **Keep five diagnostic families independent.** They are displayed together
   for orientation but are never reduced to one score:

   - **Broad-money impulse:** for primary US M2, euro-area M3, Swedish M3,
     monthly UK M4ex and Japan broadly-defined liquidity `L`, calculate
     `100 × ln(x[t] / x[t-12m])` and the three-month change in that annual growth
     rate. An equal-country median and positive/accelerating breadth are shown
     only when all five are ready **and share one common period**; otherwise the
     summary fields remain empty and the constituent periods remain visible. No
     currency levels are summed. Quarterly UK M4ex, headline UK M4 and Japan M3
     remain excluded diagnostics rather than additional votes.
   - **Money versus central-bank assets:** for the United States and euro area,
     calculate annual log growth in broad money and in the relevant central-bank
     assets, anchored to the latest weekly observation on or before each month
     end, then report
     `money annual growth − central-bank-assets annual growth` and its
     three-month change. The gap is descriptive, not proof of transmission.
   - **US MMF relative expansion and asset allocation:** calculate annual log
     growth in total MMF investments, its difference from US M2 annual log
     growth, MMF-assets-to-M2 scale and named asset shares
     `100 × component / declared parent` with twelve-month changes. Keep the
     separately available OFR-published repo counterparty/clearing-category
     ratios explicitly non-additive. FICC is a clearing category, not an ultimate
     borrower, and the ratios do not identify an end-to-end cash map. These are
     outstanding stocks and relative measures, not observed deposit flows.
   - **Repo pricing and activity:** for selected DVP, GCF and tri-party-ex-Fed
     venues, calculate five-aligned-business-day median premiums to EFFR in
     basis points and the five-day median cross-venue rate range. Robust anomaly
     context is `0.67448975 × (current − median(previous 252)) /
     MAD(previous 252)`. Venue volumes remain separate context with no assumed
     risk direction; neither measure is a pure credit spread or a stress score.
   - **Offshore reserve-currency credit:** for BIS USD, EUR and JPY stocks,
     calculate `100 × ln(x[t] / x[t-4q])` and the one-quarter change in that
     annual growth rate. Each currency remains separate. Negative growth means
     arithmetic contraction and a negative change means deceleration; neither is
     a claim of financial tightness or an investment verdict.
5. **Fail rather than manufacture comparability.** Nonpositive stocks,
   non-finite values, missing exact calendar lags, stale inputs, insufficient
   aligned history and invalid evidence manifests make a diagnostic unavailable.
   Publisher missing values are not zero-filled or forward-filled. Stocks,
   rates, transaction volumes and different currencies are not added.
6. **Use horizons as context, not forecasts.** Repo pricing/fragmentation and
   MMF allocation provide 0–12-month market-plumbing context. Broad-money,
   central-bank, MMF and offshore-credit momentum provide 1–3-year backdrop.
   The 3–5-year-and-longer horizon is explicitly not assessed by this layer
   alone and still needs debt, fiscal, pension, demographic, productivity and
   human-reviewed report evidence.
7. **Forbid causal and portfolio claims.** The brief contains no composite
   liquidity, risk, M5 or investment score; no automatic allocation or buy/sell
   instruction; and no claim that deposits moved into MMFs, that an OFR repo
   category identifies the ultimate lender or borrower, or that central-bank
   balance-sheet changes caused broad-money changes. It cannot yet answer
   “money moved from A to B because X.”

## Consequences

The Observatory gains a reproducible first interpretation layer that a small
investor can read without losing the audit trail. JSON supports downstream
analysis, while Markdown states arithmetic direction, coverage, horizon use and
limitations in plain language. Separate release and economic-date cutoffs make
future point-in-time comparisons possible as new releases accumulate, while the
hash-addressed dated name preserves distinct snapshots for the same economic
date and different known-at contents.

The output is intentionally less decisive than a single liquidity gauge. It
cannot turn revised pre-cutover history into a real-time backtest, identify
causal transmission, infer deposit migration, forecast returns or replace the
still-open scenario and SEK-household portfolio work. Institutional-report
conclusions also remain unavailable until named human review creates verified
claims.

## Rejected alternatives

- Sum official money, central-bank assets, MMF assets, repo volume and offshore
  credit into a global liquidity or M5 total.
- Convert heterogeneous levels to one currency and treat the result as a stock
  with a stable economic perimeter.
- Blend the five families into a composite risk or investment score.
- Describe MMF-versus-M2 relative expansion as a measured deposit-to-fund flow.
- Treat cross-venue repo pricing as a pure credit spread or venue volume as
  having an automatic risk direction.
- Key every dated export only by `as_of`, causing distinct known-at snapshots for
  one economic date to overwrite each other.
- Backfill releases before the first complete-snapshot cutover or present
  current-vintage history as a known-at-the-time backtest.
- Persist derived diagnostics in the source database before the methodology and
  use case require a separate immutable derived-release design.
