# ADR 0008 — National broad money and the liquidity frontier

**Status:** Accepted
**Date:** 2026-09-08

## Context

The first money/liquidity slice retained US M2, euro-area M3, Swedish M3 and
central-bank balance-sheet quantities in their native definitions. That is not
enough to observe market-based finance. Repo, money-market funds and offshore
reserve-currency credit can transmit stress even when conventional broad money
looks stable.

M4 and M5 are not international rungs of one consistent ladder. The current UK
M4 perimeter differs from historical US M4, while US and UK M5 were discontinued
and had different definitions. Euro-area M3 already contains repos, money-market
fund shares and short MFI debt. A synthetic global M5 would therefore mix
perimeters, currencies and duplicate claims.

## Decision

1. **Retain ten publisher-defined official-money series.** Keep US M2, Federal
   Reserve total assets, euro-area M3, Eurosystem total assets and Swedish M3.
   Add monthly UK M4 excluding intermediate other financial corporations
   (`RPMB53Q`) as the primary UK broad-money history, headline M4 (`LPMAUYN`)
   as a diagnostic, and the native quarterly M4ex series (`RPQB53Q`) as a
   diagnostic historical bridge back to 1997-Q4. The quarterly bridge has the
   same perimeter as monthly M4ex and overlapping quarter ends agree; it is not
   a second observation or vote. Add the Bank of Japan's broadly-defined
   liquidity `L` as Japan's primary extended-liquidity history and M3 as its
   narrower diagnostic. Publisher labels, perimeters, parent relationships,
   non-additivity groups, native currency, unit, adjustment, observation basis
   and definition breaks remain explicit.
2. **Do not create a live generic M5.** Historical official M5 observations may
   later be archived for regime research, but no constructed series may be
   presented as a current official aggregate.
3. **Create a separate 22-series liquidity-frontier domain.** Three BIS series
   retain USD-, EUR- and JPY-denominated credit to non-bank borrowers outside
   each issuing currency area. Nineteen OFR series retain eleven monthly US
   money-market-fund asset totals/components and eight daily observations of
   rates, outstanding volume or transaction volume in selected DVP, GCF and
   tri-party repo venues. These are credit stocks, asset stocks, market volumes
   and rates—not extra units of national broad money.
4. **Represent only the links the source actually identifies.** Every frontier
   series records its measure kind, claim side, published counterparty or issuer
   category where available, instrument, collateral scope, aggregation role,
   parent series where applicable and non-additivity groups. OFR MMF asset
   series can describe allocations and published repo-counterparty categories.
   OFR's aggregate repo-venue rate and volume series do not identify the cash
   provider, dealer or ultimate borrower, so both directional sector fields are
   explicitly unavailable. Totals and components, overlapping MMF and venue
   quantities, outstanding stocks, transaction volumes and rates must never be
   summed as if independent money.
5. **Keep native quantities and cadence.** Currency stocks remain in native
   units. Quarterly BIS credit, monthly MMF positions and daily repo observations
   are not silently resampled or converted. Cross-currency aggregation requires
   a later explicit constant-FX method.
6. **Use complete immutable native-series releases with fail-closed guards.**
   Retrieval time is the conservative availability clock where publishers do
   not expose an unambiguous historical publication timestamp. Exact identity,
   metadata, start, minimum history, cadence, recency, numeric validity and
   stored-history contraction are checked before projection. Complete monthly
   histories must have no gaps; sparse OFR counterparty histories retain native
   null periods rather than filling them. Daily repo histories must meet
   source-specific maximum internal-gap and minimum weekday-coverage thresholds,
   and measures from the same venue must carry aligned native dates. Quarterly
   BIS and BoE histories remain quarterly.
7. **Retain reconstructable source evidence.** Validated official-money and
   BIS/OFR response bodies are archived by the SHA-256 of their exact retained
   bytes. Every official-money release retains a canonical non-imputing
   missingness ledger; each OFR release also retains its canonical native-series
   payload and missingness ledger. Immutable `data_release_artifacts` manifests
   bind each source response and per-series evidence file, its full hash and
   canonical missingness provenance to the release; inventory readiness reopens
   and rehashes those files instead of trusting a locator or abbreviated vintage
   tag. Provider-specific frontier catalogues are archived separately, and
   catalogue-semantic hashes bind every release to its interpretation.
8. **Do not create an additive headline or score the new data yet.** Inventory
   reports raw readiness and economic roles only. Growth, acceleration,
   policy-rate spreads, deposit-to-MMF migration and a multidimensional
   liquidity regime require a later decision and point-in-time validation.

## Initial source perimeter

- Bank of England Interactive Statistical Database: monthly M4ex, its native
  quarterly historical bridge and headline M4.
- Bank of Japan Time-Series Data Search: M3 and `L`.
- BIS Global Liquidity Indicators: three currency-specific credit totals covering
  bank loans plus international debt securities owed by non-bank borrowers
  outside each issuing currency area.
- US Office of Financial Research Short-term Funding Monitor: eleven MMF asset
  totals/components and eight selected repo-venue rate/volume series.

FSB annual non-bank financial intermediation, ECB money-market/MMF detail, SEC
fund-level N-MFP filings, commercial paper, dealer financing, collateral reuse,
FX swaps and stablecoins remain later layers. They require their own coverage
and double-counting decisions; NBFI assets and derivative notionals are not money
stocks.

## Consequences

The Observatory can distinguish official money from market-based funding and
can observe how MMFs allocate assets, the counterparty categories OFR publishes,
and the price and scale of activity in selected repo venues. It cannot infer an
end-to-end cash route, identify the ultimate lender or borrower in aggregate
venue data, or establish why money moved. UK and Japan gain official
extended-liquidity coverage without pretending their aggregate names are
cross-country equivalents.

The catalogue is deliberately more verbose than a single liquidity number. That
extra structure is necessary to avoid false precision, cross-currency distortion
and double counting. The initial layer improves evidence for funding stress and
forced-sale risk but does not by itself establish causation or an investment
signal.

## Rejected alternatives

- Treat M1 through M5 as a universal ordinal taxonomy.
- Reconstruct and publish a current global M5 from whichever components exist.
- Collapse official money and the liquidity frontier into one headline number.
- Add MMF assets, repo volumes, broad money and central-bank assets together.
- Treat all non-bank financial assets as runnable shadow money.
- Treat transaction volume or derivative notional as an outstanding money stock.
- Convert current stocks at spot FX and call the sum global liquidity.
