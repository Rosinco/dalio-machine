# ADR 0004 — Dalio-machine as the canonical Macro History & Risk Observatory

**Date:** 2026-09-08 · **Status:** accepted · **Slices:** 27+

## Context

The desired product is one historical account of the world economy: what
happened, what important institutions said at the time, how debt and large
institutional portfolios developed, where capital moved, which explanations are
supported, and what the resulting 1–5 and longer-term risks mean for a small
SEK-based investor.

Dalio-machine already owns the country registry, provider mappings, macro ETL,
cycle and fundamentals history, trade relationships, pressure chains, snapshots,
tests and UI. The Börsdata repositories contain useful Swedish adapters and a
measured megatrend catalogue, but their company, branch and regression judgments
have a different responsibility. A new umbrella repository or a whole-history
merge would duplicate infrastructure and blur the source of truth.

The existing `observations` table is also insufficient for a genuine historical
record. It keeps only the latest value for `(country, indicator, date, source)`;
updates overwrite revisions and an IMF refresh removes superseded forecast rows.
Historical replay can therefore answer “what is now reported for that period,”
but not “what was knowable then.”

## Decision

1. **One upstream macro product.** Dalio-machine becomes the sole canonical
   source of macro facts and macro interpretations. The user-facing name broadens
   to **Macro History & Risk Observatory** while repository, package and CLI
   names remain stable during migration.
2. **One-way ownership.** Dalio publishes versioned context; Börsdata and stock
   analysis consume it. Company verdicts, branch crosswalks, B2/B4 chains and
   company regressions never flow back into the macro fact base.
3. **Selective migration.** Port Börsdata's useful Riksbank/SCB adapters,
   catalogues, fixtures and tests after mapping them to Dalio semantic indicators.
   Reconcile rather than duplicate FRED/BIS series; clearly quarantine proxies.
4. **Megatrends become testable topics.** Generic topic, mechanism, measured
   series, evidence class and verification fields move into an upstream topic
   catalogue. Company-chain links remain downstream. The April cascade is a
   dated hypothesis archive, not current truth or an automatic score.
5. **Separate evidence types.** Measured facts, institution forecasts,
   institution judgments, stated intentions and our inferences remain distinct.
   “Why money moved from A to B” is a linked explanation with alternatives and
   confidence, never silently stored as an observed flow.
6. **Four histories.** The product preserves what happened, what institutions
   said, what the system inferred, and what the investor decided. Later
   corrections never erase earlier knowable states.
7. **Complete append-only releases.** `observations` remains the compatible
   latest-value projection. New `data_releases` and `release_observations` tables
   store immutable, complete snapshots of stable source partitions. Historical
   queries select the latest complete release available per partition; they do
   not select the latest row independently. This makes rows omitted or retracted
   by a later release disappear without destructive tombstones. One source event
   is identified by its stable partition plus availability and retrieval clocks;
   reusing those clocks with different content or provenance is an integrity
   error. A partition cannot silently change source family or scalar identity.
8. **Three clocks.** A release records the period represented by each
   observation, when the release became available, and when it was retrieved.
   Publication time is stored separately when known. Unknown publication times
   stay null; they are never inferred from the observation period.
9. **Data-preserving migration.** SQLite remains sufficient initially. Schema
   changes are additive where possible. A constraint-only table rebuild is
   permitted when SQLite cannot alter it in place, but only with an exact
   backup, copied-row and foreign-key checks, and no rewriting of evidence
   values or clocks. The rebuild preserves explicit indexes and triggers and
   restores append-only protection before committing. A conservative explicit
   bootstrap seeds legacy current values into the release ledger at the chosen
   cutover time.
10. **Broader future schema.** Later slices add provider-series metadata,
    entities, documents and page-cited claims, events, financial-flow
    observations and horizon-specific risk cases. JSON/CSV/Parquet remain derived
    exports rather than competing sources of truth.

## Compatibility and honesty gates

- Existing CLI names, cycle/fundamentals behavior and snapshot-v1 fields remain
  valid until an explicitly versioned cutover.
- `fundamentals_latest.json` and `jurisdiction_tier.csv` retain their downstream
  meanings and paths.
- Every source partition is independently replaceable; a subset refresh cannot
  imply deletion of another country or series.
- Empty or ambiguous snapshots fail closed and cannot erase current rows.
- An incoming snapshot equal in content, vintage label and publication time to
  its chronological predecessor is idempotent, while a real A → B → A sequence
  remains three vintages. A URL locator alone is not semantic release identity.
  Out-of-order backfills may leave adjacent equal immutable events when both are
  needed to restore the earlier availability boundary; they cannot roll the
  current projection backwards.
- Release ingestion owns and commits or rolls back its supplied session
  transaction. A multi-partition refresh that requires all-or-nothing behavior
  binds that session to one explicit outer connection transaction.
- Report conclusions require document and page provenance plus extraction/review
  status before display.
- New prose, report claims and megatrend hypotheses cannot enter numeric scoring
  without another explicit decision.
- Point-in-time replay must use releases available at the chosen date and
  expanding-window or pre-fixed thresholds; full-future-history calibration is
  forbidden.
- Capital visuals label positions, transactions and valuation/FX-adjusted
  estimates separately. Motives are stated, derived or hypothetical.
- Tests remain offline and all existing tests plus Ruff must pass.

## Staged delivery

1. Boundary, release-ledger contracts, conservative bootstrap and one FRED
   end-to-end tracer.
2. Cut remaining numeric pipelines over to stable complete partitions; make IMF
   history/forecast replacement atomic; add honest point-in-time replay.
3. Consolidate official Riksbank/SCB ingestion.
4. Add Riksbank, ECB, Fed, IMF and BIS document/release/claim history.
5. Add sovereign debt anatomy and Swedish AP-fund allocation history.
6. Add separately labelled bank, portfolio, FDI, reserve and institutional-flow
   networks.
7. Add horizon risk cases, SEK household exposure and the recurring “what
   changed?” investor brief.

## Consequences

There is one coherent history of the world economy without turning macro
narratives into stock scores. Storage cost rises because source snapshots are
immutable and sometimes repeat values; the gain is reproducibility, omission
semantics, auditability and honest evaluation of forecasts and decisions.

## Rejected alternatives

- Merge whole repositories or git histories.
- Create a fourth umbrella repository.
- Maintain two canonical macro stores.
- Keep only changed observation rows without explicit tombstones.
- Encode financial counterparties inside indicator-name strings.
- Treat holdings changes as transactions without valuation/FX reconciliation.
- Automatically score report prose or megatrend narratives.
