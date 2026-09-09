# ADR 0017 — Bounded Version 1 completion gate and analysis pivot

**Date:** 2026-09-09 · **Status:** accepted · **Milestone:** Version 1

## Context

The Observatory now has a substantial raw-data foundation, but collection can
expand indefinitely: every new economy, allocator, funding market and
institution adds another plausible archive. More records do not by themselves
answer the product's central question—what the next one to five years may look
like, how shocks could travel, and what that means for a small SEK-based
investor.

The current foundation is informally estimated at roughly 80 percent of what a
useful Version 1 needs. That percentage is a planning estimate, not a measured
coverage statistic. Version 1 therefore needs a bounded stopping rule based on
named evidence packages and explicit residual gaps.

The user also chose to continue without a human communications-rights review.
That choice permits safe link and availability metadata work, but it does not
clear transcript, subtitle, CEO-letter or other communication content for
capture, extraction or analysis.

## Decision

### 1. Cap the remaining foundation work

Before the analysis pivot, collection is limited to these five packages:

| Package | Planning bound |
|---|---:|
| Riksbank and Reserve Bank of Australia 2025 communication metadata | about 16 closed-denominator events |
| Debt maturity and refinancing structure for the core economies | about 25–50 series or source partitions |
| Historical pension- and sovereign-fund allocation evidence | about 40–80 annual fund snapshots |
| Cross-border banking, collateral, FX-swap and funding-stress evidence | about 20–40 series |
| Selected historical bank and commodity-company communication metadata | about 100–250 records |

The ranges are capacity bounds, not quotas and not invitations to fill missing
official evidence with estimates. A package may close below its range when the
first-party source is unavailable, definitions are not comparable, or an exact
absence is the honest result. Any materially different collection programme
requires a later decision.

### 2. Prefer decision value over record count

Each numeric package must preserve source-native definitions, point-in-time
clocks where available, missingness and content-addressed first-party evidence.
Stock, transaction, rate, collateral and estimated-change evidence remain
distinct. Coverage breadth never becomes an equal-weight score merely because
more series exist.

Communication packages remain metadata-only while rights are pending. They may
close event denominators and record exact first-party landing or artifact links,
representation type, availability and provenance. They may not download or
retain communication content, infer clearance from personal use, generate local
transcripts, summarize conclusions, score language or feed scenarios.

### 3. Define the Version 1 exit gate

Version 1 foundation work is finished when:

1. the core numeric, debt, allocator and flow evidence is sufficient to build
   traceable scenarios, with every important remaining blind spot named;
2. the priority 2025 central-bank communication denominators and
   representation metadata are complete under the metadata-only contract;
3. the five bounded packages are completed, explicitly unavailable, or
   documented as lower-value than proceeding to analysis;
4. no unresolved source gap is silently imputed and no communications content
   is treated as analysis-ready; and
5. the next deliverables are the horizon risk cases, transmission pathways,
   warning indicators and SEK-investor brief—not another general collection
   expansion.

The analysis layer must distinguish evidence, inference and scenario. Each risk
case will state its horizon, causal pathway, supporting evidence, signposts,
invalidators, confidence and SEK-household exposure. It remains decision support,
not an automatic buy/sell instruction.

### 4. Defer the complete communications vision

A ten-year archive across major central banks, banks and commodity companies is
a separate later expansion, plausibly involving hundreds of events and
documents. Until a suitable rights and semantic-review route exists, Version 1
does not require that archive and cannot represent it as an analysis-ready
corpus.

## Consequences

The project now has a stopping rule. Work can be prioritized by whether it
closes one of five named blind spots or enables the risk layer, rather than by
whether another dataset is collectible. The ranges make the remaining workload
visible while preserving the right to stop early when first-party evidence does
not support comparable data.

Version 1 may retain meaningful gaps. That is preferable to delaying all
analysis or manufacturing false completeness. The longer communications archive
and any content-based institutional-language research remain separately gated.

## Rejected alternatives

- Continue collecting every plausible macro or institutional source before
  producing scenarios.
- Treat the approximate 80-percent planning estimate as a measured readiness
  score.
- Require an arbitrary record quota even when first-party evidence is absent or
  incomparable.
- Infer communications-content permission from personal research use.
- Treat metadata links as a transcript corpus or as evidence of an
  institution's conclusions.
- Produce a single opaque risk score or automatic portfolio instruction.
