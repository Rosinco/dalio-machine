# ADR 0044 — Survival and portfolio stress before accepting valuation

Date: 2026-09-26. Status: accepted by user direction.

## Context

The user asked whether Buffett and Munger's criticism of estimated tail risk
changes DCF/NPV or the analysis funnel, then approved the proposed improvements.
Existing scenario ranges, terminal-credit sensitivities and a temporary crisis
path do not establish probabilities of permanent loss or survival until recovery.

## Decision

Keep all existing valuation arithmetic, starter calibration, authored drafts,
saved studies, source packs and position-size thresholds. Add separate local
research worksheets accessible in Company → Research notes:

1. A company liquidity schedule, reverse stress and evidence review. Record
   unrestricted cash, minimum operating liquidity, period cash after mandatory
   spending, debt principal, incremental draws from available committed funding and combined
   shock drains. Each draw uses remaining capacity, never the full facility again.
   Show every period's headroom and preserve an earlier breach even if later cash
   recovers. Uniform additional drain reaching the liquidity boundary is a
   sensitivity, not an estimated probability or a complete insolvency model.
2. A permanent-impairment alternative for cash reaching the existing shareholder
   claim after financing and rescue dilution. Discount annual payments and final
   net proceeds once. Keep the result separate from continuing-business values.
3. A manual portfolio shared-shock worksheet with disjoint starting weights and
   assumed losses. Sum weight × loss, disclose unmodeled residual exposure, and
   compare with the user's own stated tolerable loss. Do not infer diversification,
   assign probabilities or silently change the existing concentration policy.

Valuation displays the saved company review status and links to the review.
Unassessed, incomplete, stale and material-failure states remain visible; a
completed conditional worksheet is not an investability score or formal gate pass.
Formal Börsdata structural vetoes continue to precede valuation and price reveal.

Records have independent versioned storage keys, explicit saves, source identity,
and protection for unreadable/future records. Opening the review writes nothing.
No data migration or automatic holding/financial-input import is introduced.

## Consequences and limits

The worksheet makes assumptions inspectable without pretending to calibrate
unseen catastrophes. Period-end balances can miss intra-period shortages;
covenants, collateral, refinancing feasibility and dilution require evidence and
analyst judgment. Multiple stress cases can still omit important outcomes.
DCF values, margins of safety and modeled bear losses never become loss bounds.

The framework/template move to methodology version 2. Documentation is mirrored
between canonical main and the desktop worktree; application code stays on
`feat/offline-atlas`. This source change is verified separately from publishing or installation;
the installed Windows release remains unchanged.
