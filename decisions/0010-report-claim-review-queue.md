# ADR 0010 — Read-only report-claim review queue

**Status:** Accepted and implemented

**Date:** 2026-09-09

## Context

ADR 0005 established an immutable report ledger with original PDFs, complete
physical-page extraction, atomic claims, exact citations and named human
semantic review. The live ledger contains ten documents and 852 extracted
pages, but no verified claims. Those pages are useful source material, not yet
approved conclusions.

The implemented increment is a small, inspectable review queue. It
must help a human find and evaluate the most important candidate statements
without allowing a model-generated paraphrase to become a verified claim, a
risk score or an input to investor guidance. It must also avoid silently
preferring an older, easier-to-summarise report when a newer eligible issue is
already present.

## Decision

### 1. Use a read-only packet builder

The `dalio-report-review-packet` command opens `data/dalio.db` in
SQLite read-only mode. It may read report documents, complete extractions and
page text, but it will not insert, update or delete documents, claims, citations
or review decisions. Its only writes will be generated files under
`data/review/`.

### 2. Select one current issue from each pinned family

At the explicit `as_known_at` cutoff versioned in the candidate catalogue, the
builder selects the latest eligible issue from each of these five report families
and then requires its hash-bound extraction to be complete:

1. Riksbank Monetary Policy Report;
2. ECB/Eurosystem staff macroeconomic projections;
3. Federal Reserve Monetary Policy Report;
4. IMF World Economic Outlook; and
5. BIS Annual Economic Report.

Selection uses document availability before report period or page filtering.
The selected source/issue identity, artifact SHA-256, extractor name/version and
corpus SHA-256, availability clock and cited page hashes remain visible in the
packet. The default command fails visibly if any enabled family is missing or if
the latest issue lacks the declared complete extraction; it never substitutes an
older issue merely because that issue is easier to process.

The queue is deliberately small: at most four atomic candidate claims may be
included for each selected issue, for a maximum of twenty candidates in one
five-family packet. More candidates do not create more evidence or voting
weight.

### 3. Use a checked, versioned candidate catalogue

A checked-in catalogue, `data/reference/report_claim_candidates.json`, declares
the source and issue, atomic draft proposition, epistemic type, topic, geography,
relevant period or horizon, material conditions, physical page locator, bounded
supporting excerpt and model-generation provenance. The stable candidate ID is
derived from the validated semantic-and-evidence payload; report-family identity
is resolved from the trusted manifest and stored document rather than repeated as
unchecked catalogue text.

“Checked” means that the implementation validates the catalogue schema,
uniqueness, family/issue identity, four-candidate limit, document and extraction
hash bindings, physical-page range, exact excerpt occurrence and deterministic
catalogue SHA-256. It does **not** mean that the paraphrase is semantically fair
or that a human has approved it. A stale catalogue that does not target the
latest selected eligible issue fails visibly rather than being presented as a
current review queue.

### 4. Label every candidate as an unverified model draft

Every JSON candidate carries the machine-readable `draft_label` and exact
display label `UNVERIFIED MODEL DRAFT`. Every Markdown candidate heading or
card repeats that label. The packet title and summary state that nothing in the
packet is an approved conclusion. Deliberately, no `status` or approval field can
appear in the candidate schema.

Draft text remains attributed to the named publisher as a proposed paraphrase;
this queue admits only `fact`, `forecast` and `judgment`. It never renders a
draft as the system's accepted conclusion. Observatory inference is excluded
from this single-report queue and still requires evidence from at least two
independent publishers under ADR 0005.

Candidates from this queue are ineligible for cycle/fundamental scores,
scenario probabilities, risk flags, portfolio conclusions and dashboard
headlines.

### 5. Publish deterministic review artifacts

A successful run writes fixed aliases:

- `data/review/report_claims_latest.json`; and
- `data/review/report_claims_latest.md`.

It also writes content-addressed copies:

- `data/review/report_claims_YYYY-MM-DD_<packet-hash-prefix>.json`; and
- `data/review/report_claims_YYYY-MM-DD_<packet-hash-prefix>.md`.

`<packet-hash-prefix>` is the first 16 lowercase hexadecimal characters of the
full deterministic packet SHA-256. The JSON retains the full hash,
contract/methodology version, query cutoff, catalogue hash, selected issue and
extraction identities, candidate coverage and all citation locators. Markdown
is a human review rendering of the same contract, not a second source of truth.
The fixed aliases may move after a successful run; hash-addressed copies must
not be overwritten with different content.

`data/review/` is part of the generated-output ignore policy. Review packets are
regenerable derivatives, not source evidence and not substitutes for the preserved
PDFs or page text.

### 6. Keep approval, revision and rejection human-only

The packet may present three decisions for each candidate—`approve`, `revise`
or `reject`—but the read-only builder cannot execute any of them.

- **Approve** creates a verified claim only through a separate explicit write
  path carrying a real human reviewer identity and review time.
- **Revise** preserves the model draft and records the human-authored replacement
  plus its review lineage; it is not an in-place edit disguised as approval.
- **Reject** preserves the rejected candidate and reason as review history and
  creates no verified claim.

No model may choose an outcome, supply a human identity, approve its own draft
or turn a packet file into database truth. Blank, automated or model-shaped
reviewer identities fail closed. Only verified database claims—not catalogue
candidates or generated packets—may later enter report conclusions or scenario
analysis.

## Consequences

The first five-family review pass is bounded, reproducible and practical for a
human to inspect. Each draft carries its exact source context while the
system preserves the distinction between mechanical excerpt validation and
semantic approval. Fixed aliases make the current queue easy to open, and
hash-addressed copies make the reviewed input reproducible.

The implementation intentionally leaves the report layer with zero verified
conclusions until a human acts through the later review-write path. The packet
builder does not complete the approval workflow, feed report prose into risk
scenarios or justify investment advice.

## Rejected alternatives

- Generate free-form summaries of all 852 pages and present them as conclusions.
- Select whichever issue already has convenient candidate text instead of the
  latest eligible issue in each family.
- Allow more candidates from a longer report, creating accidental publisher or
  document weighting.
- Treat schema, hash or exact-excerpt checks as semantic verification.
- Let the packet builder write draft or verified claims to the database.
- Omit the `UNVERIFIED MODEL DRAFT` label from compact or machine-readable views.
- Let a model approve, revise or reject its own output or impersonate a reviewer.
- Use generated review packets as source evidence or commit them as canonical
  conclusions.
