# ADR 0014 — One byte lineage, ordered multi-section communication scopes

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30F

## Context

ADR 0013 fixed one official ECB HTML candidate for each 2025 monetary-policy
press conference. Each page is one representation and would produce one
byte-capture lineage, but it contains two analytically different sections: publisher-authored
prepared remarks and an elicited Q&A transcript. Storing the page twice would
duplicate evidence and permit the two copies to drift. Giving the whole page
only its provisional Q&A classification would erase section-level provenance.

The section mapping must also exist before content acquisition. Otherwise a
metadata-only artifact cannot state what it covers, and a later extraction can
silently redefine the representation. Section order matters because the text is
interpreted in document order; `{q_and_a, prepared_remarks}` is not equivalent
to `[prepared_remarks, q_and_a]`.

No named-human rights clearance has been recorded. This design change must not
authorize retrieval, archive source bytes, insert pilot metadata or make the
communications corpus analysis-ready.

## Decision

### 1. Bind one atomic ordered scope set to one artifact version

`communication_artifact_section_scope_sets` stores at most one immutable scope
set for each `communication_artifacts` row. The row binds the exact
`artifact_version_sha256`, an explicit scope count, a canonical ordered JSON
array, its SHA-256, the metadata-known clock and a pinned canonicalization
version.

Each scope entry records:

- a dense positive `section_ordinal` and stable `scope_key`;
- the section's artifact role and material type;
- its origin and provenance tier; and
- its transcriber plus attribution state.

The complete array is validated and inserted atomically. Exact replay is
idempotent; a conflicting second set is rejected. A set cannot be attached once
content has been captured for the artifact. Corrected scope semantics therefore
require a successor artifact version rather than a mutation or a late rewrite.
Sources known to publish mixed representations, including the ECB pilot source,
must receive an explicit ordered scope sequence; the metadata API will not infer
a singleton if a caller omits it.

The scope set owns no URL, retrieval, byte hash or blob path. Every retrieval and
content capture continues to belong to the single artifact version, so adding
logical sections can never duplicate its byte-capture lineage.

### 2. Preserve heterogeneous provenance inside a mixed representation

The 2025 ECB pilot mapping is ordered as follows:

| ordinal | scope_key | artifact role | material type | origin | provenance |
|---:|---|---|---|---|---|
| 1 | `prepared_remarks` | `prepared_remarks` | `monetary_policy_statement` | `publisher_authored` | `official_authored_text` |
| 2 | `q_and_a` | `q_and_a_transcript` | `questions_and_answers` | `official_published_transcript` | `official_published_transcript` |

The prepared section uses no transcriber and `not_applicable`; the Q&A section
keeps a null transcriber and `not_disclosed`. The latter describes transcript
provenance, not authorship: questions remain external speech and must not become
ECB claims merely because the ECB hosts the page.

The Fed pilot uses one `full_transcript` scope with its existing official
published-transcript provenance. A singleton scope and a mixed scope set use
the same contract; document count is not evidence weight.

### 3. Make extracted segments resolve to declared scopes

`communication_segments.section_ordinal` binds a segment, through its
extraction and content capture, to the matching artifact scope. Database guards
reject an unknown scope ordinal, a scope from another artifact, or section
ordinals that move backwards through the globally ordered segment stream.

Every v2 artifact must have a declared scope set before retrieval or content
capture, every segment must have a section ordinal, and finalization requires
every declared section to be represented. Populated v1 communications ledgers
are refused rather than silently reinterpreted, so v2 has no unscoped legacy
segment exception.

Existing segment-kind and speaker-side constraints still apply: prepared remarks
are publisher-side, questions cannot be publisher-side, and Q&A answers remain
publisher or unknown unless exact speaker evidence says more.

Segment kinds are also constrained by scope: letter scopes accept letter text,
prepared-remarks and Q&A scopes accept their corresponding kinds, annual reports
accept letter or narrative text, and transcript/media-derived scopes accept
prepared remarks, Q&A or narrative text. Headings and explicitly classified
`other` framing remain allowed in every scope, but cannot by themselves satisfy
finalization's substantive-text requirement.

This is a structural completeness rule only. It does not establish extraction
fidelity, semantic correctness or analytical eligibility.

### 4. Make pilot section order part of the semantic contract

The checked pilot loader, manifest hash and pending-rights packet preserve
`section_coverage` order. Organization order, event order, denominator event-key
order and rights-question order remain set-like and canonicalized. Reversing the
ECB section sequence is now a semantic change rather than a hash-neutral edit.

### 5. Upgrade only a known communications-v1 database

The pinned communications schema advances from version 1 to version 2. Database
initialization recognizes only a fresh install, the exact current schema or the
exact prior v1 table/index and trigger fingerprints. A v1 upgrade is
backup-first and transactional; unknown, partial, tampered or unsafe layouts are
rejected before schema mutation. The hard-coded schema-contract check is rebuilt
for version 2, and foreign-key plus SQLite integrity checks must pass before
commit.

This migration changes only the structural communications contract. It does not
backfill scope meaning, pilot events, artifacts, captures or segments. The live
v1 communications evidence tables are empty, so its controlled upgrade creates
only the empty scope structure and guards while preserving all unrelated macro,
flow, commodity, liquidity, allocator and report data.

### 6. Keep acquisition and interpretation closed

This slice adds no downloader or content-ingestion API. The catalogue still has
`automated_collection_allowed = false`, the 16 pilot candidates remain pending
rights review, and no human decision is inferred from personal-use intent or
from a request to proceed without review. There are still zero communication
source bytes, content captures, extracted segments, claims, scores or
investment-facing conclusions.

## Consequences

One source representation can now carry ordered, provenance-specific semantic
sections without duplicating an artifact version or its content lineage. Future
extraction can prove that every segment resolves to the declared representation
structure, while metadata and content permissions remain separate.

The additional contract deliberately makes ingestion stricter: artifacts with
missing or inconsistent scope metadata are inventory-invalid, and future scope
corrections require append-only artifact lineage. Acquisition remains blocked
until a separate permitted path exists.

## Rejected alternatives

- Store the ECB prepared remarks and Q&A as two artifacts pointing to duplicate
  copies of the same HTML.
- Keep only an unordered set of coverage labels.
- Attach section meaning only to captured bytes or extracted segments.
- Put heterogeneous prepared/Q&A provenance solely on the artifact header.
- Allow a late scope set to reinterpret already captured or segmented content.
- Treat the structural mapping as permission to retrieve or analyze content.
