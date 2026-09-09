# ADR 0012 — Institutional communications as a rights-gated evidence ledger

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30D

## Context

Reports and numeric releases show what institutions published and what measured
conditions were. Press conferences, earnings calls, prepared remarks and
CEO/Chair letters add a different kind of evidence: what policymakers and
management teams said they observed, expected, feared and planned at that time.
That history can help identify changes in policy reaction functions, bank credit
and funding conditions, and commodity supply, investment and demand constraints.

These communications are not homogeneous. A prepared statement is not a Q&A
answer, an edited transcript is not necessarily verbatim, an official website
may host a third-party transcript, and machine captions are not publisher-authored
text. Access and reuse terms also vary by publisher and artifact. Treating every
hosted file as equivalent official prose, or treating permission to view as
permission to build and redistribute a corpus, would create both analytical and
rights risk.

The existing report manifest cannot absorb these sources safely. It is an exact
five-family, PDF/page-only contract used to build the current twenty-candidate
human-review packet. Enlarging that manifest would silently change its required
universe and invalidate the established review boundary.

## Decision

### 1. Add a sibling communications domain

Institutional communications use a separate static source-policy catalogue and
a separate immutable event, artifact, extraction and segment ledger. They do not
become numeric observations, data releases or entries in the five-family report
manifest. A future reviewed claim may cite either evidence domain, but this slice
does not change the existing claim or review contracts.

One logical event may have several representations: a landing page, prepared
remarks, Q&A transcript, annual report, CEO or Chair letter, slides, regulatory
filing, audio, video or subtitles. The event binds those representations without
pretending that they share authorship, fidelity, publication time or rights.

### 2. Make source policy fail closed

Every source records a stable source and organization identity, official-domain
allowlist, material types, host, publisher, transcript attribution, provenance
tier, commodity-family coverage where applicable, rights status, rights basis
and acquisition status. Source metadata authorizes no network collection by
itself.

Each catalogue version and exact source policy used by a row is also preserved
as an immutable database snapshot. Artifacts and commodity mappings reference
that `(catalogue hash, source id)` pair. This makes an old policy resolvable
after the checked-in catalogue advances; it does not make direct write access to
SQLite a trusted authorization boundary.

Database initialization preflights every pre-existing communications table
against a pinned schema fingerprint covering tables, constraints and explicit
indexes, and refuses an unknown or partial layout. It also recreates the custom
policy and lineage triggers transactionally and checks a separate trigger
fingerprint, so an older same-named but weaker guard cannot survive a later
initialization unnoticed.

The initial catalogue is deliberately metadata-only. Automated acquisition is
false for every entry until the relevant artifact and current terms have been
reviewed. Sources whose terms prohibit systematic retrieval or centralized
storage remain metadata-only or blocked pending permission. A later collector
must bind each artifact to the exact policy-catalogue version and refuse any
source or representation that is not explicitly authorized.

For this MVP, the checked-in current catalogue is the only trusted catalogue
manifest. Inventory treats rows bound to any other catalogue hash as untrusted
and analysis-ineligible, even if their internal hashes are self-consistent. A
later catalogue revision must retain its full predecessor manifests in a trusted,
versioned registry before historical hashes can remain eligible.

This catalogue is an operational safeguard, not legal advice or a conclusion
that a work is or is not protected by copyright.

### 3. Preserve exact artifact provenance and fidelity

Representation facts are artifact-specific even when their source archive is
mixed: the actual host, historical publisher, origin and provenance tier are
inspected and stored for each artifact rather than copied from today's landing
page defaults. Link metadata remains immutable even if bytes are collected later.
When acquisition is separately authorized, each retrieval can append a separate
content capture with its exact SHA-256, size and content-addressed path. That
child lineage preserves changed bytes at an unchanged URL without rewriting the
artifact or confusing a newly archived local copy with a new public edition.
Retain MIME type, landing and artifact URLs, host, publisher, transcriber,
language and translation status on the representation.
Record distinct artifact classes and origins rather than a generic
"transcript" label. At minimum the model distinguishes issuer-authored text,
officially hosted vendor text, official captions, automatic captions and local
speech-to-text.

Prepared remarks and Q&A remain separate. Segments retain speaker name, role and
side (`publisher`, `external`, `moderator` or `unknown`) plus page, paragraph or
timecode locators as applicable. A journalist's or analyst's question can never
silently become a publisher claim.

### 4. Keep point-in-time clocks and corrections explicit

Event/reference dates remain separate from each artifact's `published_at`,
`available_at` and `retrieved_at` clocks. A separate metadata-known clock gates
catalogue corrections so a title, attribution or rights correction learned later
cannot leak backward to the artifact's original public date. Point-in-time text
access is governed by both the public artifact clock and the metadata knowledge
clock, not merely by the event date. Unknown availability is set conservatively
to retrieval time rather than inferred from a fiscal period. Repeated retrievals
append observations rather than rewriting the first retrieval.

Corrections, restatements, changed transcripts, official translations and re-runs
of an extractor append versions; they do not overwrite prior evidence. Locally
produced human or machine translations remain deferred until a derived-artifact
lineage and translator/model provenance contract exists. Extracted segments are
reproducible derivatives bound to one exact content capture and one
extractor version. A distinct immutable run key preserves each invocation even
when an exact rerun produces the same output hash. An extraction is not complete merely because a header exists:
the writer inserts dense ordered segments, rechecks count/length/hashes, then
appends a one-time finalization row. Segments cannot be added after finalization,
and only finalized, revalidated runs are structurally eligible for semantic
review. Hashes prove byte and segment stability, not transcription fidelity or
that the named extractor actually executed; a trusted-execution or named-human
review contract is required before extracted text becomes analytical evidence.

### 5. Treat organization and commodity mappings as coverage taxonomy

Company identity and commodity exposure change through mergers, divestments and
portfolio shifts. Any mapping therefore needs an effective period and explicit
provenance. A stable coverage key anchors corrections independently of the dates
being corrected, while overlapping current lineage heads for the same
organization, family and role are rejected. Until a mapping is backed by
preserved evidence and reviewed, it is selection/coverage taxonomy only and
cannot support an analytical conclusion.

Document volume is never voting weight. A diversified producer is one issuer
view even when it touches several commodity families, and an institution that
publishes frequently does not receive more influence merely because more text
was collected.

### 6. Defer interpretation until evidence is reviewable

This slice stores no sentiment, forecast, causal conclusion, risk score or
portfolio instruction. Later extraction should favour explicit production,
capacity, capital expenditure, inventories, costs, credit, deposits, funding,
policy conditions and time-bounded guidance. The useful derived comparisons are
change from the prior communication, prepared-versus-Q&A differences,
cross-source agreement or disagreement, guidance revisions and forecast error
against subsequently observed data.

Any proposed fact, forecast, judgment or Observatory inference remains an
unverified draft until it passes a human semantic-review boundary equivalent to
the existing report-claim gate. Machine captions and local speech-to-text carry
lower provenance and require exact media/timecode references.

## Initial catalogue boundary

The source catalogue covers a bounded, family-balanced research universe:

- central-bank communication archives for the Federal Reserve, ECB, Bank of
  England and Reserve Bank of Australia;
- major-bank reporting archives spanning the United States, Europe and Asia;
  and
- commodity-company archives covering energy, food and beverages,
  agricultural raw materials, fertilizers, base metals and precious metals.

This is source-discovery and policy metadata, not a claim that any historical
corpus has been downloaded or is complete. Aggregate commodity indices have no
issuer proxy.

## MVP boundary

Slice 30D establishes:

`validated source policy -> immutable event/artifact/content/segment foundation`

It deliberately does **not** include:

- a crawler or bulk downloader;
- archived communication bytes or a completed historical corpus;
- vendor transcript ingestion;
- subtitle discovery or speech-to-text;
- automatic claim generation, sentiment or topic scores;
- changes to the existing report-review packet; or
- communication-driven risk cases or portfolio guidance.

The next acquisition slice must first pin event/artifact metadata, preserve an
archive-index discovery artifact, review rights per selected representation and
preflight the complete batch before any database write.

## Consequences

The design costs more metadata and delays broad collection, but it prevents
rights ambiguity, transcription fidelity and corporate-history changes from
being flattened into false evidence. It also lets later analysis reconstruct
exactly what text was publicly available, who said it, whether it was prepared
or elicited in Q&A, and which version the Observatory actually reviewed.

## Rejected alternatives

- Add all communication sources to `REPORT_SOURCES`.
- Treat an official host as proof of issuer authorship or redistribution rights.
- Store only generated summaries or sentiment scores.
- Treat captions, edited transcripts and verbatim official Q&A as equivalent.
- Use today's company/commodity mapping for historical events.
- Weight a commodity family by the number of documents or benchmarks collected.
- Download first and decide provenance or rights later.
