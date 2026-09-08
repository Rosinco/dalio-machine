# ADR 0005 — Report evidence ledger: immutable documents, page-cited claims

**Date:** 2026-09-08 · **Status:** accepted · **Slice:** 30

## Context

The numeric release ledger answers how a measured series was reported at a
point in time. Central-bank and institutional reports need a different evidence
model. Their source is an edition of a document, their important content is
prose, charts and forecasts, and a useful conclusion depends on both a precise
page locator and a semantic judgment about what that passage supports.

Putting report prose into `DataRelease` or `Observation` would erase those
differences. Storing only a generated summary would also make it impossible to
check the source, distinguish the publisher's position from the Observatory's
inference, or reconstruct what had been reviewed at an earlier date.

## Decision

### 1. Keep a separate immutable report ledger

Reports use their own document, extraction, claim, citation and review records.
They do not become numeric observations or numeric source releases. A future
link between a claim and a numeric series may be additive, but neither ledger
owns or mutates the other.

The trusted source identity is the static report manifest, not caller-supplied
metadata. A submitted source ID must match the manifest's publisher, report
family, jurisdiction, language, landing page and official-domain allowlist.
Issue-specific artifacts may vary, but must remain on an allowlisted HTTPS
domain.

### 2. Anchor every edition in the original artifact

- Preserve the exact original PDF bytes in content-addressed storage.
- Compute SHA-256 over those bytes before extraction and retain the source,
  issue identity, landing URL, artifact URL, MIME type and byte size.
- Re-fetching identical bytes is idempotent. Different bytes at the same URL or
  for the same issue create a new immutable document version.
- Corrections and replacements point to the document they supersede. They never
  overwrite its bytes, metadata, extraction or claims.

The original content hash, not a mutable remote URL or extracted-text hash, is
the durable identity of the evidence artifact.

### 3. Make extraction complete and page-addressable

An extraction is a versioned derivative tied to one document hash and records
the extractor name, extractor version, run time, status and page count. A
complete extraction contains every physical PDF page exactly once, numbered
from 1 through the document page count. Page text and its hash are retained so
a citation can be rechecked against the exact extraction used during review.

In the dependency-free storage core, returning every physical page is part of
the injected extractor's contract. The ledger verifies a gap-free 1-based
sequence and deterministic output for each extractor name/version. A production
ingestion adapter must obtain and compare the PDF's physical page count with its
parser; that independent parser check is deferred with the live downloader and
is not claimed by this slice.

A partial or failed extraction may be retained for diagnostics, but it cannot
support a verified claim. Physical PDF page numbers are canonical; printed page
labels and section names are supplementary locators.

### 4. Preserve availability clocks in UTC

Each document records distinct UTC clocks:

- `published_at`: the publisher's stated publication time, nullable when it
  cannot be established;
- `available_at`: when the artifact was publicly obtainable; and
- `retrieved_at`: when this system obtained the preserved bytes.

The ordering is `published_at <= available_at <= retrieved_at` whenever
`published_at` is known. Unknown publication or public-availability times are
not guessed from the report date. If availability cannot be established,
retrieval time is the conservative availability time. Reference periods inside
a report remain separate from all three clocks.

Claims and human reviews also carry immutable creation/availability times.
Publisher claims may inherit the document's verified public-availability time:
that reconstructs what the institution had said, even when the Observatory
archived and reviewed it later. The retrieval and review clocks remain visible
so this is never misrepresented as something the system had already processed.
An Observatory inference cannot be available before its creation, its cited
documents, or its human review.

### 5. Store atomic claims with explicit epistemic type

One claim contains one proposition. Compound passages are split rather than
mixing observation, forecast and explanation in one record.

| Type | Meaning |
|------|---------|
| `fact` | A historical/current observation or completed decision stated by the publisher; attribution is not a declaration of universal truth. |
| `forecast` | The publisher's future projection, with its target period and material conditions. |
| `judgment` | The publisher's qualitative risk assessment, causal view or estimate of an unobservable state. |
| `inference` | The Observatory's own synthesis across evidence, visibly attributed to the Observatory. |

Every publisher claim (`fact`, `forecast` or `judgment`) requires at least one
direct citation to its source document. A citation identifies the extraction,
physical page or page range, and a bounded evidence excerpt that mechanically
matches the cited page text.

Every Observatory `inference` requires explicit reasoning and supporting
citations from at least two independent publishers. Multiple documents or
editions from one publisher do not satisfy that independence rule. Inference is
never silently presented as a publisher conclusion.

### 6. Require human semantic review

Locator and excerpt checks prove where words occur; they do not prove that the
claim is a fair reading. Promotion from draft to verified therefore requires a
named human reviewer to confirm attribution, claim type, semantic support,
scope, units and relevant conditions.

Models may extract candidate passages and create drafts. A model-created claim
cannot approve itself, write a human-review identity, or become verified without
a later human review event. Rejection and supersession are also retained as
history rather than deleting the draft or prior verdict.

SQLite foreign-key enforcement and database triggers protect the report,
extraction, page, claim and citation tables even when writes bypass the ORM.
Partial unique indexes permit only one verified review per draft and one
verified semantic successor per prior claim. Application validation remains
responsible for the richer type, period, unit and evidence rules.

### 7. Make point-in-time claim queries honest

A verified public-history query may return a claim only if, by the requested
cutoff:

1. every underlying document was publicly available; and
2. the attributed publisher statement was publicly available in that document,
   or an Observatory inference had been created and human-reviewed.

Publisher statements can therefore be backfilled into their true public
timeline after review, just as a numeric release retrieved later may retain a
known earlier availability clock. Their retrieval and review timestamps are
always returned. Observatory inferences are different: they belong to the
system's own history and are hidden before review. A correction, retraction,
changed extraction or superseding claim never overwrites its predecessor.
Current views may prefer the newest non-superseded version, but historical views
retain and expose the prior state. A future strict "what had this installation
already ingested?" query can additionally gate publisher claims on retrieval
and review time without changing this public-history ledger.

## MVP boundary

Slice 30 establishes the smallest auditable path:

`original PDF -> immutable document -> complete page extraction -> atomic draft
claim + page citation -> human verification -> as-of claim query`

The slice deliberately does **not** include:

- automatic report summaries;
- automatic risk scores or changes to existing cycle/fundamental scoring;
- OCR for scanned or image-only reports;
- broad historical archive backfill; or
- autonomous publication of model-written claims or inferences.

Those capabilities require separate evidence, quality and product decisions
after the manual, page-cited path has proved reliable.

## Consequences

The report history costs more storage and review effort than a summary table,
but every displayed conclusion remains traceable to preserved bytes, an exact
page and a human review available at that time. Institution statements and
Observatory synthesis remain visibly different, corrections remain auditable,
and prose cannot leak into numeric scoring merely because it was extracted.

## Rejected alternatives

- Store reports as free-text rows in the numeric release ledger.
- Keep only the latest PDF or latest generated summary.
- Treat a matching excerpt as sufficient semantic verification.
- Allow the model that drafted a claim to approve it.
- Call a single-publisher paraphrase an Observatory inference.
- Rewrite old claims or documents when a publisher issues a correction.
- Ship automated summaries, OCR, archive backfill and risk scoring in the first
  report-ledger slice.
