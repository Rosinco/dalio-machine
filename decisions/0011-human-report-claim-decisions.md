# ADR 0011 — Human-only report-claim decisions

**Status:** Accepted and implemented

**Date:** 2026-09-09

## Context

ADR 0010 created a deterministic, read-only packet of page-bound model drafts.
That packet deliberately has no approval fields and cannot write to the report
ledger. A separate boundary is required before any draft can become evidence:
the reviewer must inspect the original page, explicitly attest to the semantic
checks, and choose `approve`, `revise` or `reject` without losing the original
proposal or its provenance.

The existing claim ledger can append a draft and an unchanged verified
successor. On its own it cannot durably represent rejection, bind a verdict to a
packet candidate, or commit a multi-item review atomically. Calling its public
proposal and verification functions in sequence would also create a partial
review if a later item failed.

## Decision

### 1. Keep packet construction permanently read-only

`dalio-report-review-packet` remains a read-only evidence-preparation command.
It does not gain an approval switch. Human decisions use the separate
`dalio-report-review` workflow and a separate JSON decision document. The
checked candidate catalogue and generated packet continue to contain no review
status.

Before a decision is checked or applied, the workflow rebuilds the packet from
the checked catalogue and live report ledger. It repeats the official-source,
document, extraction, page-hash and exact-excerpt checks and requires the full
packet and catalogue SHA-256 values in the decision document to match. A
mutable `latest` alias is never accepted as authority by itself.

### 2. Make the editable decision document explicit but unsigned

`dalio-report-review prepare` creates a separate, hash-bound JSON template with
exactly one entry for every candidate in the selected packet. Each entry starts
pending and carries four explicit attestations:

- attribution is a fair reading of the cited page;
- claim type is correct;
- scope, periods, units and material conditions are correct; and
- the proposition is important enough for macro-risk analysis.

The editable file contains neither reviewer identity nor review time. It is an
unsigned local work sheet, not proof of identity and not evidence by itself.
The implementation rejects duplicate JSON keys, missing or extra candidates,
unknown fields, stale hashes and partially completed batches.

An approval keeps the proposed semantic fields unchanged and requires all four
attestations. A revision requires all attestations against a complete,
non-identical replacement plus a reason and note. It may change semantic claim
fields, but retains the same official document, extraction and citations; a
needed evidence change instead requires rejection and a new candidate. A
rejection requires at least one failed attestation, a reason and a note.

### 3. Treat the local interactive boundary honestly

The CLI application path requires a TTY. It prints the resolved decision-file
path and every candidate-to-outcome assignment, then the operator must enter a
`human:<id>` reviewer attribution and type the canonical SHA-256 of that complete
decision document. Because the document itself binds the packet and catalogue
hashes, this confirmation covers both the evidence set and the actual verdict
map. The application clock is stamped by the command, not read from the editable
file.

This is deliberate operator confirmation for the current single-user local
workstation; the `human:` prefix is attribution syntax, not cryptographic
authentication. The CLI rejects non-TTY invocation, while the prohibition on
model or automated review is operating policy rather than authenticated
enforcement. Remote multi-user review remains unsupported until a trusted
identity mechanism, such as signed decision documents or an authenticated UI,
is designed. Models may generate the blank template but may not fill verdicts,
enter a human identity, confirm the hash or invoke the write boundary.

### 4. Preserve every verdict in an append-only decision ledger

`report_candidate_reviews` stores one immutable row per candidate. It binds the
candidate, packet and catalogue SHA-256 values; canonical original candidate
JSON; outcome; attestations; note; request fingerprint; reviewer and clocks; and
the resulting claim identifiers. A candidate ID is the idempotency key. An
exact replay returns the existing receipt, while a changed outcome, revision,
reviewer or rationale conflicts.

Every outcome materialises the original model proposal as an immutable draft
with its mechanically located citations:

- `approve` appends a verified successor to that original draft;
- `revise` appends a human-authored replacement draft with the inherited
  citations, then a verified successor to the replacement; and
- `reject` appends no verified claim.

The review row links the original draft, optional revised draft and optional
verified claim. Revision review lineage belongs in this decision row; it does
not use `supersedes_claim_id`, which remains reserved for semantic succession
between report claims or editions.

### 5. Commit a complete review as one protected operation

The claim storage layer exposes transaction-neutral internal append helpers,
while its established public proposal and verification APIs retain their
commit/rollback behaviour. The decision workflow validates the complete batch,
then appends all claim, citation and decision rows in one caller-owned
transaction. One invalid item rolls back the whole batch.

A read-only preflight returns an exact replay receipt or rejects a conflicting
review before creating a backup; the same check is repeated in the write
transaction for race safety. An exact replay therefore creates neither database
rows nor a misleading second backup. Immediately before a genuinely new schema
or evidence write, the CLI creates and verifies a SQLite-native backup under
`data/backups/`. The review table and its claim links have database constraints,
foreign keys, unique keys and both ORM and SQLite update/delete guards. Review
inventory reports outstanding and approve/revise/reject counts separately from
verified claim counts.

### 6. Keep analysis behind verified claims

Only the verified successors created by approval or revision are eligible for
`load_claims(..., verified_only=True)`. Rejected candidates and unreviewed
drafts cannot feed risk cases, scores, probabilities, dashboard headlines or
portfolio guidance. Scenario construction remains a later slice after a human
has actually reviewed the initial packet.

## Consequences

The system gains a reproducible path from a checked draft to a named human
verdict without weakening the packet's read-only contract. Original model text,
human changes and rejection reasons remain distinguishable, and a failed batch
cannot leave half of the packet promoted.

The first live application still requires real work by the user. Shipping this
workflow does not make any of the twenty current drafts verified, and an
interactive local identity does not provide cryptographic non-repudiation.

## Rejected alternatives

- Add `approve` fields to the catalogue or generated packet.
- Encode rejection as `claims.status = 'rejected'` without a packet-bound audit
  record.
- Mutate model drafts in place when a human revises wording or structure.
- Treat `supersedes_claim_id` as review-edit lineage.
- Apply candidates one by one with independently committed claim operations.
- Trust the editable decision file without rebuilding the packet and evidence.
- Allow a model or unattended command to provide reviewer identity or confirm
  the write.
- Feed the decision worksheet, rejected rows or draft claims into analysis.
