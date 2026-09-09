# ADR 0016 — Immutable communication source-policy catalogue snapshots

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30H

## Context

The checked Fed/ECB and Bank of England manifests bind the communication
source-policy catalogue by its semantic SHA-256. Their loaders previously
required that hash to equal the single current module-level catalogue hash and
then interpreted every source through the current source tuple.

That is safe only while the catalogue never changes. Adding Riksbank, adding a
new representation policy, or correcting a future source would otherwise make
the already checked manifests unloadable. Merely accepting a list of old hashes
would be worse: an old manifest could then be reinterpreted through newer host,
publisher, transcriber or rights metadata.

The user chose to continue without a human rights review. That choice leaves
content acquisition and analysis closed, but it does permit metadata-only
coverage work. Catalogue growth must therefore preserve the exact policy
vintage behind every prior metadata record before another institution is added.

## Decision

### 1. Address complete catalogue snapshots by semantic hash

`CommunicationCatalogueSnapshot` binds one exact tuple of frozen source-policy
objects to the schema version and evaluation clock used in its canonical hash.
An immutable registry maps a full lowercase SHA-256 to that complete snapshot.

The shipped 2026-09-09 catalogue is retained as a separately named tuple and
asserted at import time against its existing hash:

`67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6`.

The public `COMMUNICATION_SOURCES` and `COMMUNICATION_CATALOGUE_SHA256` names
continue to mean the current vintage. Their values are unchanged in this slice.

### 2. Resolve before interpreting or persisting metadata

Both checked-manifest parsers now resolve the manifest-supplied catalogue hash
before parsing any source-bound scope, locator or artifact candidate. The
resolved snapshot's source map is passed through the full validation chain.
The database metadata helpers use the same resolution path and persist the
resolved snapshot's own hash, evaluation clock and source-policy values in the
immutable policy row.
The observatory audit recognizes every registered vintage, but only after
reconstructing and exact-comparing each persisted policy row with the source
object in that snapshot. A known hash is therefore not an allowlist shortcut.

An unknown, malformed or mixed hash fails closed. There is no fallback to the
current catalogue and no use of a current global source map after resolution.
Consequently, a future source is valid only for manifests bound to the snapshot
that contains it, while an old source retains its historical publisher and
rights semantics.

### 3. Preserve checked evidence identities

This slice does not edit either checked manifest, generated review output or
the live database. The Fed/ECB and BoE raw-file hashes, semantic hashes and
inventory hashes remain unchanged. The snapshot mechanism authorizes no
network access, content capture, extraction, rights decision or analysis.

Every snapshot validates its schema version, timezone-aware evaluation clock,
source tuple and recomputed semantic digest. A manifest cannot bind a snapshot
evaluated after its own creation or knowledge cutoff, and source rights clocks
are validated against their snapshot's evaluation time rather than today's.

## Consequences

The catalogue can now grow append-only without forcing old manifests to be
repinned or silently reinterpreted. A new source-policy vintage must retain the
old tuple and register both full snapshots. This costs some deliberate data
duplication, which is preferable to deriving historical policy from mutable
current state.

The next metadata-only cohort may add Riksbank policies in a new current
snapshot. A separate checked-cohort registry is still needed before publishing
multiple institution/year inventories without moving the Bank of England's
legacy `communication_metadata_latest` aliases.

## Rejected alternatives

- Rewrite old manifest hashes whenever the catalogue grows.
- Accept historical hashes while validating their records against current
  source policies.
- Trust a stored historical policy merely because its catalogue hash is known.
- Resolve an unknown hash to the newest catalogue.
- Store only a source-policy delta and reconstruct old policy at load time.
- Treat a hash match as permission to acquire or analyze communication content.
