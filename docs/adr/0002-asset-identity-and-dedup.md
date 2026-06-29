# ADR 0002 — Asset identity is the file path; content hashes drive dedup

- **Status:** Accepted
- **Date:** 2026-06-28
- **Deciders:** Ahsan Nawazish

## Context

Stage 2 adds the crash-safe write lifecycle (outbox ordering, recovery sweep,
update/delete, three-tier dedup). All of it hinges on one question: **what is an
asset's stable identity** — its content hash or its file path?

- **Content-hash identity (sha256):** any pixel edit changes the hash, so an
  "update" is really *insert-new + delete-old*. There is no stable handle for a
  file across edits, which makes the round-3 "pixel-update dark window" fix —
  explicitly about *updating an existing row's vectors* — meaningless.
- **Path identity:** the file's location is its stable handle. Re-indexing the
  same path with changed pixels is a true **update** of one row; content hashes
  become *attributes* used to detect change and duplicates.

## Decision

**Asset identity = the file path.** It is the SQLite primary key and the
ChromaDB vector id. The three content hashes are stored as attributes:

- `sha256` — raw bytes (dedup tier 1, and change detection for a known path)
- `pixel_hash` — sha256 of decoded RGB pixels (dedup tier 2: metadata-only edits)
- `phash` — perceptual hash, dHash (dedup tier 3: near-duplicates)

Ingest decision flow (see `ingest.py`):

1. Path already indexed and `sha256` unchanged → **unchanged**, skip.
2. Path already indexed, `sha256` differs → **update** (dark-window-safe).
3. New path, but a content hash matches an existing asset → **duplicate**, skip.
4. Otherwise → **insert** (outbox-ordered).

## Consequences

- **Positive:** gives a stable handle, so the update lifecycle and its
  crash-recovery test are well-defined; matches how a file index is actually
  used (re-scan a folder, files change in place).
- **Positive:** the three hashes cleanly map to the three dedup tiers without
  overloading identity.
- **Trade-off:** the same bytes at two different paths are two rows by identity;
  they're caught at ingest by the tier-1 sha match and skipped (not aliased).
  Multi-path aliasing (record both paths, store one SIF) is deferred — noted as
  future work, not needed for correctness now.
- **Note:** `phash` is implemented as **dHash** (difference hash) — robust to
  scaling/minor edits and dependency-light (no DCT/scipy). Near-duplicate is a
  Hamming distance under a threshold (default 10/64).
