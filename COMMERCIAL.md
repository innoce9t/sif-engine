# Commercial / Open-Core

The SIF Engine core — everything in this repository — is **open source under
[Apache-2.0](LICENSE)**. You can run it, embed it, and modify it, including in
commercial products, subject to the license.

## What's open (this repo)

The full local-first engine:

- Multi-model extraction (objects, OCR, scene captioning, optional faces) with
  transparent stub fallback
- Multi-vector semantic retrieval (RRF fusion + gated cross-encoder re-rank)
- Crash-safe storage (outbox ordering, recovery, three-tier dedup)
- Concurrent ingestion pipeline + RAM-aware sizing + benchmark harness
- Multi-page PDF ingestion (hierarchical SIF, page-level retrieval)
- CLI and a FastAPI web inspector

This is enough to self-host the engine end to end.

## What a commercial layer would add (not in this repo)

The open-core model keeps the engine open and reserves an operational layer for
a future commercial offering, **deferred until there is real demand**:

- Managed multi-tenant deployment (RBAC, SSO, audit logs)
- Cloud-hybrid sync (local extraction, optional managed index/replication)
- A hardened admin/operations UI and SLAs
- Connectors (DAM/CMS/object-storage integrations) and support

None of that is required to use the core, and none of it is built yet — it is a
direction, not a product.

## Status

This is currently a **portfolio and reference implementation**. There is no
commercial offering today. If you're interested in the engine for a real
use case, open an issue.

— Ahsan Nawazish
