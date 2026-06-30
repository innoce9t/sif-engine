# ADR 0003 — Threads (not processes) for the extraction pool

- **Status:** Accepted
- **Date:** 2026-06-30
- **Deciders:** Ahsan Nawazish

## Context

Stage 4 parallelizes ingestion with the decoupled architecture from the design
review: N extraction workers → `vlm_queue` → 1 VLM worker → `write_queue` →
1 writer. The open question was the concurrency primitive for the extraction
pool: **threads** or **processes**.

The extractors are CPU-bound native inference (ONNX Runtime for YOLO, PaddleOCR,
InsightFace), which would normally argue for processes to dodge the GIL. But:

- ONNX Runtime, PaddleOCR, and torch **release the GIL during inference**, so
  threads already get real parallelism for the hot path.
- The hard constraint is **RAM** (12GB target). Processes can't share the loaded
  model weights, so each worker process would reload YOLO + OCR (+ faces),
  multiplying the memory footprint and blowing the budget — the exact failure
  the staged design exists to avoid.
- The VLM is held **once** by the single dedicated VLM worker; extraction
  workers never load it.

## Decision

Use a **thread pool** for extraction, plus one dedicated VLM thread and one
dedicated writer thread, connected by `queue.Queue` micro-queues
(`runner.index_paths`). Models are loaded **once per process** and shared across
threads; each real extractor guards its lazy load with a lock so concurrent
first-calls don't double-load.

All storage mutation goes through the **single writer thread**, which is what
keeps the Stage 2 outbox ordering correct under concurrency. SQLite is opened
`check_same_thread=False` because access is already serialized (a single-threaded
dedup pre-pass for reads; one writer thread for writes).

## Consequences

- **Positive:** one in-memory copy of each model → fits the RAM budget; real
  parallelism on the GIL-releasing inference calls; trivial shared-memory
  hand-off between stages (no IPC serialization of SIFs / vectors).
- **Positive:** RAM-aware sizing (`workers.recommended_workers`) stays simple —
  per-worker cost is just the cheap extractors, not a full model set.
- **Trade-off:** pure-Python work between inference calls is serialized by the
  GIL, but that work is negligible next to model inference; it isn't the
  bottleneck.
- **Revisit trigger:** distributing ingestion across **machines** (not cores) —
  that's a task-queue/worker-process concern (Arq/Dramatiq/Celery), separate
  from this in-process pool.
