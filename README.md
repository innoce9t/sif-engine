# SIF Engine

Local-first visual asset intelligence. Pre-compute semantic metadata once,
query forever — without sending raw images to a cloud API.

## Build Stages

| Stage | Name  | Status | What it delivers |
|-------|-------|--------|------------------|
| 0 | Walking Skeleton | ✅ DONE | End-to-end pipeline with stub models. Index + search round-trips. |
| 1 | Alpha | pending | Real models: YOLOv10n, InsightFace, Moondream2, PaddleOCR, nomic-embed. |
| 2 | Beta  | pending | Outbox storage, crash recovery, deletion lifecycle, 3-tier dedup. |
| 3 | MVP   | pending | Multi-vector retrieval, page-entity fusion, RRF, gated re-rank. |
| 4 | v1.0  | pending | Decoupled stages: extraction pool → vlm_queue → dedicated VLM worker. |
| 5 | v1.1  | pending | Multi-page PDF ingestion + FastAPI query layer. |
| 6 | v1.2  | pending | CLI polish, results UI, generated docs. |

## Stage 0 — Walking Skeleton

```bash
pip install chromadb pillow
python -m sif.cli index sample_photos
python -m sif.cli search "outdoor urban scene building"
python -m sif.cli stats
python tests/test_stage0.py
```

### What's real in Stage 0
- Project structure & package layout
- SIF schema (the metadata format) — full shape
- Dual-store wiring: SQLite (source of truth) + ChromaDB (2 collections)
- Multi-vector split: visual input excludes OCR text (the v2 dilution fix, baked in from day one)
- SQLite-validation of every search hit (orphan vectors can't leak)
- CLI: index / search / stats

### What's stubbed (becomes real in Stage 1)
- All models return deterministic placeholder output
- Embedding is a hashing function, not a semantic model
