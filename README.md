# SIF Engine

Local-first visual asset intelligence. Pre-compute semantic metadata once,
query forever — without sending raw images to a cloud API.

**Docs:** [architecture decisions](docs/adr/) ·
[Stage 1 real-model findings](docs/stage1-findings.md)

## Build Stages

| Stage | Name  | Status | What it delivers |
|-------|-------|--------|------------------|
| 0 | Walking Skeleton | ✅ DONE | End-to-end pipeline with stub models. Index + search round-trips. |
| 1 | Alpha | ✅ DONE | Real models behind identical interfaces (YOLOv10n, InsightFace, Moondream2, PaddleOCR, nomic-embed) with transparent stub fallback. |
| 2 | Beta  | ✅ DONE | Outbox storage, crash recovery, deletion lifecycle, 3-tier dedup, WAL. |
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

## Stage 1 — Alpha (real models)

Real extractors and embedder replace the stubs *behind the identical
interfaces*, so storage and retrieval are untouched. Each backend loads lazily
and **falls back to its stub** when its dependency (or, for the VLM, the Ollama
daemon) is unavailable — the engine always runs. `meta.models_used` in every
SIF reports which backend actually answered.

```bash
pip install -r requirements-stage1.txt   # heavy; ~2.5GB of weights on first use
ollama pull moondream                     # local VLM daemon for scene captions
python -m sif.cli index sample_photos     # now uses real models when present
```

| Stage | Model | Module | Notes |
|-------|-------|--------|-------|
| Objects | YOLOv10n (Ultralytics) | `extractors/objects.py` | weights auto-download |
| OCR | PaddleOCR | `extractors/ocr.py` | multilingual incl. Arabic |
| Scene | Moondream2 via Ollama | `extractors/scene.py` | needs running Ollama daemon |
| Faces | InsightFace buffalo_l | `extractors/faces.py` | **off by default** (biometric privacy) |
| Embedding | nomic-embed-text v1.5 | `embedding.py` | 384-d; hash fallback is 64-d |

### Environment flags
- `SIF_USE_STUBS=1` — force deterministic stubs everywhere (CI / no-model envs).
- `SIF_ENABLE_FACES=1` — opt in to face embeddings (GDPR/BIPA/UAE PDPL surface).
- `SIF_VLM_MODEL`, `SIF_YOLO_WEIGHTS`, `SIF_OCR_LANG`, `SIF_FACE_MODEL` — backend overrides.
- `SIF_OCR_MKLDNN=1` — re-enable PaddleOCR's MKL-DNN (off by default; it crashes
  on some CPUs with PaddleOCR 3.x — `Unimplemented ... onednn_instruction`).

## Stage 2 — Beta (storage hardening)

SQLite is the crash-safe source of truth; ChromaDB is a derived, rebuildable
index. Asset identity is the file path; content hashes drive dedup ([ADR 0002](docs/adr/0002-asset-identity-and-dedup.md)).

- **Outbox ordering** — write the SQLite row `indexed=0` → vectors → `indexed=1`.
  A crash never leaves a live row without vectors.
- **Recovery sweep** — on startup, replay any `indexed=0` row from its stored
  SIF JSON (no models needed — the vectors live in the JSON).
- **Update dark-window fix** — an update flips `indexed=0` *before* purging old
  vectors, so an interrupted update is recoverable too.
- **Delete + reconcile** — deletes tombstone then purge; `reconcile` finishes
  interrupted tombstones and purges orphan vectors (ChromaDB ids with no live
  SQLite parent). Query results are also validated against SQLite, so orphans
  never surface even before a sweep.
- **Three-tier dedup** — sha256 (exact) → pixel hash (metadata-only edits) →
  perceptual dHash (near-duplicates, Hamming threshold).
- **WAL** + `busy_timeout=5000` + a single write path.

```bash
python -m sif.cli index sample_photos    # re-running skips unchanged/duplicate files
python -m sif.cli reconcile              # purge orphans + finish tombstones
```

> The single dedicated **writer coroutine** is deferred to Stage 4, where
> concurrency actually exists — the ordering guarantees here are what make
> wrapping these methods in one async writer safe. Building the async machinery
> now, against a single-threaded pipeline, would be premature.

## Web UI

A FastAPI demo frontend: upload an image, see the full SIF JSON (objects,
caption, OCR, embeddings, backends used), and run semantic search over the
index. Uses real models if available, stubs otherwise.

```bash
pip install -r requirements.txt          # fastapi/uvicorn are in the base set
python -m sif.cli serve                   # http://127.0.0.1:8000
# real models: run from the Stage-1 venv with Ollama up
```

Endpoints: `POST /api/process` (upload → SIF), `GET /api/search?q=`,
`GET /api/stats`. Interactive API docs at `/docs`.
