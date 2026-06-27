# SIF ENGINE — HANDOVER / CONTEXT FILE

> **Purpose of this file:** This is a complete handover for continuing the SIF
> Engine build inside Claude Code. Read it fully before writing any code. It
> contains the project intent, the architecture, the full design-review history
> (three rounds of fixes that are already baked into the spec), what currently
> exists (Stage 0), and the precise staged plan for what to build next.
>
> **Owner:** Ahsan Nawazish — AI Architect (targeting US/Canada roles).
> This repo is both a portfolio artifact AND a potential open-core product.
> Treat code quality, commit hygiene, and documentation accordingly.

---

## 1. WHAT THIS PROJECT IS

The **Semantic Index File (SIF) Engine** is a local-first, privacy-preserving
pipeline that extracts rich semantic metadata from images (and later PDFs) using
lightweight on-device models, and stores it as a structured index. Any downstream
AI system can then query the visual assets using natural language **without
sending raw image data to a cloud API**.

**Core value proposition:** Pre-compute vision intelligence ONCE, locally, at
ingest. Then every downstream query is cheap text/vector work instead of an
expensive per-image vision-API call. Result: ~90-95% token-cost reduction per
query, sub-second search, full data sovereignty.

**Key insight that defines the architecture:** intelligence *extraction* ("what
is in this image?") and *querying* ("find images with X") are two different
operations. Today they're conflated into one repeated expensive cloud call. The
SIF Engine separates them: extract once, query forever.

---

## 2. THE GOLDEN STACK (component choices + rationale)

| Concern | Tool | Why |
|---|---|---|
| Object detection | **YOLOv10n** (ONNX) | NMS-free, ~300 FPS CPU, tiny weights |
| Face embeddings | **InsightFace buffalo_l** | 512-d, millisecond CPU inference. **OPTIONAL / off by default** (biometric-privacy surface) |
| Scene captioning | **Moondream2** (~1.8B, via Ollama) | Edge-built VLM, accepts constrained prompts, ~1.4GB RAM |
| OCR | **PaddleOCR** | Multilingual incl. Arabic (MENA market), fully local |
| Runtime | **ONNX Runtime** | Unified, hardware-accelerated, no PyTorch at inference |
| Vector store | **ChromaDB** | Local-first, persistent, dual-collection capable |
| Metadata store | **SQLite (WAL + FTS5)** | Zero-infra source of truth, full-text search |
| Embeddings | **nomic-embed-text v1.5** | 384-d, local, strong quality |
| API | **FastAPI** | Async, auto OpenAPI |
| Language | **Python 3.11+** | Best ML ecosystem |

**Hard constraint:** total memory footprint must be respected per deployment.
The current target hardware is a **12GB laptop** (see §6 for the worker math).

---

## 3. DESIGN-REVIEW HISTORY — FIXES ALREADY IN THE SPEC

This design went through THREE rounds of senior-level review. Every fix below is
**already part of the intended architecture** — do not reintroduce the original
flaws. This history is the single most important part of this handover.

### Round 1 fixes (structural)
1. **Memory/concurrency** → staged pipeline; VLM inference is memory-bound and
   must be serialized, cheap extractors can be parallel.
2. **Dual-store split-brain** → **SQLite is the single source of truth** via an
   OUTBOX pattern: write SQLite row `indexed=false` → write vectors → flip
   `indexed=true`. A startup **recovery sweep** replays any `indexed=false` rows.
   ChromaDB is a *derived, rebuildable* index.
3. **Embedding dilution ("semantic soup")** → **MULTI-VECTOR**: a VISUAL vector
   (caption + scene tags + object labels, **NO OCR text**) and a separate TEXT
   vector (OCR only). Different ChromaDB collections.
4. **SQLite write contention** → WAL mode + `busy_timeout=5000` + a SINGLE
   dedicated writer coroutine draining a write queue.
5. **Brittle dedup** → THREE-TIER: (1) raw SHA-256, (2) decoded-pixel hash
   (catches metadata-only edits), (3) perceptual hash / pHash (near-duplicates).

### Round 2 fixes (calibration & lifecycle)
1. **Memory arithmetic was actually wrong** → **RAM-AWARE ADAPTIVE
   CONCURRENCY**. Worker count is computed at startup:
   `max_workers = floor((RAM_budget - VLM_reserve - base_overhead) / per_worker_cost)`.
   Per-worker cost ≈ 900MB (YOLO 180 + OCR 400 + InsightFace 320).
   VLM reserve = 1400MB. Base overhead ≈ 300MB.
2. **PDF vector ID mismatch** → text vectors are page-level, visual vectors are
   sub-page region-level; they are NOT 1:1. Fuse at the **PAGE ENTITY** after
   aggregating child region scores up to their parent `page_id` (max or mean).
3. **Raw distance scale mismatch** → use **Reciprocal Rank Fusion (RRF)**, not
   weighted raw-score addition. RRF depends only on rank, immune to cross-space
   scale. `RRF_score = Σ weight[s] / (k + rank[s])`, k=60.
4. **Tombstone / orphan vectors** → full deletion + update LIFECYCLE outbox +
   a **daily reconciliation sweep** that purges ChromaDB ids with no live SQLite
   parent. ALSO: every query result is validated against SQLite before return,
   so orphans can never surface even pre-sweep.
5. **Latency contradiction** → no single latency number. Report per stage.
   Neural re-rank is **GATED**: only run the cross-encoder when the top-candidate
   margin is below a confidence threshold.

### Round 3 fixes (the four that must be in code)
1. **RRF "absentee penalty" / single-modality starvation** → when an asset is in
   one collection's top-K but ABSENT from the other, do NOT assign rank=infinity.
   Assign a **baseline default rank of `top_k + 1`** (e.g. 51) so a perfect
   single-modality match (e.g. a pure image with no text) isn't unfairly beaten
   by a mediocre dual-modality asset. **This needs a unit test** (the "drone
   shot ranks #1" test).
2. **Pipeline stalls + weight thrashing** → do NOT have extraction workers block
   on the VLM semaphore (idle workers + weight reload = thrashing). Instead
   **DECOUPLE INTO STAGES with micro-queues**: N extraction workers parse files
   and push results to an intermediate `vlm_queue`; a **single dedicated VLM
   worker** drains that queue sequentially. Extraction pool never blocks.
3. **Brittle absolute re-rank margin** → RRF scores are tiny and tightly
   compressed by k=60; an absolute margin threshold misfires constantly. Use a
   **RELATIVE percentage gap** between top-1 and top-2 instead.
4. **Pixel-update dark window** → on a pixel update, the FIRST step must flip the
   SQLite row to `indexed=false` (or `state='updating'`) BEFORE touching
   ChromaDB. Otherwise a crash after purging old vectors leaves an `indexed=true`
   row with no vectors, invisible to the recovery sweep until the daily sweep.
   Mirror the insert ordering. **This needs a crash-recovery test.**

### IMPORTANT META-NOTE
The paper design has CONVERGED. Do not start a fourth round of paper review —
remaining questions are empirical and must be answered by running code and
benchmarks, not more prose. Build, test, measure.

---

## 4. SIF JSON SCHEMA (the contract everything depends on)

```json
{
  "sif_version": "3.0",
  "file": { "path","sha256","pixel_hash","phash","size_bytes","resolution","format","indexed_at" },
  "objects": [ { "label","confidence","bbox":[x1,y1,x2,y2] } ],
  "faces":   [ { "face_id","embedding":[512],"bbox","confidence" } ],
  "scene":   { "caption","tags":[...] },
  "ocr":     { "full_text","text_blocks":[...],"has_text" },
  "embeddings": {
    "visual":[384], "text":[384],
    "visual_input":"caption + tags + object labels (NO OCR)",
    "text_input":"OCR text only",
    "model":"nomic-embed-text-v1.5"
  },
  "meta": { "processing_ms","stage","models_used":[...] }
}
```

For **PDFs** the schema is HIERARCHICAL: a parent `doc` record with child
`pages[]`, each page typed (`born_digital_text` | `scanned` | `mixed` |
`figure`), text vectors bound to pages, visual vectors bound to sub-page
`regions[]`. Retrieval returns a PAGE (with best-matching region highlighted),
not a raw vector id.

---

## 5. CURRENT STATE — STAGE 0 (walking skeleton, DONE)

A runnable end-to-end skeleton exists with STUB models. Structure:

```
sif-engine/
├── README.md
├── sif/
│   ├── __init__.py
│   ├── schema.py        # SIF dataclasses — the metadata format
│   ├── store.py         # SQLite (source of truth) + ChromaDB (2 collections)
│   ├── pipeline.py      # orchestration; build_visual_input / build_text_input
│   ├── query.py         # semantic search, validates hits against SQLite
│   ├── cli.py           # index / search / stats
│   ├── embedding.py     # STUB hashing embedder (becomes nomic in Stage 1)
│   └── extractors/
│       ├── __init__.py
│       └── stub.py      # STUB models (become real in Stage 1)
└── tests/
    └── test_stage0.py   # 4 passing tests
```

**What's already REAL in Stage 0 (do not regress these):**
- SIF schema full shape
- Dual-store wiring (SQLite + 2 Chroma collections)
- Multi-vector split: `build_visual_input()` EXCLUDES OCR text (round-1 fix #3,
  already tested)
- Every search hit validated against SQLite (orphan vectors can't leak, tested)

**What's STUBBED (replace in Stage 1):**
- All 4 extractors return deterministic placeholder output
- Embedding is a hashing function, not semantic

**Stage 0 tests passing:**
1. pipeline produces populated SIF
2. visual input excludes OCR text (no dilution)
3. index + search round-trip
4. orphan vector excluded from results

Run: `pip install chromadb pillow` then `python tests/test_stage0.py`.

---

## 6. TARGET HARDWARE: 12GB LAPTOP

RAM-aware worker math at 70% budget (8400MB usable):
`(8400 - 1400 VLM - 300 overhead) / 900 per-worker = 7 extraction workers`
(then cap to CPU core count). Note: with the Stage-4 **dedicated-VLM-worker**
architecture, extraction workers never hold the VLM in memory simultaneously, so
this is comfortable. Moondream2 runs via a local **Ollama** instance.

---

## 7. THE STAGED BUILD PLAN

Build stage by stage. Each stage is independently runnable + testable. Commit
each stage as its own labeled commit (the git history is part of the portfolio
value).

### Stage 1 — ALPHA (real models, single-threaded)  ← NEXT
Replace the 4 stubs + embedder with real implementations behind the IDENTICAL
interfaces (so nothing downstream changes):
- `extractors/objects.py` — YOLOv10n via ONNX Runtime (ultralytics or onnx)
- `extractors/faces.py` — InsightFace buffalo_l (**optional, off by default**)
- `extractors/ocr.py` — PaddleOCR
- `extractors/scene.py` — Moondream2 via Ollama client
- `embedding.py` — nomic-embed-text via sentence-transformers
Keep stub.py as a fallback when models/Ollama are unavailable (env flag
`SIF_USE_STUBS=1`). Goal: real SIF JSON from a real photo, all fields populated.
**Note:** models need ~2.5GB download + a running Ollama — likely run on the
real laptop, not in a sandbox. Write real integration code with clean fallback.

### Stage 2 — BETA (storage hardening + lifecycle)  ← most testable
- Outbox insert ordering (indexed=false → vectors → indexed=true)
- Recovery sweep on startup
- Delete + update outbox (update flips indexed=false FIRST — round-3 fix #4)
- Daily reconciliation sweep
- Three-tier dedup (SHA-256 → pixel hash → pHash)
- SQLite WAL mode + busy_timeout=5000 + single writer coroutine
- **Tests:** crash recovery, orphan purge, dedup skips, update-dark-window

### Stage 3 — MVP (retrieval that works)
- Multi-vector retrieval (query both collections)
- Page-entity aggregation (for later PDF use)
- **RRF fusion with absentee rank = top_k+1** (round-3 fix #1)
- **Relative-margin gated re-ranking** (round-3 fix #3), cross-encoder
  ms-marco-MiniLM
- **Tests:** the "drone shot ranks #1" starvation test; gating fires only on
  ambiguous queries

### Stage 4 — v1.0 (concurrency + perf)
- Decoupled stages: N extraction workers → `vlm_queue` → single dedicated VLM
  worker (round-3 fix #2, no thrashing)
- RAM-aware worker sizing (§6)
- **Benchmark harness:** measure real memory + latency; confirm/correct the
  spec numbers. THIS is where paper claims become measured facts.

### Stage 5 — v1.1 (PDF + API)
- Multi-page PDF ingestion: page-classification router (pdfplumber), hierarchical
  SIF, region extraction by bbox
- FastAPI query layer

### Stage 6 — v1.2 (polish + portfolio)
- CLI ergonomics, watch mode, config
- Minimal results UI
- Product-grade README (lead with enterprise problem + token-cost story)
- Final docs generated FROM the working system (every number measured)
- LICENSE (Apache-2.0), .gitignore, requirements.txt, COMMERCIAL.md (open-core)

---

## 8. REPO / PUBLISHING DECISIONS (already made)

- **Public repo, Apache-2.0, open-core model.** Core engine is open (portfolio +
  top-of-funnel). Future commercial layer (managed UI, multi-tenant, cloud-hybrid
  sync, RBAC/audit, support) stays private and is DEFERRED until there's demand.
- README framed as a PRODUCT, not a toy. Lead with the enterprise problem.
- Face-ID module ships **optional + off by default** (biometric-privacy: GDPR,
  BIPA, UAE PDPL). Reads well to privacy-conscious reviewers.
- Commit hygiene matters: stage-labeled commits tell the architectural-evolution
  story. Example: `feat: Stage 1 alpha — real models (YOLOv10n, PaddleOCR,
  Moondream2, InsightFace, nomic-embed)`.
- `.gitignore` must exclude: `__pycache__/`, `*.pyc`, `sif_data/`, `chroma/`,
  `*.db`, `.venv/`, `models/`.

---

## 9. IMMEDIATE NEXT ACTION FOR CLAUDE CODE

1. Confirm Stage 0 files are present and `python tests/test_stage0.py` passes.
2. Create `.gitignore`, `requirements.txt`, `LICENSE` (Apache-2.0) if missing.
3. Begin **Stage 1**: implement the real extractors + embedder behind the
   existing interfaces, with `SIF_USE_STUBS` fallback. Wire Moondream2 via
   Ollama. Keep each model in its own `extractors/<name>.py`.
4. Add Stage 1 tests (real-output shape; fallback works when models absent).
5. Commit as a labeled Stage 1 commit.
6. STOP and report results before Stage 2 (owner reviews each stage).

**Do NOT** reintroduce any flaw listed in §3. **Do NOT** start a fourth round of
paper redesign. Build, test, measure.
