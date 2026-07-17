# SIF Engine

**Search your images and documents by meaning — locally, cheaply, and without
sending a single pixel to the cloud.**

The SIF (Semantic Index File) Engine pre-computes rich vision intelligence
**once, on-device, at ingest** — objects, scene captions, OCR text, and
embeddings — and stores it as a structured, queryable index. Every later search
is cheap text/vector work instead of a fresh, expensive vision-API call.

> Apache-2.0 · local-first · CPU-friendly · open-core ([COMMERCIAL.md](COMMERCIAL.md))

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-4f46e5)
![Ollama](https://img.shields.io/badge/Ollama-local%20models-000000)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---

## The problem

Teams sitting on large visual archives — marketing libraries, scanned document
stores, claims photos, medical/legal imagery — want to *search them by meaning*.
The default path is a cloud vision API, which has two hard problems:

1. **Cost compounds.** Vision APIs bill **per call, every time**. Searching the
   same library repeatedly pays for the same images over and over.
2. **The data leaves.** In healthcare, legal, finance, and government, sending
   raw images/documents to a third party is often simply **not allowed**.

The key insight: *extracting* intelligence ("what's in this image?") and
*querying* it ("find images about X") are different operations that today get
conflated into one repeated, expensive cloud call.

**SIF separates them: extract once, query forever.**

```
                        cloud vision API                 SIF Engine
  cost of N queries     N × (vision call)                1 × (local extract) + N × (cheap vector search)
  data location         leaves your infra                never leaves the machine
```

For a library queried many times, the per-query vision cost trends to **~zero**
after a one-time local ingest — the amortized saving is large (the design target
is ~90%+), and the raw data stays put. *(Exact savings depend on your query/image
ratio and provider pricing; the structural shift — pay once, not per query — is
the point.)*

---

## What it does

- **Multi-model extraction** at ingest — object detection (YOLOv10n), scene
  captioning (Moondream2 via Ollama), OCR (PaddleOCR, multilingual incl.
  Arabic), optional face embeddings (**off by default** — biometric privacy),
  and text embeddings (nomic-embed-text v1.5).
- **Semantic search** — natural-language queries over a **multi-vector** index
  fused with Reciprocal Rank Fusion + a cross-encoder re-rank. Three spaces:
  a **description** vector (caption + objects), an **OCR-text** vector, and a
  **CLIP pixel** vector (raw visual content, so queries can match what the
  caption never described). CLIP is optional — the engine runs on the first two
  when it's unavailable.
- **Multi-page PDFs** — page-classified ingestion into a hierarchical index;
  search returns the matching **page**.
- **Crash-safe storage** — SQLite is the source of truth; the ChromaDB vector
  index is derived and rebuildable. Outbox ordering + a startup recovery sweep
  mean a crash never corrupts the index.
- **Concurrent ingestion** — a decoupled pipeline (extraction pool → VLM worker
  → single writer) with RAM-aware worker sizing and a benchmark harness.
- **Runs without the models** — every extractor falls back to a deterministic
  stub when its dependency is missing, so the engine always runs (great for CI).
- **Interfaces** — a CLI, a **FastAPI web app**, and an **MCP server**.

### Web app (`python -m sif.cli serve`)
A tabbed local UI over the engine:
- **Library** — browse everything indexed (thumbnails incl. rendered PDF pages);
  per-item *more-like-this*, *reveal in folder*, and *delete*.
- **Search** — text search with filters (type / has-text / object) and match
  badges (which space hit), **plus reverse image search** (find visually similar
  via CLIP).
- **Ask** — RAG: retrieves the top matches and has a local LLM answer from them
  (cited), so raw data stays on-device.
- **Add** — analyze one image, or bulk-index a folder (async, live progress) and
  **Watch** a folder for auto-indexing.
- **Duplicates** — scan a folder for exact / pixel / perceptual duplicate files.
- **Settings** — model health, VLM picker, and the faces on/off toggle.

### MCP server (`python -m sif.mcp_server`)
Exposes `sif_search` / `sif_stats` as tools so an AI agent can query your local
archive — the engine does the retrieval, the agent reasons over results, and the
images never leave the machine. Needs `pip install "mcp[cli]"`.

---

## Quickstart

```bash
pip install -r requirements.txt           # base engine (runs on stubs)

# Real models (heavier; best on Python 3.11/3.12):
pip install -r requirements-stage1.txt    # YOLO, PaddleOCR, nomic, etc.
ollama pull moondream                      # local VLM for captions

python -m sif.cli index ./my_photos        # concurrent, dedup-aware ingest
python -m sif.cli search "public transport in the city"
python -m sif.cli serve                    # web UI at http://127.0.0.1:8000
```

Other commands: `index <pdf>` (PDFs), `watch <dir>` (incremental), `stats`,
`reconcile`, `bench <dir>`, `version`. Set `SIF_USE_STUBS=1` to force the
model-free path anywhere.

---

## How it works

```
INGEST
  file ─▶ hashes + objects + OCR ─▶ [vlm_queue] ─▶ scene caption ─▶ embeddings
         (N parallel workers)                       (1 VLM worker)        │
                                                                          ▼
                                              SQLite row (source of truth) + 2 vector
                                              collections (visual / text)  ── 1 writer

QUERY
  text ─▶ embed ─▶ retrieve top-K from BOTH collections ─▶ aggregate to page/asset
       ─▶ RRF fuse (absentee-safe) ─▶ validate vs SQLite ─▶ gated re-rank ─▶ results
```

The **SIF schema** is the contract every layer depends on: per asset it records
`file` (hashes, resolution), `objects`, `faces`, `scene` (caption + tags),
`ocr`, and the multi-vector `embeddings` — with PDFs adding a hierarchical
`pages[] → regions[]`. The visual vector deliberately **excludes OCR text** to
avoid embedding dilution; OCR gets its own text vector.

### Design decisions on record (ADRs)
- [0001](docs/adr/0001-no-orchestrator-for-core-pipeline.md) — no agent
  orchestrator (LangChain/LangGraph) for a fixed pipeline
- [0002](docs/adr/0002-asset-identity-and-dedup.md) — asset identity = file
  path; content hashes drive three-tier dedup
- [0003](docs/adr/0003-threads-not-processes-for-concurrency.md) — threads (not
  processes) so models load once and fit the RAM budget

### Measured & honest findings
This was built **build → test → measure**, and the measurements corrected the
design. Those write-ups are part of the artifact:
- [Stage 1](docs/stage1-findings.md) — real-model integration surprises
- [Stage 3](docs/stage3-findings.md) — an RRF over-crediting case (known
  limitation; tunable via `SIF_RERANK_GAP`)
- [Stage 4](docs/stage4-findings.md) — a concurrency bug the benchmark caught;
  measured warm throughput (~0.45 img/s with all real models, peak RSS ~2GB on a
  20GB/12-core box) and the parallelism ceiling that follows from serialized
  model inference

---

## Build stages

Built in independently runnable, design-reviewed stages — the commit history is
the architectural-evolution story.

| Stage | Name | Delivers |
|-------|------|----------|
| 0 | Walking skeleton | End-to-end pipeline with stub models |
| 1 | Alpha | Real models behind identical interfaces + stub fallback |
| 2 | Beta | Crash-safe storage, lifecycle, three-tier dedup, WAL |
| 3 | MVP | Multi-vector retrieval, RRF fusion, gated re-rank |
| 4 | v1.0 | Concurrent pipeline, RAM-aware sizing, benchmark harness |
| 5 | v1.1 | Multi-page PDF ingestion (hierarchical, page-level retrieval) |
| 6 | v1.2 | CLI ergonomics (watch/config), portfolio docs, license |

## Configuration (env)

| Variable | Purpose |
|---|---|
| `SIF_USE_STUBS=1` | force deterministic stubs everywhere |
| `SIF_ENABLE_FACES=1` | opt in to face embeddings (off by default) |
| `SIF_VLM_MODEL` | scene VLM served by Ollama (default `moondream`) |
| `SIF_CLIP_MODEL` | CLIP pixel-embedding model (default `clip-ViT-B-32`) |
| `SIF_OCR_LANG` / `SIF_OCR_MKLDNN` | OCR language / re-enable MKL-DNN |
| `SIF_RERANK_GAP` | re-rank gate (raise to re-rank more aggressively) |
| `SIF_RAM_BUDGET_FRAC`, `SIF_PER_WORKER_MB`, … | RAM-aware worker sizing |

## Honest limitations

- Uses **small edge models** on purpose — it wins on cost/privacy/offline, not
  on raw model quality.
- It's an **archive/search** engine, not real-time video analytics.
- The **CLIP** space improves *recall* (surfaces visual matches the caption
  missed), but the final cross-encoder re-rank scores *text* (caption/OCR/labels)
  — so a purely-visual hit with a weak caption is retrieved but may not top the
  list. Blending the CLIP score into the final ranking is a natural next step.
- Ranking precision depends on the **cross-encoder re-rank** (on by default). If
  `sentence-transformers` isn't installed it degrades to RRF-only, which has a
  known over-crediting edge case (see [Stage 3 findings](docs/stage3-findings.md)).
- If the **Ollama VLM daemon is down**, scene captions fall back to stubs
  *silently* (by design) — `meta.models_used` shows `scene: stub` so it's
  detectable, but watch for it in deployment.

## License

Apache-2.0 ([LICENSE](LICENSE)). Open-core model — see [COMMERCIAL.md](COMMERCIAL.md).
© 2026 Ahsan Nawazish.

## Author

Built by **Ahsan Nawazish** — AI / ML Engineer specializing in LLM, RAG, and semantic-search systems.
[Portfolio](https://ahsan.live) · [LinkedIn](https://linkedin.com/in/anawazish) · [GitHub](https://github.com/innoce9t)
