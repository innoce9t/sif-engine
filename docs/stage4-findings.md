# Stage 4 — Concurrency & Benchmark Findings

What the benchmark harness measured the first time it ran the concurrent
pipeline with real models — i.e. the "paper claims become measured facts" step.

## Finding 1 — model inference is not thread-safe (YOLO raced under the pool)

**Symptom:** the first real-model benchmark logged
`objects: real backend failed ('Conv' object has no attribute 'bn'); using stub`
— YOLO silently fell back to the stub. It had worked fine single-threaded in
Stage 1.

**Cause:** Ultralytics fuses Conv+BN layers lazily on first inference. With the
extraction pool, two threads hit that first inference simultaneously; one
deletes `.bn` mid-fuse while the other reads it → the race above. The load-lock
only guarded *loading*, not *inference*.

**Fix:** serialize each model's **inference** with its per-model lock
(`extract_objects` / `extract_ocr` / `extract_faces`). Different models still
overlap (YOLO of one item while OCR of another), and the dominant throughput win
— **stage pipelining** (extraction of later items while the VLM/writer handle
earlier ones) — is unaffected.

**Lesson — refines the design's "cheap extractors run in parallel" assumption:**
the *same* model can't run concurrently, but different models and the
extract→VLM→write stages still overlap. The benchmark harness existed precisely
to catch this; it did on run #1.

## Finding 2 — first measured numbers (2 images, cold, real models)

```
RAM         : 20118MB total, budget 14083MB (70%)
workers     : 12 (cpu=12)
peak RSS    : 2105MB          # well under the 14GB budget
per-stage   : vlm ~2.2s/img, finalize(embed) ~5.4s/img, write ~80ms/img
```

**Caveats (why these aren't headline numbers yet):**
- **2 images + cold start**: the `extract` stage time is dominated by one-time
  model downloads/loads (YOLO + OCR), not per-image work — it is not
  representative. A real throughput number needs a warm run over many images.
- This run also had objects on the stub fallback (Finding 1), so re-measure
  after the fix.

**Peak RSS (2.1GB) is the one solid number** and sits comfortably under the
14GB budget for this small run — consistent with the per-worker estimate, to be
confirmed at higher worker counts and corpus sizes.

## Finding 3 — warm benchmark (Stage 6): throughput + a parallelism ceiling

`sif bench --warmup` pre-loads every model so the timed run measures warm
per-image cost. Over 24 images with all real models (YOLO + OCR + Moondream +
nomic) on a 20GB / 12-core machine:

```
processed   : 24 images in 53.0s  (warm)
throughput  : ~0.45 img/s
peak RSS    : ~2.0GB              # confirms the per-worker estimate — far under
                                  # the 14GB budget at 12 workers
per-stage   : vlm ~1.3s/img, finalize(embed) ~0.85s/img, write ~33ms/img
```

**Parallelism ceiling (important):** the per-stage `extract` average looks huge
(~20s/img) but is *wall time including lock wait*, not inference time. Because
YOLO and OCR inference are **serialized by per-model locks** (Finding 1), 12
extraction workers all queue on those two locks — so worker count beyond a
**small number (~2-4)** buys little for the model stages. The real throughput is
bounded by the serialized models + the single VLM worker; the win is **stage
pipelining** (extraction of later items overlapping the VLM/writer), not wide
extraction fan-out. RAM-aware sizing still matters as a *cap*, but the practical
sweet spot is a handful of workers, not one-per-core.

*(Caveat: synthetic noise images are adversarial for OCR, which inflates the
extract stage; real photos/documents differ. The throughput/RSS shape holds.)*

## Still outstanding (carried forward)

- **Stage 3 RRF tuning** — RESOLVED in Stage 6 via retrieve-then-rerank
  (docs/stage3-findings.md). The cross-encoder now re-ranks by default.
