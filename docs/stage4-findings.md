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

## Still outstanding (carried forward)

- **Warm, multi-image benchmark** to get a representative throughput/img and to
  validate the per-worker MB estimate at scale.
- **Stage 3 RRF tuning** (docs/stage3-findings.md) — needs a small labeled
  relevance set to measure against before adjusting fusion/gating; the harness
  is now in place to support that measurement.
