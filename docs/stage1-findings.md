# Stage 1 — Real-Model Findings

Empirical notes from running the Stage 1 extractors/embedder against real
models (not stubs) for the first time, on **Windows 11 + Python 3.12** with a
local Ollama daemon.

Test environment: a `.venv` on Python 3.12 (the repo's default Python 3.14 is
too new for the torch/paddle wheels). Test images were two real photos (a
street scene with a bus, and a two-person portrait), which exercise detection,
captioning, OCR, and semantic retrieval in ways solid-color fixtures cannot.

**Outcome:** all four extractors + the embedder ran for real
(`{objects: yolov10n, scene: moondream2, ocr: paddleocr, faces: disabled}`),
the multi-vector split held with real data (OCR text stayed out of the visual
vector), and natural-language semantic search ranked every query's correct
image first. Three issues were found and fixed in the process.

---

## Finding 1 — Moondream2 returns an empty caption on a rigid format prompt

**Symptom:** the scene backend reported `moondream2` but produced an empty
caption. The pipeline ran fine; the caption was just `''`.

**Cause:** the original prompt demanded a strict structured response
(`"Respond in EXACTLY this format ... Caption: <...>\nTags: <...>"`).
Moondream2 (~1.8B) does not reliably follow rigid format instructions and
returned an **empty string**. A plain `"Describe this image in one detailed
sentence."` prompt returned good captions every time.

**Fix:** [`sif/extractors/scene.py`](../sif/extractors/scene.py) now uses a
plain caption prompt and derives tags itself (content-word extraction with a
small stopword filter). `_parse` still honors a `Caption:/Tags:` structured
response if a larger model is swapped in via `SIF_VLM_MODEL`.

**Lesson:** match prompt rigidity to model capability; tiny VLMs need simple
instructions. Only surfaced by running the actual model.

---

## Finding 2 — PaddleOCR 3.x crashes on MKL-DNN, and changed its result format

**Symptom:** `extract_ocr` raised
`NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute ...
onednn_instruction.cc` on first inference.

**Cause:** two separate PaddleOCR 3.x changes:
1. The default **MKL-DNN (oneDNN)** execution path hits an unimplemented op on
   this CPU under Paddle's new IR.
2. The result shape changed from the classic 2.x nesting
   (`[[[bbox, (text, conf)], ...]]`) to a list of `OCRResult` dicts exposing
   `rec_texts` / `rec_scores`. The 2.x-style parser silently produced no text.

**Fix:** [`sif/extractors/ocr.py`](../sif/extractors/ocr.py) constructs the
engine with `enable_mkldnn=False` by default (override with `SIF_OCR_MKLDNN=1`),
and `_texts_from_result` parses **both** the 3.x `rec_texts` dicts and the 2.x
nested lists. Verified by reading real signage off the test photo
(`MADRID`, `EMT`, `cero emisiones`).

**Lesson:** OCR stacks are environment-sensitive; pin behavior and parse
defensively across versions. The per-extractor stub fallback meant the rest of
the pipeline kept working while OCR was broken.

---

## Finding 3 — Stage 0 tests were not hermetic

**Symptom:** after real models were installed in the venv,
`test_stage0.py::test_pipeline_produces_populated_sif` failed on
`assert len(sif.objects) >= 1`.

**Cause:** the Stage 0 tests assert **stub-specific** output (deterministic
objects, the stub embedder's model name) but did not force stubs. With real
YOLO installed, a solid-color test image legitimately detects **zero** objects,
so the assertion failed. The tests implicitly depended on "no models installed"
being the default.

**Fix:** [`tests/test_stage0.py`](../tests/test_stage0.py) sets
`os.environ["SIF_USE_STUBS"] = "1"` at import — these are stub-contract tests
and should be deterministic regardless of what's installed. Suite is now green
in both the venv (real models present) and system Python 3.14 (stubs only).

**Lesson:** tests that assert a specific backend's output must pin that backend.
A fallback that auto-detects the environment makes "what's installed" an
implicit test input — eliminate it.

---

## Not a bug — deferred to later stages

- **OCR text isn't searchable yet.** Stage 1 query only hits the *visual*
  collection, so searching for OCR'd words (e.g. `MADRID`) returns nothing.
  Multi-vector retrieval + RRF fusion across both collections is **Stage 3**.
- **Faces** stay off by default (`SIF_ENABLE_FACES=1` to enable) and were not
  installed/tested here, by design (biometric-privacy surface).
