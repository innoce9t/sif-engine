"""
Ingestion pipeline.

Assembles a SIF from a file: run the extractors, build the two embedding-input
strings (the multi-vector split), embed them.

The work is split into three stage functions so Stage 4's concurrent runner can
decouple them across threads (cheap parallel extractors -> single VLM worker ->
single writer), while ``process()`` keeps the simple single-threaded path for
the API, CLI single calls, and tests:

  * ``extract_partial`` — file info/hashes + objects, faces, OCR (no VLM)
  * ``add_scene``       — the memory-bound VLM caption (serialized in Stage 4)
  * ``finalize``        — multi-vector embedding inputs + vectors + meta
"""
from __future__ import annotations

import os
import time

from .schema import SIF, new_sif
from . import extractors, dedup
from .embedding import embed, active_model


def build_visual_input(sif: SIF) -> str:
    """Synthetic string for the VISUAL vector: caption + tags + object labels.
    Deliberately EXCLUDES OCR text (that was the v2 dilution bug)."""
    parts = [sif.scene.caption]
    parts += sif.scene.tags
    parts += [o.label for o in sif.objects]
    return " ".join(p for p in parts if p).strip()


def build_text_input(sif: SIF) -> str:
    """Synthetic string for the TEXT vector: OCR text only."""
    return sif.ocr.full_text.strip()


def extract_partial(path: str, file_hashes: dedup.Hashes | None = None) -> SIF:
    """Everything EXCEPT the VLM scene caption: file info/hashes + the cheap,
    parallelizable extractors (objects, faces, OCR)."""
    sif = new_sif(path)

    # -- file info (model-free: three dedup hashes + size + resolution/format) --
    h = file_hashes if file_hashes is not None else dedup.hashes(path)
    sif.file.sha256 = h.sha256
    sif.file.pixel_hash = h.pixel_hash
    sif.file.phash = h.phash
    sif.file.size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    try:
        from PIL import Image

        with Image.open(path) as im:
            sif.file.resolution = (im.width, im.height)
            sif.file.format = (im.format or "").upper()
    except Exception:
        pass  # non-image / unreadable: leave schema defaults

    sif.objects = extractors.extract_objects(path)
    sif.faces = extractors.extract_faces(path)   # [] unless SIF_ENABLE_FACES=1
    sif.ocr = extractors.extract_ocr(path)
    return sif


def add_scene(sif: SIF) -> SIF:
    """The memory-bound VLM caption. Serialized through one worker in Stage 4."""
    sif.scene = extractors.extract_scene(sif.file.path)
    return sif


def finalize(sif: SIF, started_at: float | None = None) -> SIF:
    """Build the multi-vector embedding inputs, embed them, and stamp meta."""
    sif.embeddings.visual_input = build_visual_input(sif)
    sif.embeddings.text_input = build_text_input(sif)
    sif.embeddings.visual = embed(sif.embeddings.visual_input, kind="document")
    sif.embeddings.text = embed(sif.embeddings.text_input, kind="document")
    sif.embeddings.model = active_model()

    sif.meta = {
        "processing_ms": round((time.time() - started_at) * 1000, 2) if started_at else 0.0,
        "stage": 1,
        "models_used": extractors.backends_used(),
        "embedding": active_model(),
    }
    return sif


def process(path: str, file_hashes: dedup.Hashes | None = None) -> SIF:
    """Single-threaded composition of the three stages."""
    t0 = time.time()
    sif = extract_partial(path, file_hashes)
    add_scene(sif)
    finalize(sif, t0)
    return sif
