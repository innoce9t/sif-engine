"""
Ingestion pipeline (Stage 1).

Single-threaded orchestration: take a file path, run every extractor (real
model when available, deterministic stub otherwise), assemble a SIF, build the
two embedding-input strings (the multi-vector split), embed them, and return
the populated SIF ready to store.

The extractor/embedder interfaces are identical to Stage 0, so storage and
retrieval are untouched. Concurrency (RAM-aware workers + dedicated VLM queue)
arrives in Stage 4.
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


def process(path: str, file_hashes: dedup.Hashes | None = None) -> SIF:
    t0 = time.time()
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

    # -- extraction (real models when available, else stubs) --
    sif.objects = extractors.extract_objects(path)
    sif.faces = extractors.extract_faces(path)   # [] unless SIF_ENABLE_FACES=1
    sif.scene = extractors.extract_scene(path)
    sif.ocr = extractors.extract_ocr(path)

    # -- multi-vector embedding inputs --
    sif.embeddings.visual_input = build_visual_input(sif)
    sif.embeddings.text_input = build_text_input(sif)
    sif.embeddings.visual = embed(sif.embeddings.visual_input, kind="document")
    sif.embeddings.text = embed(sif.embeddings.text_input, kind="document")
    sif.embeddings.model = active_model()

    sif.meta = {
        "processing_ms": round((time.time() - t0) * 1000, 2),
        "stage": 1,
        "models_used": extractors.backends_used(),
        "embedding": active_model(),
    }
    return sif
