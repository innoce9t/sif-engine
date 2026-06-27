"""
Stage 0 ingestion pipeline.

Single-threaded, stub-model orchestration. Takes a file path, runs every
extractor, assembles a SIF, builds the two embedding-input strings (the v3
multi-vector split), embeds them, and returns the populated SIF ready to store.

Concurrency (RAM-aware workers + dedicated VLM queue) arrives in Stage 4.
Real models arrive in Stage 1. The shape here does not change.
"""
from __future__ import annotations

import time

from .schema import SIF, FileInfo, new_sif
from .extractors import stub
from .embedding import embed


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


def process(path: str) -> SIF:
    t0 = time.time()
    sif = new_sif(path)

    # -- file info (real even in Stage 0) --
    info = stub.read_basic_file_info(path)
    sif.file.sha256 = info["sha256"]
    sif.file.size_bytes = info["size_bytes"]

    # -- extraction (stubs in Stage 0) --
    sif.objects = stub.extract_objects(path)
    sif.faces = stub.extract_faces(path)
    sif.scene = stub.extract_scene(path)
    sif.ocr = stub.extract_ocr(path)

    # -- multi-vector embedding inputs --
    sif.embeddings.visual_input = build_visual_input(sif)
    sif.embeddings.text_input = build_text_input(sif)
    sif.embeddings.visual = embed(sif.embeddings.visual_input)
    sif.embeddings.text = embed(sif.embeddings.text_input)
    sif.embeddings.model = "stub-hash-embed-stage0"

    sif.meta = {
        "processing_ms": round((time.time() - t0) * 1000, 2),
        "stage": 0,
        "models_used": ["stub"],
    }
    return sif
