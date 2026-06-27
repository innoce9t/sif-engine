"""
Stage 0 stub extractors.

Each function has the SAME SIGNATURE it will have in Stage 1, but returns
placeholder data instead of running a real model. This lets us prove the
end-to-end pipeline shape (ingest -> extract -> assemble -> store -> query)
before pulling in heavy model weights.

In Stage 1 these get replaced one-by-one with real implementations behind
the identical interface, so nothing downstream changes.
"""
from __future__ import annotations

import hashlib
import os

from ..schema import DetectedObject, Face, Scene, OCRResult


def _stub_signal_from_path(path: str) -> int:
    """Deterministic pseudo-signal so stub output varies per file but is stable."""
    h = hashlib.sha256(path.encode()).hexdigest()
    return int(h[:8], 16)


def extract_objects(image_path: str) -> list[DetectedObject]:
    """STUB for YOLOv10n. Returns deterministic fake objects."""
    seed = _stub_signal_from_path(image_path)
    pool = ["person", "car", "tree", "building", "dog", "chair", "sign"]
    picks = [pool[seed % len(pool)], pool[(seed // 7) % len(pool)]]
    return [DetectedObject(label=p, confidence=0.9, bbox=[0, 0, 100, 100]) for p in dict.fromkeys(picks)]


def extract_faces(image_path: str) -> list[Face]:
    """STUB for InsightFace. Returns no faces by default."""
    return []


def extract_scene(image_path: str) -> Scene:
    """STUB for Moondream2. Returns a deterministic fake caption."""
    seed = _stub_signal_from_path(image_path)
    settings = ["outdoor urban scene", "indoor room", "natural landscape", "street view"]
    caption = settings[seed % len(settings)]
    tags = caption.split()
    return Scene(caption=caption, tags=tags)


def extract_ocr(image_path: str) -> OCRResult:
    """STUB for PaddleOCR. Returns no text by default."""
    return OCRResult(full_text="", text_blocks=[], has_text=False)


def read_basic_file_info(image_path: str) -> dict:
    """Real even in Stage 0: size + sha256. No model needed."""
    size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
    sha = ""
    if os.path.exists(image_path):
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha = "sha256:" + h.hexdigest()
    return {"size_bytes": size, "sha256": sha}
