"""
SIF Schema - the structured metadata format that powers the engine.

Stage 0: minimal but real schema shape. The fields here mirror the v3
documentation. Extractors fill these in; storage persists them; retrieval
reads them back. Keeping this as the single source of schema truth means
every other module imports from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json
import time


# ---------------------------------------------------------------------------
# Sub-records. Kept as plain dataclasses (no heavy pydantic dependency for
# Stage 0) so the skeleton stays light. We can swap to pydantic validation
# in Stage 2 when the storage layer needs strict guarantees.
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    path: str
    sha256: str = ""          # tier-1 dedup: exact byte hash
    pixel_hash: str = ""      # tier-2 dedup: decoded-pixel hash (Stage 2)
    phash: str = ""           # tier-3 dedup: perceptual hash (Stage 2)
    size_bytes: int = 0
    resolution: tuple[int, int] = (0, 0)
    format: str = ""
    indexed_at: float = field(default_factory=time.time)


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: list[float] = field(default_factory=list)  # [x1,y1,x2,y2]


@dataclass
class Face:
    face_id: str
    embedding: list[float] = field(default_factory=list)  # 512-d
    bbox: list[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Scene:
    caption: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class OCRResult:
    full_text: str = ""
    text_blocks: list[str] = field(default_factory=list)
    has_text: bool = False


@dataclass
class Embeddings:
    """
    v3 multi-vector design: separate visual-semantic and OCR-text vectors.
    They live in different ChromaDB collections and are fused at query time
    via RRF (Stage 3). Stored here so the vector store is fully rebuildable
    from SQLite (the outbox / source-of-truth guarantee).
    """
    visual: list[float] = field(default_factory=list)   # caption + tags + objects
    text: list[float] = field(default_factory=list)     # OCR text only
    visual_input: str = ""   # the synthetic string that produced `visual`
    text_input: str = ""     # the synthetic string that produced `text`
    model: str = ""


@dataclass
class SIF:
    """The complete Semantic Index File for a single asset."""
    sif_version: str
    file: FileInfo
    objects: list[DetectedObject] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    scene: Scene = field(default_factory=Scene)
    ocr: OCRResult = field(default_factory=OCRResult)
    embeddings: Embeddings = field(default_factory=Embeddings)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- serialization helpers --------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def id(self) -> str:
        """Stable id for this asset = its content hash (or path if unhashed)."""
        return self.file.sha256 or self.file.path


SIF_VERSION = "3.0-stage0"


def new_sif(path: str) -> SIF:
    """Factory: a blank SIF for a given file path."""
    return SIF(sif_version=SIF_VERSION, file=FileInfo(path=path))
