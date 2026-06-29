"""
Text embedding (Stage 1): nomic-embed-text v1.5 with a hashing fallback.

Stage 0 shipped a deterministic hash embedder so the plumbing could be tested
without semantics. Stage 1 swaps in a real model (nomic-embed-text-v1.5 via
sentence-transformers) when it's installed, and transparently falls back to the
hash embedder otherwise so the pipeline always runs end-to-end.

Dimension consistency: a ChromaDB collection is pinned to the dimension of its
first insert, so the real-vs-stub choice is decided ONCE per process and cached
(real = 384-d Matryoshka-truncated, stub = 64-d). Mixing the two within one
index would corrupt it.

nomic v1.5 expects task prefixes — indexed assets use ``search_document:`` and
searches use ``search_query:`` — so ``embed`` takes a ``kind`` argument. The
stub ignores it, keeping the interface identical for both backends.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os

log = logging.getLogger("sif.embedding")

DIM = 64                                  # stub dimension
REAL_DIM = 384                            # nomic v1.5, Matryoshka-truncated
REAL_MODEL = "nomic-ai/nomic-embed-text-v1.5"

_decided: bool | None = None             # None = undecided, then True/False
_model = None


def _force_stubs() -> bool:
    return os.environ.get("SIF_USE_STUBS") == "1"


def _real_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _decide() -> bool:
    """Resolve (once) whether the real model is usable, loading it if so."""
    global _decided, _model
    if _decided is not None:
        return _decided
    if _force_stubs() or not _real_available():
        _decided = False
        return False
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(REAL_MODEL, trust_remote_code=True, truncate_dim=REAL_DIM)
        _decided = True
    except Exception as e:
        log.warning("embedding: real model unavailable (%s); using hash stub", e)
        _decided = False
    return _decided


def reset() -> None:
    """Drop the cached backend decision (used by tests that toggle env flags)."""
    global _decided, _model
    _decided = None
    _model = None


def active_model() -> str:
    """Name of the embedding backend that will be used."""
    return "nomic-embed-text-v1.5" if _decide() else "stub-hash-embed-stage0"


def active_dim() -> int:
    return REAL_DIM if _decide() else DIM


def _embed_real(text: str, kind: str) -> list[float]:
    prefix = "search_query: " if kind == "query" else "search_document: "
    vec = _model.encode(prefix + text, normalize_embeddings=True)
    return [float(v) for v in vec]


def _embed_stub(text: str, dim: int = DIM) -> list[float]:
    """Deterministic hashing embedder. Same text -> same vector."""
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed(text: str, kind: str = "document") -> list[float]:
    """Embed text. ``kind='document'`` for indexed assets, ``'query'`` for searches.

    Empty input returns a zero vector (of the active dimension) so the store
    skips it — e.g. an image with no OCR text gets no TEXT vector at all.
    """
    if not text or not text.strip():
        return [0.0] * active_dim()
    if _decide():
        # Decided real: don't silently fall back on a runtime failure — that
        # would change the vector dimension mid-index and corrupt the store.
        return _embed_real(text, kind)
    return _embed_stub(text)
