"""
Stage 0 embedding stub.

A real embedding model (nomic-embed-text) lands in Stage 1. For Stage 0 we
use a tiny deterministic hashing embedder: it maps text -> a fixed-length
vector reproducibly. It is NOT semantically meaningful, but it lets the
vector store, the multi-vector split, and (later) RRF fusion all run and be
tested for *mechanics* before real semantics arrive.
"""
from __future__ import annotations

import hashlib

DIM = 64  # small for Stage 0; real model is 384-d


def embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic hashing embedder. Same text -> same vector."""
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    # Hash each token into buckets; crude bag-of-words feel, fully deterministic.
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize so cosine distance behaves
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
