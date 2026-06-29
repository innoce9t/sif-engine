"""
Semantic search (Stage 3): multi-vector retrieval with RRF fusion and gated
neural re-ranking.

Pipeline:
  1. Embed the query (search_query prefix).
  2. Retrieve top-K from BOTH vector collections (visual + text).
  3. Fuse with Reciprocal Rank Fusion incl. the absentee-baseline fix
     (see retrieval.rrf_fuse).
  4. Validate every candidate against SQLite — orphan vectors can never surface.
  5. If the top two are ambiguous (small relative gap), re-rank the candidate
     pool with a cross-encoder; otherwise return the RRF order as-is.

The cross-encoder is optional: skipped under SIF_USE_STUBS and when
sentence-transformers isn't installed, so search always works. Result dicts keep
`id`/`path`/`caption`/`objects` (Stage 0/1/2 callers) and add `score`/`reranked`.
"""
from __future__ import annotations

import os

from .store import Store
from .embedding import embed
from . import retrieval


def _stubs() -> bool:
    return os.environ.get("SIF_USE_STUBS") == "1"


# -- optional cross-encoder (lazy, cached) --------------------------------
_cross = None
_cross_failed = False


def _get_cross_encoder():
    global _cross, _cross_failed
    if _cross is not None or _cross_failed:
        return _cross
    try:
        from sentence_transformers import CrossEncoder
        _cross = CrossEncoder(os.environ.get(
            "SIF_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    except Exception:
        _cross_failed = True
    return _cross


def _doc(sif: dict) -> str:
    """Text view of an asset for the cross-encoder (includes OCR)."""
    parts = [
        sif["scene"]["caption"],
        " ".join(sif["scene"]["tags"]),
        " ".join(o["label"] for o in sif["objects"]),
        sif["ocr"]["full_text"],
    ]
    return " ".join(p for p in parts if p).strip()


def _neural_rerank(query: str, pool: list[dict]):
    ce = _get_cross_encoder()
    if ce is None:
        return pool, False
    scores = ce.predict([(query, _doc(p["_sif"])) for p in pool])
    order = sorted(range(len(pool)), key=lambda i: float(scores[i]), reverse=True)
    return [pool[i] for i in order], True


def search(store: Store, query: str, limit: int = 10,
           top_k: int = retrieval.DEFAULT_TOP_K, rerank: bool = True) -> list[dict]:
    q = embed(query, kind="query")
    if not any(q):
        return []

    # 1-2. retrieve ranks from each collection
    per_source: dict[str, dict[str, int]] = {}
    for name, coll in (("visual", store.visual), ("text", store.text)):
        n = coll.count()
        if n == 0:
            continue
        res = coll.query(query_embeddings=[q], n_results=min(top_k, n))
        ids = res.get("ids", [[]])[0]
        per_source[name] = {sid: rank for rank, sid in enumerate(ids, start=1)}
    if not per_source:
        return []

    # 3-4. fuse, then validate each candidate against SQLite
    fused = retrieval.rrf_fuse(per_source, top_k)
    pool: list[dict] = []
    for sid, score in fused:
        sif = store.get(sid)            # active rows only -> orphans excluded
        if sif is None:
            continue
        pool.append({
            "id": sid,
            "path": sif["file"]["path"],
            "score": round(score, 6),
            "caption": sif["scene"]["caption"],
            "objects": [o["label"] for o in sif["objects"]],
            "_sif": sif,
        })
        if len(pool) >= max(limit * 3, 20):
            break

    # 5. gated re-rank on a small relative margin
    reranked = False
    if rerank and not _stubs() and retrieval.should_rerank([p["score"] for p in pool]):
        pool, reranked = _neural_rerank(query, pool)

    out = []
    for p in pool[:limit]:
        p = dict(p)
        p.pop("_sif", None)
        p["reranked"] = reranked
        out.append(p)
    return out
