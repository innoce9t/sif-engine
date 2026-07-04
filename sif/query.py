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
from . import retrieval, clip_embed


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


def _entity_of(vid: str) -> tuple[str, str, int | None]:
    """Map a vector id to (entity_id, doc_id, page_index).

    Image vectors have no '#' -> entity == doc == id, page None.
    PDF vectors are ``{doc}#p{n}`` (page text) or ``{doc}#p{n}#r{m}`` (region) ->
    the entity is the PAGE ``{doc}#p{n}`` (round-2 fix: fuse at the page, not the
    raw vector, since page text and sub-page region vectors are not 1:1)."""
    if "#" not in vid:
        return vid, vid, None
    parts = vid.split("#")
    doc = parts[0]
    page_tok = parts[1]                       # 'p{n}'
    pidx = int(page_tok[1:]) if page_tok[1:].isdigit() else None
    return f"{doc}#{page_tok}", doc, pidx


def _result_for(entity: str, doc: str, pidx: int | None, sif: dict, score: float) -> dict:
    """Build a result row + the text used for re-ranking, for an image asset or
    a PDF page entity."""
    if pidx is None:  # image
        caption = sif["scene"]["caption"]
        objects = [o["label"] for o in sif["objects"]]
        text = " ".join([caption, " ".join(sif["scene"]["tags"]),
                         " ".join(objects), sif["ocr"]["full_text"]]).strip()
        return {"id": entity, "path": doc, "page": None, "score": round(score, 6),
                "caption": caption, "objects": objects, "_text": text}
    # PDF page entity
    page = next((p for p in sif.get("pages", []) if p.get("page_index") == pidx), {})
    region_caps = [r.get("scene", {}).get("caption", "") for r in page.get("regions", [])]
    page_text = page.get("text", "")
    caption = (page_text[:120] or next((c for c in region_caps if c), "")
               or f"page {pidx + 1}")
    objects = [o["label"] for r in page.get("regions", []) for o in r.get("objects", [])]
    text = " ".join([page_text, *region_caps, " ".join(objects)]).strip()
    return {"id": entity, "path": doc, "page": pidx, "score": round(score, 6),
            "caption": caption, "objects": objects, "_text": text}


def _neural_rerank(query: str, pool: list[dict]):
    ce = _get_cross_encoder()
    if ce is None:
        return pool, False
    scores = ce.predict([(query, p["_text"]) for p in pool])
    order = sorted(range(len(pool)), key=lambda i: float(scores[i]), reverse=True)
    return [pool[i] for i in order], True


def search(store: Store, query: str, limit: int = 10,
           top_k: int = retrieval.DEFAULT_TOP_K, rerank: bool = True) -> list[dict]:
    q = embed(query, kind="query")
    if not any(q):
        return []
    q_clip = clip_embed.embed_text(query)   # [] when CLIP is unavailable

    # 1. retrieve ranks from each collection (each with its matching query
    # vector — nomic for visual/text, CLIP for the pixel space), then
    # 2. aggregate raw vectors up to their PAGE/asset ENTITY (best rank).
    per_source: dict[str, dict[str, int]] = {}
    entity_meta: dict[str, tuple[str, int | None]] = {}
    for name, coll, qvec in (("visual", store.visual, q), ("text", store.text, q),
                             ("clip", store.clip, q_clip)):
        if not any(qvec):
            continue
        n = coll.count()
        if n == 0:
            continue
        res = coll.query(query_embeddings=[qvec], n_results=min(top_k, n))
        ranks: dict[str, int] = {}
        for rank, vid in enumerate(res.get("ids", [[]])[0], start=1):
            ent, doc, pidx = _entity_of(vid)
            entity_meta[ent] = (doc, pidx)
            if ent not in ranks or rank < ranks[ent]:
                ranks[ent] = rank
        per_source[name] = ranks
    if not per_source:
        return []

    # 3-4. fuse entities, then validate each against SQLite (orphans excluded)
    fused = retrieval.rrf_fuse(per_source, top_k)
    pool: list[dict] = []
    for entity, score in fused:
        doc, pidx = entity_meta.get(entity, (entity, None))
        sif = store.get(doc)
        if sif is None:
            continue
        pool.append(_result_for(entity, doc, pidx, sif, score))
        if len(pool) >= max(limit * 3, 20):
            break

    # 5. re-rank the candidate pool with the cross-encoder (retrieve-then-rerank:
    # RRF gives recall, the cross-encoder gives precision and fixes RRF's
    # over-crediting of an asset that merely appears in both collections). It's
    # cheap on a ~20-item pool, so it runs by DEFAULT. Setting SIF_RERANK_GAP
    # restores cost-gating: only re-rank when the top two are within that gap.
    reranked = False
    if rerank and not _stubs() and len(pool) > 1:
        gap_env = os.environ.get("SIF_RERANK_GAP")
        do_rerank = (gap_env is None
                     or retrieval.should_rerank([p["score"] for p in pool], float(gap_env)))
        if do_rerank:
            pool, reranked = _neural_rerank(query, pool)

    out = []
    for p in pool[:limit]:
        p = dict(p)
        p.pop("_text", None)
        p["reranked"] = reranked
        out.append(p)
    return out
