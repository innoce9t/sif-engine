"""
Semantic search: multi-vector retrieval with RRF fusion and cross-encoder
re-ranking, plus image-similarity search over the CLIP space.

Public entry points:
  * search(store, text)          — text query across visual + OCR + CLIP spaces
  * search_by_image(store, vec)  — reverse image search (CLIP pixel similarity)
  * search_similar(store, id)    — "more like this" for an indexed asset

All share the same fuse -> validate-vs-SQLite -> filter -> re-rank machinery.
Results are page/asset entities (round-2 aggregation) and carry `matched`
(which vector spaces hit) so the UI can explain a result.
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
    """Map a vector id to (entity_id, doc_id, page_index). Images: entity==doc,
    page None. PDF page/region vectors aggregate to the PAGE entity."""
    if "#" not in vid:
        return vid, vid, None
    parts = vid.split("#")
    doc, page_tok = parts[0], parts[1]
    pidx = int(page_tok[1:]) if page_tok[1:].isdigit() else None
    return f"{doc}#{page_tok}", doc, pidx


def _result_for(entity: str, doc: str, pidx: int | None, sif: dict, score: float) -> dict:
    if pidx is None:  # image
        caption = sif["scene"]["caption"]
        objects = [o["label"] for o in sif["objects"]]
        text = " ".join([caption, " ".join(sif["scene"]["tags"]),
                         " ".join(objects), sif["ocr"]["full_text"]]).strip()
        return {"id": entity, "path": doc, "page": None, "kind": sif.get("kind", "image"),
                "score": round(score, 6), "caption": caption, "objects": objects, "_text": text}
    page = next((p for p in sif.get("pages", []) if p.get("page_index") == pidx), {})
    region_caps = [r.get("scene", {}).get("caption", "") for r in page.get("regions", [])]
    page_text = page.get("text", "")
    caption = (page_text[:120] or next((c for c in region_caps if c), "") or f"page {pidx + 1}")
    objects = [o["label"] for r in page.get("regions", []) for o in r.get("objects", [])]
    text = " ".join([page_text, *region_caps, " ".join(objects)]).strip()
    return {"id": entity, "path": doc, "page": pidx, "kind": "pdf",
            "score": round(score, 6), "caption": caption, "objects": objects, "_text": text}


def _passes(sif: dict, pidx: int | None, filters: dict) -> bool:
    if not filters:
        return True
    kind = filters.get("kind")
    if kind in ("image", "pdf") and sif.get("kind", "image") != kind:
        return False
    if filters.get("has_text"):
        if pidx is None:
            if not sif.get("ocr", {}).get("has_text"):
                return False
        else:
            page = next((p for p in sif.get("pages", []) if p.get("page_index") == pidx), {})
            if not page.get("text", "").strip():
                return False
    obj = (filters.get("object") or "").strip().lower()
    if obj:
        labels = {o["label"].lower() for o in sif.get("objects", [])}
        labels |= {o["label"].lower() for p in sif.get("pages", [])
                   for r in p.get("regions", []) for o in r.get("objects", [])}
        if obj not in labels:
            return False
    return True


def _neural_rerank(query: str, pool: list[dict]):
    ce = _get_cross_encoder()
    if ce is None:
        return pool, False
    scores = ce.predict([(query, p["_text"]) for p in pool])
    order = sorted(range(len(pool)), key=lambda i: float(scores[i]), reverse=True)
    return [pool[i] for i in order], True


def _blend_clip(pool: list[dict], clip_ranks: dict[str, int], k: int = 60, absent: int = 51):
    """Fuse the (text) re-rank order with the CLIP visual rank so a strong pixel
    match nudges upward without overriding text relevance (rerank weighted 2:1)."""
    scored = []
    for i, row in enumerate(pool, start=1):
        cr = clip_ranks.get(row["id"], absent)
        scored.append((2.0 / (k + i) + 1.0 / (k + cr), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _collect_ranks(store, sources):
    per_source: dict[str, dict[str, int]] = {}
    entity_meta: dict[str, tuple[str, int | None]] = {}
    for name, coll, qvec in sources:
        if not any(qvec):
            continue
        n = coll.count()
        if n == 0:
            continue
        res = coll.query(query_embeddings=[qvec],
                         n_results=min(retrieval.DEFAULT_TOP_K, n))
        ranks: dict[str, int] = {}
        for rank, vid in enumerate(res.get("ids", [[]])[0], start=1):
            ent, doc, pidx = _entity_of(vid)
            entity_meta[ent] = (doc, pidx)
            if ent not in ranks or rank < ranks[ent]:
                ranks[ent] = rank
        per_source[name] = ranks
    return per_source, entity_meta


def _assemble(store, per_source, entity_meta, limit, rerank_query, filters):
    if not per_source:
        return []
    fused = retrieval.rrf_fuse(per_source, retrieval.DEFAULT_TOP_K)
    pool: list[dict] = []
    for entity, score in fused:
        doc, pidx = entity_meta.get(entity, (entity, None))
        sif = store.get(doc)
        if sif is None or not _passes(sif, pidx, filters):
            continue
        row = _result_for(entity, doc, pidx, sif, score)
        row["matched"] = [s for s in per_source if entity in per_source[s]]
        pool.append(row)
        if len(pool) >= max(limit * 3, 20):
            break

    reranked = False
    if rerank_query and not _stubs() and len(pool) > 1:
        gap_env = os.environ.get("SIF_RERANK_GAP")
        if gap_env is None or retrieval.should_rerank([p["score"] for p in pool], float(gap_env)):
            pool, reranked = _neural_rerank(rerank_query, pool)
            if reranked and "clip" in per_source:
                pool = _blend_clip(pool, per_source["clip"])

    out = []
    for p in pool[:limit]:
        p = dict(p)
        p.pop("_text", None)
        p["reranked"] = reranked
        out.append(p)
    return out


def search(store: Store, query: str, limit: int = 10,
           top_k: int = retrieval.DEFAULT_TOP_K, rerank: bool = True,
           filters: dict | None = None) -> list[dict]:
    q = embed(query, kind="query")
    if not any(q):
        return []
    q_clip = clip_embed.embed_text(query)
    sources = [("visual", store.visual, q), ("text", store.text, q),
               ("clip", store.clip, q_clip)]
    per_source, entity_meta = _collect_ranks(store, sources)
    return _assemble(store, per_source, entity_meta, limit,
                     rerank_query=(query if rerank else None), filters=filters)


def search_by_image(store: Store, clip_vec: list[float], limit: int = 20,
                    filters: dict | None = None) -> list[dict]:
    """Reverse image search: rank by CLIP pixel similarity (no text re-rank)."""
    if not any(clip_vec):
        return []
    per_source, entity_meta = _collect_ranks(store, [("clip", store.clip, clip_vec)])
    return _assemble(store, per_source, entity_meta, limit,
                     rerank_query=None, filters=filters)


def search_similar(store: Store, asset_id: str, limit: int = 12) -> list[dict]:
    """'More like this' for an indexed image, via its stored CLIP vector."""
    doc = asset_id.split("#")[0]
    sif = store.get(doc)
    if sif is None:
        return []
    clip_vec = (sif.get("embeddings") or {}).get("clip") or []
    if not any(clip_vec):  # PDF or no CLIP vector
        return []
    hits = search_by_image(store, clip_vec, limit=limit + 1)
    return [h for h in hits if h["path"] != doc][:limit]
