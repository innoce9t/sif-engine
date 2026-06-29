"""
Stage 0 query.

Minimal semantic search: embed the query, look it up in the visual
collection, return matches validated against SQLite (the source of truth).

Multi-vector RRF fusion, page-entity aggregation, and gated re-ranking all
arrive in Stage 3. Stage 0 proves the round-trip: a query string finds the
asset we indexed.
"""
from __future__ import annotations

from .store import Store
from .embedding import embed


def search(store: Store, query: str, limit: int = 10) -> list[dict]:
    q = embed(query, kind="query")
    if not any(q):
        return []

    res = store.visual.query(query_embeddings=[q], n_results=limit)
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for sid, dist in zip(ids, dists):
        # CRITICAL even in Stage 0: validate every hit against SQLite before
        # returning it. This is the guarantee that (later) stranded vectors
        # can never surface in results.
        sif = store.get(sid)
        if sif is None:
            continue
        out.append({
            "id": sid,
            "path": sif["file"]["path"],
            "distance": round(dist, 4),
            "caption": sif["scene"]["caption"],
            "objects": [o["label"] for o in sif["objects"]],
        })
    return out
