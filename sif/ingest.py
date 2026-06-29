"""
Dedup-aware ingestion (Stage 2).

Decides, cheaply and BEFORE running any models, whether a file needs to be
processed at all — then routes it through the crash-safe store lifecycle.

Flow (see ADR 0002):
  1. path known + sha unchanged  -> 'unchanged'  (idempotent re-scan)
  2. path known + sha changed    -> 'updated'    (dark-window-safe update)
  3. content matches another id  -> 'duplicate'  (skip; tier sha/pixel/phash)
  4. otherwise                   -> 'indexed'    (outbox insert)

The hashes are computed once here and handed to the pipeline so they aren't
recomputed. Only cases 2 and 4 pay for model inference.
"""
from __future__ import annotations

from typing import NamedTuple

from . import dedup
from .pipeline import process
from .store import Store


class Result(NamedTuple):
    status: str          # indexed | updated | unchanged | duplicate
    path: str
    detail: str = ""     # dup target id, or dedup tier


def ingest(store: Store, path: str) -> Result:
    h = dedup.hashes(path)

    meta = store.get_meta(path)
    if meta is not None:
        if meta["sha256"] == h.sha256:
            return Result("unchanged", path)
        store.update(process(path, file_hashes=h))
        return Result("updated", path)

    dup = store.find_duplicate(h)
    if dup is not None:
        return Result("duplicate", path, detail=f"{dup[1]}->{dup[0]}")

    store.insert(process(path, file_hashes=h))
    return Result("indexed", path)
