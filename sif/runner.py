"""
Concurrent indexing runner (Stage 4).

Decoupled stages connected by micro-queues — the anti-thrashing architecture
from the design review (round-3 fix #2):

    N extraction workers  ->  vlm_queue  ->  1 VLM worker  ->  write_queue  ->  1 writer

* Extraction workers (a thread pool, sized by ``workers.recommended_workers``)
  run the cheap, parallelizable extractors and push a partial SIF onto the VLM
  queue. They NEVER touch the VLM, so they never block on it or trigger weight
  thrashing.
* A single dedicated VLM worker drains the queue and runs the memory-bound
  scene model sequentially, then finalizes embeddings.
* A single dedicated writer applies all SQLite/Chroma writes — the one place
  storage is mutated, which is what makes the Stage 2 outbox ordering safe under
  concurrency.

Threads (not processes) are used deliberately: ONNX Runtime, PaddleOCR, and
torch release the GIL during inference, so threads give real parallelism while
sharing one in-memory copy of each model (processes would reload weights per
worker and blow the RAM budget). See ADR 0003.

Dedup runs as a single-threaded pre-pass so the expensive pipeline only touches
files that actually need indexing, and store reads stay off the worker threads.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import dedup, pipeline, workers
from .store import Store

_SENTINEL = object()


def _near(phashes: list[str], ph: str,
          threshold: int = dedup.DEFAULT_PHASH_THRESHOLD) -> bool:
    return bool(ph) and any(dedup.hamming(ph, p) <= threshold for p in phashes)


def _dedup_plan(store: Store, paths: list[str], force: bool = False):
    """Single-threaded pre-pass: decide insert/update/skip per path, against the
    store and within the batch itself. Returns (plan, tally). ``force`` bypasses
    the dedup skips and re-processes everything (re-index)."""
    plan: list[tuple] = []
    tally: dict[str, int] = {}
    seen_sha: set[str] = set()
    seen_pixel: set[str] = set()
    seen_phash: list[str] = []

    if force:
        for path in paths:
            h = dedup.hashes(path)
            action = "update" if store.get_meta(path) is not None else "insert"
            plan.append((path, h, action))
        return plan, tally

    for path in paths:
        h = dedup.hashes(path)
        meta = store.get_meta(path)
        if meta is not None:
            if meta["sha256"] == h.sha256:
                tally["unchanged"] = tally.get("unchanged", 0) + 1
                continue
            plan.append((path, h, "update"))
            continue
        if store.find_duplicate(h) is not None:
            tally["duplicate"] = tally.get("duplicate", 0) + 1
            continue
        if (h.sha256 in seen_sha or (h.pixel_hash and h.pixel_hash in seen_pixel)
                or _near(seen_phash, h.phash)):
            tally["duplicate"] = tally.get("duplicate", 0) + 1
            continue
        seen_sha.add(h.sha256)
        if h.pixel_hash:
            seen_pixel.add(h.pixel_hash)
        if h.phash:
            seen_phash.append(h.phash)
        plan.append((path, h, "insert"))
    return plan, tally


def _summarize(timings: dict[str, list[float]]) -> dict:
    out = {}
    for stage, samples in timings.items():
        if samples:
            out[stage] = {
                "count": len(samples),
                "avg_ms": round(1000 * sum(samples) / len(samples), 2),
                "total_s": round(sum(samples), 3),
            }
        else:
            out[stage] = {"count": 0, "avg_ms": 0.0, "total_s": 0.0}
    return out


def index_paths(store: Store, paths: list[str], max_workers: int | None = None,
                progress=None, force: bool = False) -> dict:
    """Index ``paths`` through the decoupled concurrent pipeline. Returns a
    report dict: tally of statuses, worker count, processed count, per-stage
    timings, and wall time. ``force`` re-processes already-indexed files."""
    if max_workers is None:
        max_workers = workers.recommended_workers()

    plan, tally = _dedup_plan(store, paths, force=force)

    vlm_q: queue.Queue = queue.Queue(maxsize=max(2, max_workers * 4))
    write_q: queue.Queue = queue.Queue()
    timings: dict[str, list[float]] = {"extract": [], "vlm": [], "finalize": [], "write": []}
    lock = threading.Lock()
    wall0 = time.time()

    def writer():
        while True:
            item = write_q.get()
            if item is _SENTINEL:
                break
            sif, action = item
            ts = time.time()
            (store.update if action == "update" else store.insert)(sif)
            label = "updated" if action == "update" else "indexed"
            with lock:
                timings["write"].append(time.time() - ts)
                tally[label] = tally.get(label, 0) + 1
            if progress:
                progress(label, sif.file.path)

    def vlm_worker():
        while True:
            item = vlm_q.get()
            if item is _SENTINEL:
                break
            sif, action, t0 = item
            ts = time.time()
            pipeline.add_scene(sif)
            tv = time.time() - ts
            tf = time.time()
            pipeline.finalize(sif, t0)
            with lock:
                timings["vlm"].append(tv)
                timings["finalize"].append(time.time() - tf)
            write_q.put((sif, action))

    wt = threading.Thread(target=writer, name="sif-writer", daemon=True)
    vt = threading.Thread(target=vlm_worker, name="sif-vlm", daemon=True)
    wt.start()
    vt.start()

    def extract(item):
        path, h, action = item
        t0 = time.time()
        sif = pipeline.extract_partial(path, h)
        with lock:
            timings["extract"].append(time.time() - t0)
        vlm_q.put((sif, action, t0))

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sif-extract") as ex:
        list(ex.map(extract, plan))

    # Drain in order: extraction done -> stop VLM worker -> stop writer.
    vlm_q.put(_SENTINEL)
    vt.join()
    write_q.put(_SENTINEL)
    wt.join()

    return {
        "stats": tally,
        "workers": max_workers,
        "processed": len(plan),
        "wall_s": round(time.time() - wall0, 3),
        "timings": _summarize(timings),
    }
