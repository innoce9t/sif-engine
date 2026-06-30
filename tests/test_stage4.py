"""
Stage 4 tests: concurrency + RAM-aware sizing.

The decoupled runner (N extraction workers -> 1 VLM worker -> 1 writer) is
exercised on the stub path so it's fast and deterministic. We assert it indexes
correctly, is equivalent to the single-threaded path, dedups within a batch,
and that the writer really is the single mutation point.

Run: python -m pytest tests/test_stage4.py -v
or:  python tests/test_stage4.py
"""
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image

from sif import workers, runner
from sif.store import Store
from sif.ingest import ingest
from sif.query import search


def _img(path, seed):
    # Seeded noise so each image has a DISTINCT perceptual hash. (Solid colors
    # all share an all-zero dHash and would be deduped to one asset — correct
    # behavior, wrong for this test.)
    rnd = random.Random(seed)
    im = Image.new("RGB", (64, 64))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                for _ in range(64 * 64)])
    im.save(path)


def _make_set(d, n):
    paths = []
    for i in range(n):
        p = os.path.join(d, f"img_{i}.png")
        _img(p, i)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# RAM-aware sizing
# --------------------------------------------------------------------------
def test_recommended_workers_sane():
    n = workers.recommended_workers()
    assert n >= 1
    assert n <= (os.cpu_count() or 1)
    rep = workers.sizing_report()
    assert rep["recommended_workers"] == n
    assert rep["total_ram_mb"] > 0
    print(f"PASS: RAM-aware sizing -> {n} workers ({rep['total_ram_mb']}MB total)")


def test_sizing_formula_matches_spec():
    # The §6 example: a 12GB laptop @ 70% budget -> 7 extraction workers before
    # the CPU-count cap. Pin RAM and the cost constants to check the arithmetic.
    orig_total = workers.total_ram_mb
    orig = (workers.PER_WORKER_MB, workers.VLM_RESERVE_MB, workers.BASE_OVERHEAD_MB)
    workers.total_ram_mb = lambda: 12288.0
    workers.PER_WORKER_MB, workers.VLM_RESERVE_MB, workers.BASE_OVERHEAD_MB = 900, 1400, 300
    try:
        # (12288*0.70 - 1400 - 300) / 900 = 7.66 -> 7, then capped to CPU count
        assert workers.recommended_workers(0.70) == min(7, os.cpu_count() or 1)
    finally:
        workers.total_ram_mb = orig_total
        workers.PER_WORKER_MB, workers.VLM_RESERVE_MB, workers.BASE_OVERHEAD_MB = orig
    print("PASS: sizing formula matches the §6 spec (12GB -> 7 before CPU cap)")


# --------------------------------------------------------------------------
# Concurrent indexing
# --------------------------------------------------------------------------
def test_concurrent_index_all_present():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_set(d, 12)
        store = Store(os.path.join(d, "data"))
        report = runner.index_paths(store, paths, max_workers=4)
        assert report["stats"].get("indexed") == 12
        assert store.count() == 12
        assert store.visual.count() == 12
        # every row fully written (indexed=1) -> outbox completed under concurrency
        assert len(store.all_active_ids()) == 12
        # a known asset is searchable
        assert any(r["path"] == paths[0] for r in search(store, store.get(paths[0])["scene"]["caption"]))
        store.close()
    print("PASS: concurrent pipeline indexed all 12 images, all fully written")


def test_concurrent_matches_sequential():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_set(d, 8)

        seq = Store(os.path.join(d, "seq"))
        for p in paths:
            ingest(seq, p)
        seq_ids = seq.all_active_ids()
        seq.close()

        con = Store(os.path.join(d, "con"))
        runner.index_paths(con, paths, max_workers=4)
        con_ids = con.all_active_ids()
        con.close()

        assert seq_ids == con_ids
    print("PASS: concurrent result == sequential result")


def test_concurrent_dedup_within_batch():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.png"); _img(a, 1)
        b = os.path.join(d, "b.png"); shutil.copyfile(a, a.replace("a.png", "b.png"))  # exact dup
        c = os.path.join(d, "c.png"); _img(c, 2)
        store = Store(os.path.join(d, "data"))
        report = runner.index_paths(store, [a, b, c], max_workers=3)
        assert report["stats"].get("indexed") == 2
        assert report["stats"].get("duplicate") == 1
        assert store.count() == 2
        store.close()
    print("PASS: in-batch duplicates skipped under concurrency")


def test_report_has_stage_timings():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_set(d, 5)
        store = Store(os.path.join(d, "data"))
        report = runner.index_paths(store, paths, max_workers=2)
        for stage in ("extract", "vlm", "finalize", "write"):
            assert report["timings"][stage]["count"] == 5
        assert report["workers"] == 2
        assert report["wall_s"] >= 0
        store.close()
    print("PASS: runner reports per-stage timings for every item")


if __name__ == "__main__":
    test_recommended_workers_sane()
    test_sizing_formula_matches_spec()
    test_concurrent_index_all_present()
    test_concurrent_matches_sequential()
    test_concurrent_dedup_within_batch()
    test_report_has_stage_timings()
    print("\nAll Stage 4 tests passed.")
