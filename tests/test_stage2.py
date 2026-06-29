"""
Stage 2 tests: storage hardening + lifecycle.

Pure logic, no models — forced onto the stub path so it's fast and deterministic.
Covers the design-review durability fixes: outbox ordering, crash recovery, the
update dark-window, delete/tombstone + reconciliation, orphan purge, three-tier
dedup, and WAL/busy_timeout.

Run: python -m pytest tests/test_stage2.py -v
or:  python tests/test_stage2.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Storage logic is model-independent; force stubs for speed + determinism.
os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image

from sif.store import Store
from sif.pipeline import process
from sif.ingest import ingest
from sif.query import search


def _grad(path, shift=0, compress_level=6, patch=0):
    """A non-trivial gradient PNG (so phash is meaningful). `compress_level`
    changes bytes without changing pixels; `patch` paints a small corner block."""
    im = Image.new("RGB", (64, 64))
    px = im.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = ((x * 4 + shift) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
    for y in range(patch):
        for x in range(patch):
            px[x, y] = (255, 0, 0)
    im.save(path, format="PNG", compress_level=compress_level)


# --------------------------------------------------------------------------
# WAL / pragmas
# --------------------------------------------------------------------------
def test_wal_and_busy_timeout():
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "data"))
        mode = store.db.execute("PRAGMA journal_mode").fetchone()[0]
        bt = store.db.execute("PRAGMA busy_timeout").fetchone()[0]
        assert mode.lower() == "wal"
        assert bt == 5000
        store.close()
    print("PASS: SQLite in WAL mode with busy_timeout=5000")


# --------------------------------------------------------------------------
# Outbox insert + crash recovery
# --------------------------------------------------------------------------
def test_insert_is_outbox_ordered():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png"); _grad(p)
        store = Store(os.path.join(d, "data"))
        sif = process(p)
        store.insert(sif)
        # row indexed, vector present
        assert store.get_meta(p)["indexed"] == 1
        assert store.visual.count() == 1
        assert p in store.all_active_ids()
        store.close()
    print("PASS: insert lands row then vectors then indexed=1")


def test_crash_recovery_replays_unindexed_rows():
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        p = os.path.join(d, "a.png"); _grad(p)

        store = Store(data)
        sif = process(p)
        store.insert(sif)
        # Simulate a crash mid-write: row exists as indexed=0, vectors gone.
        store.db.execute("UPDATE sif SET indexed=0 WHERE id=?", (p,)); store.db.commit()
        store._purge_vectors(p)
        assert store.visual.count() == 0
        store.close()

        # Restart -> recovery sweep replays vectors from the stored SIF JSON.
        store2 = Store(data)
        assert store2.visual.count() == 1
        assert p in store2.all_active_ids()
        assert any(r["id"] == p for r in search(store2, sif.scene.caption))
        store2.close()
    print("PASS: recovery sweep replays unindexed rows after a crash")


def test_update_dark_window_is_recoverable():
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        p = os.path.join(d, "a.png")

        _grad(p, shift=0)
        store = Store(data)
        store.insert(process(p))
        sha_a = store.get_meta(p)["sha256"]

        # Content at the same path changes; simulate an update interrupted AFTER
        # indexed was flipped to 0 and old vectors purged, BEFORE new vectors.
        _grad(p, shift=80)
        sif_b = process(p)
        store.db.execute(
            "UPDATE sif SET indexed=0, sif_json=?, sha256=? WHERE id=?",
            (sif_b.to_json(indent=None), sif_b.file.sha256, p))
        store.db.commit()
        store._purge_vectors(p)
        assert sif_b.file.sha256 != sha_a
        assert store.visual.count() == 0
        store.close()

        # Restart -> recovery restores the NEW content's vectors.
        store2 = Store(data)
        assert store2.visual.count() == 1
        assert store2.get_meta(p)["indexed"] == 1
        assert store2.get_meta(p)["sha256"] == sif_b.file.sha256
        store2.close()
    print("PASS: interrupted update (dark window) recovered to new content")


# --------------------------------------------------------------------------
# Delete + reconcile
# --------------------------------------------------------------------------
def test_delete_removes_row_and_vectors():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png"); _grad(p)
        store = Store(os.path.join(d, "data"))
        store.insert(process(p))
        assert store.count() == 1 and store.visual.count() == 1
        store.delete(p)
        assert store.count() == 0 and store.visual.count() == 0
        assert store.get(p) is None
        store.close()
    print("PASS: delete removes both the row and its vectors")


def test_reconcile_finishes_tombstone():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png"); _grad(p)
        store = Store(os.path.join(d, "data"))
        store.insert(process(p))
        # Interrupted delete: tombstone set but vectors not yet purged.
        store.db.execute("UPDATE sif SET state='deleted' WHERE id=?", (p,)); store.db.commit()
        assert store.visual.count() == 1
        assert store.get(p) is None              # tombstoned: never surfaces
        stats = store.reconcile()
        assert stats["tombstones_finished"] == 1
        assert store.visual.count() == 0
        assert store.count() == 0
        store.close()
    print("PASS: reconcile finishes an interrupted delete (tombstone)")


def test_reconcile_purges_orphan_vectors():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png"); _grad(p)
        store = Store(os.path.join(d, "data"))
        sif = process(p)
        store.insert(sif)
        # Inject a vector with no SQLite parent.
        store.visual.upsert(ids=["ORPHAN"], embeddings=[sif.embeddings.visual],
                            metadatas=[{"path": "/ghost.png", "sif_id": "ORPHAN"}])
        assert store.visual.count() == 2
        stats = store.reconcile()
        assert stats["orphans_purged"] >= 1
        assert store.visual.count() == 1
        store.close()
    print("PASS: reconcile purges orphan vectors")


# --------------------------------------------------------------------------
# Three-tier dedup
# --------------------------------------------------------------------------
def test_dedup_tiers():
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "data"))

        a = os.path.join(d, "a.png"); _grad(a, shift=0, compress_level=0)
        assert ingest(store, a).status == "indexed"
        # same path, unchanged bytes -> idempotent
        assert ingest(store, a).status == "unchanged"

        # tier 1: exact byte copy at a different path
        b = os.path.join(d, "b.png"); shutil.copyfile(a, b)
        r = ingest(store, b)
        assert r.status == "duplicate" and r.detail.startswith("sha256")

        # tier 2: identical pixels, different bytes (different PNG compression)
        c = os.path.join(d, "c.png"); _grad(c, shift=0, compress_level=9)
        r = ingest(store, c)
        assert r.status == "duplicate" and r.detail.startswith("pixel")

        # tier 3: near-duplicate (small corner patch -> pixels & bytes differ,
        # perceptual hash stays within threshold)
        e = os.path.join(d, "e.png"); _grad(e, shift=0, compress_level=0, patch=6)
        r = ingest(store, e)
        assert r.status == "duplicate" and r.detail.startswith("phash")

        assert store.count() == 1  # only the original was actually stored
        store.close()
    print("PASS: three-tier dedup (sha256 / pixel / phash) skips duplicates")


def test_changed_content_updates():
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "data"))
        p = os.path.join(d, "a.png")
        _grad(p, shift=0)
        assert ingest(store, p).status == "indexed"
        # genuinely different image at the same path -> update, not duplicate
        _grad(p, shift=120, patch=20)
        assert ingest(store, p).status == "updated"
        assert store.count() == 1
        assert store.get_meta(p)["indexed"] == 1
        store.close()
    print("PASS: changed content at a known path triggers an update")


if __name__ == "__main__":
    test_wal_and_busy_timeout()
    test_insert_is_outbox_ordered()
    test_crash_recovery_replays_unindexed_rows()
    test_update_dark_window_is_recoverable()
    test_delete_removes_row_and_vectors()
    test_reconcile_finishes_tombstone()
    test_reconcile_purges_orphan_vectors()
    test_dedup_tiers()
    test_changed_content_updates()
    print("\nAll Stage 2 tests passed.")
