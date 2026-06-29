"""
Stage 0 smoke test: prove the walking skeleton round-trips.

Run: python -m pytest tests/ -v   (from sif-engine/)
or:  python tests/test_stage0.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These tests validate the Stage 0 stub skeleton and assert stub-specific output
# (deterministic objects, the stub embedder's model name). Force stubs so they
# stay hermetic even when real Stage 1 models are installed in the environment.
os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image
from sif.store import Store
from sif.pipeline import process, build_visual_input, build_text_input
from sif.query import search


def _make_image(path, color):
    Image.new("RGB", (320, 240), color).save(path)


def test_pipeline_produces_populated_sif():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jpg")
        _make_image(p, (200, 30, 30))
        sif = process(p)
        assert sif.file.sha256.startswith("sha256:")
        assert sif.file.size_bytes > 0
        assert sif.scene.caption != ""
        assert len(sif.objects) >= 1
        assert any(sif.embeddings.visual)        # visual vector populated
        assert sif.embeddings.model == "stub-hash-embed-stage0"
    print("PASS: pipeline produces populated SIF")


def test_visual_input_excludes_ocr():
    # The v2 dilution bug: OCR text must NOT be in the visual input string.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jpg")
        _make_image(p, (30, 30, 200))
        sif = process(p)
        sif.ocr.full_text = "SOME LONG DOCUMENT TEXT THAT SHOULD NOT DILUTE VISUAL"
        vis = build_visual_input(sif)
        assert "DOCUMENT" not in vis.upper()
        txt = build_text_input(sif)
        assert "DOCUMENT" in txt.upper()
    print("PASS: visual input excludes OCR text (no dilution)")


def test_index_and_search_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        img = os.path.join(d, "scene.jpg")
        _make_image(img, (30, 180, 30))

        store = Store(data)
        sif = process(img)
        store.upsert(sif)

        # search using the asset's own caption should return it
        results = search(store, sif.scene.caption, limit=5)
        assert len(results) >= 1
        assert results[0]["path"] == img
        store.close()
    print("PASS: index + search round-trip returns the indexed asset")


def test_search_validates_against_sqlite():
    # A vector with no SQLite row must never appear in results.
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        img = os.path.join(d, "scene.jpg")
        _make_image(img, (90, 90, 90))

        store = Store(data)
        sif = process(img)
        store.upsert(sif)

        # Inject an orphan vector directly into Chroma with no SQLite row
        store.visual.upsert(ids=["ORPHAN"], embeddings=[sif.embeddings.visual],
                            metadatas=[{"path": "/ghost.jpg", "sif_id": "ORPHAN"}])

        results = search(store, sif.scene.caption, limit=10)
        ids = [r["id"] for r in results]
        assert "ORPHAN" not in ids, "orphan vector leaked into results!"
        store.close()
    print("PASS: orphan vector excluded (SQLite validation works)")


if __name__ == "__main__":
    test_pipeline_produces_populated_sif()
    test_visual_input_excludes_ocr()
    test_index_and_search_roundtrip()
    test_search_validates_against_sqlite()
    print("\nAll Stage 0 tests passed.")
