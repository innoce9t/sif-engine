"""
Stage 3 tests: multi-vector retrieval, RRF fusion, and re-rank gating.

Fusion/gating logic is tested as pure functions (deterministic); the search
path is tested on the stub embedder. The neural re-rank itself is gated off
under SIF_USE_STUBS, so these never touch the network.

Run: python -m pytest tests/test_stage3.py -v
or:  python tests/test_stage3.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image

from sif import retrieval
from sif.store import Store
from sif.pipeline import process, build_text_input
from sif.embedding import embed
from sif.query import search


def _img(path, color=(40, 90, 160)):
    Image.new("RGB", (64, 64), color).save(path)


# --------------------------------------------------------------------------
# RRF fusion (pure)
# --------------------------------------------------------------------------
def test_rrf_orders_by_fused_rank():
    ranks = {"visual": {"a": 1, "b": 2}, "text": {"b": 1, "a": 3}}
    fused = retrieval.rrf_fuse(ranks, top_k=50)
    ids = [i for i, _ in fused]
    assert set(ids) == {"a", "b"}
    # b: 1/52 + 1/61 ; a: 1/61 + 1/63  -> b wins
    assert ids[0] == "b"


def test_rrf_absentee_prevents_single_modality_starvation():
    # "drone shot": perfect in visual (rank 1), absent from text.
    # "dual": mediocre in both (visual rank 2, text rank 50).
    ranks = {"visual": {"drone": 1, "dual": 2}, "text": {"dual": 50}}

    with_fix = retrieval.rrf_fuse(ranks, top_k=50, absentee=True)
    without_fix = retrieval.rrf_fuse(ranks, top_k=50, absentee=False)

    assert with_fix[0][0] == "drone", "absentee baseline should let the drone shot win"
    assert without_fix[0][0] == "dual", "without the fix the dual-modality asset wins"


# --------------------------------------------------------------------------
# Re-rank gating (pure)
# --------------------------------------------------------------------------
def test_relative_gap_and_gating():
    assert retrieval.relative_gap([0.9, 0.1]) > 0.5
    assert retrieval.relative_gap([0.50, 0.49]) < 0.05
    assert not retrieval.should_rerank([0.9, 0.1])      # clear -> no rerank
    assert retrieval.should_rerank([0.500, 0.495])      # ambiguous -> rerank
    assert not retrieval.should_rerank([0.9])           # single result


# --------------------------------------------------------------------------
# Multi-vector search through the store
# --------------------------------------------------------------------------
def test_search_roundtrip_visual():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "scene.jpg"); _img(img)
        store = Store(os.path.join(d, "data"))
        sif = process(img)
        store.insert(sif)
        res = search(store, sif.scene.caption, limit=5)
        assert res and res[0]["path"] == img
        assert "score" in res[0] and res[0]["reranked"] is False
        store.close()
    print("PASS: visual round-trip via RRF")


def test_text_vector_is_searchable():
    # Stage 3 queries BOTH collections, so OCR text becomes retrievable even
    # when the visual side doesn't match. Stub OCR is empty, so set it directly.
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "doc.jpg"); _img(img)
        store = Store(os.path.join(d, "data"))
        sif = process(img)
        sif.ocr.full_text = "madrid metro timetable"
        sif.ocr.has_text = True
        sif.embeddings.text_input = build_text_input(sif)
        sif.embeddings.text = embed(sif.embeddings.text_input, kind="document")
        store.insert(sif)
        assert store.text.count() == 1            # text vector now exists
        res = search(store, "madrid", limit=5)
        assert any(r["id"] == sif.file.path for r in res), "OCR text should be retrievable"
        store.close()
    print("PASS: text (OCR) vector is searchable via multi-vector retrieval")


def test_search_excludes_orphan_vectors():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "s.jpg"); _img(img)
        store = Store(os.path.join(d, "data"))
        sif = process(img)
        store.insert(sif)
        store.visual.upsert(ids=["ORPHAN"], embeddings=[sif.embeddings.visual],
                            metadatas=[{"path": "/ghost.jpg", "sif_id": "ORPHAN"}])
        res = search(store, sif.scene.caption, limit=10)
        assert "ORPHAN" not in [r["id"] for r in res]
        store.close()
    print("PASS: orphan vectors still excluded after fusion")


if __name__ == "__main__":
    test_rrf_orders_by_fused_rank()
    test_rrf_absentee_prevents_single_modality_starvation()
    test_relative_gap_and_gating()
    test_search_roundtrip_visual()
    test_text_vector_is_searchable()
    test_search_excludes_orphan_vectors()
    print("\nAll Stage 3 tests passed.")
