"""
Stage 1 tests: real-model wiring + the stub-fallback guarantee.

The heavy models aren't present in CI, so these tests focus on the contract
that must hold regardless: the pipeline runs end-to-end via the stub fallback,
the backend selection is reported honestly, faces stay off by default, and the
embedder's dimension/empty-input behavior is correct. The real-model output
shape is exercised on a machine where the models are installed (the fallback
machinery means the same code path produces real SIFs there).

Run: python -m pytest tests/ -v   (from sif-engine/)
or:  python tests/test_stage1.py
"""
import contextlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from sif import embedding, extractors
from sif.pipeline import process
from sif.store import Store
from sif.query import search


@contextlib.contextmanager
def env(**kw):
    """Temporarily set/clear env vars and reset the cached embedder decision."""
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    embedding.reset()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        embedding.reset()


def _make_image(path, color=(120, 90, 200)):
    Image.new("RGB", (320, 240), color).save(path)


def test_stub_fallback_produces_populated_sif():
    # With models forced off, the pipeline must still produce a full SIF.
    with env(SIF_USE_STUBS="1", SIF_ENABLE_FACES=None):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jpg")
            _make_image(p)
            sif = process(p)
            assert sif.file.sha256.startswith("sha256:")
            assert sif.file.resolution == (320, 240)   # populated via PIL in Stage 1
            assert sif.file.format == "JPEG"
            assert sif.scene.caption != ""
            assert any(sif.embeddings.visual)
            assert sif.embeddings.model == "stub-hash-embed-stage0"
            assert sif.meta["stage"] == 1
            backends = sif.meta["models_used"]
            assert backends["objects"] == "stub"
            assert backends["scene"] == "stub"
            assert backends["ocr"] == "stub"
    print("PASS: stub fallback produces a populated Stage 1 SIF")


def test_faces_off_by_default():
    with env(SIF_USE_STUBS="1", SIF_ENABLE_FACES=None):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jpg")
            _make_image(p)
            sif = process(p)
            assert sif.faces == []
            assert extractors.backends_used()["faces"] == "disabled"
    print("PASS: faces are off by default (biometric privacy)")


def test_embed_empty_returns_zero_vector():
    with env(SIF_USE_STUBS="1"):
        assert not any(embedding.embed(""))
        assert not any(embedding.embed("   "))
        # non-empty produces a populated vector of the active dimension
        v = embedding.embed("a cat on a mat")
        assert any(v)
        assert len(v) == embedding.active_dim()
    print("PASS: empty text -> zero vector (no spurious text vectors)")


def test_embed_query_and_document_same_dim():
    with env(SIF_USE_STUBS="1"):
        d = embedding.embed("red car", kind="document")
        q = embedding.embed("red car", kind="query")
        assert len(d) == len(q) == embedding.active_dim()
    print("PASS: query and document embeddings share a dimension")


def test_index_search_roundtrip_stage1():
    with env(SIF_USE_STUBS="1"):
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "data")
            img = os.path.join(d, "scene.jpg")
            _make_image(img, (30, 180, 30))

            store = Store(data)
            sif = process(img)
            store.upsert(sif)

            results = search(store, sif.scene.caption, limit=5)
            assert len(results) >= 1
            assert results[0]["path"] == img
            assert sif.meta["stage"] == 1
            store.close()
    print("PASS: Stage 1 index + search round-trip")


if __name__ == "__main__":
    test_stub_fallback_produces_populated_sif()
    test_faces_off_by_default()
    test_embed_empty_returns_zero_vector()
    test_embed_query_and_document_same_dim()
    test_index_search_roundtrip_stage1()
    print("\nAll Stage 1 tests passed.")
