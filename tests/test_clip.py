"""
CLIP visual-embedding tests (fallback contract).

The real CLIP model is a ~600MB download, so CI runs the fallback path: under
SIF_USE_STUBS the CLIP space is skipped everywhere and the engine behaves
exactly as before (two vector spaces). The real 3-way fusion is validated
manually with the model installed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image

from sif import clip_embed
from sif.store import Store
from sif.ingest import ingest
from sif.query import search


def _img(path, seed):
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", (48, 48))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                for _ in range(48 * 48)])
    im.save(path)


def test_clip_disabled_under_stubs():
    clip_embed.reset()
    assert clip_embed.active() is False
    assert clip_embed.embed_text("anything") == []
    assert clip_embed.embed_image("nope.png") == []
    print("PASS: CLIP is skipped under stubs (fallback contract)")


def test_index_and_search_without_clip():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "a.png"); _img(img, 1)
        store = Store(os.path.join(d, "data"))
        assert ingest(store, img).status == "indexed"
        assert store.clip.count() == 0          # no CLIP vectors under stubs
        res = search(store, "anything", limit=3)
        assert res and res[0]["path"] == img    # search still works (2 spaces)
        store.close()
    print("PASS: index + search work with the CLIP space empty")


if __name__ == "__main__":
    test_clip_disabled_under_stubs()
    test_index_and_search_without_clip()
    print("\nAll CLIP fallback tests passed.")
