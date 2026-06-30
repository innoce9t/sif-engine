"""
Stage 6 tests: CLI integration + ergonomics.

Drives the CLI end-to-end (the same entry users hit) on the stub path, covering
the index -> search round-trip, idempotent re-index (the basis of watch mode),
and the configurable re-rank gate.

Run: python -m pytest tests/test_stage6.py -v
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SIF_USE_STUBS"] = "1"

from PIL import Image

from sif import cli, retrieval


def _img(path, seed):
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", (48, 48))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                for _ in range(48 * 48)])
    im.save(path)


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(argv)
    return buf.getvalue()


def test_cli_index_search_and_idempotent_reindex():
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        imgs = os.path.join(d, "imgs"); os.makedirs(imgs)
        for i in range(3):
            _img(os.path.join(imgs, f"{i}.png"), i)

        out = _run(["--data", data, "index", imgs])
        assert "indexed=3" in out

        # re-index: dedup makes it idempotent (this is what watch relies on)
        out2 = _run(["--data", data, "index", imgs])
        assert "unchanged=3" in out2

        out3 = _run(["--data", data, "search", "anything", "--limit", "2"])
        assert "Results for" in out3
    print("PASS: CLI index/search + idempotent re-index")


def test_cli_version():
    out = _run(["version"])
    assert "SIF Engine" in out
    print("PASS: version command")


def test_rerank_gap_is_configurable():
    # default conservative gate doesn't fire on a 10% gap...
    os.environ.pop("SIF_RERANK_GAP", None)
    assert not retrieval.should_rerank([0.50, 0.45])
    # ...but a higher configured gate does
    os.environ["SIF_RERANK_GAP"] = "0.30"
    try:
        assert retrieval.should_rerank([0.50, 0.45])
    finally:
        os.environ.pop("SIF_RERANK_GAP", None)
    print("PASS: re-rank gate honors SIF_RERANK_GAP")


if __name__ == "__main__":
    test_cli_index_search_and_idempotent_reindex()
    test_cli_version()
    test_rerank_gap_is_configurable()
    print("\nAll Stage 6 tests passed.")
