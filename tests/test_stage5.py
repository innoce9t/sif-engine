"""
Stage 5 tests: multi-page PDF ingestion + page-level retrieval.

Born-digital text PDFs (the core document use case) need no rendering, so these
run on the stub path. They're skipped when the PDF deps (pdfplumber to read,
fpdf2 to author the fixture) aren't installed.

Run (from the Stage-5 venv): python -m pytest tests/test_stage5.py -v
"""
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SIF_USE_STUBS"] = "1"

_HAVE = (importlib.util.find_spec("pdfplumber") is not None
         and importlib.util.find_spec("fpdf") is not None)

if not _HAVE:
    import pytest
    pytestmark = pytest.mark.skip(reason="pdf deps not installed (pdfplumber, fpdf2)")
else:
    from sif import pdf
    from sif.store import Store
    from sif.ingest import ingest
    from sif.query import search


def _make_pdf(path, page_texts):
    from fpdf import FPDF
    doc = FPDF()
    for t in page_texts:
        doc.add_page()
        doc.set_font("helvetica", size=14)
        doc.multi_cell(0, 10, t)
    doc.output(path)


def test_classify_page():
    assert pdf.classify_page(200, 0) == "born_digital_text"
    assert pdf.classify_page(200, 1) == "mixed"
    assert pdf.classify_page(5, 1) == "scanned"
    assert pdf.classify_page(5, 3) == "figure"
    print("PASS: page classification router")


def test_pdf_ingest_is_hierarchical():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.pdf")
        _make_pdf(p, [
            "Quarterly revenue report for the northern region and logistics.",
            "Appendix covering warehouse safety procedures and compliance.",
        ])
        sif = pdf.process_pdf(p)
        assert sif.kind == "pdf"
        assert len(sif.pages) == 2
        assert all(pg.page_type == "born_digital_text" for pg in sif.pages)
        assert "revenue" in sif.pages[0].text.lower()
        assert any(sif.pages[0].text_vector)        # page text vector populated
        assert sif.meta["stage"] == 5
    print("PASS: PDF ingests into a hierarchical SIF with page text vectors")


def test_pdf_page_level_retrieval():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.pdf")
        _make_pdf(p, [
            "Quarterly revenue report for the northern region and logistics.",
            "Appendix covering warehouse safety procedures and compliance.",
        ])
        store = Store(os.path.join(d, "data"))
        assert ingest(store, p).status == "indexed"
        assert store.count() == 1                   # one doc entity
        assert store.text.count() == 2              # one text vector per page

        # a query that matches page 2 should return page 2 of this doc
        res = search(store, "warehouse safety compliance", limit=3)
        assert res and res[0]["path"] == p
        assert res[0]["page"] == 1
        store.close()
    print("PASS: PDF search returns the matching PAGE (entity aggregation)")


def test_pdf_delete_purges_all_page_vectors():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.pdf")
        _make_pdf(p, ["Page one about budgets.", "Page two about hiring plans."])
        store = Store(os.path.join(d, "data"))
        ingest(store, p)
        assert store.text.count() == 2
        store.delete(p)
        assert store.count() == 0
        assert store.text.count() == 0              # all page vectors purged
        store.close()
    print("PASS: deleting a PDF purges every page/region vector")


def test_pdf_mixed_page_regions():
    # Exercises the rendering + region path (pypdfium2 crop -> region visual
    # vector). Needs the renderer; skipped otherwise.
    if not pdf.renderer_available():
        print("SKIP: pypdfium2 not installed")
        return
    from fpdf import FPDF
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        imgp = os.path.join(d, "pic.png")
        Image.new("RGB", (200, 150), (40, 120, 200)).save(imgp)
        p = os.path.join(d, "mixed.pdf")
        doc = FPDF()
        doc.add_page()
        doc.set_font("helvetica", size=14)
        doc.multi_cell(0, 10, "Figure 1 shows the proposed system architecture in detail.")
        doc.image(imgp, x=10, y=60, w=80)
        doc.output(p)

        store = Store(os.path.join(d, "data"))
        assert ingest(store, p).status == "indexed"
        # a region visual vector exists and is searchable back to the page
        assert store.visual.count() >= 1
        res = search(store, "system architecture diagram", limit=3)
        assert res and res[0]["path"] == p and res[0]["page"] == 0
        store.close()
    print("PASS: mixed PDF page renders regions with visual vectors")


def test_pdf_crash_recovery_replays_pages():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.pdf")
        _make_pdf(p, ["Alpha section.", "Beta section.", "Gamma section."])
        data = os.path.join(d, "data")
        store = Store(data)
        ingest(store, p)
        # simulate crash: row indexed=0, vectors gone
        store.db.execute("UPDATE sif SET indexed=0 WHERE id=?", (p,)); store.db.commit()
        store._purge_vectors(p)
        assert store.text.count() == 0
        store.close()

        store2 = Store(data)                         # recovery replays page vectors
        assert store2.text.count() == 3
        store2.close()
    print("PASS: PDF crash recovery replays page vectors from the stored SIF")


if __name__ == "__main__":
    if not _HAVE:
        print("SKIP: pdf deps not installed (pdfplumber, fpdf2)")
    else:
        test_classify_page()
        test_pdf_ingest_is_hierarchical()
        test_pdf_page_level_retrieval()
        test_pdf_delete_purges_all_page_vectors()
        test_pdf_crash_recovery_replays_pages()
        print("\nAll Stage 5 tests passed.")
