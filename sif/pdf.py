"""
Multi-page PDF ingestion (Stage 5).

Produces a HIERARCHICAL SIF: a parent doc -> child pages[] -> sub-page regions[].
Text vectors bind to pages; visual vectors bind to regions (they are NOT 1:1).
Retrieval fuses at the PAGE entity and returns a page (see query.py).

A page-classification router decides what work each page needs:
  * born_digital_text — real text layer, no images -> text only (no rendering)
  * mixed             — text + embedded images -> page text + region visuals
  * scanned           — little/no text, image(s) -> OCR the rendered page
  * figure            — image-dominant -> region visuals

Text + structure come from **pdfplumber** (pure-python). Page rendering for the
vision path uses **pypdfium2** (pip wheel, no system deps, Apache/BSD-licensed —
unlike Poppler/pdf2image or AGPL PyMuPDF). Rendering is OPTIONAL: without it,
born-digital text still indexes fully and image pages degrade to text-only.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

from .schema import SIF, Page, Region, new_sif
from . import dedup, extractors, clip_embed
from .embedding import embed, active_model

DPI = int(os.environ.get("SIF_PDF_DPI", "150"))
MAX_REGIONS_PER_PAGE = int(os.environ.get("SIF_PDF_MAX_REGIONS", "8"))
_TEXT_MIN = 40  # chars of real text below which a page is treated as image-based


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def deps_available() -> bool:
    return importlib.util.find_spec("pdfplumber") is not None


def renderer_available() -> bool:
    return importlib.util.find_spec("pypdfium2") is not None


def classify_page(text_len: int, n_images: int) -> str:
    if text_len >= _TEXT_MIN:
        return "mixed" if n_images >= 1 else "born_digital_text"
    if n_images >= 2:
        return "figure"
    if n_images == 1:
        return "scanned"
    return "born_digital_text"  # empty/sparse page


def _render_page(path: str, index: int):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[index]
        pil = page.render(scale=DPI / 72).to_pil().convert("RGB")
        page.close()
        return pil
    finally:
        pdf.close()


def _crop(pil, img_meta, page_height_pt: float):
    """Crop a pdfplumber image bbox (PDF points, top-down) out of the render."""
    s = DPI / 72.0
    x0 = max(0, int(img_meta["x0"] * s))
    x1 = min(pil.width, int(img_meta["x1"] * s))
    top = max(0, int(img_meta["top"] * s))
    bottom = min(pil.height, int(img_meta["bottom"] * s))
    if x1 <= x0 or bottom <= top:
        return None
    return pil.crop((x0, top, x1, bottom))


def _region_from_pil(pil, region_id: str, bbox=None) -> Region:
    """Run the visual extractors on a region image and build its visual vector."""
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil.save(tmp)
        objects = extractors.extract_objects(tmp)
        scene = extractors.extract_scene(tmp)
        clip_vec = clip_embed.embed_image(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    region = Region(region_id=region_id, bbox=bbox or [], objects=objects, scene=scene)
    parts = [scene.caption, *scene.tags, *(o.label for o in objects)]
    region.visual_input = " ".join(p for p in parts if p).strip()
    region.visual = embed(region.visual_input, kind="document") if region.visual_input else []
    region.clip = clip_vec
    return region


def _ocr_pil(pil) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil.save(tmp)
        return extractors.extract_ocr(tmp).full_text
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def process_pdf(path: str, file_hashes: dedup.Hashes | None = None) -> SIF:
    import pdfplumber

    sif = new_sif(path)
    sif.kind = "pdf"
    h = file_hashes if file_hashes is not None else dedup.hashes(path)
    sif.file.sha256 = h.sha256          # pixel/phash are empty for PDFs (sha dedup only)
    sif.file.size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    sif.file.format = "PDF"

    can_render = renderer_available()

    with pdfplumber.open(path) as pdf:
        sif.file.resolution = (0, len(pdf.pages))   # (—, page count)
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            images = page.images or []
            ptype = classify_page(len(text), len(images))
            prec = Page(page_index=i, page_type=ptype, text=text)

            rendered = None
            if can_render and ptype in ("scanned", "mixed", "figure"):
                try:
                    rendered = _render_page(path, i)
                except Exception:
                    rendered = None

            # Scanned page with no text layer -> OCR the whole render.
            if ptype == "scanned" and not text and rendered is not None:
                prec.text = _ocr_pil(rendered).strip()

            # Regions: one per embedded image, else the whole page for image-only.
            if rendered is not None and images:
                for j, im in enumerate(images[:MAX_REGIONS_PER_PAGE]):
                    crop = _crop(rendered, im, page.height)
                    if crop is None:
                        continue
                    prec.regions.append(_region_from_pil(
                        crop, f"{path}#p{i}#r{j}",
                        bbox=[im["x0"], im["top"], im["x1"], im["bottom"]]))
            elif rendered is not None and ptype in ("scanned", "figure"):
                prec.regions.append(_region_from_pil(rendered, f"{path}#p{i}#r0"))

            # Page text vector.
            if prec.text.strip():
                prec.text_input = prec.text
                prec.text_vector = embed(prec.text_input, kind="document")

            sif.pages.append(prec)

    sif.meta = {
        "stage": 5,
        "kind": "pdf",
        "pages": len(sif.pages),
        "page_types": [p.page_type for p in sif.pages],
        "models_used": extractors.backends_used(),
        "embedding": active_model(),
    }
    return sif
