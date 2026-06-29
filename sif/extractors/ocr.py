"""
PaddleOCR text extraction (Stage 1).

Real implementation behind the SAME interface as ``stub.extract_ocr``.
PaddleOCR is fully local and multilingual (incl. Arabic for the MENA market).
Language via ``SIF_OCR_LANG`` (default ``en``).

The extracted text feeds ONLY the TEXT vector (never the visual vector) — that
separation is enforced upstream in ``pipeline.build_visual_input`` and is the
anti-dilution guarantee from the design review.
"""
from __future__ import annotations

import importlib.util
import os

from ..schema import OCRResult

LABEL = "paddleocr"

_engine = None


def is_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def _get_engine():
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR

        kwargs = {"lang": os.environ.get("SIF_OCR_LANG", "en")}
        # MKL-DNN triggers an "Unimplemented ... onednn_instruction" crash on
        # some CPUs with PaddleOCR 3.x, so it's disabled by default. Re-enable
        # (faster where it works) with SIF_OCR_MKLDNN=1.
        if os.environ.get("SIF_OCR_MKLDNN") != "1":
            kwargs["enable_mkldnn"] = False
        try:
            _engine = PaddleOCR(**kwargs)
        except TypeError:
            kwargs.pop("enable_mkldnn", None)  # older builds lack the kwarg
            _engine = PaddleOCR(**kwargs)
    return _engine


def _texts_from_result(result) -> list[str]:
    """Flatten PaddleOCR output to text blocks, across 3.x and 2.x formats."""
    blocks: list[str] = []
    for page in (result or []):
        # PaddleOCR 3.x: each page is an OCRResult dict with a 'rec_texts' list.
        if hasattr(page, "get"):
            rec = page.get("rec_texts")
            if rec is not None:
                blocks.extend(str(t) for t in rec if t)
                continue
        # PaddleOCR 2.x: each page is [ [bbox, (text, conf)], ... ].
        for line in (page or []):
            try:
                text = line[1][0]
            except (IndexError, TypeError, KeyError):
                continue
            if text:
                blocks.append(str(text))
    return blocks


def extract_ocr(image_path: str) -> OCRResult:
    engine = _get_engine()
    try:
        result = engine.ocr(image_path)
    except TypeError:
        result = engine.ocr(image_path, cls=True)

    blocks = _texts_from_result(result)
    full = "\n".join(blocks)
    return OCRResult(full_text=full, text_blocks=blocks, has_text=bool(full.strip()))
