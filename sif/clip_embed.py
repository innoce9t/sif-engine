"""
CLIP visual embedding — pixels and text in ONE shared space.

The rest of the engine is "describe-then-embed": it turns an image into text
(caption/OCR/labels) and embeds that text. CLIP is complementary — it embeds the
raw **pixels** directly into a space shared with a text encoder, so a text query
can match visual content the caption never mentioned. This becomes a third
vector space, fused with the description and OCR spaces at query time.

Uses sentence-transformers' CLIP (``clip-ViT-B-32``, 512-d) — no new dependency.
Optional and lazy, like every other model: skipped under SIF_USE_STUBS and when
sentence-transformers isn't installed, so the engine still runs without it.
"""
from __future__ import annotations

import importlib.util
import logging
import os

log = logging.getLogger("sif.clip")

MODEL = os.environ.get("SIF_CLIP_MODEL", "clip-ViT-B-32")
DIM = 512

_decided: bool | None = None
_model = None


def _force_stubs() -> bool:
    return os.environ.get("SIF_USE_STUBS") == "1"


def _available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _decide() -> bool:
    global _decided, _model
    if _decided is not None:
        return _decided
    if _force_stubs() or not _available():
        _decided = False
        return False
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL)
        _decided = True
    except Exception as e:
        log.warning("clip: unavailable (%s); visual embedding disabled", e)
        _decided = False
    return _decided


def active() -> bool:
    return _decide()


def reset() -> None:
    global _decided, _model
    _decided = None
    _model = None


def embed_text(query: str) -> list[float]:
    """CLIP text-side embedding of a query (for searching the image space)."""
    if not _decide() or not query:
        return []
    v = _model.encode(query, normalize_embeddings=True)
    return [float(x) for x in v]


def embed_image(path: str) -> list[float]:
    """CLIP image embedding from raw pixels. [] if unavailable/unreadable."""
    if not _decide():
        return []
    try:
        from PIL import Image
        with Image.open(path) as im:
            v = _model.encode(im.convert("RGB"), normalize_embeddings=True)
        return [float(x) for x in v]
    except Exception as e:
        log.warning("clip: image encode failed (%s)", e)
        return []
