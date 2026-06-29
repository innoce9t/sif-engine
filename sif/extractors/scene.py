"""
Moondream2 scene captioning via Ollama (Stage 1).

Real implementation behind the SAME interface as ``stub.extract_scene``.
Moondream2 is a small (~1.8B) edge VLM served by a local Ollama daemon, which
keeps raw pixels on-device (the privacy promise of the engine).

Requires BOTH the ``ollama`` Python client AND a running daemon serving the
model (default ``moondream``, override with ``SIF_VLM_MODEL``). ``is_available``
pings the daemon so a missing/stopped Ollama cleanly falls back to the stub.
"""
from __future__ import annotations

import importlib.util
import os
import re

from ..schema import Scene

LABEL = "moondream2"

# A plain caption request. We learned empirically that Moondream2 (a tiny VLM)
# returns an EMPTY response when handed a rigid "Caption:/Tags:" format prompt,
# but answers reliably to a simple instruction. So we ask for one sentence and
# derive tags ourselves. _parse still honors a structured response if a larger
# model (via SIF_VLM_MODEL) chooses to emit one.
_PROMPT = "Describe this image in one detailed sentence."

# Minimal stopword set so derived tags are content words, not glue words.
_STOPWORDS = frozenset(
    "a an the of on in at by to is are was were be been being and or but with "
    "for from as it its this that these those there their they them he she his "
    "her you your we our i over under near next two one some".split()
)


def is_available() -> bool:
    if importlib.util.find_spec("ollama") is None:
        return False
    try:
        import ollama

        ollama.list()  # ping the daemon; raises if it isn't running
        return True
    except Exception:
        return False


def _derive_tags(caption: str, limit: int = 8) -> list[str]:
    """Content-word keywords from a free-form caption (dedup, order-preserving)."""
    tags: list[str] = []
    for w in re.findall(r"[A-Za-z]{3,}", caption.lower()):
        if w not in _STOPWORDS and w not in tags:
            tags.append(w)
    return tags[:limit]


def _parse(text: str) -> Scene:
    """Turn a VLM response into a Scene. Free-form by default; honors a
    structured 'Caption:/Tags:' response if a model chooses to emit one."""
    text = text.strip()
    caption, tags = "", []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("caption:"):
            caption = line.split(":", 1)[1].strip()
        elif low.startswith("tags:"):
            tags = [t.strip() for t in re.split(r"[,;]", line.split(":", 1)[1]) if t.strip()]

    if not caption:
        caption = next((l.strip() for l in text.splitlines() if l.strip()), "")[:300]
    if not tags:
        tags = _derive_tags(caption)
    return Scene(caption=caption, tags=tags)


def extract_scene(image_path: str) -> Scene:
    import ollama

    model = os.environ.get("SIF_VLM_MODEL", "moondream")
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _PROMPT, "images": [image_path]}],
    )
    return _parse(resp["message"]["content"])
