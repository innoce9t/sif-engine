"""
InsightFace buffalo_l face embeddings (Stage 1) — OPTIONAL, OFF BY DEFAULT.

Real implementation behind the SAME interface as ``stub.extract_faces``.
Produces 512-d face embeddings with millisecond CPU inference.

Faces are a biometric-privacy surface (GDPR, BIPA, UAE PDPL), so this extractor
is DISABLED unless ``SIF_ENABLE_FACES=1`` — the gate lives in the dispatch
facade, which simply returns no faces when the flag is unset. Model variant via
``SIF_FACE_MODEL`` (default ``buffalo_l``).
"""
from __future__ import annotations

import importlib.util
import os
import threading

from ..schema import Face

LABEL = "insightface-buffalo_l"

_app = None
_lock = threading.Lock()  # Stage 4: extraction runs in parallel; load once


def is_available() -> bool:
    return importlib.util.find_spec("insightface") is not None


def _get_app():
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name=os.environ.get("SIF_FACE_MODEL", "buffalo_l"))
                app.prepare(ctx_id=-1)  # -1 = CPU
                globals()["_app"] = app
    return _app


def extract_faces(image_path: str) -> list[Face]:
    import numpy as np
    from PIL import Image

    app = _get_app()
    # InsightFace expects BGR (OpenCV convention); convert from PIL RGB.
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    bgr = rgb[:, :, ::-1]

    # Serialize inference across extraction threads (see Stage 4 findings).
    with _lock:
        faces = app.get(bgr)

    out: list[Face] = []
    for i, f in enumerate(faces):
        out.append(
            Face(
                face_id=f"{os.path.basename(image_path)}#{i}",
                embedding=[float(v) for v in f.normed_embedding],
                bbox=[float(v) for v in f.bbox],
                confidence=float(getattr(f, "det_score", 0.0)),
            )
        )
    return out
