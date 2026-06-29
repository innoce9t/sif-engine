"""
YOLOv10n object detection (Stage 1).

Real implementation of the object detector behind the SAME interface as
``stub.extract_objects``. Uses Ultralytics, which runs YOLOv10n on the ONNX
Runtime / Torch backend. Weights download automatically on first use; override
the path or variant with ``SIF_YOLO_WEIGHTS`` (default ``yolov10n.pt``).

Heavy imports are deferred into the functions so importing this module is cheap
and never fails when Ultralytics isn't installed — ``is_available()`` lets the
dispatch facade decide whether to use this or fall back to the stub.
"""
from __future__ import annotations

import importlib.util
import os

from ..schema import DetectedObject

LABEL = "yolov10n"

_model = None  # cached across calls so weights load once per process


def is_available() -> bool:
    """True when the Ultralytics package is importable."""
    return importlib.util.find_spec("ultralytics") is not None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(os.environ.get("SIF_YOLO_WEIGHTS", "yolov10n.pt"))
    return _model


def extract_objects(image_path: str) -> list[DetectedObject]:
    """Detect objects, returning the same DetectedObject list shape as the stub."""
    model = _get_model()
    out: list[DetectedObject] = []
    for r in model(image_path, verbose=False):
        names = r.names
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(
                DetectedObject(
                    label=str(names[cls]),
                    confidence=round(conf, 4),
                    bbox=[x1, y1, x2, y2],
                )
            )
    return out
