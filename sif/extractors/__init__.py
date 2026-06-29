"""
Extractor dispatch facade.

The pipeline calls ``extract_objects`` / ``extract_faces`` / ``extract_scene``
/ ``extract_ocr`` from here without caring whether a real model or the Stage-0
stub answered. Selection is per-extractor and resolved live on each call:

  * ``SIF_USE_STUBS=1``  → force the deterministic stubs everywhere
    (CI, no-model environments, reproducible tests).
  * otherwise            → use the real model when its deps are importable
    (and, for the VLM, its daemon is up); else transparently fall back to stub.

Faces are a biometric-privacy surface and stay OFF unless
``SIF_ENABLE_FACES=1``.

Resolution is cheap (an ``importlib`` spec check); the expensive model load is
cached inside each real module. ``backends_used()`` reports which backend
actually answered for the most recent extraction, for the SIF ``meta`` block.
"""
from __future__ import annotations

import importlib
import logging
import os

from . import stub
from .stub import read_basic_file_info  # model-free; re-exported for the pipeline

log = logging.getLogger("sif.extractors")

# stage -> backend label that actually produced the most recent output
_last_backend: dict[str, str] = {}


def _force_stubs() -> bool:
    return os.environ.get("SIF_USE_STUBS") == "1"


def faces_enabled() -> bool:
    return os.environ.get("SIF_ENABLE_FACES") == "1"


def _real(module_name: str):
    """Return the sibling real-extractor module iff its deps are available."""
    mod = importlib.import_module(f".{module_name}", __package__)
    return mod if mod.is_available() else None


def _run(stage: str, module_name: str, image_path: str, stub_fn):
    """Try the real backend for ``stage``; fall back to the stub on any issue."""
    if not _force_stubs():
        try:
            mod = _real(module_name)
            if mod is not None:
                result = getattr(mod, stub_fn.__name__)(image_path)
                _last_backend[stage] = mod.LABEL
                return result
        except Exception as e:  # model present but failed → don't kill the run
            log.warning("%s: real backend failed (%s); using stub", stage, e)
    _last_backend[stage] = "stub"
    return stub_fn(image_path)


def extract_objects(image_path: str):
    return _run("objects", "objects", image_path, stub.extract_objects)


def extract_scene(image_path: str):
    return _run("scene", "scene", image_path, stub.extract_scene)


def extract_ocr(image_path: str):
    return _run("ocr", "ocr", image_path, stub.extract_ocr)


def extract_faces(image_path: str):
    # OFF by default: no biometric extraction unless explicitly enabled.
    if not faces_enabled():
        _last_backend["faces"] = "disabled"
        return []
    return _run("faces", "faces", image_path, stub.extract_faces)


def backends_used() -> dict[str, str]:
    """Backend label per stage from the most recent extraction pass."""
    return dict(_last_backend)
