"""
Three-tier deduplication hashing (Stage 2).

Each tier catches a different class of "same image":

  1. **sha256**     — raw file bytes. Exact duplicate.
  2. **pixel_hash** — sha256 of the decoded RGB pixels. Catches metadata-only
                      edits (EXIF stripped, re-saved losslessly) where the bytes
                      differ but the picture is identical.
  3. **phash**      — a perceptual hash (dHash). Catches near-duplicates
                      (re-compression, light crop/resize, small tweaks) via a
                      Hamming-distance threshold rather than exact equality.

dHash is used for the perceptual tier: it's robust to scaling and minor edits
and needs no DCT/scipy — just PIL. See ADR 0002.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

from PIL import Image

# Near-duplicate cutoff: dHash is 64 bits, so distances run 0..64. <=10 reliably
# groups re-compressions / small edits without colliding distinct images.
DEFAULT_PHASH_THRESHOLD = 10


class Hashes(NamedTuple):
    sha256: str
    pixel_hash: str
    phash: str


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _dhash_hex(gray_9x8: Image.Image) -> str:
    """Difference hash of a 9x8 grayscale image -> 16-hex (64-bit) string."""
    px = gray_9x8.tobytes()  # row-major, width=9, one byte per pixel ('L' mode)
    bits = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return format(bits, "016x")


def hashes(path: str) -> Hashes:
    """All three hashes for a file. sha256 always; pixel/phash best-effort
    (empty strings if the file can't be decoded as an image)."""
    sha = sha256_file(path)
    pixel_hash = ""
    phash = ""
    try:
        with Image.open(path) as im:
            pixel_hash = "px:" + hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
            phash = _dhash_hex(im.convert("L").resize((9, 8), Image.LANCZOS))
    except Exception:
        pass  # non-image / unreadable: tiers 2 and 3 simply don't apply
    return Hashes(sha, pixel_hash, phash)


def hamming(a_hex: str, b_hex: str) -> int:
    """Bit distance between two hex phashes. Max distance if either is empty."""
    if not a_hex or not b_hex:
        return 64
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")


def duplicate_groups(paths: list[str],
                     threshold: int = DEFAULT_PHASH_THRESHOLD) -> list[list[str]]:
    """Group files on disk into duplicate/near-duplicate clusters via the three
    tiers (exact bytes / identical pixels / perceptual). Returns groups of >1
    path. Useful for cleaning a folder BEFORE indexing (the index itself dedups)."""
    from collections import defaultdict
    hs = [(p, hashes(p)) for p in paths]
    used: set[str] = set()
    groups: list[list[str]] = []

    by_key: dict[str, list[str]] = defaultdict(list)
    for p, h in hs:
        by_key[h.pixel_hash or h.sha256].append(p)   # tiers 1-2
    for members in by_key.values():
        if len(members) > 1:
            groups.append(list(members))
            used.update(members)

    rest = [(p, h) for p, h in hs if p not in used and h.phash]   # tier 3
    for i, (p, h) in enumerate(rest):
        if p in used:
            continue
        grp = [p]
        for q, h2 in rest[i + 1:]:
            if q not in used and hamming(h.phash, h2.phash) <= threshold:
                grp.append(q)
                used.add(q)
        if len(grp) > 1:
            used.add(p)
            groups.append(grp)
    return groups
