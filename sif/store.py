"""
Storage layer (Stage 2): SQLite as the crash-safe source of truth, ChromaDB as
a derived, rebuildable index.

Durability model (the design-review fixes, now in code):

* **Outbox ordering** — a write lands the SQLite row as ``indexed=0`` FIRST,
  then the vectors, then flips ``indexed=1``. A crash at any point leaves a
  recoverable ``indexed=0`` row; it never leaves a "live" row with no vectors.
* **Update dark-window fix** — an update flips ``indexed=0`` BEFORE purging the
  old vectors (mirror of the insert ordering), so a crash mid-update is also
  caught by recovery.
* **Recovery sweep** — on startup, every ``indexed=0`` active row is replayed
  from its stored SIF JSON (no models needed: the vectors live in the JSON).
* **Reconciliation** — finishes ``deleted`` tombstones and purges ChromaDB ids
  with no live SQLite parent.
* **WAL + busy_timeout + single writer** — all mutations go through this one
  object/connection. Stage 4 wraps these methods in a dedicated writer
  coroutine; the ordering guarantees here are what make that safe.

Identity is the file path (ADR 0002). Content hashes are dedup attributes.
"""
from __future__ import annotations

import json
import os
import sqlite3

import chromadb

from .schema import SIF
from . import dedup


class Store:
    def __init__(self, root: str = "./sif_data", *, recover: bool = True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, "sif.db")
        self._init_sqlite()
        self._init_chroma()
        if recover:
            self.recover()

    # -- SQLite (source of truth) -----------------------------------------
    def _init_sqlite(self):
        # check_same_thread=False: Stage 4 runs all writes through one dedicated
        # writer thread (and reads in a single-threaded pre-pass), so access is
        # already serialized — we just need the handle usable across threads.
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        # WAL + busy_timeout: concurrent readers don't block the writer, and a
        # contended write waits instead of failing fast. synchronous=NORMAL is
        # the WAL-recommended durability/throughput balance.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sif (
                id          TEXT PRIMARY KEY,   -- = file path (stable identity)
                path        TEXT NOT NULL,
                sha256      TEXT,
                pixel_hash  TEXT,
                phash       TEXT,
                sif_json    TEXT NOT NULL,
                indexed     INTEGER NOT NULL DEFAULT 0,
                state       TEXT NOT NULL DEFAULT 'active',
                updated_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_sha   ON sif(sha256)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_pixel ON sif(pixel_hash)")
        self.db.commit()

    # -- ChromaDB (derived index) -----------------------------------------
    def _init_chroma(self):
        self.chroma = chromadb.PersistentClient(path=os.path.join(self.root, "chroma"))
        self.visual = self.chroma.get_or_create_collection("visual")
        self.text = self.chroma.get_or_create_collection("text")
        self.clip = self.chroma.get_or_create_collection("clip")   # CLIP pixel vectors

    # -- vector helpers ---------------------------------------------------
    def _write_vectors_from_dict(self, sid: str, d: dict):
        """Write an asset's vectors from its SIF dict. Handles both an image
        (one visual + one text vector keyed by the doc id) and a PDF (page text
        vectors + region visual vectors keyed under the doc). Every vector
        carries metadata ``path=sid`` so purge/reconcile can find them."""
        if d.get("kind") == "pdf":
            for page in d.get("pages", []):
                pidx = page.get("page_index")
                meta = {"path": sid, "sif_id": sid, "page": pidx}
                tv = page.get("text_vector") or []
                if any(tv):
                    self.text.upsert(ids=[f"{sid}#p{pidx}"], embeddings=[list(tv)], metadatas=[meta])
                for region in page.get("regions", []):
                    rid = region["region_id"]
                    if any(region.get("visual") or []):
                        self.visual.upsert(ids=[rid], embeddings=[list(region["visual"])], metadatas=[meta])
                    if any(region.get("clip") or []):
                        self.clip.upsert(ids=[rid], embeddings=[list(region["clip"])], metadatas=[meta])
        else:
            emb = d.get("embeddings", {})
            meta = {"path": sid, "sif_id": sid}
            if any(emb.get("visual") or []):
                self.visual.upsert(ids=[sid], embeddings=[list(emb["visual"])], metadatas=[meta])
            if any(emb.get("text") or []):
                self.text.upsert(ids=[sid], embeddings=[list(emb["text"])], metadatas=[meta])
            if any(emb.get("clip") or []):
                self.clip.upsert(ids=[sid], embeddings=[list(emb["clip"])], metadatas=[meta])

    def _purge_vectors(self, sid: str):
        # Delete every vector belonging to this asset (image = 1 each; PDF = many
        # page/region vectors). where=path catches them all; ids=[sid] is a
        # belt-and-suspenders for the image case. Absent deletes are no-ops.
        for coll in (self.visual, self.text, self.clip):
            try:
                coll.delete(where={"path": sid})
            except Exception:
                pass
            try:
                coll.delete(ids=[sid])
            except Exception:
                pass

    # -- write lifecycle (outbox-ordered) ---------------------------------
    def insert(self, sif: SIF):
        """New asset. Row(indexed=0) -> vectors -> indexed=1."""
        sid = sif.file.path
        self.db.execute(
            "INSERT OR REPLACE INTO sif "
            "(id, path, sha256, pixel_hash, phash, sif_json, indexed, state, updated_at) "
            "VALUES (?,?,?,?,?,?,0,'active',strftime('%s','now'))",
            (sid, sif.file.path, sif.file.sha256, sif.file.pixel_hash,
             sif.file.phash, sif.to_json(indent=None)),
        )
        self.db.commit()
        self._write_vectors_from_dict(sid, sif.to_dict())
        self.db.execute("UPDATE sif SET indexed=1 WHERE id=?", (sid,))
        self.db.commit()

    def update(self, sif: SIF):
        """Existing asset, content changed. Flip indexed=0 FIRST (close the
        dark window), then purge old vectors, write new, flip indexed=1."""
        sid = sif.file.path
        self.db.execute(
            "UPDATE sif SET sha256=?, pixel_hash=?, phash=?, sif_json=?, "
            "indexed=0, state='active', updated_at=strftime('%s','now') WHERE id=?",
            (sif.file.sha256, sif.file.pixel_hash, sif.file.phash,
             sif.to_json(indent=None), sid),
        )
        self.db.commit()
        self._purge_vectors(sid)
        self._write_vectors_from_dict(sid, sif.to_dict())
        self.db.execute("UPDATE sif SET indexed=1 WHERE id=?", (sid,))
        self.db.commit()

    def upsert(self, sif: SIF):
        """Insert or update by path. Preserves the Stage 0/1 write interface."""
        if self.get_meta(sif.file.path) is not None:
            self.update(sif)
        else:
            self.insert(sif)

    def delete(self, path: str):
        """Tombstone -> purge vectors -> hard delete. A crash after the
        tombstone is finished by reconcile()."""
        self.db.execute("UPDATE sif SET state='deleted' WHERE id=?", (path,))
        self.db.commit()
        self._purge_vectors(path)
        self.db.execute("DELETE FROM sif WHERE id=?", (path,))
        self.db.commit()

    # -- recovery & reconciliation ----------------------------------------
    def recover(self) -> int:
        """Replay every active, not-yet-indexed row from its stored SIF JSON.
        Returns how many rows were recovered."""
        rows = self.db.execute(
            "SELECT id, path, sif_json FROM sif WHERE state='active' AND indexed=0"
        ).fetchall()
        for sid, path, sif_json in rows:
            self._purge_vectors(sid)  # idempotent: clear any partial write
            self._write_vectors_from_dict(sid, json.loads(sif_json))
            self.db.execute("UPDATE sif SET indexed=1 WHERE id=?", (sid,))
        self.db.commit()
        return len(rows)

    def reconcile(self) -> dict:
        """Finish deleted tombstones and purge orphan vectors (ChromaDB ids with
        no live SQLite parent). Returns counts for reporting."""
        # 1. finish any tombstones left by an interrupted delete
        tombs = [r[0] for r in self.db.execute(
            "SELECT id FROM sif WHERE state='deleted'").fetchall()]
        for sid in tombs:
            self._purge_vectors(sid)
            self.db.execute("DELETE FROM sif WHERE id=?", (sid,))
        self.db.commit()

        # 2. purge orphan vectors — a vector is an orphan when its parent doc
        # (metadata 'path', which works for PDF page/region ids too) is not a
        # live SQLite row. Fall back to the id itself for legacy vectors.
        active = {r[0] for r in self.db.execute(
            "SELECT id FROM sif WHERE state='active'").fetchall()}
        orphans = 0
        for coll in (self.visual, self.text, self.clip):
            got = coll.get()
            ids = got.get("ids", [])
            metas = got.get("metadatas", []) or [{}] * len(ids)
            stranded = [ids[i] for i in range(len(ids))
                        if (metas[i] or {}).get("path", ids[i]) not in active]
            if stranded:
                coll.delete(ids=stranded)
                orphans += len(stranded)
        return {"tombstones_finished": len(tombs), "orphans_purged": orphans}

    # -- dedup lookups ----------------------------------------------------
    def find_duplicate(self, h: dedup.Hashes,
                       threshold: int = dedup.DEFAULT_PHASH_THRESHOLD):
        """Return (existing_id, tier) for the first dedup match, else None."""
        row = self.db.execute(
            "SELECT id FROM sif WHERE state='active' AND sha256=? LIMIT 1",
            (h.sha256,)).fetchone()
        if row:
            return (row[0], "sha256")
        if h.pixel_hash:
            row = self.db.execute(
                "SELECT id FROM sif WHERE state='active' AND pixel_hash=? LIMIT 1",
                (h.pixel_hash,)).fetchone()
            if row:
                return (row[0], "pixel")
        if h.phash:
            for sid, ph in self.db.execute(
                    "SELECT id, phash FROM sif WHERE state='active' AND phash!=''"):
                if dedup.hamming(h.phash, ph) <= threshold:
                    return (sid, "phash")
        return None

    # -- read -------------------------------------------------------------
    def get(self, sid: str) -> dict | None:
        """Full SIF JSON for an ACTIVE asset (deleted/tombstoned rows return
        None, so they can never surface in results)."""
        row = self.db.execute(
            "SELECT sif_json FROM sif WHERE id=? AND state='active'", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_assets(self, limit: int = 200, offset: int = 0) -> list[dict]:
        """Indexed assets, newest first, as light summaries for the Library UI."""
        rows = self.db.execute(
            "SELECT id, path, sif_json FROM sif WHERE state='active' AND indexed=1 "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        out = []
        for sid, path, sj in rows:
            d = json.loads(sj)
            kind = d.get("kind", "image")
            caption = d.get("scene", {}).get("caption", "")
            if kind == "pdf":
                caption = caption or f"PDF · {len(d.get('pages', []))} pages"
            out.append({
                "id": sid, "path": path, "kind": kind, "caption": caption,
                "has_text": bool(d.get("ocr", {}).get("has_text")),
                "n_faces": len(d.get("faces", [])),
            })
        return out

    def get_meta(self, path: str) -> dict | None:
        """Lightweight row state for a path (or None if not active)."""
        row = self.db.execute(
            "SELECT sha256, indexed, state FROM sif WHERE id=? AND state='active'",
            (path,)).fetchone()
        if not row:
            return None
        return {"sha256": row[0], "indexed": row[1], "state": row[2]}

    def count(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM sif WHERE state='active'").fetchone()[0]

    def all_active_ids(self) -> set[str]:
        rows = self.db.execute(
            "SELECT id FROM sif WHERE state='active' AND indexed=1").fetchall()
        return {r[0] for r in rows}

    def close(self):
        self.db.close()
        # Release ChromaDB's file handles too. On POSIX an open file can still
        # be unlinked, so this is a no-op there; on Windows the handles must be
        # dropped or the data dir can't be deleted (e.g. temp dirs in tests).
        try:
            self.chroma._system.stop()
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
