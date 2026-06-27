"""
Stage 0 storage layer.

Establishes the dual-store shape that all later stages build on:
  - SQLite  = source of truth (full SIF JSON persisted here)
  - ChromaDB = derived vector index (two collections: visual + text)

Stage 0 does a simple write to both. The crash-safe OUTBOX ordering
(indexed=false -> vectors -> indexed=true), recovery sweep, deletion
lifecycle, and reconciliation all arrive in Stage 2. The interface here
is deliberately shaped so that hardening slots in without callers changing.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import chromadb

from .schema import SIF


class Store:
    def __init__(self, root: str = "./sif_data"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, "sif.db")
        self._init_sqlite()
        self._init_chroma()

    # -- SQLite (source of truth) -----------------------------------------
    def _init_sqlite(self):
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sif (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                sha256      TEXT,
                sif_json    TEXT NOT NULL,
                indexed     INTEGER NOT NULL DEFAULT 0,
                state       TEXT NOT NULL DEFAULT 'active',
                updated_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self.db.commit()

    # -- ChromaDB (derived index) -----------------------------------------
    def _init_chroma(self):
        self.chroma = chromadb.PersistentClient(path=os.path.join(self.root, "chroma"))
        # Two collections = the v3 multi-vector design.
        self.visual = self.chroma.get_or_create_collection("visual")
        self.text = self.chroma.get_or_create_collection("text")

    # -- write ------------------------------------------------------------
    def upsert(self, sif: SIF):
        """
        Stage 0: straight write to SQLite then both vector collections.
        (Outbox ordering + crash-safety added in Stage 2.)
        """
        sid = sif.id
        # 1. SQLite row (source of truth)
        self.db.execute(
            "INSERT OR REPLACE INTO sif (id, path, sha256, sif_json, indexed, state) "
            "VALUES (?, ?, ?, ?, 1, 'active')",
            (sid, sif.file.path, sif.file.sha256, sif.to_json(indent=None)),
        )
        self.db.commit()

        # 2. Vectors -> two collections (skip empties)
        meta = {"path": sif.file.path, "sif_id": sid}
        if any(sif.embeddings.visual):
            self.visual.upsert(ids=[sid], embeddings=[sif.embeddings.visual], metadatas=[meta])
        if any(sif.embeddings.text):
            self.text.upsert(ids=[sid], embeddings=[sif.embeddings.text], metadatas=[meta])

    # -- read -------------------------------------------------------------
    def get(self, sid: str) -> dict | None:
        row = self.db.execute("SELECT sif_json FROM sif WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM sif WHERE state='active'").fetchone()[0]

    def all_active_ids(self) -> set[str]:
        rows = self.db.execute("SELECT id FROM sif WHERE state='active' AND indexed=1").fetchall()
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
