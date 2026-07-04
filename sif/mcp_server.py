"""
MCP server — expose the SIF index to AI agents as tools.

Lets an MCP client (Claude Desktop, Claude Code, etc.) search your local visual
+ document archive: the engine does the cheap, local, pre-computed retrieval and
the agent reasons over the results. The raw images never leave the machine.

Run:
    pip install "mcp[cli]"
    SIF_DATA=./sif_data python -m sif.mcp_server        # stdio server

Register it with a client, e.g. Claude Desktop config:
    "sif-engine": { "command": "python", "args": ["-m", "sif.mcp_server"],
                    "env": { "SIF_DATA": "C:\\path\\to\\your\\index" } }

Tools:
  * sif_search(query, limit)  — semantic search; returns path/page/caption/score
  * sif_stats()               — index size + vector counts
"""
from __future__ import annotations

import os

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The MCP SDK isn't installed. Run:  pip install \"mcp[cli]\"\n"
        f"(import error: {e})"
    )

from .store import Store
from .query import search

DATA_ROOT = os.environ.get("SIF_DATA", "./sif_data")

mcp = FastMCP("sif-engine")
_store: Store | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(DATA_ROOT)
    return _store


@mcp.tool()
def sif_search(query: str, limit: int = 10) -> list[dict]:
    """Semantically search the local image/PDF archive. Returns matches with
    file path, page (for PDFs), caption, and which vector spaces matched."""
    results = search(_get_store(), query, limit=limit)
    return [{
        "path": r["path"],
        "page": r.get("page"),
        "caption": r["caption"],
        "objects": r["objects"],
        "matched": r.get("matched", []),
        "score": r["score"],
    } for r in results]


@mcp.tool()
def sif_stats() -> dict:
    """Report the SIF index size and vector-space counts."""
    s = _get_store()
    return {
        "data_root": os.path.abspath(DATA_ROOT),
        "indexed": s.count(),
        "visual_vectors": s.visual.count(),
        "text_vectors": s.text.count(),
        "clip_vectors": s.clip.count(),
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
