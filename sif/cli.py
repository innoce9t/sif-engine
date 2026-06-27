"""
Stage 0 CLI.

  python -m sif.cli index <path-or-dir>
  python -m sif.cli search "<query>"
  python -m sif.cli stats

Minimal argparse interface. Real ergonomics (progress bars, watch mode,
config) arrive in Stage 6.
"""
from __future__ import annotations

import argparse
import os
import sys

from .store import Store
from .pipeline import process
from .query import search


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def cmd_index(args):
    store = Store(args.data)
    targets = []
    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    targets.append(os.path.join(root, f))
    else:
        targets = [args.path]

    if not targets:
        print(f"No images found at {args.path}")
        return

    for i, path in enumerate(targets, 1):
        sif = process(path)
        store.upsert(sif)
        print(f"[{i}/{len(targets)}] indexed {os.path.basename(path)} "
              f"-> caption='{sif.scene.caption}' objects={[o.label for o in sif.objects]} "
              f"({sif.meta['processing_ms']}ms)")
    print(f"\nDone. {store.count()} assets in index.")
    store.close()


def cmd_search(args):
    store = Store(args.data)
    results = search(store, args.query, limit=args.limit)
    if not results:
        print("No results.")
        store.close()
        return
    print(f"Results for '{args.query}':\n")
    for r in results:
        print(f"  {r['distance']:.4f}  {os.path.basename(r['path'])}")
        print(f"          caption: {r['caption']}")
        print(f"          objects: {', '.join(r['objects'])}\n")
    store.close()


def cmd_stats(args):
    store = Store(args.data)
    print(f"Indexed assets: {store.count()}")
    print(f"Visual vectors: {store.visual.count()}")
    print(f"Text vectors:   {store.text.count()}")
    store.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="sif", description="SIF Engine (Stage 0)")
    p.add_argument("--data", default="./sif_data", help="data directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index an image or directory")
    pi.add_argument("path")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="semantic search")
    ps.add_argument("query")
    ps.add_argument("--limit", type=int, default=10)
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("stats", help="index stats")
    pst.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
