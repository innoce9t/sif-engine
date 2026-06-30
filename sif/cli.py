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
from .ingest import ingest
from .query import search


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DOC_EXTS = {".pdf"}


def _collect(path: str) -> list[str]:
    if os.path.isdir(path):
        out = []
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS | DOC_EXTS:
                    out.append(os.path.join(root, f))
        return out
    return [path]


def cmd_index(args):
    store = Store(args.data)
    targets = _collect(args.path)
    if not targets:
        print(f"No images or PDFs found at {args.path}")
        return

    images = [t for t in targets if os.path.splitext(t)[1].lower() in IMAGE_EXTS]
    pdfs = [t for t in targets if os.path.splitext(t)[1].lower() in DOC_EXTS]
    tally: dict[str, int] = {}

    def bump(status):
        tally[status] = tally.get(status, 0) + 1

    if images and not args.sequential:
        # Concurrent decoupled pipeline (Stage 4) for images.
        from . import runner
        done = [0]

        def progress(label, path):
            done[0] += 1
            print(f"[{done[0]}] {label:9} {os.path.basename(path)}")

        report = runner.index_paths(store, images, max_workers=args.workers, progress=progress)
        for k, v in report["stats"].items():
            tally[k] = tally.get(k, 0) + v
        images = []  # handled

    # PDFs (and images under --sequential) go through the single-threaded path.
    for path in images + pdfs:
        r = ingest(store, path)
        bump(r.status)
        extra = f" ({r.detail})" if r.detail else ""
        print(f"  {r.status:9} {os.path.basename(path)}{extra}")

    summary = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    print(f"\nDone. {store.count()} assets in index. [{summary}]")
    store.close()


def cmd_search(args):
    store = Store(args.data)
    results = search(store, args.query, limit=args.limit)
    if not results:
        print("No results.")
        store.close()
        return
    tag = " (re-ranked)" if results and results[0].get("reranked") else ""
    print(f"Results for '{args.query}'{tag}:\n")
    for r in results:
        loc = os.path.basename(r['path'])
        if r.get("page") is not None:
            loc += f"  (page {r['page'] + 1})"
        print(f"  {r['score']:.5f}  {loc}")
        print(f"          caption: {r['caption']}")
        print(f"          objects: {', '.join(r['objects'])}\n")
    store.close()


def cmd_stats(args):
    store = Store(args.data)
    print(f"Indexed assets: {store.count()}")
    print(f"Visual vectors: {store.visual.count()}")
    print(f"Text vectors:   {store.text.count()}")
    store.close()


def cmd_reconcile(args):
    store = Store(args.data)
    stats = store.reconcile()
    print(f"Reconciled: tombstones finished={stats['tombstones_finished']}, "
          f"orphan vectors purged={stats['orphans_purged']}")
    store.close()


def cmd_serve(args):
    os.environ.setdefault("SIF_DATA", args.data)
    import uvicorn
    print(f"SIF Engine UI at http://{args.host}:{args.port}  (data: {args.data})")
    uvicorn.run("sif.api:app", host=args.host, port=args.port, reload=False)


def cmd_bench(args):
    from . import bench
    targets = _collect(args.path)
    if not targets:
        print(f"No images found at {args.path}")
        return
    print(bench.format_report(bench.benchmark(args.data, targets, max_workers=args.workers)))


def cmd_watch(args):
    import time
    store = Store(args.data)
    print(f"Watching {args.path} every {args.interval}s (Ctrl+C to stop)…")
    try:
        while True:
            changed = 0
            for path in _collect(args.path):
                r = ingest(store, path)         # dedup makes re-scans idempotent
                if r.status in ("indexed", "updated"):
                    changed += 1
                    print(f"  {r.status:9} {os.path.basename(path)}")
            if changed:
                print(f"  -> {store.count()} assets indexed")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        store.close()


def cmd_version(args):
    from . import __version__
    print(f"SIF Engine {__version__}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sif", description="SIF Engine")
    p.add_argument("--data", default="./sif_data", help="data directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index an image or directory")
    pi.add_argument("path")
    pi.add_argument("--workers", type=int, default=None,
                    help="extraction workers (default: RAM-aware auto)")
    pi.add_argument("--sequential", action="store_true",
                    help="single-threaded ingest instead of the concurrent pipeline")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="semantic search")
    ps.add_argument("query")
    ps.add_argument("--limit", type=int, default=10)
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("stats", help="index stats")
    pst.set_defaults(func=cmd_stats)

    pr = sub.add_parser("reconcile", help="purge orphan vectors + finish tombstones")
    pr.set_defaults(func=cmd_reconcile)

    pv = sub.add_parser("serve", help="launch the web UI")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.set_defaults(func=cmd_serve)

    pb = sub.add_parser("bench", help="benchmark the concurrent indexer")
    pb.add_argument("path")
    pb.add_argument("--workers", type=int, default=None)
    pb.set_defaults(func=cmd_bench)

    pw = sub.add_parser("watch", help="watch a folder and index new/changed files")
    pw.add_argument("path")
    pw.add_argument("--interval", type=float, default=5.0)
    pw.set_defaults(func=cmd_watch)

    pver = sub.add_parser("version", help="print version")
    pver.set_defaults(func=cmd_version)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
