"""
Benchmark harness (Stage 4).

Runs the concurrent indexer over a set of images and reports measured numbers —
per-stage latency, throughput, and peak RSS — alongside the RAM-aware sizing
inputs. This is where the design's paper estimates (per-worker MB, throughput)
get confirmed or corrected with real measurements rather than guessed.

Run with real models for meaningful numbers:
    python -m sif.cli bench <dir>            # from the Stage-1 venv, Ollama up
"""
from __future__ import annotations

import threading
import time

from . import runner, workers
from .store import Store


class _PeakRSS:
    """Sample this process's RSS on a background thread to capture a peak."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = False
        self._thread = None

    def __enter__(self):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            return self

        def loop():
            while not self._stop:
                self.peak_mb = max(self.peak_mb, proc.memory_info().rss / (1024 ** 2))
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1)


def benchmark(data_root: str, paths: list[str], max_workers: int | None = None) -> dict:
    store = Store(data_root)
    with _PeakRSS() as peak:
        report = runner.index_paths(store, paths, max_workers=max_workers)
    store.close()

    n = report["processed"]
    wall = report["wall_s"]
    report["sizing"] = workers.sizing_report()
    report["peak_rss_mb"] = round(peak.peak_mb)
    report["throughput_imgs_per_s"] = round(n / wall, 3) if wall > 0 else 0.0
    return report


def format_report(r: dict) -> str:
    s = r["sizing"]
    lines = [
        "SIF Engine — benchmark",
        f"  RAM            : {s['total_ram_mb']}MB total, "
        f"budget {s['budget_mb']}MB ({int(s['budget_frac']*100)}%)",
        f"  workers        : {r['workers']} (cpu={s['cpu_count']}, "
        f"per-worker~{s['per_worker_mb']}MB, vlm_reserve={s['vlm_reserve_mb']}MB)",
        f"  processed      : {r['processed']} images in {r['wall_s']}s",
        f"  throughput     : {r['throughput_imgs_per_s']} img/s",
        f"  peak RSS       : {r['peak_rss_mb']}MB",
        "  per-stage (avg ms / total s):",
    ]
    for stage in ("extract", "vlm", "finalize", "write"):
        t = r["timings"].get(stage, {})
        lines.append(f"    {stage:9}: {t.get('avg_ms', 0):>8} ms   "
                     f"{t.get('total_s', 0):>7} s   (n={t.get('count', 0)})")
    lines.append(f"  result         : {r['stats']}")
    return "\n".join(lines)
