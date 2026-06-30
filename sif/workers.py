"""
RAM-aware concurrency sizing (Stage 4).

The extraction pool is bounded by memory, not just CPU. Each extraction worker
holds its own copy of the cheap models (YOLO + OCR + optional faces); the VLM
is held once by the single dedicated VLM worker, so extraction workers never
hold it. Worker count is computed from available RAM at a configurable budget:

    max_workers = floor((RAM_budget - VLM_reserve - base_overhead) / per_worker)

then capped to the CPU count. Per the design review (HANDOVER §6):
per-worker ~= 900MB (YOLO 180 + OCR 400 + faces 320), VLM reserve ~= 1400MB,
base overhead ~= 300MB, budget = 70% of total RAM.
"""
from __future__ import annotations

import os

# Megabyte costs from the design review. Overridable via env for benchmarking.
PER_WORKER_MB = int(os.environ.get("SIF_PER_WORKER_MB", "900"))
VLM_RESERVE_MB = int(os.environ.get("SIF_VLM_RESERVE_MB", "1400"))
BASE_OVERHEAD_MB = int(os.environ.get("SIF_BASE_OVERHEAD_MB", "300"))
RAM_BUDGET_FRAC = float(os.environ.get("SIF_RAM_BUDGET_FRAC", "0.70"))


def total_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 2)
    except Exception:
        # psutil missing: assume a conservative 8GB so we still pick a sane count
        return 8192.0


def recommended_workers(budget_frac: float = RAM_BUDGET_FRAC) -> int:
    """Memory-aware extraction-worker count, capped to CPU cores, floored at 1."""
    budget = total_ram_mb() * budget_frac
    usable = budget - VLM_RESERVE_MB - BASE_OVERHEAD_MB
    by_ram = int(usable // PER_WORKER_MB)
    cpus = os.cpu_count() or 1
    return max(1, min(by_ram, cpus))


def sizing_report(budget_frac: float = RAM_BUDGET_FRAC) -> dict:
    """The inputs and result, for logging / the benchmark harness."""
    total = total_ram_mb()
    return {
        "total_ram_mb": round(total),
        "budget_frac": budget_frac,
        "budget_mb": round(total * budget_frac),
        "vlm_reserve_mb": VLM_RESERVE_MB,
        "base_overhead_mb": BASE_OVERHEAD_MB,
        "per_worker_mb": PER_WORKER_MB,
        "cpu_count": os.cpu_count() or 1,
        "recommended_workers": recommended_workers(budget_frac),
    }
