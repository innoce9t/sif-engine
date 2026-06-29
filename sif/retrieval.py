"""
Retrieval fusion + re-rank gating (Stage 3) — pure, testable logic.

The engine stores two vector spaces (visual, text) whose raw distances live on
incomparable scales, so results are combined with **Reciprocal Rank Fusion**
(rank-based, scale-immune) rather than weighted distance addition.

Design-review fixes baked in here:

* **RRF, not weighted raw scores** — ``score = Σ weight[s] / (k + rank[s])``,
  k=60.
* **Absentee baseline (no single-modality starvation)** — an asset in one
  space's top-K but absent from the other is assigned a baseline rank of
  ``top_k + 1`` instead of infinity, so a perfect single-modality match (e.g. a
  pure image with no text) isn't unfairly beaten by a mediocre dual-modality
  asset. (The "drone shot ranks #1" fix.)
* **Relative-margin re-rank gating** — RRF scores are tiny and tightly
  compressed by k=60, so an *absolute* margin misfires. Gate the (expensive)
  neural re-rank on the **relative** gap between the top two scores instead.
"""
from __future__ import annotations

K_RRF = 60
DEFAULT_TOP_K = 50
DEFAULT_WEIGHTS = {"visual": 1.0, "text": 1.0}
RERANK_GAP = 0.05  # rerank when top-1 vs top-2 differ by less than this fraction


def rrf_fuse(per_source_ranks: dict[str, dict[str, int]], top_k: int,
             weights: dict[str, float] | None = None, k: int = K_RRF,
             absentee: bool = True) -> list[tuple[str, float]]:
    """Fuse per-source rank maps into a single ranked list of (id, score).

    ``per_source_ranks`` maps a source name -> {id: rank} (rank starts at 1).
    With ``absentee=True``, an id missing from a source contributes as if it
    sat at rank ``top_k + 1`` rather than contributing zero.
    """
    weights = weights or DEFAULT_WEIGHTS
    absent_rank = top_k + 1

    ids: set[str] = set()
    for ranks in per_source_ranks.values():
        ids |= set(ranks)

    scored: list[tuple[str, float]] = []
    for sid in ids:
        score = 0.0
        for source, ranks in per_source_ranks.items():
            w = weights.get(source, 1.0)
            if sid in ranks:
                score += w / (k + ranks[sid])
            elif absentee:
                score += w / (k + absent_rank)
            # else: absent contributes 0 (the old infinity behavior)
        scored.append((sid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def relative_gap(scores: list[float]) -> float:
    """Fractional gap between the top two scores, in [0, 1]. 1.0 if <2 scores."""
    if len(scores) < 2 or scores[0] <= 0:
        return 1.0
    return (scores[0] - scores[1]) / scores[0]


def should_rerank(scores: list[float], threshold: float = RERANK_GAP) -> bool:
    """True when the top two results are close enough to be ambiguous."""
    if len(scores) < 2:
        return False
    return relative_gap(scores) < threshold
