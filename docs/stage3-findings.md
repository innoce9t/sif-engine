# Stage 3 — Retrieval Findings

Empirical notes from running multi-vector search with real models. The fusion
and gating logic match the converged design and pass their unit tests; this
documents a retrieval-quality issue that only surfaces with real embeddings.

## Finding 1 — RRF over-credits an asset that has a (sparse) text vector

**Symptom:** querying `"two men in dark suits"` ranked `street.jpg` (a bus)
above `zidane.jpg` (literally two men in black suits).

**Diagnosis (per-collection raw distances, lower = better):**

| collection | zidane.jpg | street.jpg |
|---|---|---|
| visual | **0.60** (strong) | 1.04 (weak) |
| text   | — (no OCR → no vector) | 1.10 (irrelevant) |

`zidane` is the clear visual match. But `street` is the **only** asset with a
text vector, so it takes **rank 1** in the text collection *even though its
text match (1.10) is irrelevant*. RRF scores rank, not similarity, so:

- zidane: `1/(60+1)` visual + `1/(60+51)` absentee = 0.02540
- street: `1/(60+2)` visual + `1/(60+1)` text = 0.03252  ← wins, wrongly

So an asset that merely *possesses* a text vector gets unearned rank-credit for
an irrelevant match. The re-rank gate (relative gap < 5%) did **not** fire,
because RRF produced a confident-looking 22% gap — on the wrong answer.

**Confirmation:** forcing the cross-encoder re-rank puts `zidane.jpg` #1
correctly — it reads the caption "two men dressed in black suits" and scores it
far above the bus.

**Severity / scope:** amplified by the tiny 2-document corpus (with many text
vectors, an irrelevant one wouldn't sit at rank 1), but the structural bias is
real at any scale: rank-only fusion gives credit to a collection's top hit even
when that hit is irrelevant.

**Headline still works:** OCR text became searchable for the first time —
`"madrid public transport"` and `"cero emisiones"` (Spanish text off the bus)
correctly return `street.jpg`. The multi-vector retrieval itself is sound.

## Candidate fixes (to calibrate against the Stage 4 benchmark harness)

1. **Relevance floor on fusion** — a collection only contributes a candidate's
   rank if its distance is within a cutoff, so an irrelevant sparse-collection
   hit doesn't inflate the score. Principled (addresses the root cause) but the
   cutoff is embedder/metric-specific and should be *measured*, not guessed.
2. **Run the re-rank more often** — lower the gate, or always re-rank small
   candidate pools. More accurate, more cost; the gate exists precisely to
   bound that cost.
3. **Distance-aware fusion** — blend rank with a normalized score signal so a
   bad-but-top hit contributes less than a great one.

These are tuning decisions that need real measurements. They're deferred to the
**Stage 4 benchmark harness** (where the spec's numbers become measured facts)
rather than hard-coded with an untuned threshold now. The cross-encoder is the
quality backstop and already corrects the case when it runs.
