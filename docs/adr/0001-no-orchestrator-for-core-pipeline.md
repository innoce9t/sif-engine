# ADR 0001 — No agent orchestrator (LangChain/LangGraph) for the core pipeline

- **Status:** Accepted
- **Date:** 2026-06-28
- **Deciders:** Ahsan Nawazish

## Context

The question came up of whether to build the SIF ingestion/query pipeline on an
agent-orchestration framework such as LangChain or LangGraph.

Those frameworks are designed for workflows where **an LLM decides the control
flow at runtime** — agentic loops, dynamic tool selection, branching on model
output, human-in-the-loop, multi-step reasoning. LangGraph in particular is a
stateful-graph engine for *non-deterministic* agent workflows.

The SIF Engine is the opposite shape. It is a **fixed, deterministic DAG**:

```
ingest:  file -> [objects, faces, scene, ocr] -> build visual/text input -> embed -> store
query:   text -> embed -> vector search -> RRF fuse -> gated rerank -> validate vs SQLite
```

Nothing in this graph makes a runtime decision about *what to do next*. The
graph is known at design time and never changes. The intelligence lives in
**pre-computed extraction** (extract once, query forever), not in agentic
orchestration — which is the engine's whole thesis.

The real engineering concerns here are throughput and memory: serialize the
memory-bound VLM, parallelize cheap extractors, backpressure via queues,
RAM-aware worker sizing (Stage 4). Those are data-pipeline concerns, not
agent-orchestration concerns.

## Decision

Do **not** adopt LangChain/LangGraph for the core ingestion or query pipeline.

- Stage 4 concurrency (N extraction workers -> `vlm_queue` -> single dedicated
  VLM worker) will be built on plain `asyncio` + `queue` / `concurrent.futures`.
- If durable/distributed ingestion is ever needed, reach for a task or workflow
  engine (Arq, Dramatiq, Celery, or Prefect/Dagster/Temporal) — not LangChain.

An agent orchestrator is reconsidered **only** if/when an agentic
natural-language query layer is added (an LLM planning multi-step retrieval and
tool calls over the SIF index). That is a Stage 5+/commercial concern and would
live as an optional layer **above** the core engine, firewalled so it never
contaminates the ingestion pipeline.

## Consequences

- **Positive:** keeps the engine dependency-light and aligned with its value
  prop (local-first, minimal, ~90% query-cost reduction); avoids abstraction
  overhead and the churn of a fast-moving framework; control flow stays explicit
  and debuggable.
- **Positive (portfolio):** avoids the over-orchestration smell of putting an
  agent graph on top of a static DAG.
- **Negative / accepted:** we hand-roll the Stage 4 concurrency primitives
  instead of getting them from a framework. This is the intended trade — those
  primitives are small and the design (dedicated VLM worker, micro-queues) is
  already specified.
- **Revisit trigger:** a genuine agentic NL query interface — see above.
