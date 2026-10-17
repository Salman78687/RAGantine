# Architecture

RAGantine wraps a standard hybrid RAG pipeline with defense hooks at ingestion, retrieval and
generation. Everything a defense blocks lands in one quarantine store. The benchmark (`bench/`)
drives the pipeline (`ragantine/`) from the outside. It is the only code that ever reads ground
truth.

## Packages

| Package | Role | May import |
|---|---|---|
| `ragantine/` | the library: ingestion, index, retrieval, generation, defenses, quarantine, query graph, API, CLI, MCP | itself |
| `bench/` | corpus build, questions, poison, run protocol, scoring, metrics, reports, ablation, adaptive attacker, smoke gate, dataset release | `ragantine` |
| `demo/` | Gradio app and its snapshot builder | `ragantine` (the snapshot builder also uses `bench`) |

The `ragantine` CLI gets its `corpus`, `index` and `bench` commands from the `bench` package
through the `ragantine.cli` entry-point group. That keeps the dependency pointing one way:
`tests/unit/test_leakage.py` fails if anything in `ragantine/` imports `bench` or names the
poison files.

## Data contracts

Every module boundary passes Pydantic v2 models: `ragantine/schemas.py` for the library,
`bench/schemas.py` for ground truth.

```
RawDocument ──load/D1──► text ──chunk──► Chunk ──D2──► index | QuarantineRecord
question ──► RetrievedChunk[] (bm25_rank, knn_rank, rrf_score, rerank_score, final_score)
          ──► DefenseDecision[] (defense_id, stage, target_id, action, score, reason)
          ──► Answer (text, citations, decisions, retrieved, candidate_ids, draft, tokens, latency, trace_id)
bench: BenchQuestion · PoisonSample (ground truth) · JudgedAnswer · Metrics
```

## Ingestion (`ragantine/ingest/`)

`load → D1.on_load → chunk → D2.on_chunks → embed → bulk index`

- The baseline loaders are deliberately naive. They keep HTML comments, `alt`/`title`, meta
  descriptions, `display:none` text and white PDF text, like most real pipelines.
- The chunker packs whole sentences up to 350 cl100k tokens with about 50 tokens of overlap,
  and prefixes `Title: …`.
- Ingestion is idempotent by `content_hash`, and precomputed embeddings are reused by hash.
- A chunk with any `quarantine` decision goes to the quarantine index instead of the main one.
  D1 quarantines whole documents.

## Index (`ragantine/index/`, `ragantine/retrieve/memory.py`)

`IndexBackend` is implemented by `OpenSearchIndex` (BM25 and HNSW k-NN on Lucene, one index)
and by `MemoryIndex` (rank_bm25 and numpy cosine, used for tests, the CI smoke run and the HF
Space). Both hold the quarantine store. Each ingestion configuration gets its own index pair,
`ragantine_chunks__<hash>`, because ingestion defenses change what is indexed.

## Query graph (`ragantine/graph.py`)

```
retrieve ─► filter_context ─► build_prompt ─► generate ─► verify ─► finalize
 BM25+kNN    D2 · D3 · D6       D0 or D4        LLM          D5        Answer + trace
 RRF, rerank
```

- `retrieve` returns the reranked `fused_k` (20) candidates. `filter_context` runs the
  retrieval hooks and keeps `final_k` (5).
- `generate` streams tokens when called through `stream_query`. That is the SSE path:
  `retrieval`, `defense*`, `token*`, then `done` or `error`.
- `verify` hands D5 a `QueryContext` with a `regenerate` callback, so rewrites go through the
  same prompt path, and their tokens are counted.
- Public entry points: `run_query`, `arun_query`, `stream_query`, `asafe_retrieve`.

## LLM access (`ragantine/generate/llm.py`)

All calls go through `LLMClient`: the OpenAI SDK pointed at each provider's `base_url` (Groq,
OpenAI, or vLLM later). It has tenacity retries on 429/5xx and connection errors (5 attempts),
a disk cache at temperature 0 keyed by sha256(provider, model, messages, params), per-loop
concurrency limits, and token accounting. Models that reject `temperature` are detected
and remembered. Placeholder keys in `.env` count as missing.

## Benchmark (`bench/`)

```
corpus build ─► questions (filtered by retrieval + clean correctness) ─► poison (A1–A6, seeds)
     └────────────── run: Workspace.prepare(index per ingestion config + poison set)
                     ─► arun_query per question ─► rule → judge scoring ─► metrics.json, report.md
ablation = runs over {D0, singles, full, leave-one-out} × models × seeds ─► report table + CSV + PNG
```

`OpenSearchWorkspace` writes a manifest under `data/indexes/` so repeated runs reuse a finished
index. `MemoryWorkspace` builds everything in-process.

## Observability

Each answer can be sent to Langfuse as one trace: a chain root, a retriever span, one guardrail
per defense decision, and a generation with token usage. This is a no-op unless both Langfuse
keys are set.
