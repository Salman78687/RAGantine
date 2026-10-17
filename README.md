# RAGantine

**Quarantine for retrieval: benchmarking and defending RAG against knowledge-base poisoning.**

> **Finding:** _pending the first full benchmark run._ This line and the table below are filled
> only from `results/<run_id>/metrics.json` (`ragantine bench report`). No number in this README
> is typed by hand.

Plant poison in a RAG corpus, measure how often it wins, then quarantine it, and publish the numbers.

## Results

| Configuration | ASR (%) | Clean accuracy (%) | Detection recall (%) | False quarantine (%) | p95 latency (ms) |
|---|---|---|---|---|---|
| D0 Baseline | — | — | — | — | — |
| + D1 Sanitizer | — | — | — | — | — |
| + D2 Injection classifier | — | — | — | — | — |
| + D3 Source trust | — | — | — | — | — |
| + D4 Spotlighting | — | — | — | — | — |
| + D5 Consistency check | — | — | — | — | — |
| Full stack | — | — | — | — | — |

ASR = attack success rate on poisoned **test** questions; 3 seeds, 95% bootstrap CIs; three generator
models (`open-large` = gpt-oss-120b and `open-small` = gpt-oss-20b on Groq, `closed-small` = OpenAI
gpt-6-luna). Reproduce with `make ablation report`.

## What it is

RAG pipelines trust whatever is in the knowledge base. Wikis, tickets, web crawls and uploads mean
that is no longer only trusted admins. RAGantine ships three things:

- **RAGantine-Bench**: a Wikipedia corpus with planted poison for six attacks, plus target, clean
  and paraphrase question sets with dev/test splits. Questions are kept only if the clean system
  already answers them correctly, so every point of ASR is caused by poison.
- **A layered defense** that can be switched on one layer at a time:

  | | Defense | Stage | Main attacks |
  |---|---|---|---|
  | D1 | Ingestion sanitizer: strips hidden HTML/PDF/zero-width text, quarantines documents that hide content | ingestion | A5 |
  | D2 | Injection classifier: DeBERTa-v3-small fine-tuned on train-only poison templates | ingestion + retrieval | A2, A5, A6 |
  | D3 | Source trust: tier-weighted re-ranking, at most 2 web/user-upload chunks in the prompt | retrieval | A1, A3, A4 |
  | D4 | Spotlighting: documents wrapped and datamarked as data, instructions kept separate | generation | A2, A6 |
  | D5 | Cross-source consistency: NLI check that each claim is backed by ≥ 2 docs or a curated one | generation | A1, A3, A4 |
  | D6 | Retrieval anomaly detector: filters "hub" chunks that rank for unrelated queries (P2) | retrieval | A4 |

- **A quarantine, not a delete.** Flagged chunks go to a review queue with their reasons and can
  be released by a human (`/quarantine`, `ragantine quarantine release`).

### Attacks

| ID | Attack | The poison… |
|---|---|---|
| A1 | Factual override | states a false fact in encyclopedic style |
| A2 | Indirect instruction | hides an instruction to the model in normal text (canary goal) |
| A3 | Authority spoof | poses as an "official correction / errata" |
| A4 | Retrieval stuffing | repeats the question and rewordings to rank first |
| A5 | Hidden text | puts the instruction in an HTML comment, `display:none`, `alt`, zero-width or white PDF text |
| A6 | Split payload | spreads the instruction across two documents |

Poison is harmless and machine-checkable: a specific wrong answer, or a canary such as `ZEBRA-7741`.

## Architecture

```mermaid
flowchart LR
  src[Document sources<br/>curated · partner · web · user-upload] --> ing[Ingestion<br/>D1 sanitizer · D2 classifier]
  ing --> idx[(OpenSearch<br/>BM25 + k-NN)]
  ing -- flagged --> q[(Quarantine store<br/>reason + score<br/>human release)]
  ask([question]) --> ret[Retrieval<br/>RRF + bge reranker<br/>D2 · D3 · D6]
  idx --> ret
  ret -- filtered --> q
  ret --> gen[Generation<br/>D4 spotlight · LLM · D5 check]
  gen --> ans[Answer + trace<br/>citations · decisions]
  ans --> bench[RAGantine-Bench scorer<br/>ASR · accuracy]
```

The query pipeline is a LangGraph graph with fixed nodes, `retrieve → filter_context → build_prompt
→ generate → verify → finalize`. A disabled defense is a pass-through, so the graph shape never
changes between configurations. Ground truth (`poison.jsonl`, attack labels, canaries) never
reaches the library or the index, and a unit test enforces that. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Needs Python 3.11 via [uv](https://docs.astral.sh/uv/) and Docker (about 8 GB RAM). A GPU is optional.

```bash
git clone <this repo> ragantine && cd ragantine
make setup                       # uv sync + pre-commit
cp .env.example .env             # then put your GROQ_API_KEY / OPENAI_API_KEY in .env
```

**No keys? Run the offline smoke benchmark.** It replays recorded responses on the bundled
fixture: no network, no OpenSearch, no model downloads.

```bash
uv run ragantine bench smoke     # baseline vs full stack on tests/fixtures/mini_bench
make check                       # lint, types, ~350 unit/component tests
```

**A small real run** costs well under $1 with `--limit`. Every paid command prints a cost
estimate and asks before calling any model.

```bash
make up                                           # OpenSearch 2.19 in Docker
uv run ragantine corpus build --n-articles 300    # stream + sample Wikipedia, embed (CPU)
uv run ragantine index build --no-poison
uv run ragantine ask "Where is The Market Place shopping center located?"
uv run ragantine bench questions --limit 24       # 4 target questions per attack + clean + paraphrase
uv run ragantine bench poison --limit 24
uv run ragantine bench run --defenses none --sets target,clean --split dev --limit 60
uv run ragantine bench run --defenses all  --sets target,clean --split dev --limit 60
uv run ragantine bench report --runs all
```

**The full benchmark:** `make corpus bench-data`, train D2 (below), then `make ablation report`.

| Command | What it does |
|---|---|
| `make serve` | FastAPI on :8000: SSE `/ask`, `/quarantine`, `/runs`, `/health` |
| `make demo` | Gradio side-by-side demo, Baseline vs RAGantine, on a 500-article snapshot |
| `ragantine bench d2-dataset` → [notebooks/train_d2.ipynb](notebooks/train_d2.ipynb) → `ragantine bench d2-threshold --write` | fine-tune D2 on Colab, then pick its threshold on dev |
| `ragantine bench run --defenses d2 --set defenses.d2.variant=zeroshot` | D2 zero-shot baseline (`protectai/deberta-v3-base-prompt-injection-v2`) |
| `ragantine bench adaptive --defenses all` | adaptive attacker: rewrites blocked poison from the defense's reason, ≤ 3 rounds |
| `ragantine mcp` | MCP server with `safe_retrieve` and `audit_corpus` tools |
| `ragantine bench quality --runs latest` | clean-set context recall and NLI faithfulness (RAGAS-style, local models) |
| `ragantine.adapters.AdapterRetriever` | benchmark **your own** retriever (LangChain, LlamaIndex or a function); see [docs/ADAPTERS.md](docs/ADAPTERS.md) |
| `ragantine bench run … --max-asr 0.1` | CI mode: exits non-zero when ASR or clean-accuracy thresholds are breached |

## Methodology

- **Splits:** defenses are tuned on dev only, and the published table is test only. Poison
  templates t01–t04 train D2; t05–t08 are the benchmark. They never overlap, and a test enforces it.
- **Protocol (§10):** D0, each defense alone, the full stack, and leave-one-out, for 3 models
  × 3 seeds. Seeds vary template choice, false answers, canaries, tiers and hiding technique.
- **Scoring:** rule-based first (false answer, canary, gold or alias, refusal), then an LLM
  judge from a different model family than the generator. 10% of judgements are exported
  for hand-labelling, and `bench report` prints agreement.
- **Metrics:** ASR (overall and per attack), conditional ASR (poison reached the prompt),
  clean accuracy, detection recall, ingestion false-quarantine, retrieval false-filter, p50/p95
  latency, and tokens and cost per model.

## Limitations

- **D5 trades clean accuracy for safety by design.** A claim needs ≥ 2 independent documents
  or one curated document. Correct answers whose only source is a partner, web or user-upload
  article become "I don't know". On the offline fixture this refused 5 of 7 clean questions;
  the real cost is whatever the benchmark measures. The knobs are `support_threshold` and
  `min_supporting_docs`, tuned on dev.
- **Poison is LLM-written from templates.** A detector trained on t01–t04 may still learn
  generator style. The held-out templates, the paraphrase set and the adaptive attacker probe
  this, but they do not remove it.
- **The 20% of poison filed under `partner`** simulates a compromised trusted source, so D3
  cannot be perfect by construction.
- **Model family diversity.** Groq retired the Llama models named in the spec, so both open
  models are gpt-oss, which is OpenAI-trained ([docs/DECISIONS.md](docs/DECISIONS.md)).
- **The CI smoke gate tests plumbing, not models.** It uses a scripted compliant model and
  heuristic stand-ins for D2 and D5.
- **CPU embedding is slow.** About 2 chunks/s on a laptop CPU, so a 5,000-article corpus takes
  hours to embed once. Embeddings are cached.

## Ethics

Public data only. Never point RAGantine at systems you do not own. Every attack ships with at
least one defense and its measured effect. Poison samples are labelled, kept under
`data/bench/`, and aimed at a local benchmark corpus: wrong answers and canary codes only, no
URLs, code or real-world harm.

## Project layout

```
ragantine/   library: ingest, index, retrieve, generate, defenses, quarantine, graph.py, api.py, cli.py, mcp_server.py
bench/       corpus, questions, poison, run/score/metrics/report, ablation, adaptive, smoke, dataset
configs/     default.yaml (run config) · models.yaml (model aliases, prices)
demo/        Gradio app, snapshot builder, HF Spaces packager
notebooks/   train_d2.ipynb (Colab)
docs/        SPEC.md · ARCHITECTURE.md · ADDING_A_DEFENSE.md · DECISIONS.md
```

## License

Code: Apache-2.0. RAGantine-Bench data: CC BY-SA 4.0 (Wikipedia-derived).
