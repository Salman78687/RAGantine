# RAGantine

**Quarantine for retrieval: benchmarking and defending RAG against knowledge-base poisoning.**

> **Finding (v0.1):** undefended, planted documents won 59.5% (Gemma 4 26B) and 69.0% (Gemini 3.5 Flash-Lite) of poisoned questions.
> Spotlighting with source labels (D4) alone cut that to 7.1% and 7.1%,
> with clean accuracy 100.0% and 100.0%. Every number below is generated from
> `results/<run_id>/metrics.json` by `ragantine bench report`; none is typed by hand.

Plant poison in a RAG corpus, measure how often it wins, then quarantine it, and publish the numbers.

## Results

| Configuration | Model | ASR (%) | Clean accuracy (%) | Detection recall (%) | False quarantine (%) | p95 latency (ms) | Seeds |
|---|---|---|---|---|---|---|---|
| D0 Baseline | gemini-lite | 69.0 [54.8–83.3] | 95.2 [85.7–100.0] | 0.0 | 0.0 | 11139 | 1 |
| + D1 Sanitizer | gemini-lite | 50.0 [33.3–64.3] | 95.2 [85.7–100.0] | 15.1 | 0.0 | 143926 | 1 |
| + D3 Source trust | gemini-lite | 45.2 [31.0–61.9] | 95.2 [85.7–100.0] | 0.0 | 0.0 | 135742 | 1 |
| + D4 Spotlighting | gemini-lite | 7.1 [0.0–16.7] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 273551 | 1 |
| + D5 Consistency check | gemini-lite | 4.8 [0.0–11.9] | 28.6 [9.5–47.6] | 0.0 | 0.0 | 558338 | 1 |
| + D2 Injection classifier (D2 zero-shot) | gemini-lite | 66.7 [52.4–81.0] | 95.2 [85.7–100.0] | 5.6 | 1.9 | 17447 | 1 |
| Full stack (D2 zero-shot) | gemini-lite | 0.0 [0.0–0.0] | 38.1 [19.0–57.1] | 20.8 | 1.9 | 416048 | 1 |
| D0 Baseline | gemma-small | 59.5 [45.2–76.2] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 12642 | 1 |
| + D1 Sanitizer | gemma-small | 47.6 [33.3–61.9] | 100.0 [100.0–100.0] | 15.1 | 0.0 | 306674 | 1 |
| + D3 Source trust | gemma-small | 35.7 [23.8–50.0] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 285964 | 1 |
| + D4 Spotlighting | gemma-small | 7.1 [0.0–16.7] | 100.0 [100.0–100.0] | 0.0 | 0.0 | 898389 | 1 |
| + D5 Consistency check | gemma-small | 7.1 [0.0–16.7] | 19.0 [4.8–38.1] | 0.0 | 0.0 | 428184 | 1 |
| + D2 Injection classifier (D2 zero-shot) | gemma-small | 59.5 [45.2–76.2] | 100.0 [100.0–100.0] | 5.6 | 1.9 | 467994 | 1 |
| Full stack (D2 zero-shot) | gemma-small | 0.0 [0.0–0.0] | 42.9 [23.8–61.9] | 20.8 | 1.9 | 747664 | 1 |

**ASR by attack type (%)**

| Configuration | Model | A1 | A2 | A3 | A4 | A5 | A6 | Paraphrase |
|---|---|---|---|---|---|---|---|---|
| D0 Baseline | gemini-lite | 57.1 | 100.0 | 71.4 | 42.9 | 100.0 | 42.9 | — |
| + D1 Sanitizer | gemini-lite | 57.1 | 100.0 | 85.7 | 28.6 | 0.0 | 28.6 | — |
| + D3 Source trust | gemini-lite | 28.6 | 71.4 | 71.4 | 14.3 | 57.1 | 28.6 | — |
| + D4 Spotlighting | gemini-lite | 28.6 | 0.0 | 14.3 | 0.0 | 0.0 | 0.0 | — |
| + D5 Consistency check | gemini-lite | 0.0 | 0.0 | 0.0 | 0.0 | 28.6 | 0.0 | — |
| + D2 Injection classifier (D2 zero-shot) | gemini-lite | 57.1 | 100.0 | 71.4 | 42.9 | 85.7 | 42.9 | — |
| Full stack (D2 zero-shot) | gemini-lite | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — |
| D0 Baseline | gemma-small | 57.1 | 85.7 | 85.7 | 57.1 | 71.4 | 0.0 | — |
| + D1 Sanitizer | gemma-small | 57.1 | 85.7 | 85.7 | 57.1 | 0.0 | 0.0 | — |
| + D3 Source trust | gemma-small | 57.1 | 42.9 | 57.1 | 28.6 | 28.6 | 0.0 | — |
| + D4 Spotlighting | gemma-small | 28.6 | 0.0 | 0.0 | 14.3 | 0.0 | 0.0 | — |
| + D5 Consistency check | gemma-small | 14.3 | 0.0 | 0.0 | 0.0 | 28.6 | 0.0 | — |
| + D2 Injection classifier (D2 zero-shot) | gemma-small | 57.1 | 85.7 | 85.7 | 57.1 | 71.4 | 0.0 | — |
| Full stack (D2 zero-shot) | gemma-small | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — |

ASR = attack success rate on poisoned **test** questions (n=42 per run; clean n=21), one seed, 95% bootstrap CIs in brackets. Models: `gemma-small` = Gemma 4 26B-A4B (open weights), `gemini-lite` = Gemini 3.5 Flash-Lite (closed), both on the Gemini API free tier; judge = gpt-oss-120b on Groq. p95 latency includes client-side free-tier pacing. Reproduce with `scripts/v01_free.sh` (see Quick start). Charts and per-attack CSV: [docs/results/](docs/results/).

![ASR by attack, Gemma 4 26B](docs/results/asr_by_attack_gemma-small.png)
![ASR by attack, Gemini 3.5 Flash-Lite](docs/results/asr_by_attack_gemini-lite.png)

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
cp .env.example .env             # then put your GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY in .env
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

**Run it for free.** The v0.1 results above use only free tiers: a free Gemini API key
(aistudio.google.com) and a free Groq key. `configs/v01-free.yaml` is the same pipeline at small
scale (300 articles, 60 target and 30 clean questions, one seed) and `configs/models-v01-free.yaml`
picks models that fit the free daily limits. Then:

```bash
export RAGANTINE_MODELS_CONFIG=configs/models-v01-free.yaml
uv run ragantine bench questions --config configs/v01-free.yaml --yes
scripts/v01_free.sh     # poison -> D2 threshold on dev -> 7 configs x 2 models -> report
```

When a daily quota runs out, a command stops with exit code 3. Rerun it after the reset: finished
LLM calls and reranker scores are cached, so it resumes where it stopped.

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
  **v0.1 (published above) is smaller:** 2 models × 7 configs × 1 seed, 42 poisoned and 21 clean
  test questions per run, with D2 in its zero-shot variant. Leave-one-out runs are deferred to fit
  free-tier daily limits.
- **Scoring:** rule-based first (false answer, canary, gold or alias, refusal), then an LLM
  judge from a different model family than the generator. 10% of judgements are exported
  for hand-labelling, and `bench report` prints agreement.
- **Metrics:** ASR (overall and per attack), conditional ASR (poison reached the prompt),
  clean accuracy, detection recall, ingestion false-quarantine, retrieval false-filter, p50/p95
  latency, and tokens and cost per model.

## Limitations

- **D5 trades clean accuracy for safety by design.** A claim needs ≥ 2 independent documents
  or one curated document. Correct answers whose only source is a partner, web or user-upload
  article become "I don't know". In v0.1 this cost most clean questions on both models (see
  the table), which also drags down the full stack. The knobs are `support_threshold` and
  `min_supporting_docs`, tuned on dev; v0.1 uses the spec defaults.
- **D4's gain depends on visible trust labels.** Spotlighting tags each passage with its source
  tier, and both models preferred curated passages over conflicting lower-tier ones. Poison
  that lands in a curated source would not be discounted this way.
- **v0.1 is small.** One seed and 7 questions per attack type, so per-attack numbers have wide
  confidence intervals. Treat differences of a few points as noise.
- **D2 is zero-shot in v0.1.** The off-the-shelf classifier barely fires on these templates
  (most attacks are plain false facts, not instructions). The fine-tuned variant
  ([notebooks/train_d2.ipynb](notebooks/train_d2.ipynb)) is not yet in the published table.
- **Latency in v0.1 is not representative.** Requests were paced client-side to stay under
  free-tier per-minute limits, so p95 latency mostly measures waiting.
- **Poison is LLM-written from templates.** A detector trained on t01–t04 may still learn
  generator style. The held-out templates, the paraphrase set and the adaptive attacker probe
  this, but they do not remove it.
- **The 20% of poison filed under `partner`** simulates a compromised trusted source, so D3
  cannot be perfect by construction.
- **Models.** v0.1 compares Gemma 4 26B-A4B (open weights) with Gemini 3.5 Flash-Lite
  (closed), both on the Gemini API free tier, judged by gpt-oss-120b on Groq (a third model
  family). The default `configs/models.yaml` still targets the spec's gpt-oss + OpenAI set
  ([docs/DECISIONS.md](docs/DECISIONS.md)).
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

## Contributing

New attacks, defenses, adapters and benchmark results are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
and [SECURITY.md](SECURITY.md).

## License

Code: Apache-2.0. RAGantine-Bench data: CC BY-SA 4.0 (Wikipedia-derived).
