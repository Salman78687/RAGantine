# Benchmark your own pipeline (FR-10)

**LLMs:** any OpenAI-compatible endpoint, such as vLLM, Ollama, Azure OpenAI or a gateway, is
one entry in `configs/models.yaml`:

```yaml
providers:
  local: { base_url: http://localhost:8000/v1, api_key_env: LOCAL_LLM_KEY }
models:
  my-model: { provider: local, model: meta-llama/Llama-3.1-8B-Instruct, price_in_per_1m: 0, price_out_per_1m: 0 }
```

Then pass `--model my-model` to `bench run` or `bench ablation`.

**Retrievers:** wrap yours with `ragantine.adapters.AdapterRetriever` and run the benchmark
through `bench.workspace.AdapterWorkspace`:

```python
import asyncio
from bench.corpus.build import load_articles
from bench.run import RunSpec, execute_run
from bench.store import load_poison, load_questions
from bench.workspace import AdapterWorkspace
from ragantine.adapters import AdapterRetriever
from ragantine.config import Settings, load_config, load_models_config
from ragantine.generate.llm import LLMClient

cfg, models = load_config(), load_models_config()
retriever = AdapterRetriever.from_langchain(my_vectorstore.as_retriever(search_kwargs={"k": 5}))
# or AdapterRetriever.from_llamaindex(index.as_retriever()) / AdapterRetriever(my_search_fn)

def load_documents(docs):  # REPLACE your store's contents with these RawDocuments
    my_vectorstore.delete_all()
    my_vectorstore.add_texts([str(d.raw) for d in docs],
                             metadatas=[{"doc_id": d.doc_id, "title": d.title, "source_tier": d.source_tier.value} for d in docs])

ws = AdapterWorkspace(cfg, LLMClient(models, Settings(), cfg.paths.llm_cache_dir), retriever, load_documents, load_articles(cfg))
spec = RunSpec(run_id="my-pipeline-d0", defenses=[], model_alias="open-large", sets=["target", "clean"], split="dev", seed=cfg.seed)
metrics, out = asyncio.run(execute_run(cfg, models, spec, load_questions(cfg), load_poison(cfg), ws))
```

Notes:

- Keep `doc_id` and `source_tier` in your metadata. D3 trust scoring and the benchmark's poison
  tracking (conditional ASR) depend on them. A hit with unknown provenance counts as `web`.
- Ingestion defenses (D1, D2) screen documents before `load_documents` receives them. A document
  with a quarantined chunk is withheld whole, because RAGantine cannot see your chunking.
- Retrieval and generation defenses (D2, D3, D4, D5) run on your hits unchanged. D6 needs
  RAGantine's own index statistics, so it is not available through adapters.
- Detection recall is chunk-based and therefore reported only for RAGantine's own index.
  ASR, conditional ASR and clean accuracy work for adapters.
