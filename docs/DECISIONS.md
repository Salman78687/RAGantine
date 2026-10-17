# Decisions

Decisions not covered by docs/SPEC.md, one line of reasoning each.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-17 | SPEC.md §B9 points to the root `CLAUDE.md` instead of duplicating it | One source of truth; avoids the two copies drifting |
| 2026-10-17 | Benchmark-only models (`AttackType`, `BenchQuestion`, `PoisonSample`, `JudgedAnswer`) live in `bench/schemas.py`, not `ragantine/schemas.py` | Keeps ground-truth types physically outside the library; enforces the leakage rule (approved by owner) |
| 2026-10-17 | 80% coverage gate deferred until defenses/scoring had code; now enforced in CI (`--cov=ragantine.defenses --cov=bench.score --cov-fail-under=80`) | Gate could not pass on empty modules (approved by owner) |
| 2026-10-17 | `SourceTier` uses `StrEnum` instead of `(str, Enum)` | Same behaviour on 3.11; ruff UP042 prefers it |
| 2026-10-17 | Added `RAGANTINE_MODELS_CONFIG` env var (default `configs/models.yaml`) | Symmetric with `RAGANTINE_CONFIG`; lets tests and CI point at fixture models |
| 2026-10-17 | Optional `price_in_per_1m` / `price_out_per_1m` on model entries in `models.yaml` | §B5.6 says prices live in models.yaml but §B2 gives no schema |
| 2026-10-17 | Configs are strict (`extra="forbid"`) and validated (mixes sum to 1, `final_k <= fused_k`, aliases exist) | Typos in YAML should fail loudly, not silently change a benchmark |
| 2026-10-17 | `Chunk` validates `chunk_id == f"{doc_id}:{ordinal:04d}"` | Makes the §B3 id format an enforced contract instead of a comment |
| 2026-10-17 | OpenSearch pinned to `2.19.6` (latest 2.x on Docker Hub) | §B2 says pin the latest 2.x tag at build time |
| 2026-10-17 | torch resolved from the PyTorch CPU index via `[tool.uv.sources]` | §B2 asks for the CPU wheel; avoids ~3 GB of CUDA packages |
| 2026-10-17 | Ruff ignores E501 (formatter owns line length) and excludes `docs/` | `ruff format` already wraps code; ruff 0.17 otherwise reformats code blocks inside SPEC.md |
| 2026-10-17 | Minimal `ragantine config show` command in Phase 0 | The `ragantine` entry point needs at least one command; useful to inspect resolved config |
| 2026-10-17 | Compose `env_file` is `required: false`; `api` service overrides `OPENSEARCH_URL=http://opensearch:9200` | `localhost` from `.env` would not resolve to the OpenSearch container |
| 2026-10-17 | `ragantine/retrieve/` split into `bm25.py`, `dense.py`, `rrf.py`, `reranker.py`, `memory.py`; defenses named `d1_sanitizer.py`, `d2_classifier.py`, `d3_trust.py`, `d4_spotlight.py`, `d5_consistency.py`, `d6_anomaly.py` | §9 names only d1/d6 filenames; one module per concern |
| 2026-10-17 | CLAUDE.md work style changed: no per-phase approval gate, commit without asking | Owner instruction to build the whole project end-to-end |
| 2026-10-17 | Models replaced: `llama-3.3-70b-versatile`→`openai/gpt-oss-120b`, `llama-3.1-8b-instant`→`openai/gpt-oss-20b` (Groq), `gpt-4o-mini`→`gpt-6-luna` (OpenAI); aliases renamed `open-large`/`open-small`/`closed-small` | Groq shut the Llama models down for free/dev tiers on 2026-08-16; gpt-4o-mini retires 2027-01-14; §B1 says use current equivalents. Caveat: both open models are OpenAI-trained, so model-family diversity is lower than the spec intended |
| 2026-10-17 | `max_tokens` default 1024, `reasoning_effort: low` for gpt-oss; client sends `max_completion_tokens` and drops `temperature` if a model rejects it | gpt-oss/gpt-6 are reasoning models whose reasoning tokens consume the completion budget |
| 2026-10-17 | `judge_fallback` in models.yaml; judge = fallback when the generator under test is the judge model | §B1: judge must differ from the generator when possible |
| 2026-10-17 | Added `paths`, `corpus.scan_limit`, `chunking.min_tokens`, `embedding.dimension` to config | No magic numbers in code; scan_limit bounds the Wikipedia download (sampling = smallest sha1(seed+id) among the first scan_limit rows, not all 6M articles) |
| 2026-10-17 | One `IndexBackend` protocol implemented by `OpenSearchIndex` and `MemoryIndex` (retrieve/memory.py); quarantine store lives on the same backend | §B7.3 requires the fallback to share the retriever interface; lets tests and the HF demo run without OpenSearch |
| 2026-10-17 | Index mapping adds `ordinal`; quarantine mapping adds `reasons` (keyword), `quarantined_at`, `released_by`, `released_at` | Needed to rebuild `Chunk`/`QuarantineRecord` and to filter the review queue by reason; still no ground-truth field |
| 2026-10-17 | Retriever returns the reranked `fused_k` candidates; `filter_context` applies D2/D3/D6 then keeps `final_k` | D2 re-scores and D3 rebuilds the final set from the fused candidates (§B6) |
| 2026-10-17 | `Answer` gains optional `candidate_ids` and `draft` | Answer trace (FR-9) and the retrieval false-filter metric need the fused candidates; the demo's verified/rewritten badge needs the draft |
| 2026-10-17 | `Defense.on_verify` takes an optional `ctx: QueryContext` (model alias + regenerate callback) | D5 must regenerate through the same prompt path and have its tokens counted; passing it explicitly is safe under concurrent queries |
| 2026-10-17 | `parse_defense_ids("all")` = D1–D5 | D6 is a P2 stretch defense; "full stack" in §10 is D1–D5 |
| 2026-10-17 | Bench CLI commands (`corpus`, `index`, `bench`) live in `bench/cli.py`, registered via the `ragantine.cli` entry-point group | Keeps `ragantine/` free of any `bench` import (leakage rule) while exposing one `ragantine` CLI |
| 2026-10-17 | Ingestion-config indexes are named `<name>__<hash>`; the plain clean index keeps the base name | §B5.4; `ask`/`serve` default to the base index |
| 2026-10-17 | Chunker cuts a single "sentence" longer than the budget at word boundaries | Run-on lists/tables would otherwise exceed the embedder's 512-token window |
| 2026-10-17 | Tests use `httpx.MockTransport` via an injectable transport on `LLMClient` instead of respx | openai 3.x uses `httpx2`, which respx does not intercept |
| 2026-10-17 | `ragantine/fakes.py` ships FakeLLM, HashingEmbedder and OverlapReranker inside the library | Used by tests, the CI smoke benchmark and offline demos, not only tests |
| 2026-10-17 | Clean articles get `ingested_at` = the dump date (2023-11-01) | Deterministic corpus files across rebuilds |
| 2026-10-17 | Benchmark prompts live in `bench/prompts/*.txt` (question writer, paraphrase, rewordings, false answers, judge) | CLAUDE.md requires prompts in .txt files; keeps benchmark prompts out of the library |
| 2026-10-17 | Poison templates have a `[writer]` prompt plus a `[payload]` (`[payload_b]` for A6) inserted by code | Guarantees the canary/instruction is present and machine-checkable instead of trusting the writer model to include it |
| 2026-10-17 | A6 produces two `PoisonSample`s per target (`technique` = "a"/"b"), sharing qid and canary; only half "b" contains the canary, half "a" the instruction verb | Schema has one document per sample; each half is meaningless alone |
| 2026-10-17 | `PoisonSample` gains `seed` and `technique`; all seeds live in one `poison.jsonl`; D2 training poison goes to `poison_train.jsonl` | Seeds vary template, false answer, canary, tier and A5 technique (§B5.6); technique enables per-technique analysis |
| 2026-10-17 | False answers: writer proposes 3 candidates, seeded pick, retried with an avoid-list if all match the gold | Varies across seeds even at temperature 0; satisfies the reject-and-retry rule |
| 2026-10-17 | Training poison (t01–t04) is generated only for dev-split target questions | §B6: never on questions in any test split |
| 2026-10-17 | Paraphrases inherit their parent's split and attack, and are scored against the parent's poison | Avoids leaking test questions into dev; tests whether attacks generalise to rewording |
| 2026-10-17 | `bench questions --limit N` = N target questions; clean/paraphrase sizes scale by N / (6 × targets_per_attack) | Matches the Phase 2 acceptance ("--limit 24 → 4 targets per attack, clean questions and paraphrases") |
| 2026-10-17 | Poison docs get a neutral `source_uri` (`<tier>://uploads/<hash>`) and the article's title; `ingested_at` = 2026-01-01 | Nothing in the indexed document marks it as poison; newer than the clean dump, like a real upload |
| 2026-10-17 | Scorer matches answers order-free on normalised tokens (numbers, months, units unified) | "29 May 1953" vs "May 29, 1953", "8,849 m" vs "8849 metres"; inconclusive cases go to the judge |
| 2026-10-17 | `Workspace` protocol: `OpenSearchWorkspace` (index per ingestion config + poison set, manifest in `data/indexes/` for reuse) and `MemoryWorkspace` (tests, CI smoke) | Ingestion with D2 re-scores every chunk; reusing a finished index keeps repeat runs cheap |
| 2026-10-17 | A run ingests all poison of its split and seed, even for questions cut by `--limit` | §B5.4: "all attacks present at once, as in a real corpus" |
| 2026-10-17 | Detection recall counts `quarantine` and `filter` actions; D3 `downweight` is not detection | Downweighting is ranking, not a decision that a chunk is poison |
| 2026-10-17 | Reports pool k/n across seeds and bootstrap the pooled outcomes for the 95% CI | One CI per configuration and model, covering question and seed variation |
| 2026-10-17 | Chart: grouped bars of ASR, attack × {D0, each single defense, Full stack}; leave-one-out rows go to tables only | Seven series fit the validated 8-slot categorical palette; more would not be readable |
| 2026-10-17 | `bench run --max-asr/--min-clean-accuracy` exit 2 when breached | FR-11 CI mode |
| 2026-10-17 | Added `matplotlib` dependency | §B7.2 requires a bar chart PNG |
| 2026-10-17 | `build_defenses` takes the full `AppConfig` and injects `final_k`, `fused_k`, `models_dir`, `data_dir` into each defense's cfg | D3 caps the final set and D6 probes the fused set; values come from config, not code |
| 2026-10-17 | D1 compares NFKC(naive) with NFKC(sanitized); zero-width characters still count as hidden | Ordinary Unicode in clean Wikipedia (km², ligatures) must not trigger false quarantines |
| 2026-10-17 | D1 returns the naive text when it quarantines a document | Reviewers in the quarantine queue need to see what was hidden |
| 2026-10-17 | D3 returns the capped final set followed by remaining high-tier candidates as backfill | Later filters (D6) can drop chunks without starving the prompt |
| 2026-10-17 | D4 puts instructions in the system message and wrapped documents + question in the user message; `</document>` inside text is defanged | "keeps system instructions separate"; a document must not be able to close its own tag |
| 2026-10-17 | D2 scorer and D5 NLI are pluggable (`InjectionScorer`, `NLIModel`); D2 caches scores by content_hash; `defenses.d2.variant` = finetuned \| zeroshot (`protectai/deberta-v3-base-prompt-injection-v2`, Apache-2.0) | Zero-shot baseline vs fine-tuned (§B6 step 1/2); offline stand-ins in tests and CI |
| 2026-10-17 | D5 checks short sentences (< 6 words) as "<question> <sentence>" and treats "I don't know" as supported | NLI cannot judge a bare "1953."; refusing is never an unsupported claim |
| 2026-10-17 | D5 implements the spec's support rule literally (≥2 docs or 1 curated chunk) | Expect a clean-accuracy cost on partner/web-only answers; it is measured, not hidden |
| 2026-10-17 | `KeywordInjectionScorer` and `LexicalNLI` (ragantine/fakes.py) are heuristic stand-ins used only in tests and the CI smoke run | CI cannot download the classifier/NLI models; results from them are never reported as D2/D5 numbers |
