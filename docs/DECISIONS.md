# Decisions

Decisions not covered by docs/SPEC.md, one line of reasoning each.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-17 | SPEC.md §B9 points to the root `CLAUDE.md` instead of duplicating it | One source of truth; avoids the two copies drifting |
| 2026-10-17 | Benchmark-only models (`AttackType`, `BenchQuestion`, `PoisonSample`, `JudgedAnswer`) live in `bench/schemas.py`, not `ragantine/schemas.py` | Keeps ground-truth types physically outside the library; enforces the leakage rule (approved by owner) |
| 2026-10-17 | 80% coverage gate is reported but not enforced until defenses/scoring have code (Phase 3/4) | Gate cannot pass on empty modules (approved by owner) |
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
