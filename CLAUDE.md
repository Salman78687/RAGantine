# RAGantine — instructions for Claude Code

RAGantine benchmarks and defends RAG pipelines against knowledge-base poisoning.
Read docs/SPEC.md before any task: Part A (sections 1–13) is the PRD (what and why),
Part B (sections B0–B10) is the build spec (how).
Part B wins on implementation details; Part A wins on goals and scope.

## Work style
- Build phases in order (SPEC section B10). The owner has authorised building straight through:
  no approval gate between phases, and commit each concern without asking.
- After each phase: run `make check` and verify that phase's acceptance checks (those that need
  no paid API calls); report anything that could not be verified and why.
- Record any decision not covered by the spec in docs/DECISIONS.md (date, decision, one-line reason).
- If the spec is ambiguous, pick the reading closest to Part A's goals and log it in DECISIONS.md;
  ask only when the choice is costly to reverse.

## Commands
- Install: `uv sync` · Lint/types/tests: `make check` · OpenSearch: `make up`
- Single test: `uv run pytest tests/unit/test_x.py -k name`
- Never run `bench run` / `bench ablation` without `--limit` unless the user asks (they cost money).

## Code rules
- Python 3.11, full type hints, mypy clean. Pydantic v2 models from ragantine/schemas.py at every module boundary.
- Config comes from configs/*.yaml via pydantic-settings; no magic numbers in code.
- Every defense subclasses `Defense`, registers by id, and returns `DefenseDecision`s with a reason.
- All LLM calls go through `LLMClient` (cache, retries, token accounting). No direct SDK calls elsewhere.
- Prompts live in .txt files under prompts/ or bench/poison/templates/, never inline strings.
- Tests ship with every feature. Unit tests must not hit the network or OpenSearch (use FakeLLM).

## Hard rules
- LEAKAGE: nothing in ragantine/ may import from bench/ or read data/bench/poison.jsonl.
  Ground-truth fields (is_poison, attack, canary) never go into OpenSearch.
- Train/test separation: poison templates t01–t04 are for D2 training only; t05–t08 for the benchmark.
  Tune thresholds on the dev split only.
- Never fabricate or edit results. Numbers in README/report come only from results/<run_id>/metrics.json.
- Secrets only from .env; never print, log, commit, or hard-code keys. Keep .env.example updated.
- Poison content stays harmless: wrong answers or canary strings only, no URLs, code or real-world harm.
- Public data only. No employer, customer or private data anywhere in this repo.

## Layout
ragantine/ (library: ingest, index, retrieve, generate, defenses, quarantine, graph.py, api.py, cli.py)
bench/ (corpus, questions, poison, run.py, score.py, report.py) · configs/ · demo/ · notebooks/ · tests/ · results/

## Git
- Small commits, one concern each, conventional messages (feat:, fix:, test:, docs:, chore:).
- Never commit data/, models/, results/*/answers.jsonl, or .env.
