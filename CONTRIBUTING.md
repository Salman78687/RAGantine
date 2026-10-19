# Contributing to RAGantine

Thanks for helping make RAG pipelines harder to poison. Contributions of every size are
welcome: new attacks, new defenses, adapters for other stacks, better docs, bug reports, and
benchmark results from models we have not run.

## Set up

```bash
git clone <your fork> && cd RAGantine
make setup            # uv sync + pre-commit hooks (Python 3.11 via uv)
make check            # lint, types, ~360 tests; no API keys, network or OpenSearch needed
uv run ragantine bench smoke    # offline end-to-end check of the defense pipeline
```

For anything that calls a model, copy `.env.example` to `.env` and add a key. A free
[Groq](https://console.groq.com) key is enough for small runs, using `--config configs/v01-free.yaml`.

## Ground rules

These keep the benchmark trustworthy. PRs that break them will not be merged.

- **No leakage.** Nothing in `ragantine/` may import from `bench/` or read poison ground truth.
  `tests/unit/test_leakage.py` enforces this.
- **Train/test separation.** Poison templates `t01–t04` are for training detectors only;
  `t05–t08` are the benchmark. Tune thresholds on the dev split only.
- **Honest numbers.** Results come only from `results/<run_id>/metrics.json` produced by
  `ragantine bench run`/`ablation`. Never edit numbers by hand. Report failures too.
- **Harmless poison.** Attack goals stay as wrong answers or canary codes. No URLs, code, real
  products or real-world harmful instructions.
- **Public data only.** Never add private, customer or employer data.
- **No secrets.** Keys live in `.env` (git-ignored). Never commit, log or print them.

## Adding things

- **A defense:** see [docs/ADDING_A_DEFENSE.md](docs/ADDING_A_DEFENSE.md). Subclass `Defense`,
  register an id, return a `DefenseDecision` with a reason for every block, and add tests.
  Coverage on `ragantine/defenses` must stay ≥ 80%.
- **An attack:** add `A7` to the attack types, write ≥ 8 templates under
  `bench/poison/templates/` (`A7-t01…t08`, train/test split as above), extend the generator
  and the scorer, and add at least one defense that measurably helps against it.
- **Your own retriever or model:** see [docs/ADAPTERS.md](docs/ADAPTERS.md). Results PRs are
  welcome; include the run's `config.yaml` and `metrics.json`.

## Pull requests

- One concern per PR. Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Tests ship with the change; `make check` must pass.
- Record decisions the spec doesn't cover in `docs/DECISIONS.md`, with one line of reasoning.
- If you change a prompt, re-record the CI fixture: `uv run ragantine bench smoke --build-fixture`.

By contributing you agree that your code is licensed under Apache-2.0, and benchmark data under
CC BY-SA 4.0.
