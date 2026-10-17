"""Benchmark commands contributed to the ``ragantine`` CLI (SPEC §B7.2).

Registered through the ``ragantine.cli`` entry-point group so that ``ragantine/`` never
imports ``bench`` (leakage rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ragantine.config import Settings, load_config

corpus_app = typer.Typer(help="Build the clean Wikipedia corpus.", no_args_is_help=True)
index_app = typer.Typer(help="Build OpenSearch indexes.", no_args_is_help=True)
bench_app = typer.Typer(
    help="RAGantine-Bench: questions, poison, runs, reports.", no_args_is_help=True
)

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Run config YAML (default: $RAGANTINE_CONFIG).")
]


def register(app: typer.Typer) -> None:
    app.add_typer(corpus_app, name="corpus")
    app.add_typer(index_app, name="index")
    app.add_typer(bench_app, name="bench")


@corpus_app.command("build")
def corpus_build(
    n_articles: Annotated[int | None, typer.Option(help="Override corpus.n_articles.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Stream Wikipedia, sample deterministically, chunk and embed (SPEC §B5.1)."""
    from bench.corpus.build import build_corpus
    from ragantine.retrieve.dense import SentenceTransformerEmbedder

    cfg = load_config(config)
    report = build_corpus(cfg, SentenceTransformerEmbedder(cfg.embedding), n_articles=n_articles)
    typer.echo(report.model_dump_json(indent=2))


@index_app.command("build")
def index_build(
    defenses: Annotated[str, typer.Option(help="Ingestion defenses, e.g. d1,d2.")] = "none",
    with_poison: Annotated[
        bool, typer.Option("--with-poison/--no-poison", help="Also ingest the test poison docs.")
    ] = False,
    drop: Annotated[bool, typer.Option(help="Delete the target indexes first.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Ingest the corpus (and optionally poison docs) into OpenSearch."""
    from bench.indexing import build_index

    cfg = load_config(config)
    report, names = build_index(cfg, Settings(), defenses, with_poison=with_poison, drop=drop)
    typer.echo(f"index={names[0]} quarantine={names[1]}")
    typer.echo(report.model_dump_json(indent=2, exclude={"decisions", "quarantined_chunk_ids"}))


YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation prompt.")]
LimitOpt = Annotated[int | None, typer.Option(help="Smoke-run size (see command help).")]


def _confirm(text: str, yes: bool) -> None:
    typer.echo(text)
    if not yes and not typer.confirm("Proceed with these LLM calls?", default=False):
        raise typer.Exit(1)


@bench_app.command("questions")
def bench_questions(
    limit: LimitOpt = None,
    yes: YesOpt = False,
    writer: Annotated[str | None, typer.Option(help="Question/poison writer model alias.")] = None,
    model: Annotated[str | None, typer.Option(help="Generator used to filter questions.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Generate + filter target/clean/paraphrase questions (SPEC §B5.2).

    --limit N makes N target questions (N/6 per attack) and scales clean/paraphrase sets.
    """
    import asyncio
    import json

    from bench.corpus.build import corpus_paths
    from bench.cost import CallEstimate, estimate
    from bench.io import read_jsonl, write_jsonl
    from bench.questions.generate import generate_questions, set_sizes
    from bench.store import bench_paths
    from ragantine.config import load_models_config
    from ragantine.graph import Pipeline
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    models = load_models_config()
    writer_alias = writer or models.defaults.poison_writer
    generator = model or models.defaults.generator
    judge = models.judge_for(generator)
    sizes = set_sizes(cfg, limit)
    n_cand = 2 * (sizes.targets + sizes.clean)
    _confirm(
        estimate(
            models,
            [
                CallEstimate(
                    purpose="write question",
                    model_alias=writer_alias,
                    calls=n_cand,
                    prompt_tokens_each=520,
                    completion_tokens_each=250,
                ),
                CallEstimate(
                    purpose="answer on clean index",
                    model_alias=generator,
                    calls=n_cand,
                    prompt_tokens_each=2200,
                    completion_tokens_each=250,
                ),
                CallEstimate(
                    purpose="judge (when rules inconclusive)",
                    model_alias=judge,
                    calls=n_cand // 3,
                    prompt_tokens_each=200,
                    completion_tokens_each=80,
                ),
                CallEstimate(
                    purpose="paraphrase",
                    model_alias=writer_alias,
                    calls=sizes.paraphrases,
                    prompt_tokens_each=80,
                    completion_tokens_each=150,
                ),
            ],
        ).render(),
        yes,
    )
    chunks = list(read_jsonl(corpus_paths(cfg)["chunks"], Chunk))
    pipeline = Pipeline.from_config(cfg, models, Settings())
    questions, report = asyncio.run(
        generate_questions(
            cfg, pipeline, chunks, writer_alias, generator, judge, limit, log=typer.echo
        )
    )
    paths = bench_paths(cfg)
    write_jsonl(paths["questions"], questions)
    out = {**report.model_dump(), "rejection_rate": round(report.rejection_rate, 4)}
    paths["questions_report"].write_text(json.dumps(out, indent=2))
    typer.echo(json.dumps(out, indent=2))
    typer.echo(f"rejection rate: {report.rejection_rate:.1%} ({report.candidates} candidates)")
    typer.echo(f"wrote {len(questions)} questions -> {paths['questions']}")


@bench_app.command("poison")
def bench_poison(
    limit: LimitOpt = None,
    yes: YesOpt = False,
    writer: Annotated[str | None, typer.Option(help="Poison writer model alias.")] = None,
    seeds: Annotated[
        str | None, typer.Option(help="Comma list of seeds (default: bench.seeds).")
    ] = None,
    train: Annotated[
        bool,
        typer.Option(
            "--train/--no-train", help="Also write D2 training poison (t01-t04, dev only)."
        ),
    ] = True,
    config: ConfigOpt = None,
) -> None:
    """Generate poison docs for A1-A6 (SPEC §B5.3). --limit N: first N target questions."""
    import asyncio

    from bench.corpus.build import corpus_paths
    from bench.cost import CallEstimate, estimate
    from bench.io import read_jsonl, write_jsonl
    from bench.poison.generator import generate_poison, write_poison_files
    from bench.store import bench_paths, load_questions
    from ragantine.config import load_models_config
    from ragantine.generate.llm import LLMClient
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    models = load_models_config()
    alias = writer or models.defaults.poison_writer
    seed_list = [int(s) for s in seeds.split(",")] if seeds else list(cfg.bench.seeds)
    questions = load_questions(cfg)
    targets = [q for q in questions if q.qset == "target"]
    if limit is not None:
        keep = {q.qid for q in targets[:limit]}
        questions = [q for q in questions if q.qid in keep]
        targets = [q for q in targets if q.qid in keep]
    n_dev = sum(q.split == "dev" for q in targets)
    per_target = 3  # body + false answer or rewordings/second half, on average
    _confirm(
        estimate(
            models,
            [
                CallEstimate(
                    purpose="test poison",
                    model_alias=alias,
                    calls=per_target * len(targets) * len(seed_list),
                    prompt_tokens_each=180,
                    completion_tokens_each=450,
                ),
                CallEstimate(
                    purpose="train poison (D2)",
                    model_alias=alias,
                    calls=per_target * n_dev * (1 if train else 0),
                    prompt_tokens_each=180,
                    completion_tokens_each=450,
                ),
            ],
        ).render(),
        yes,
    )
    subjects = {c.chunk_id: c.title for c in read_jsonl(corpus_paths(cfg)["chunks"], Chunk)}
    llm = LLMClient(models, Settings(), cache_dir=cfg.paths.llm_cache_dir)
    paths = bench_paths(cfg)
    samples, report = asyncio.run(
        generate_poison(cfg, llm, alias, questions, subjects, seed_list, "test", typer.echo)
    )
    write_jsonl(paths["poison"], samples)
    write_poison_files(samples, paths["poison_docs"])
    typer.echo(f"test poison: {report.model_dump_json()} -> {paths['poison']}")
    if train:
        tsamples, treport = asyncio.run(
            generate_poison(cfg, llm, alias, questions, subjects, [cfg.seed], "train", typer.echo)
        )
        write_jsonl(paths["poison_train"], tsamples)
        typer.echo(f"train poison: {treport.model_dump_json()} -> {paths['poison_train']}")
    typer.echo(f"rendered files -> {paths['poison_docs']}")
