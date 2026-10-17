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
