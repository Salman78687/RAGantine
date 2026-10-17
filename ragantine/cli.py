"""Typer CLI, entry point ``ragantine`` (SPEC §B7.2).

A thin layer over the library. Library commands live here (``ask``, ``retrieve``,
``serve``, ``quarantine``, ``defenses``, ``config``). Benchmark commands (``corpus``,
``index``, ``bench``) are contributed by the ``bench`` package through the
``ragantine.cli`` entry-point group, so this package never imports benchmark code
(leakage rule, SPEC §B3).
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated

import typer
import yaml

from ragantine.config import Settings, load_config, load_models_config

app = typer.Typer(
    help="RAGantine: benchmark and defend RAG against knowledge-base poisoning.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # never dump locals: they can hold API keys
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
quarantine_app = typer.Typer(help="Review the quarantine store.", no_args_is_help=True)
defenses_app = typer.Typer(help="Defense utilities.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(quarantine_app, name="quarantine")
app.add_typer(defenses_app, name="defenses")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Run config YAML (default: $RAGANTINE_CONFIG).")
]
DefensesOpt = Annotated[
    str, typer.Option("--defenses", help="Comma list of d1..d6, 'all' (D1-D5) or 'none'.")
]
ModelOpt = Annotated[str | None, typer.Option("--model", help="Model alias from models.yaml.")]
IndexOpt = Annotated[
    str | None, typer.Option("--index", help="Main index name (default: config index.name).")
]


@config_app.command("show")
def config_show(
    config: ConfigOpt = None,
    models: Annotated[Path | None, typer.Option(help="Models YAML.")] = None,
) -> None:
    """Print the resolved, validated configuration (no secrets)."""
    resolved = {
        "config": load_config(config).model_dump(mode="json"),
        "models": load_models_config(models).model_dump(mode="json"),
    }
    typer.echo(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))


def _pipeline(config: Path | None, index: str | None, quarantine_index: str | None = None):  # type: ignore[no-untyped-def]
    from ragantine.graph import Pipeline

    cfg = load_config(config)
    qname = quarantine_index
    if index and not qname:
        qname = index.replace(cfg.index.name, cfg.index.quarantine_name, 1)
    return Pipeline.from_config(cfg, load_models_config(), Settings(), index, qname)


@app.command()
def ask(
    question: str,
    defenses: DefensesOpt = "none",
    model: ModelOpt = None,
    config: ConfigOpt = None,
    index: IndexOpt = None,
    show_context: Annotated[bool, typer.Option(help="Print the chunks given to the LLM.")] = False,
) -> None:
    """Answer a question with citations through the query graph."""
    from ragantine.graph import run_query

    answer = run_query(question, _pipeline(config, index), defenses, model)
    typer.echo(answer.text)
    typer.echo("")
    for c in answer.citations:
        typer.echo(f"  [{c.chunk_id}] {c.title} ({c.source_tier.value})")
    for d in answer.decisions:
        typer.echo(f"  {d.defense_id} {d.action} {d.target_id}: {d.reason}")
    if show_context:
        for h in answer.retrieved:
            typer.echo(
                f"\n--- {h.chunk.chunk_id} ({h.chunk.source_tier.value}) score={h.final_score:.3f}"
            )
            typer.echo(h.chunk.text)
    typer.echo(
        f"\nmodel={answer.model_alias} latency={answer.latency_ms}ms "
        f"tokens={answer.prompt_tokens}+{answer.completion_tokens}"
    )


@app.command()
def retrieve(
    question: str,
    config: ConfigOpt = None,
    index: IndexOpt = None,
    k: Annotated[int, typer.Option(help="How many candidates to print.")] = 10,
) -> None:
    """Show hybrid retrieval candidates for a question (no LLM call)."""
    pipeline = _pipeline(config, index)
    for h in pipeline.retriever.search(question)[:k]:
        typer.echo(
            f"{h.final_score:8.3f}  bm25={h.bm25_rank!s:>4} knn={h.knn_rank!s:>4}  "
            f"{h.chunk.chunk_id}  {h.chunk.title} ({h.chunk.source_tier.value})"
        )


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
) -> None:
    """Run the FastAPI service (SSE /ask, quarantine review, runs)."""
    import uvicorn

    uvicorn.run("ragantine.api:create_app", factory=True, host=host, port=port)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio (tools: safe_retrieve, audit_corpus)."""
    from ragantine.mcp_server import main

    main()


@quarantine_app.command("list")
def quarantine_list(
    status: Annotated[str | None, typer.Option(help="quarantined | released")] = None,
    reason: Annotated[str | None, typer.Option(help="Reason prefix filter.")] = None,
    limit: int = 50,
    config: ConfigOpt = None,
    index: IndexOpt = None,
) -> None:
    """List quarantined chunks with their reasons."""
    items, total = _pipeline(config, index).backend.list_quarantine(status, reason, limit, 0)
    typer.echo(f"{total} record(s)")
    for r in items:
        reasons = "; ".join(f"{d.defense_id}:{d.reason}" for d in r.decisions)
        typer.echo(f"{r.status:11}  {r.chunk.chunk_id}  {r.chunk.title}  [{reasons}]")


@quarantine_app.command("release")
def quarantine_release(
    chunk_id: str,
    released_by: Annotated[str, typer.Option(help="Who approved the release (logged).")],
    config: ConfigOpt = None,
    index: IndexOpt = None,
) -> None:
    """Release a quarantined chunk back into the main index."""
    record = _pipeline(config, index).backend.release(chunk_id, released_by)
    if record is None:
        typer.echo(f"{chunk_id} is not in quarantine", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@defenses_app.command("build-d6")
def defenses_build_d6(
    config: ConfigOpt = None,
    index: IndexOpt = None,
    n_titles: Annotated[int, typer.Option(help="Chunks sampled for probe-query titles.")] = 2000,
) -> None:
    """Build the D6 hub list for an index (re-run after each ingestion batch)."""
    from ragantine.defenses.base import build_defenses
    from ragantine.defenses.d6_anomaly import build_hubs, hub_file

    pipeline = _pipeline(config, index)
    cfg = pipeline.cfg
    d6 = build_defenses(["d6"], cfg)[0]
    titles = sorted({c.title for c in pipeline.backend.sample(n_titles, cfg.seed)})
    hubs = build_hubs(pipeline.backend, pipeline.retriever.embedder, titles, d6.cfg, cfg.seed)
    path = hub_file(d6.cfg, index or cfg.index.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hubs, indent=2))
    typer.echo(f"{len(hubs)} hub chunk(s) above z={d6.cfg['hub_z_threshold']} -> {path}")


def _load_plugins() -> None:
    for ep in entry_points(group="ragantine.cli"):
        ep.load()(app)


_load_plugins()


if __name__ == "__main__":
    app()
