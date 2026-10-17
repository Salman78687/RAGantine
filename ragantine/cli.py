"""Typer CLI, entry point ``ragantine`` (SPEC §B7.2).

Thin layer over the library; commands are added phase by phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from ragantine.config import load_config, load_models_config

app = typer.Typer(help="RAGantine: benchmark and defend RAG against knowledge-base poisoning.")
config_app = typer.Typer(help="Inspect configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Annotated[
        Path | None, typer.Option(help="Run config YAML (default: $RAGANTINE_CONFIG).")
    ] = None,
    models: Annotated[Path | None, typer.Option(help="Models YAML.")] = None,
) -> None:
    """Print the resolved, validated configuration (no secrets)."""
    resolved = {
        "config": load_config(config).model_dump(mode="json"),
        "models": load_models_config(models).model_dump(mode="json"),
    }
    typer.echo(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    app()
