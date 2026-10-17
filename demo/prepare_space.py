"""Assemble a Hugging Face Space (Gradio SDK) in build/space/.

    uv run python demo/build_snapshot.py      # once, needs data/ from the benchmark
    uv run python demo/prepare_space.py
    # then: cd build/space && git init && git remote add space https://huggingface.co/spaces/<you>/ragantine && ...

The Space gets the ragantine library (no bench/ code, no ground truth), demo/app.py as
app.py, the snapshot, configs and a requirements.txt. Add GROQ_API_KEY / OPENAI_API_KEY as
Space secrets; never commit them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"
REQUIREMENTS = [
    "gradio>=6", "pydantic>=2", "pydantic-settings", "pyyaml", "numpy", "openai", "tenacity", "diskcache",
    "httpx", "langgraph", "sentence-transformers", "transformers", "torch", "rank-bm25", "tiktoken",
    "beautifulsoup4", "lxml", "markdown-it-py", "pymupdf", "langfuse", "opensearch-py", "sse-starlette", "fastapi",
]  # fmt: skip
SPACE_README = """---
title: RAGantine
emoji: 🧪
colorFrom: gray
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# RAGantine demo

The same question through a baseline RAG pipeline and through RAGantine's layered defenses,
side by side. Poisoned documents are clearly labelled benchmark samples on a local Wikipedia
snapshot (harmless wrong answers and canary codes only). Source: see the GitHub repository.
"""


def main() -> None:
    snapshot = ROOT / "demo" / "snapshot"
    if not (snapshot / "snapshot.json").exists():
        raise SystemExit("build the snapshot first: uv run python demo/build_snapshot.py")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(
        ROOT / "ragantine", OUT / "ragantine", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "configs", OUT / "configs")
    shutil.copytree(snapshot, OUT / "demo" / "snapshot")
    shutil.copy(ROOT / "demo" / "app.py", OUT / "app.py")
    (OUT / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    (OUT / "README.md").write_text(SPACE_README)
    (OUT / ".gitattributes").write_text("*.npy filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Space ready in {OUT} (set RAGANTINE_DEMO_SNAPSHOT=demo/snapshot if you move it)")


if __name__ == "__main__":
    main()
