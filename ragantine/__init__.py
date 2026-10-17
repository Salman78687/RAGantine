"""RAGantine: quarantine for retrieval. Library for defending RAG against knowledge-base poisoning.

Nothing in this package may import from ``bench`` (see the leakage rule in docs/SPEC.md §B3).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragantine")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"
