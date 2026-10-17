"""Leakage rule (SPEC §B3, CLAUDE.md hard rules).

Nothing in ragantine/ may import from bench/ or read the poison ground truth, so the
system under test can never see which documents are poison.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "ragantine"
GROUND_TRUTH_FIELDS = ("is_poison", "attack", "canary", "target_false_answer", "poison_id")


def _library_files() -> list[Path]:
    files = sorted(LIBRARY.rglob("*.py"))
    assert files, "ragantine/ has no python files; leakage test would be vacuous"
    return files


def _bench_imports(source: str) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] == "bench"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "bench":
                hits.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name | ast.Attribute)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).split(".")[0] == "bench"
        ):
            hits.append(str(node.args[0].value))
    return hits


@pytest.mark.parametrize("path", _library_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_library_never_imports_bench(path: Path) -> None:
    assert _bench_imports(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", _library_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_library_never_reads_poison_ground_truth(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "poison.jsonl" not in text
    assert "bench/poison" not in text


def test_detector_catches_bench_imports() -> None:
    src = "import bench\nfrom bench.score import x\nimport importlib\nimportlib.import_module('bench.run')\n"
    assert _bench_imports(src) == ["bench", "bench.score", "bench.run"]
    assert _bench_imports("from .bench import x\nimport benchmark\n") == []


def test_index_mappings_have_no_ground_truth_fields() -> None:
    """Index mappings are JSON dicts in ragantine/index; none may name a ground-truth field."""
    from ragantine.index import opensearch

    mappings = [v for k, v in vars(opensearch).items() if k.isupper() and isinstance(v, dict)]
    blob = json.dumps(mappings)
    for field in GROUND_TRUTH_FIELDS:
        assert f'"{field}"' not in blob
