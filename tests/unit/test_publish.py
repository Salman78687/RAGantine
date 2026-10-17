"""Dataset release folder + card (no network)."""

from pathlib import Path

from bench.dataset.publish import build_release
from bench.io import write_jsonl
from tests.fakes_bench import build_bench_data


def test_release_folder_and_card(tmp_path: Path) -> None:
    cfg, _, docs, questions, poison = build_bench_data(limit=12)
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"data_dir": tmp_path / "data"})}
    )
    write_jsonl(tmp_path / "data" / "corpus" / "articles.jsonl", docs)
    write_jsonl(tmp_path / "data" / "bench" / "questions.jsonl", questions)
    write_jsonl(tmp_path / "data" / "bench" / "poison.jsonl", poison)
    out = tmp_path / "release"
    values = build_release(cfg, out, "v0.1")
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-sa-4.0")
    assert "{" + "n_articles}" not in card and f"| {values['n_articles']} |" in card
    assert "RAGantine-Bench v0.1" in card
    assert (out / "questions.jsonl").exists() and (out / "corpus" / "articles.jsonl").exists()
