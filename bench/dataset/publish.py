"""Assemble RAGantine-Bench for the Hugging Face Hub and (optionally) push it.

    uv run ragantine bench publish-dataset --repo-id <you>/ragantine-bench          # build only
    HF_TOKEN=... uv run ragantine bench publish-dataset --repo-id <you>/ragantine-bench --push

The token is read from the environment and never printed.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from bench.corpus.build import corpus_paths
from bench.store import bench_paths, load_poison, load_questions
from ragantine.config import AppConfig

CARD = Path(__file__).parent / "CARD.md"


def _fill(template: str, values: dict[str, object]) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def build_release(cfg: AppConfig, out: Path, version: str) -> dict[str, object]:
    if out.exists():
        shutil.rmtree(out)
    (out / "corpus").mkdir(parents=True)
    bp, cp = bench_paths(cfg), corpus_paths(cfg)
    shutil.copy(cp["articles"], out / "corpus" / "articles.jsonl")
    shutil.copy(bp["questions"], out / "questions.jsonl")
    shutil.copy(bp["poison"], out / "poison.jsonl")
    if bp["poison_train"].exists():
        shutil.copy(bp["poison_train"], out / "poison_train.jsonl")
    if bp["poison_docs"].exists():
        shutil.copytree(bp["poison_docs"], out / "poison_docs")
    questions = load_questions(cfg)
    sets = Counter(q.qset for q in questions)
    poison = load_poison(cfg)
    train = load_poison(cfg, "train") if bp["poison_train"].exists() else []
    n_articles = sum(1 for _ in cp["articles"].open())
    values: dict[str, object] = {
        "version": version, "n_articles": n_articles, "hf_config": cfg.corpus.hf_config,
        "n_questions": len(questions), "n_target": sets["target"], "n_clean": sets["clean"],
        "n_paraphrase": sets["paraphrase"], "n_poison": len(poison), "n_poison_train": len(train),
        "seeds": ", ".join(str(s) for s in sorted({p.seed for p in poison})), "min_chars": cfg.corpus.min_chars,
        "max_chars": cfg.corpus.max_chars, "scan_limit": cfg.corpus.scan_limit,
        "tier_mix": ", ".join(f"{k.value} {v:.0%}" for k, v in cfg.corpus.tier_mix.items()),
    }  # fmt: skip
    (out / "README.md").write_text(_fill(CARD.read_text(), values))
    (out / "build_info.json").write_text(
        json.dumps({"version": version, **values}, indent=2, default=str)
    )
    return values


def push(out: Path, repo_id: str, token: str, version: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=out,
        repo_type="dataset",
        commit_message=f"RAGantine-Bench {version}",
    )
    api.create_tag(repo_id, tag=version, repo_type="dataset", exist_ok=True)
    return f"https://huggingface.co/datasets/{repo_id}"
