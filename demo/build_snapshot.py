"""Build the demo snapshot: 500 articles + preset poison, baseline and defended indexes.

    uv run python demo/build_snapshot.py [--articles 500] [--d2-variant zeroshot]

Reads data/corpus and data/bench (run the benchmark data steps first). Writes
demo/snapshot/ (git-ignored; copied into the HF Space by demo/prepare_space.py):
  baseline/  clean articles + the presets' poison docs, no ingestion defenses
  defended/  same documents through D1 + D2 at ingestion (quarantine store included)
  presets.json  12 poisoned TEST questions, 2 per attack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.corpus.build import corpus_paths, load_articles, load_embedding_lookup
from bench.store import load_poison, load_questions
from ragantine.config import apply_overrides, load_config
from ragantine.defenses.base import build_defenses
from ragantine.ingest.pipeline import ingest
from ragantine.retrieve.dense import SentenceTransformerEmbedder
from ragantine.retrieve.memory import MemoryIndex

OUT = Path(__file__).parent / "snapshot"
ATTACK_NAMES = {
    "A1": "Factual override", "A2": "Indirect instruction", "A3": "Authority spoof",
    "A4": "Retrieval stuffing", "A5": "Hidden text", "A6": "Split payload",
}  # fmt: skip
PER_ATTACK = 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--articles", type=int, default=500)
    ap.add_argument("--d2-variant", choices=["finetuned", "zeroshot"], default=None,
                    help="default: finetuned if models/d2-injection exists, else zeroshot")  # fmt: skip
    args = ap.parse_args()

    cfg = load_config()
    variant = args.d2_variant or (
        "finetuned"
        if (cfg.paths.models_dir / "d2-injection" / "config.json").exists()
        else "zeroshot"
    )
    cfg = apply_overrides(cfg, [f"defenses.d2.variant={variant}"])
    questions = load_questions(cfg)
    poison = [p for p in load_poison(cfg) if p.seed == cfg.seed]
    targets = sorted(
        (q for q in questions if q.qset == "target" and q.split == "test"), key=lambda q: q.qid
    )
    presets = []
    for attack in ATTACK_NAMES:
        presets += [
            q for q in targets if q.attack == attack and any(p.qid == q.qid for p in poison)
        ][:PER_ATTACK]
    preset_ids = {q.qid for q in presets}
    preset_poison = [p for p in poison if p.qid in preset_ids]

    articles = load_articles(cfg)
    needed = {q.gold_chunk_id.split(":")[0] for q in presets}
    chosen = [a for a in articles if a.doc_id in needed]
    chosen += [a for a in articles if a.doc_id not in needed][: max(0, args.articles - len(chosen))]
    docs = [*chosen, *(p.document for p in preset_poison)]

    embedder = SentenceTransformerEmbedder(cfg.embedding)
    lookup = load_embedding_lookup(corpus_paths(cfg))
    for name, ids in (("baseline", []), ("defended", ["d1", "d2"])):
        idx = MemoryIndex(cfg.embedding.dimension)
        report = ingest(docs, build_defenses(ids, cfg), backend=idx, embedder=embedder,
                        chunking=cfg.chunking, precomputed=lookup)  # fmt: skip
        idx.save(OUT / name)
        print(f"{name}: {report.n_indexed} indexed, {report.n_quarantined} quarantined")
    (OUT / "presets.json").write_text(json.dumps(
        [{"qid": q.qid, "question": q.question, "attack": q.attack,
          "label": f"{q.attack} · {ATTACK_NAMES[str(q.attack)]}: {q.question}"} for q in presets], indent=2))  # fmt: skip
    (OUT / "snapshot.json").write_text(json.dumps(
        {"articles": len(chosen), "poison_docs": len(preset_poison), "presets": len(presets),
         "d2_variant": variant, "seed": cfg.seed}, indent=2))  # fmt: skip
    print(f"snapshot -> {OUT}")


if __name__ == "__main__":
    main()
