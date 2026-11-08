"""Build demo/replay/examples.json: recorded benchmark answers for the replay demo.

The replay demo shows real answers from finished v0.1 runs instead of calling models live, so
the public Space needs no API keys and cannot exhaust a free quota. For each example it keeps
the question, what the attacker planted (the poison chunks the baseline actually retrieved),
the baseline (D0) answer, and the RAGantine (full stack) answer with every defense decision
that touched the question or its poison. Selection is deterministic (sorted question ids):

* one "stopped" example per attack type (baseline poisoned, RAGantine correct),
* one "still fooled" example (both poisoned) and one "cost" example (clean question that
  RAGantine refused), so the demo shows the limits as well as the wins.

    uv run python demo/build_replay.py      # needs results/ and data/bench/ from the benchmark
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "replay" / "examples.json"
SEED = 1337
MODELS = ("gemma-small", "gemini-lite")
MODEL_NAMES = {"gemma-small": "Gemma 4 26B (open)", "gemini-lite": "Gemini 3.5 Flash-Lite (closed)"}
BASELINE_RUN = "v01-{model}-none"
DEFENDED_RUN = "v01-{model}-full-final"
ATTACKS = {
    "A1": "Factual override", "A2": "Indirect instruction", "A3": "Authority spoof",
    "A4": "Retrieval stuffing", "A5": "Hidden text", "A6": "Split payload",
}  # fmt: skip
DEFENSES = {
    "d1": "D1 sanitizer", "d2": "D2 injection classifier", "d3": "D3 source trust",
    "d4": "D4 spotlighting", "d5": "D5 consistency check", "d6": "D6 anomaly",
}  # fmt: skip
MAX_CHUNK_CHARS = 900

Row = Mapping[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clip(text: str, needle: str | None = None, limit: int = MAX_CHUNK_CHARS) -> str:
    """Shorten ``text`` to ``limit`` chars, centred on ``needle`` (the planted claim) if present."""
    text = text.strip()
    if len(text) <= limit:
        return text
    at = text.find(needle) if needle else -1
    start = 0 if at < 0 else max(0, min(at - limit // 2, len(text) - limit))
    end = start + limit
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _decisions(
    answer: Row, ingestion: Iterable[Row], poison_docs: set[str]
) -> list[dict[str, str]]:
    """Answer-time decisions plus ingestion decisions that hit this question's poison."""
    hits = [d for d in ingestion if str(d["target_id"]).split(":")[0] in poison_docs]
    return [
        {
            "defense": DEFENSES.get(d["defense_id"], d["defense_id"]),
            "stage": d["stage"],
            "action": d["action"],
            "reason": d["reason"],
        }
        for d in [*hits, *answer["answer"]["decisions"]]
    ]


def make_example(
    kind: str,
    model: str,
    question: Row,
    baseline: Row,
    defended: Row,
    poison: list[Row],
    ingestion: list[Row],
) -> dict[str, Any]:
    poison_docs = {p["document"]["doc_id"] for p in poison}
    goal = poison[0] if poison else {}
    needle = goal.get("canary") or goal.get("target_false_answer")
    planted = [
        {
            "title": h["chunk"]["title"],
            "tier": h["chunk"]["source_tier"],
            "text": _clip(h["chunk"]["text"], needle),
        }
        for h in baseline["answer"]["retrieved"]
        if h["chunk"]["doc_id"] in poison_docs
    ]
    attack = question.get("attack")
    return {
        "kind": kind,
        "model": MODEL_NAMES.get(model, model),
        "attack": attack,
        "attack_name": ATTACKS.get(attack or "", "No attack (clean question)"),
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "attacker_goal": goal.get("target_false_answer") or goal.get("canary"),
        "planted": planted,
        "baseline": {
            "text": baseline["answer"]["text"],
            "verdict": baseline["verdict"],
            "reason": baseline["judge_reason"],
        },
        "defended": {
            "text": defended["answer"]["text"],
            "verdict": defended["verdict"],
            "reason": defended["judge_reason"],
            "decisions": _decisions(defended, ingestion, poison_docs),
        },
    }


def select(
    questions: Mapping[str, Row],
    runs: Mapping[str, tuple[Mapping[str, Row], Mapping[str, Row]]],
    poison_by_qid: Mapping[str, list[Row]],
    ingestion: Mapping[str, list[Row]],
) -> list[dict[str, Any]]:
    """Pick examples deterministically; ``runs`` maps model -> (baseline, defended) by qid."""
    out: list[dict[str, Any]] = []

    def add(kind: str, attack: str | None, qset: str, base_v: str, def_v: str) -> bool:
        """Append the first matching question (sorted qids, models in MODELS order)."""
        for model in MODELS:
            if model not in runs:
                continue
            base, dfd = runs[model]
            for qid in sorted(base):
                q = questions[qid]
                got = (q.get("attack"), q["qset"], base[qid]["verdict"], dfd[qid]["verdict"])
                if got == (attack, qset, base_v, def_v):
                    out.append(make_example(kind, model, q, base[qid], dfd[qid],
                                            poison_by_qid.get(qid, []), ingestion.get(model, [])))  # fmt: skip
                    return True
        return False

    for attack in ATTACKS:
        add("stopped", attack, "target", "poisoned", "correct")
    for attack in ATTACKS:
        if add("still fooled", attack, "target", "poisoned", "poisoned"):
            break
    add("cost", None, "clean", "correct", "refused")
    return out


HEADLINE_RUNS = {  # label -> run ids pooled over seeds 1337, 2024, 7
    "No defenses": ["v01-{m}-none", "v01-{m}-none-s2024", "v01-{m}-none-s7"],
    "D4 spotlighting": ["v01-{m}-d4", "v01-{m}-d4-s2024", "v01-{m}-d4-s7"],
    "Full stack": ["v01-{m}-full-final", "v01-{m}-full-final-s2024", "v01-{m}-full-final-s7"],
}


def headline(results: Path) -> dict[str, list[dict[str, Any]]]:
    """Pooled ASR and clean accuracy per model and configuration, from metrics.json (k/n sums)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        rows = []
        for name, ids in HEADLINE_RUNS.items():
            metrics = [
                json.loads((results / r.format(m=model) / "metrics.json").read_text()) for r in ids
            ]
            asr_k, asr_n = sum(x["asr"]["k"] for x in metrics), sum(x["asr"]["n"] for x in metrics)
            ok_k = sum(x["clean_accuracy"]["k"] for x in metrics)
            ok_n = sum(x["clean_accuracy"]["n"] for x in metrics)
            rows.append({"config": name, "asr": asr_k / asr_n, "clean": ok_k / ok_n, "n": asr_n})
        out[MODEL_NAMES[model]] = rows
    return out


def main() -> None:
    questions = {q["qid"]: q for q in read_jsonl(ROOT / "data" / "bench" / "questions.jsonl")}
    poison_by_qid: dict[str, list[Row]] = {}
    for p in read_jsonl(ROOT / "data" / "bench" / "poison.jsonl"):
        if p["seed"] == SEED:
            poison_by_qid.setdefault(p["qid"], []).append(p)
    runs, ingestion = {}, {}
    for model in MODELS:
        base_dir = ROOT / "results" / BASELINE_RUN.format(model=model)
        def_dir = ROOT / "results" / DEFENDED_RUN.format(model=model)
        runs[model] = (
            {a["qid"]: a for a in read_jsonl(base_dir / "answers.jsonl")},
            {a["qid"]: a for a in read_jsonl(def_dir / "answers.jsonl")},
        )
        ingestion[model] = [
            d for d in read_jsonl(def_dir / "decisions.jsonl") if d.get("qid") is None
        ]
    examples = select(questions, runs, poison_by_qid, ingestion)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"seed": SEED, "headline": headline(ROOT / "results"), "examples": examples},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    kinds = [f"{e['kind']}:{e['attack'] or 'clean'}" for e in examples]
    print(f"wrote {len(examples)} examples -> {OUT.relative_to(ROOT)}: {', '.join(kinds)}")


if __name__ == "__main__":
    main()
