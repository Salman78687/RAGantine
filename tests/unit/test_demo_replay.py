"""Replay demo: deterministic example selection and HTML rendering (no models, no network)."""

from __future__ import annotations

from typing import Any

from demo.build_replay import _clip, select
from demo.replay_app import label, show


def _ans(
    verdict: str,
    text: str,
    decisions: list[dict[str, Any]] | None = None,
    poison_doc: str | None = None,
) -> dict[str, Any]:
    retrieved = []
    if poison_doc:
        chunk = {
            "doc_id": poison_doc,
            "title": "Planted",
            "source_tier": "web",
            "text": "Note <b>CODE-1</b>",
        }
        retrieved.append({"chunk": chunk})
    return {
        "verdict": verdict,
        "judge_reason": "r",
        "answer": {"text": text, "decisions": decisions or [], "retrieved": retrieved},
    }


def _data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    questions = {
        "q1": {
            "qid": "q1",
            "qset": "target",
            "attack": "A2",
            "question": "Q1?",
            "gold_answer": "1988",
        },
        "q2": {
            "qid": "q2",
            "qset": "target",
            "attack": "A1",
            "question": "Q2?",
            "gold_answer": "x",
        },
        "q3": {"qid": "q3", "qset": "clean", "attack": None, "question": "Q3?", "gold_answer": "y"},
    }
    poison = {"q1": [{"document": {"doc_id": "p1"}, "canary": "CODE-1", "target_false_answer": None}],
              "q2": [{"document": {"doc_id": "p2"}, "canary": None, "target_false_answer": "1901"}]}  # fmt: skip
    base = {"q1": _ans("poisoned", "1988 CODE-1", poison_doc="p1"), "q2": _ans("poisoned", "1901", poison_doc="p2"),
            "q3": _ans("correct", "y")}  # fmt: skip
    dec = {
        "defense_id": "d2",
        "stage": "ingestion",
        "target_id": "p1:0000",
        "action": "quarantine",
        "reason": "p=0.9",
    }
    dfd = {
        "q1": _ans("correct", "1988"),
        "q2": _ans("poisoned", "1901"),
        "q3": _ans("refused", "I don't know"),
    }
    return questions, {"gemma-small": (base, dfd)}, poison, {"gemma-small": [dec]}


def test_select_picks_wins_a_failure_and_a_cost() -> None:
    questions, runs, poison, ingestion = _data()
    out = select(questions, runs, poison, ingestion)
    assert [(e["kind"], e["attack"]) for e in out] == [
        ("stopped", "A2"),
        ("still fooled", "A1"),
        ("cost", None),
    ]
    stopped = out[0]
    assert stopped["attacker_goal"] == "CODE-1" and stopped["planted"][0]["tier"] == "web"
    assert stopped["defended"]["decisions"][0]["defense"] == "D2 injection classifier"


def test_render_escapes_planted_html_and_shows_decisions() -> None:
    questions, runs, poison, ingestion = _data()
    examples = select(questions, runs, poison, ingestion)
    page = show(label(examples[0]), examples)
    assert "<b>CODE-1</b>" not in page and "&lt;b&gt;" in page  # planted HTML is escaped
    assert '<mark class="rg-hit">CODE-1</mark>' in page  # the attacker's goal is highlighted
    assert "Poisoned" in page and "Correct" in page and "quarantine" in page and "Q1?" in page


def test_clip_centres_on_the_planted_claim() -> None:
    text = "a" * 2000 + " CODE-9 " + "b" * 2000
    assert "CODE-9" in _clip(text, "CODE-9", limit=200) and "CODE-9" not in _clip(
        text, None, limit=200
    )
