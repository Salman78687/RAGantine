"""Replay demo: deterministic example selection and the static page (no models, no network)."""

from __future__ import annotations

from typing import Any

from demo.build_replay import _clip, select
from demo.build_site import TEMPLATE, render


def _ans(verdict: str, text: str, poison_doc: str | None = None) -> dict[str, Any]:
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
        "answer": {"text": text, "decisions": [], "retrieved": retrieved},
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
    base = {"q1": _ans("poisoned", "1988 CODE-1 [abcdef0123456789:0001]", "p1"),
            "q2": _ans("poisoned", "1901", "p2"), "q3": _ans("correct", "y")}  # fmt: skip
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


def _page() -> str:
    questions, runs, poison, ingestion = _data()
    examples = select(questions, runs, poison, ingestion)
    row = {"config": "No defenses", "asr": 0.5, "clean": 1.0, "n": 42, "seeds": 1}
    d4 = {**row, "config": "D4 spotlighting", "asr": 0.1}
    data = {"headline": {"Model": [row, d4]}, "chart": {"Model": [row, d4]}, "examples": examples}
    return render(data, TEMPLATE.read_text(encoding="utf-8"))


def test_select_picks_wins_a_failure_and_a_cost() -> None:
    questions, runs, poison, ingestion = _data()
    out = select(questions, runs, poison, ingestion)
    assert [(e["kind"], e["attack"]) for e in out] == [
        ("stopped", "A2"),
        ("still fooled", "A1"),
        ("cost", None),
    ]
    assert out[0]["attacker_goal"] == "CODE-1" and out[0]["planted"][0]["tier"] == "web"
    assert out[0]["defended"]["decisions"][0]["defense"] == "D2 injection classifier"


def test_page_escapes_recorded_text_and_fills_every_slot() -> None:
    page = _page()
    assert "<b>CODE-1</b>" not in page and "&lt;b&gt;" in page  # planted HTML never runs
    assert '<mark class="hit">CODE-1</mark>' in page  # the attacker's goal is highlighted
    assert '<span class="cite">abcdef0123456789:0001</span>' in page  # citations quieted
    assert "{{" not in page and page.count('class="record"') == 3 and page.count('role="tab"') == 3
    assert 'data-count="10.0"' in page and "50.0%" in page  # readings come from the data


def test_clip_centres_on_the_planted_claim() -> None:
    text = "a" * 2000 + " CODE-9 " + "b" * 2000
    assert "CODE-9" in _clip(text, "CODE-9", limit=200) and "CODE-9" not in _clip(
        text, None, limit=200
    )
