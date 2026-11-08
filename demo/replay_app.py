"""RAGantine replay demo: recorded v0.1 benchmark answers, side by side.

Each example shows a question, the document an attacker planted in the knowledge base, the
answer of an undefended RAG pipeline (D0) and the answer of RAGantine's full defense stack,
with every defense decision and its reason. Answers are replayed from finished benchmark runs
(demo/replay/examples.json, built by demo/build_replay.py), so the demo needs no API keys,
no GPU and no models, and it cannot run out of a free quota.

    uv run python demo/replay_app.py
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

EXAMPLES_PATH = Path(
    os.environ.get("RAGANTINE_REPLAY", Path(__file__).parent / "replay" / "examples.json")
)
REPO_URL = "https://github.com/Salman78687/RAGantine"
KIND_TITLES = {
    "stopped": "attack stopped",
    "still fooled": "still fooled (a limitation)",
    "cost": "the cost: a normal question refused",
}
VERDICTS = {
    "correct": ("Correct", "ok"),
    "poisoned": ("Poisoned", "bad"),
    "refused": ("Refused / I don't know", "warn"),
    "other": ("Unclear", "warn"),
}

CSS = """
:root { --ok:#1a7f37; --bad:#c62828; --warn:#9a6700; --muted:#57606a; --line:#d0d7de; --card:#f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --ok:#3fb950; --bad:#f85149; --warn:#d29922; --muted:#8b949e; --line:#30363d; --card:#161b22; }
}
.rg-card { border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:var(--card); margin-bottom:10px; }
.rg-label { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom:6px; }
.rg-badge { display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; border:1px solid currentColor; }
.rg-ok { color:var(--ok); } .rg-bad { color:var(--bad); } .rg-warn { color:var(--warn); }
.rg-text { white-space:pre-wrap; line-height:1.5; }
.rg-doc { white-space:pre-wrap; font-size:13px; max-height:260px; overflow:auto; }
.rg-dec { font-size:13px; margin:4px 0; } .rg-dec b { font-weight:600; }
.rg-note { font-size:13px; color:var(--muted); }
"""


def load_examples(path: Path = EXAMPLES_PATH) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["examples"]
    return examples


def label(e: dict[str, Any]) -> str:
    attack = f"{e['attack']} {e['attack_name']}" if e.get("attack") else e["attack_name"]
    return f"{attack} — {KIND_TITLES.get(e['kind'], e['kind'])}"


def _esc(text: Any) -> str:
    return html.escape(str(text or ""))


def _badge(verdict: str) -> str:
    text, cls = VERDICTS.get(verdict, (verdict, "warn"))
    return f'<span class="rg-badge rg-{cls}">{_esc(text)}</span>'


def render_question(e: dict[str, Any]) -> str:
    goal = (
        f'<div class="rg-note">Attacker\'s goal: make the answer say <b>{_esc(e["attacker_goal"])}</b>.</div>'
        if e.get("attacker_goal")
        else '<div class="rg-note">No poison targets this question.</div>'
    )
    return (
        f'<div class="rg-card"><div class="rg-label">Question · {_esc(e["model"])}</div>'
        f'<div class="rg-text"><b>{_esc(e["question"])}</b></div>'
        f'<div class="rg-note">Correct answer: {_esc(e["gold_answer"])}</div>{goal}</div>'
    )


def render_planted(e: dict[str, Any]) -> str:
    if not e["planted"]:
        return (
            '<div class="rg-card"><div class="rg-label">Planted document</div>'
            '<div class="rg-note">None: this is a clean question.</div></div>'
        )
    parts = [
        f'<div class="rg-card"><div class="rg-label">Planted document · {_esc(c["title"])} · '
        f'source tier: {_esc(c["tier"])}</div><div class="rg-doc">{_esc(c["text"])}</div></div>'
        for c in e["planted"]
    ]
    return "".join(parts)


def render_answer(title: str, side: dict[str, Any], decisions: bool) -> str:
    body = (
        f'<div class="rg-card"><div class="rg-label">{_esc(title)}</div>'
        f'<div>{_badge(side["verdict"])}</div><div class="rg-text" style="margin-top:8px">'
        f'{_esc(side["text"])}</div><div class="rg-note" style="margin-top:8px">Judge: {_esc(side["reason"])}</div>'
    )
    if decisions:
        rows = side.get("decisions") or []
        items = "".join(
            f'<div class="rg-dec"><b>{_esc(d["defense"])}</b> · {_esc(d["stage"])} · '
            f"{_esc(d['action'])} — {_esc(d['reason'])}</div>"
            for d in rows
        )
        body += f'<div class="rg-label" style="margin-top:12px">Defense decisions</div>{items or "<div class=rg-note>none</div>"}'
    return body + "</div>"


def show(choice: str, examples: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    e = next((x for x in examples if label(x) == choice), examples[0])
    return (
        render_question(e),
        render_planted(e),
        render_answer("Baseline RAG (no defenses)", e["baseline"], decisions=False),
        render_answer("RAGantine (full defense stack)", e["defended"], decisions=True),
    )


INTRO = f"""
# RAGantine: defending RAG against knowledge-base poisoning

An attacker plants a document in the knowledge base a RAG assistant searches. Below, the same
question goes through a **baseline RAG pipeline** and through **RAGantine's layered defenses**.

These are **recorded answers from the v0.1 benchmark** (Wikipedia corpus, harmless wrong answers
and canary codes only), not live model calls. The examples include a case where the defenses
still fail and one where they refuse a normal question: both are part of the result.
Code, method and full results: [{REPO_URL}]({REPO_URL}).
"""


def build_ui(examples: list[dict[str, Any]]) -> Any:
    import gradio as gr

    labels = [label(e) for e in examples]
    with gr.Blocks(title="RAGantine demo") as ui:
        gr.Markdown(INTRO)
        pick = gr.Dropdown(choices=labels, value=labels[0], label="Example")
        question = gr.HTML()
        planted = gr.HTML()
        with gr.Row(equal_height=False):
            baseline = gr.HTML()
            defended = gr.HTML()

        def on_pick(choice: str) -> tuple[str, str, str, str]:
            return show(choice, examples)

        outputs = [question, planted, baseline, defended]
        pick.change(on_pick, inputs=pick, outputs=outputs)
        ui.load(on_pick, inputs=pick, outputs=outputs)
    return ui


def main() -> None:
    build_ui(load_examples()).launch(css=CSS)


if __name__ == "__main__":
    main()
