"""Build the static demo site (build/site/index.html) from demo/replay/examples.json.

The public demo is one self-contained HTML page (Hugging Face "static" Space): a stage hero
with the headline readings, a working diagram of how a planted page travels a RAG pipeline
(with a defenses switch), a chart of every defense, and an explorer of recorded answers. All
figures come from examples.json, which demo/build_replay.py pools from metrics.json; nothing
is typed by hand. Every recorded string is HTML-escaped. The page works with scripting off
(every record is shown); scripting adds the motion, the switch and the tabs.

    uv run python demo/build_replay.py && uv run python demo/build_site.py
    python -m http.server -d build/site 7862      # preview
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "demo" / "site" / "template.html"
DATA = ROOT / "demo" / "replay" / "examples.json"
OUT = ROOT / "build" / "site" / "index.html"
REPO_URL = "https://github.com/Salman78687/RAGantine"
KINDS = {"stopped": ("stopped", "Attack stopped"), "still fooled": ("fooled", "Still fooled"),
         "cost": ("cost", "The cost")}  # fmt: skip
VERDICTS = {"correct": ("Correct", "v-ok"), "poisoned": ("Poisoned", "v-bad"),
            "refused": ("Refused", "v-warn"), "other": ("Unclear", "v-warn")}  # fmt: skip
CITATION = re.compile(r"[\[【][^\]】]*?[0-9a-f]{4,}[^\]】]*?[\]】]")


def esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def highlight(text: str, needle: str | None, cites: bool = False) -> str:
    """Escape ``text``, mark the attacker's goal string and (optionally) quiet the citations."""
    out = esc(text)
    if needle:
        target = esc(needle)
        out = out.replace(target, f'<mark class="hit">{target}</mark>')
    if cites:
        out = CITATION.sub(lambda m: f'<span class="cite">{m.group(0)[1:-1]}</span>', out)
    return out


def readings(headline: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for model, configs in headline.items():
        by = {r["config"]: r for r in configs}
        base, d4 = by.get("No defenses"), by.get("D4 spotlighting")
        if not base or not d4:
            continue
        table = "".join(
            f'<tr><td>{esc(r["config"])}</td><td class="measured">{pct(r["asr"])}</td>'
            f'<td class="measured">{pct(r["clean"])}</td></tr>'
            for r in configs
        )
        rows.append(
            f'<div><div class="reading-row"><div><div class="label">{esc(model)}</div>'
            '<div class="note">poisoned answers with D4 spotlighting alone</div></div>'
            f'<div><span class="struck">{pct(base["asr"])}</span>'
            f'<div class="reading" data-count="{100 * d4["asr"]:.1f}">{pct(d4["asr"])}</div></div>'
            f'<div class="note">{d4["n"]} attacked questions, <span class="cool">{d4["seeds"]} seeds</span></div></div>'
            f'<table class="mini"><tr><th>Defenses</th><th>Poisoned</th><th>Clean correct</th></tr>{table}</table></div>'
        )
    return (
        '<div class="panel"><div class="panel-head"><h2>Benchmark v0.1</h2>'
        f'<span class="note">struck: with no defenses</span></div>{"".join(rows)}</div>'
    )


def flow(example: dict[str, Any]) -> str:
    """The working diagram, told with one real recorded example."""
    goal, gold = example.get("attacker_goal"), example["gold_answer"]
    node = '<div class="node{cls}"><h3>{title}</h3><p class="note">{text}</p>{extra}</div>'
    docs = '<div class="docs" aria-hidden="true"><i></i><i></i><i></i><i class="planted"></i><i></i></div>'

    def gates(*g: str) -> str:
        return '<div class="gates">' + "".join(g) + "</div>"

    def gate(name: str, first: bool = False) -> str:
        return f'<span class="gate{" first" if first else ""}">{esc(name)}</span>'

    answer = (
        f'<p class="note">{esc(example["question"])}</p>'
        f'<div class="verdict-line out-bad"><span class="said">{esc(goal)}</span> <span class="bad">poisoned</span></div>'
        f'<div class="verdict-line out-ok"><span class="said">{esc(gold)}</span> <span class="ok">correct</span></div>'
    )
    lanes = "".join(
        [
            node.format(cls="", title="Knowledge base", text="Wikipedia articles, and one planted page.",
                        extra=docs + gates(gate("D1 sanitizer", True), gate("D2 classifier"))),
            node.format(cls="", title="Retriever", text="Finds the passages that match the question.",
                        extra=gates(gate("D3 source trust"))),
            node.format(cls="", title="Model", text="Writes the answer from what it was handed.",
                        extra=gates(gate("D4 spotlighting"))),
            '<div class="node answer-card"><h3>Answer</h3>' + answer + gates(gate("D5 consistency")) + "</div>",
        ]
    )  # fmt: skip
    return (
        '<div class="panel reveal"><div class="flow"><div class="lanes">'
        f'{lanes}</div><div class="rail-wrap" aria-hidden="true"><i class="tick"></i><i class="tick"></i><i class="tick"></i>'
        '<i class="tick"></i><i class="tick stop"></i><div class="rail"><span class="token"></span></div></div>'
        '<div class="flow-caption"><p class="note">Recorded example, Gemma 4 26B. The planted page '
        f'says <span class="bad">{esc(goal)}</span>; the true answer is <span class="ok">{esc(gold)}</span>.</p>'
        '<button class="switch" type="button" aria-pressed="false"><span class="switch-track" aria-hidden="true">'
        '<span class="switch-thumb"></span></span><span class="switch-state">Defenses off</span></button>'
        "</div></div></div>"
    )


def chart(data: dict[str, list[dict[str, Any]]]) -> str:
    cols = []
    for model, rows in data.items():
        bars = "".join(
            f'<div class="bar-row"><div class="bar-name">{esc(r["config"])}'
            f"<small>{r['seeds']} seed{'s' if r['seeds'] > 1 else ''}, {r['n']} attacked</small></div>"
            f'<div class="bars"><div class="track"><div class="fill asr" style="--v:{r["asr"]:.4f};--i:{i}"></div></div>'
            f'<div class="track thin"><div class="fill clean" style="--v:{r["clean"]:.4f};--i:{i}"></div></div></div>'
            f'<div class="bar-val"><span class="measured">{pct(r["asr"])}</span>'
            f'<span class="measured">{pct(r["clean"])}</span></div></div>'
            for i, r in enumerate(rows)
        )
        cols.append(
            f'<div><div class="panel-head" style="margin-bottom:8px"><h3>{esc(model)}</h3></div>{bars}</div>'
        )
    return f'<div class="panel"><div class="chart">{"".join(cols)}</div></div>'


def tab_label(e: dict[str, Any]) -> str:
    return f"{e['attack']} {e['attack_name'].lower()}" if e.get("attack") else "Clean question"


def tabs(examples: list[dict[str, Any]]) -> str:
    out = []
    for i, e in enumerate(examples):
        cls, text = KINDS.get(e["kind"], ("stopped", e["kind"]))
        out.append(
            f'<a class="tab" role="tab" id="tab-{i}" href="#rec-{i}" aria-controls="rec-{i}" aria-selected="false">'
            f'{esc(tab_label(e))}<span class="kind {cls}">{esc(text)}</span></a>'
        )
    return "".join(out)


def side(title: str, s: dict[str, Any], goal: str | None) -> str:
    text, cls = VERDICTS.get(s["verdict"], (s["verdict"], "v-warn"))
    return (
        f'<div class="side {cls}"><div class="side-head"><span class="label">{esc(title)}</span>'
        f'<span class="verdict {cls}">{esc(text)}</span></div>'
        f'<div class="answer">{highlight(s["text"], goal, cites=True)}</div>'
        f'<p class="note judge">Judge: {esc(s["reason"])}</p></div>'
    )


def record(i: int, e: dict[str, Any]) -> str:
    goal = e.get("attacker_goal")
    want = (
        f'<span><span class="muted">Attacker wants</span> <b class="bad">{esc(goal)}</b></span>'
        if goal
        else ""
    )
    rows = [
        f'<div><h3 class="q">{esc(e["question"])}</h3><div class="facts">'
        f'<span><span class="muted">Model</span> <b>{esc(e["model"])}</b></span>'
        f'<span><span class="muted">Correct answer</span> <b class="ok">{esc(e["gold_answer"])}</b></span>{want}</div></div>'
    ]
    for c in e["planted"]:
        rows.append(
            f'<div class="doc"><div class="panel-head"><span class="label">Planted in the knowledge base: '
            f'{esc(c["title"])}</span><span class="note">{esc(e["attack_name"])}, source tier {esc(c["tier"])}</span></div>'
            f"<pre>{highlight(c['text'], goal)}</pre></div>"
        )
    if not e["planted"]:
        rows.append('<div class="doc"><span class="note">Nothing planted: a clean question, to show what the '
                    "defenses cost on normal traffic.</span></div>")  # fmt: skip
    rows.append(
        f'<div class="diff">{side("Without defenses", e["baseline"], goal)}{side("With RAGantine", e["defended"], goal)}</div>'
    )
    trail = "".join(
        f'<div class="trail-row"><span class="cool">{esc(d["defense"])}</span>'
        f'<span>{esc(d["action"])} at {esc(d["stage"])}</span><span class="measured">{esc(d["reason"])}</span></div>'
        for d in e["defended"].get("decisions") or []
    )
    rows.append(
        '<div><div class="label" style="margin-bottom:8px">What each defense did</div>'
        f"{trail or '<span class=note>No defense acted.</span>'}</div>"
    )
    return (
        f'<article class="record" id="rec-{i}" role="tabpanel" aria-labelledby="tab-{i}">'
        f'<div class="panel">{"".join(rows)}</div></article>'
    )


def render(data: dict[str, Any], template: str) -> str:
    examples = data["examples"]
    parts = {
        "{{READINGS}}": readings(data.get("headline", {})),
        "{{FLOW}}": flow(examples[0]),
        "{{CHART}}": chart(data.get("chart", data.get("headline", {}))),
        "{{TABS}}": tabs(examples),
        "{{RECORDS}}": "".join(record(i, e) for i, e in enumerate(examples)),
        "{{REPO_URL}}": REPO_URL,
        "{{REPO_SHORT}}": REPO_URL.removeprefix("https://"),
    }
    for key, value in parts.items():
        template = template.replace(key, value)
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", template)
    if leftover:
        raise ValueError(f"unfilled placeholders: {leftover}")
    return template


def main() -> None:
    page = render(
        json.loads(DATA.read_text(encoding="utf-8")), TEMPLATE.read_text(encoding="utf-8")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
