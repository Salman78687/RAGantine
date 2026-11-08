"""RAGantine replay demo: recorded v0.1 benchmark answers, side by side.

Each example shows a question, the document an attacker planted in the knowledge base, the
answer of an undefended RAG pipeline (D0) and the answer of RAGantine's full defense stack,
with every defense decision and its reason. Answers are replayed from finished benchmark runs
(demo/replay/examples.json, built by demo/build_replay.py), so the demo needs no API keys,
no GPU and no models, and it cannot run out of a free quota. The headline numbers are pooled
from the runs' metrics.json at build time; nothing here is typed by hand.

    uv run python demo/replay_app.py
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

EXAMPLES_PATH = Path(
    os.environ.get("RAGANTINE_REPLAY", Path(__file__).parent / "replay" / "examples.json")
)
REPO_URL = "https://github.com/Salman78687/RAGantine"
KIND_TITLES = {
    "stopped": "Attack stopped",
    "still fooled": "Still fooled",
    "cost": "The cost",
}
VERDICTS = {
    "correct": ("Correct", "ok"),
    "poisoned": ("Poisoned", "bad"),
    "refused": ("Refused", "warn"),
    "other": ("Unclear", "warn"),
}

# One fixed dark palette for both Gradio light and dark modes, so text is always readable.
BG, PANEL, PANEL_2, LINE = "#0b0e15", "#121722", "#171d2b", "#252d3f"
TEXT, MUTED, ACCENT = "#e7eaf0", "#8d95a8", "#8ea2ff"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --rg-bg:{BG}; --rg-panel:{PANEL}; --rg-panel2:{PANEL_2}; --rg-line:{LINE};
  --rg-text:{TEXT}; --rg-muted:{MUTED}; --rg-accent:{ACCENT};
  --rg-ok:#43d39e; --rg-bad:#ff7070; --rg-warn:#f4b955;
}}
body, .gradio-container {{ background:var(--rg-bg) !important; color:var(--rg-text) !important;
  font-family:'Inter', system-ui, sans-serif !important; }}
.gradio-container {{ max-width:1180px !important; margin:0 auto !important; padding:0 20px !important; }}
footer {{ display:none !important; }}
.rg, .rg * {{ box-sizing:border-box; }}
.rg {{ color:var(--rg-text); font-family:'Inter', system-ui, sans-serif; line-height:1.55; }}
.rg-hero {{ padding:40px 0 8px; }}
.rg-eyebrow {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--rg-accent); font-weight:600; }}
.rg-title {{ font-size:40px; font-weight:700; letter-spacing:-.02em; margin:6px 0 8px; color:#fff; }}
.rg-sub {{ font-size:17px; color:var(--rg-muted); max-width:760px; margin:0; }}
.rg-sub b {{ color:var(--rg-text); font-weight:600; }}
.rg-stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; margin:28px 0 8px; }}
.rg-stat {{ background:var(--rg-panel); border:1px solid var(--rg-line); border-radius:14px; padding:18px 20px; }}
.rg-stat h4 {{ margin:0 0 14px; font-size:14px; font-weight:600; color:var(--rg-text); }}
.rg-bar {{ display:grid; grid-template-columns:118px 1fr 56px; align-items:center; gap:12px; margin:9px 0; font-size:13px; }}
.rg-bar span:first-child {{ color:var(--rg-muted); }}
.rg-track {{ height:8px; background:var(--rg-panel2); border-radius:99px; overflow:hidden; }}
.rg-fill {{ height:100%; border-radius:99px; background:linear-gradient(90deg, #ff9a7a, var(--rg-bad)); }}
.rg-pct {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }}
.rg-foot {{ font-size:12px; color:var(--rg-muted); margin-top:12px; }}
.rg-section {{ font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--rg-muted);
  font-weight:600; margin:30px 0 10px; }}
.rg-card {{ background:var(--rg-panel); border:1px solid var(--rg-line); border-radius:14px; padding:20px 22px; margin-bottom:16px; }}
.rg-q {{ font-size:22px; font-weight:600; color:#fff; margin:2px 0 14px; letter-spacing:-.01em; }}
.rg-meta {{ display:flex; flex-wrap:wrap; gap:8px; }}
.rg-chip {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; padding:4px 10px;
  border-radius:99px; background:var(--rg-panel2); border:1px solid var(--rg-line); color:var(--rg-text); }}
.rg-chip i {{ font-style:normal; color:var(--rg-muted); }}
.rg-chip.bad {{ border-color:rgba(255,112,112,.45); color:#ffb3b3; }}
.rg-chip.ok {{ border-color:rgba(67,211,158,.45); color:#9ff0cf; }}
.rg-doc {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-size:12.5px; line-height:1.7;
  color:#c6cbd6; white-space:pre-wrap; max-height:240px; overflow:auto; margin-top:12px;
  background:var(--rg-bg); border:1px solid var(--rg-line); border-radius:10px; padding:14px 16px; }}
.rg-doc-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
.rg-doc-head b {{ font-weight:600; color:#fff; }}
mark.rg-hit {{ background:rgba(255,112,112,.2); color:#ffc9c9; border-radius:4px; padding:0 3px;
  box-shadow:inset 0 -1px 0 rgba(255,112,112,.6); }}
.rg-vs {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:820px) {{ .rg-vs {{ grid-template-columns:1fr; }} .rg-title {{ font-size:32px; }} }}
.rg-ans {{ border-top:3px solid var(--rg-line); }}
.rg-ans.ok {{ border-top-color:var(--rg-ok); }} .rg-ans.bad {{ border-top-color:var(--rg-bad); }}
.rg-ans.warn {{ border-top-color:var(--rg-warn); }}
.rg-ans-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.rg-ans-head span {{ font-size:13px; font-weight:600; color:var(--rg-muted); letter-spacing:.02em; }}
.rg-badge {{ font-size:12px; font-weight:700; padding:3px 10px; border-radius:99px; letter-spacing:.02em; }}
.rg-badge.ok {{ background:rgba(67,211,158,.14); color:var(--rg-ok); }}
.rg-badge.bad {{ background:rgba(255,112,112,.14); color:var(--rg-bad); }}
.rg-badge.warn {{ background:rgba(244,185,85,.14); color:var(--rg-warn); }}
.rg-text {{ font-size:15px; color:var(--rg-text); white-space:pre-wrap; }}
.rg-judge {{ font-size:12.5px; color:var(--rg-muted); margin-top:12px; padding-top:12px; border-top:1px dashed var(--rg-line); }}
.rg-dec {{ margin-top:14px; }}
.rg-dec-row {{ display:grid; grid-template-columns:auto 1fr; gap:4px 12px; padding:10px 0; border-top:1px solid var(--rg-line); }}
.rg-dec-name {{ font-size:12.5px; font-weight:600; color:var(--rg-accent); }}
.rg-dec-act {{ font-size:12.5px; color:var(--rg-text); }}
.rg-dec-why {{ grid-column:1 / -1; font-family:'JetBrains Mono', ui-monospace, monospace; font-size:11.5px; color:var(--rg-muted); word-break:break-word; }}
.rg-note {{ font-size:13px; color:var(--rg-muted); }}
.rg-picker, .rg-picker > *, .rg-picker fieldset, .form:has(.rg-picker) {{ background:transparent !important;
  border:none !important; box-shadow:none !important; padding:0 !important; }}
.rg-cite {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-size:11px; color:var(--rg-muted);
  background:var(--rg-panel2); border:1px solid var(--rg-line); border-radius:6px; padding:1px 6px; white-space:nowrap; }}
.rg-picker .wrap {{ gap:8px !important; }}
.rg-picker label {{ border-radius:99px !important; font-size:13px !important; }}
.rg-picker input {{ display:none !important; }}
.rg-footer {{ font-size:12.5px; color:var(--rg-muted); margin:30px 0 40px; padding-top:16px; border-top:1px solid var(--rg-line); }}
.rg-footer a, .rg a {{ color:var(--rg-accent); text-decoration:none; }}
"""


def load_data(path: Path = EXAMPLES_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def label(e: dict[str, Any]) -> str:
    attack = f"{e['attack']} · {e['attack_name']}" if e.get("attack") else "Clean question"
    return f"{attack} — {KIND_TITLES.get(e['kind'], e['kind'])}"


def _esc(text: Any) -> str:
    return html.escape(str(text or ""))


CITATION = re.compile(r"[\[【][^\]】]*?[0-9a-f]{4,}[^\]】]*?[\]】]")


def _highlight(text: str, needle: str | None, cites: bool = False) -> str:
    """Escape ``text``, mark the attacker's goal string and (optionally) style citations."""
    escaped = _esc(text)
    if needle:
        target = _esc(needle)
        escaped = escaped.replace(target, f'<mark class="rg-hit">{target}</mark>')
    if cites:
        escaped = CITATION.sub(
            lambda m: f'<span class="rg-cite">{m.group(0)[1:-1]}</span>', escaped
        )
    return escaped


def render_header(headline: dict[str, list[dict[str, Any]]]) -> str:
    cards = []
    for model, rows in headline.items():
        bars = "".join(
            f'<div class="rg-bar"><span>{_esc(r["config"])}</span>'
            f'<div class="rg-track"><div class="rg-fill" style="width:{100 * r["asr"]:.1f}%"></div></div>'
            f'<span class="rg-pct">{100 * r["asr"]:.0f}%</span></div>'
            for r in rows
        )
        clean = " · ".join(f"{_esc(r['config'])} {100 * r['clean']:.0f}%" for r in rows)
        n = rows[0]["n"] if rows else 0
        cards.append(
            f'<div class="rg-stat"><h4>{_esc(model)}: poisoned answers</h4>{bars}'
            f'<div class="rg-foot">Clean accuracy: {clean}<br>{n} attacked questions, 3 seeds</div></div>'
        )
    return (
        '<div class="rg rg-hero"><div class="rg-eyebrow">Open-source benchmark · v0.1</div>'
        '<div class="rg-title">RAGantine</div>'
        '<p class="rg-sub">An attacker plants a document in the knowledge base a RAG assistant '
        "searches. RAGantine measures how often that works, and puts <b>layered defenses</b> in "
        "front of the model.</p>"
        f'<div class="rg-stats">{"".join(cards)}</div>'
        '<div class="rg-section">Recorded examples</div>'
        '<p class="rg-note" style="margin:0">Real answers recorded during the benchmark, not live '
        "calls. They include a case where the defenses still fail and one where they refuse a "
        "normal question.</p></div>"
    )


def render_example(e: dict[str, Any]) -> str:
    goal = e.get("attacker_goal")
    chips = [
        f'<span class="rg-chip"><i>Model</i>{_esc(e["model"])}</span>',
        f'<span class="rg-chip ok"><i>Correct answer</i>{_esc(e["gold_answer"])}</span>',
    ]
    if goal:
        chips.append(f'<span class="rg-chip bad"><i>Attacker wants</i>{_esc(goal)}</span>')
    question = (
        f'<div class="rg-section">The question</div><div class="rg-card">'
        f'<div class="rg-q">{_esc(e["question"])}</div><div class="rg-meta">{"".join(chips)}</div></div>'
    )
    if e["planted"]:
        docs = "".join(
            f'<div class="rg-card"><div class="rg-doc-head"><b>{_esc(c["title"])}</b>'
            f'<span class="rg-chip"><i>Source tier</i>{_esc(c["tier"])}</span></div>'
            f'<div class="rg-doc">{_highlight(c["text"], goal)}</div></div>'
            for c in e["planted"]
        )
        planted = f'<div class="rg-section">What the attacker planted · {_esc(e["attack_name"])}</div>{docs}'
    else:
        planted = (
            '<div class="rg-section">What the attacker planted</div>'
            '<div class="rg-card"><span class="rg-note">Nothing: this is a clean question. It '
            "shows what the defenses cost on normal traffic.</span></div>"
        )
    answers = (
        '<div class="rg-section">The answers</div><div class="rg-vs">'
        f"{render_answer('Baseline RAG · no defenses', e['baseline'], goal, decisions=False)}"
        f"{render_answer('RAGantine · full defense stack', e['defended'], goal, decisions=True)}</div>"
    )
    return f'<div class="rg">{question}{planted}{answers}</div>'


def render_answer(title: str, side: dict[str, Any], goal: str | None, decisions: bool) -> str:
    text, cls = VERDICTS.get(side["verdict"], (side["verdict"], "warn"))
    body = (
        f'<div class="rg-card rg-ans {cls}"><div class="rg-ans-head"><span>{_esc(title)}</span>'
        f'<span class="rg-badge {cls}">{_esc(text)}</span></div>'
        f'<div class="rg-text">{_highlight(side["text"], goal, cites=True)}</div>'
        f'<div class="rg-judge">Judge: {_esc(side["reason"])}</div>'
    )
    if decisions:
        rows = "".join(
            f'<div class="rg-dec-row"><span class="rg-dec-name">{_esc(d["defense"])}</span>'
            f'<span class="rg-dec-act">{_esc(d["action"])} at {_esc(d["stage"])}</span>'
            f'<span class="rg-dec-why">{_esc(d["reason"])}</span></div>'
            for d in side.get("decisions") or []
        )
        body += f'<div class="rg-dec"><div class="rg-section" style="margin:0 0 4px">Defense decisions</div>{rows or "<span class=rg-note>none</span>"}</div>'
    return body + "</div>"


def show(choice: str, examples: list[dict[str, Any]]) -> str:
    e = next((x for x in examples if label(x) == choice), examples[0])
    return render_example(e)


FOOTER = (
    f'<div class="rg rg-footer">Wikipedia corpus; poison is harmless by design (wrong answers and '
    f"canary codes only). Code, method and full results: "
    f'<a href="{REPO_URL}" target="_blank">{REPO_URL}</a></div>'
)


def theme() -> Any:
    import gradio as gr

    fixed: dict[str, str] = {}
    for key, value in {
        "body_background_fill": BG, "body_text_color": TEXT, "body_text_color_subdued": MUTED,
        "background_fill_primary": BG, "background_fill_secondary": PANEL,
        "block_background_fill": "transparent", "block_border_color": "transparent",
        "block_label_text_color": MUTED, "block_title_text_color": MUTED, "border_color_primary": LINE,
        "checkbox_label_background_fill": PANEL, "checkbox_label_background_fill_hover": PANEL_2,
        "checkbox_label_background_fill_selected": ACCENT, "checkbox_label_border_color": LINE,
        "checkbox_label_border_color_selected": ACCENT, "checkbox_label_text_color": TEXT,
        "checkbox_label_text_color_selected": BG,
    }.items():  # fmt: skip
        fixed[key] = value
        fixed[f"{key}_dark"] = value
    return gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
        **fixed
    )


def build_ui(data: dict[str, Any]) -> Any:
    import gradio as gr

    examples = data["examples"]
    labels = [label(e) for e in examples]
    with gr.Blocks(title="RAGantine: RAG poisoning demo") as ui:
        gr.HTML(render_header(data.get("headline", {})))
        pick = gr.Radio(
            choices=labels, value=labels[0], show_label=False, elem_classes=["rg-picker"]
        )
        view = gr.HTML()
        gr.HTML(FOOTER)

        def on_pick(choice: str) -> str:
            return show(choice, examples)

        pick.change(on_pick, inputs=pick, outputs=view)
        ui.load(on_pick, inputs=pick, outputs=view)
    return ui


def main() -> None:
    build_ui(load_data()).launch(css=CSS, theme=theme())


if __name__ == "__main__":
    main()
