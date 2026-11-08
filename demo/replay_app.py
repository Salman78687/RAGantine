"""RAGantine replay demo: recorded v0.1 benchmark answers, side by side.

Each example shows a question, the document an attacker planted in the knowledge base, the
answer of an undefended RAG pipeline (D0) and the answer of RAGantine's full defense stack,
with every defense decision and its reason. Answers are replayed from finished benchmark runs
(demo/replay/examples.json, built by demo/build_replay.py), so the demo needs no API keys,
no GPU and no models, and it cannot run out of a free quota. The headline numbers are pooled
from the runs' metrics.json at build time; nothing here is typed by hand.

Design: Hallmark v2 (github.com/nutlope/hallmark, MIT), theme Console, applied by hand.

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
KIND_TITLES = {"stopped": "stopped", "still fooled": "still fooled", "cost": "the cost"}
VERDICTS = {
    "correct": ("Correct", "ok"),
    "poisoned": ("Poisoned", "bad"),
    "refused": ("Refused", "warn"),
    "other": ("Unclear", "warn"),
}
CITATION = re.compile(r"[\[【][^\]】]*?[0-9a-f]{4,}[^\]】]*?[\]】]")

# Console tokens (Hallmark theme "console", graphite colourway). The Gradio theme below reuses
# them so Gradio's own light and dark modes both render this one palette.
TOKENS = {
    "paper": "oklch(17% 0.008 250)", "paper-2": "oklch(21% 0.009 250)", "paper-3": "oklch(13.5% 0.007 250)",
    "ink": "oklch(95% 0.005 250)", "ink-2": "oklch(83% 0.008 250)", "muted": "oklch(71% 0.01 250)",
    "rule": "oklch(31% 0.01 250)", "rule-2": "oklch(42% 0.012 250)",
    "accent": "oklch(82% 0.15 80)", "accent-ink": "oklch(18% 0.04 75)", "accent-soft": "oklch(29% 0.05 80)",
    "cool": "oklch(77% 0.12 245)", "cool-soft": "oklch(29% 0.05 245)",
    "ok": "oklch(76% 0.15 150)", "ok-soft": "oklch(24.5% 0.03 150)",
    "bad": "oklch(70% 0.17 25)", "bad-soft": "oklch(24.5% 0.035 25)",
}  # fmt: skip

CSS = (
    "/* Hallmark · shape: Stage hero then one record · genre: modern-minimal · theme: console (deal: manual) · job: try\n"
    " * hero: stage, headline left, readings panel right · nav: none (single screen tool) · footer: one line\n"
    " * imagery: the product (recorded answers) · signature: the struck undefended reading above the defended one · console: graphite\n"
    " * axes: dark paper, grotesk, amber person, cool instrument · contrast: pass · fold: pass · v2.0.0 */\n"
    "/* Hallmark · pre-emit critique: P4 H4 E4 S5 R4 V4 · weakest: P, a replay cannot argue; led with the struck reading so the claim is read first */\n"
    "@import url('https://fonts.googleapis.com/css2?family=Host+Grotesk:wght@400;500&family=Atkinson+Hyperlegible+Mono:wght@400;500&display=swap');\n"
    ":root {\n"
    + "".join(f"  --color-{k}: {v};\n" for k, v in TOKENS.items())
    + """  --color-focus: var(--color-accent);
  --font-display: "Host Grotesk", ui-sans-serif, system-ui, sans-serif; --font-body: var(--font-display);
  --font-outlier: "Atkinson Hyperlegible Mono", ui-monospace, monospace;
  --radius-card: 10px; --radius-pill: 6px; --radius-chip: 4px; --rule-hair: 1px;
  --tracking-display: -0.03em; --dur-short: 140ms; --ease-out: cubic-bezier(.2,.7,.2,1);
  --text-xs: 12px; --text-sm: 14px; --text-md: 16px; --text-lg: 20px; --text-h2: 34px; --text-q: 28px;
  --text-reading: 44px; --text-display: clamp(40px, 4.6vw, 58px);
  --space-xs: 6px; --space-sm: 10px; --space-md: 16px; --space-lg: 24px; --space-xl: 48px; --space-2xl: 88px;
  --measure: 1180px;
}
html, body { overflow-x: clip; }
body, .gradio-container { background: var(--color-paper) !important; color: var(--color-ink) !important;
  font-family: var(--font-body) !important; }
.gradio-container { max-width: var(--measure) !important; margin: 0 auto !important; padding: 0 var(--space-lg) !important; }
footer { display: none !important; }
.gradio-container .app, .gradio-container .html-container { padding-left: 0 !important; padding-right: 0 !important; }
::selection { background: color-mix(in oklab, var(--color-accent) 25%, transparent); }
.rg, .rg * { box-sizing: border-box; }
.rg { color: var(--color-ink); font-family: var(--font-body); font-size: var(--text-md); line-height: 1.55; }
.rg h1, .rg h2 { font-family: var(--font-display); font-weight: 500; letter-spacing: var(--tracking-display);
  line-height: 1.04; color: var(--color-ink); margin: 0; overflow-wrap: break-word; min-width: 0; }
.rg a { color: var(--color-ink); text-decoration: underline; text-decoration-color: var(--color-rule-2);
  text-underline-offset: 3px; transition: color var(--dur-short) var(--ease-out); }
.rg a:hover { color: var(--color-accent); }
.rg a:focus-visible, .rg-pick label:focus-within { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.rg-measured { font-family: var(--font-outlier); font-variant-numeric: tabular-nums slashed-zero; }

.rg-hero { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: var(--space-xl);
  align-items: center; padding: var(--space-2xl) 0 var(--space-xl); }
.rg-hero h1 { font-size: var(--text-display); }
.rg-lede { color: var(--color-ink-2); font-size: var(--text-lg); margin: var(--space-md) 0 var(--space-lg); max-width: 30ch; }
.rg-actions { display: flex; gap: var(--space-sm); flex-wrap: wrap; }
.rg .rg-btn { display: inline-block; border: var(--rule-hair) solid var(--color-rule-2); border-radius: var(--radius-pill);
  padding: var(--space-sm) var(--space-md); background: var(--color-paper-2); color: var(--color-ink);
  font-weight: 500; text-decoration: none; transition: background-color var(--dur-short) var(--ease-out); }
.rg .rg-btn:hover { background: var(--color-rule); color: var(--color-ink); }
.rg .rg-btn--primary { background: var(--color-accent); border-color: var(--color-accent); color: var(--color-accent-ink); }
.rg .rg-btn--primary:hover { background: var(--color-accent); color: var(--color-accent-ink); }
.rg .rg-btn:focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }

.rg-panel { display: grid; gap: var(--rule-hair); background: var(--color-rule); border: var(--rule-hair) solid var(--color-rule);
  border-radius: var(--radius-card); overflow: hidden; }
.rg-panel > * { background: var(--color-paper-2); padding: var(--space-md) var(--space-lg); min-width: 0; }
.rg-panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: var(--space-md); flex-wrap: wrap; }
.rg-panel-head h2 { font-size: var(--text-lg); letter-spacing: -0.01em; }
.rg-note { color: var(--color-muted); font-size: var(--text-sm); }
.rg-label { color: var(--color-muted); font-size: var(--text-sm); font-weight: 500; }

.rg-reading-row { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 1fr); gap: var(--space-md); align-items: end; }
.rg-reading { font-family: var(--font-outlier); font-variant-numeric: tabular-nums slashed-zero; font-size: var(--text-reading);
  letter-spacing: -0.04em; line-height: 1; color: var(--color-ink); }
.rg-struck { font-family: var(--font-outlier); font-variant-numeric: tabular-nums slashed-zero; font-size: var(--text-md);
  color: var(--color-bad); text-decoration: line-through; text-decoration-thickness: 1px; }
.rg-reading-who { font-size: var(--text-sm); color: var(--color-muted); margin-top: var(--space-xs); }
.rg-cool { color: var(--color-cool); }
.rg-table { width: 100%; border-collapse: collapse; font-size: var(--text-sm); margin-top: var(--space-md); }
.rg .rg-table, .rg .rg-table tr, .rg .rg-table th, .rg .rg-table td { border: none !important; background: transparent !important; }
.rg .rg-table th { text-align: left; font-weight: 500; color: var(--color-muted); padding: var(--space-xs) 0; }
.rg .rg-table td { padding: var(--space-xs) 0; border-top: var(--rule-hair) solid var(--color-rule) !important; color: var(--color-ink-2); }
.rg .rg-table td.rg-measured { color: var(--color-cool); }

.rg-section-head { padding: var(--space-xl) 0 var(--space-md); }
.rg-section-head h2 { font-size: var(--text-h2); }
.rg-section-head p { color: var(--color-ink-2); margin: var(--space-sm) 0 0; max-width: 62ch; }
.rg-pick, .rg-pick > *, .rg-pick fieldset, .form:has(.rg-pick) { background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; }
.rg-pick .wrap { gap: var(--space-xs) !important; }
.rg-pick label { border: var(--rule-hair) solid var(--color-rule) !important; border-radius: var(--radius-chip) !important;
  background: var(--color-paper-2) !important; color: var(--color-ink-2) !important; font-size: var(--text-sm) !important;
  font-weight: 500 !important; padding: var(--space-xs) var(--space-sm) !important; box-shadow: none !important;
  transition: background-color var(--dur-short) var(--ease-out); }
.rg-pick label:hover { background: var(--color-rule) !important; }
.rg-pick label.selected, .rg-pick label:has(input:checked) { background: var(--color-cool-soft) !important;
  border-color: var(--color-cool) !important; color: var(--color-ink) !important; }
.rg-pick input { position: absolute !important; opacity: 0 !important; width: 1px !important; height: 1px !important; }

.rg-record { margin-top: var(--space-md); }
.rg .rg-q { font-size: var(--text-q); }
.rg-facts { display: flex; flex-wrap: wrap; gap: var(--space-lg); margin-top: var(--space-sm); font-size: var(--text-sm); }
.rg-facts span { color: var(--color-muted); }
.rg-facts b { font-weight: 500; color: var(--color-ink); }
.rg-facts b.rg-want { color: var(--color-bad); }
.rg-facts b.rg-gold { color: var(--color-ok); }
.rg-panel > .rg-doc { background: var(--color-paper-3); }
.rg-doc pre { font-family: var(--font-outlier); font-size: var(--text-sm); line-height: 1.7; color: var(--color-ink-2);
  white-space: pre-wrap; margin: var(--space-sm) 0 0; max-height: 260px; overflow: auto; }
mark.rg-hit { background: var(--color-bad-soft); color: var(--color-bad); border-radius: 3px; padding: 0 3px; }
.rg-panel > .rg-diff { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--rule-hair);
  background: var(--color-rule); padding: 0; }
.rg-side { background: var(--color-paper-2); padding: var(--space-md) var(--space-lg); min-width: 0; }
.rg-side.bad { background: var(--color-bad-soft); } .rg-side.ok { background: var(--color-ok-soft); }
.rg-side.warn { background: var(--color-accent-soft); }
.rg-side-head { display: flex; justify-content: space-between; gap: var(--space-sm); margin-bottom: var(--space-sm); }
.rg-verdict { font-size: var(--text-sm); font-weight: 500; border: var(--rule-hair) solid currentColor;
  border-radius: var(--radius-chip); padding: 0 var(--space-xs); }
.rg-verdict.ok { color: var(--color-ok); } .rg-verdict.bad { color: var(--color-bad); } .rg-verdict.warn { color: var(--color-accent); }
.rg-answer { white-space: pre-wrap; color: var(--color-ink); }
.rg-cite { font-family: var(--font-outlier); font-size: var(--text-xs); color: var(--color-muted); }
.rg-judge { margin-top: var(--space-sm); }
.rg-trail-row { display: grid; grid-template-columns: minmax(0, 200px) minmax(0, 170px) minmax(0, 1fr); gap: var(--space-md);
  padding: var(--space-sm) 0 var(--space-sm) var(--space-md); border-left: var(--rule-hair) solid var(--color-rule-2);
  font-size: var(--text-sm); }
.rg-trail-row .rg-measured { color: var(--color-ink-2); overflow-wrap: anywhere; }
.rg-footer { padding: var(--space-xl) 0 var(--space-2xl); color: var(--color-muted); font-size: var(--text-sm); }
@media (max-width: 860px) {
  .gradio-container { padding: 0 var(--space-md) !important; }
  .rg-hero { grid-template-columns: minmax(0, 1fr); padding-top: var(--space-xl); }
  .rg-panel > * { padding: var(--space-md); }
  .rg .rg-q { font-size: var(--text-lg); }
  .rg-section-head h2 { font-size: 28px; }
  .rg-panel > .rg-diff { grid-template-columns: minmax(0, 1fr); }
  .rg-trail-row { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .rg-reading-row { grid-template-columns: minmax(0, 1fr); gap: var(--space-xs); }
  .rg-reading { font-size: 40px; }
}
@media (prefers-reduced-motion: reduce) { .rg *, .rg-pick label { transition: none !important; } }
"""
)


def load_data(path: Path = EXAMPLES_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def label(e: dict[str, Any]) -> str:
    attack = f"{e['attack']} {e['attack_name'].lower()}" if e.get("attack") else "Clean question"
    return f"{attack}, {KIND_TITLES.get(e['kind'], e['kind'])}"


def _esc(text: Any) -> str:
    return html.escape(str(text or ""))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _highlight(text: str, needle: str | None, cites: bool = False) -> str:
    """Escape ``text``, mark the attacker's goal string and (optionally) quiet the citations."""
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
    """Stage hero: headline and actions left, the readings panel right."""
    panels = []
    for model, rows in headline.items():
        by = {r["config"]: r for r in rows}
        base, d4 = by.get("No defenses"), by.get("D4 spotlighting")
        if not base or not d4:
            continue
        table = "".join(
            f'<tr><td>{_esc(r["config"])}</td><td class="rg-measured">{_pct(r["asr"])}</td>'
            f'<td class="rg-measured">{_pct(r["clean"])}</td></tr>'
            for r in rows
        )
        panels.append(
            f'<div><div class="rg-reading-row"><div><div class="rg-label">{_esc(model)}</div>'
            f'<div class="rg-reading-who">poisoned answers with D4 spotlighting alone</div></div>'
            f'<div><div class="rg-struck">{_pct(base["asr"])}</div><div class="rg-reading">{_pct(d4["asr"])}</div></div>'
            f'<div class="rg-reading-who">{d4["n"]} attacked questions, <span class="rg-cool">3 seeds</span></div></div>'
            f'<table class="rg-table"><tr><th>Defenses</th><th>Poisoned</th><th>Clean correct</th></tr>'
            f"{table}</table></div>"
        )
    readings = (
        '<div class="rg-panel"><div class="rg-panel-head"><h2>Benchmark v0.1</h2>'
        '<span class="rg-note">struck: with no defenses</span></div>'
        f"{''.join(panels)}</div>"
    )
    return (
        '<div class="rg rg-hero"><div><h1>One planted page can change the answer.</h1>'
        '<p class="rg-lede">RAGantine measures how often, and which defenses stop it.</p>'
        f'<div class="rg-actions"><a class="rg-btn rg-btn--primary" href="{REPO_URL}" target="_blank" '
        'rel="noopener">Read the code</a><a class="rg-btn" href="#examples">See the examples</a></div></div>'
        f"{readings}</div>"
        '<div class="rg rg-section-head" id="examples"><h2>Read one record</h2>'
        "<p>Answers recorded during the benchmark, not live calls. Two of them show the limits: an "
        "attack that still gets through, and a normal question the defenses refuse.</p></div>"
    )


def render_answer(title: str, side: dict[str, Any], goal: str | None) -> str:
    text, cls = VERDICTS.get(side["verdict"], (side["verdict"], "warn"))
    return (
        f'<div class="rg-side {cls}"><div class="rg-side-head"><span class="rg-label">{_esc(title)}</span>'
        f'<span class="rg-verdict {cls}">{_esc(text)}</span></div>'
        f'<div class="rg-answer">{_highlight(side["text"], goal, cites=True)}</div>'
        f'<div class="rg-note rg-judge">Judge: {_esc(side["reason"])}</div></div>'
    )


def render_example(e: dict[str, Any]) -> str:
    goal = e.get("attacker_goal")
    want = (
        f'<div><span>Attacker wants</span> <b class="rg-want">{_esc(goal)}</b></div>'
        if goal
        else ""
    )
    facts = (
        f'<div class="rg-facts"><div><span>Model</span> <b>{_esc(e["model"])}</b></div>'
        f'<div><span>Correct answer</span> <b class="rg-gold">{_esc(e["gold_answer"])}</b></div>{want}</div>'
    )
    rows = [f'<div><h2 class="rg-q">{_esc(e["question"])}</h2>{facts}</div>']
    for c in e["planted"]:
        rows.append(
            f'<div class="rg-doc"><div class="rg-panel-head"><span class="rg-label">Planted in the '
            f'knowledge base: {_esc(c["title"])}</span><span class="rg-note">{_esc(e["attack_name"])}, '
            f"source tier {_esc(c['tier'])}</span></div><pre>{_highlight(c['text'], goal)}</pre></div>"
        )
    if not e["planted"]:
        rows.append(
            '<div class="rg-doc"><span class="rg-note">Nothing planted: a clean question, to show '
            "what the defenses cost on normal traffic.</span></div>"
        )
    rows.append(
        '<div class="rg-diff">'
        f"{render_answer('Without defenses', e['baseline'], goal)}"
        f"{render_answer('With RAGantine', e['defended'], goal)}</div>"
    )
    trail = "".join(
        f'<div class="rg-trail-row"><span class="rg-cool">{_esc(d["defense"])}</span>'
        f"<span>{_esc(d['action'])} at {_esc(d['stage'])}</span>"
        f'<span class="rg-measured">{_esc(d["reason"])}</span></div>'
        for d in e["defended"].get("decisions") or []
    )
    rows.append(
        '<div><div class="rg-label" style="margin-bottom:8px">What each defense did</div>'
        f"<div>{trail or '<span class=rg-note>No defense acted.</span>'}</div></div>"
    )
    return f'<div class="rg"><div class="rg-panel rg-record">{"".join(rows)}</div></div>'


def show(choice: str, examples: list[dict[str, Any]]) -> str:
    e = next((x for x in examples if label(x) == choice), examples[0])
    return render_example(e)


FOOTER = (
    '<div class="rg rg-footer">Wikipedia corpus. The poison is harmless by design: wrong answers '
    f'and canary codes only. Method, code and every result: <a href="{REPO_URL}" target="_blank" '
    f'rel="noopener">{REPO_URL.removeprefix("https://")}</a></div>'
)


def theme() -> Any:
    import gradio as gr

    t = TOKENS
    fixed: dict[str, str] = {}
    for key, value in {
        "body_background_fill": t["paper"], "body_text_color": t["ink"], "body_text_color_subdued": t["muted"],
        "background_fill_primary": t["paper"], "background_fill_secondary": t["paper-2"],
        "block_background_fill": "transparent", "block_border_color": "transparent",
        "block_label_text_color": t["muted"], "block_title_text_color": t["muted"], "border_color_primary": t["rule"],
        "checkbox_label_background_fill": t["paper-2"], "checkbox_label_background_fill_hover": t["rule"],
        "checkbox_label_background_fill_selected": t["cool-soft"], "checkbox_label_border_color": t["rule"],
        "checkbox_label_border_color_selected": t["cool"], "checkbox_label_text_color": t["ink-2"],
        "checkbox_label_text_color_selected": t["ink"],
    }.items():  # fmt: skip
        fixed[key] = value
        fixed[f"{key}_dark"] = value
    font = [
        gr.themes.GoogleFont("Host Grotesk"),
        gr.themes.Font("system-ui"),
        gr.themes.Font("sans-serif"),
    ]
    return gr.themes.Base(font=font).set(**fixed)


def build_ui(data: dict[str, Any]) -> Any:
    import gradio as gr

    examples = data["examples"]
    labels = [label(e) for e in examples]
    with gr.Blocks(title="RAGantine: knowledge-base poisoning, measured") as ui:
        gr.HTML(render_header(data.get("headline", {})))
        pick = gr.Radio(choices=labels, value=labels[0], show_label=False, elem_classes=["rg-pick"])
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
