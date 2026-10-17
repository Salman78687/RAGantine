"""RAGantine demo (SPEC §B7.3): the same question through Baseline (D0) and RAGantine (all
defenses), side by side and streaming.

Runs on the in-memory snapshot in demo/snapshot/ (build it with demo/build_snapshot.py),
so it works on Hugging Face Spaces without OpenSearch. Needs GROQ_API_KEY / OPENAI_API_KEY
(Space secrets) for the generator models.

    uv run python demo/app.py
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import gradio as gr

from ragantine.config import Settings, apply_overrides, load_config, load_models_config
from ragantine.defenses.base import FULL_STACK
from ragantine.generate.llm import LLMClient
from ragantine.graph import Pipeline, stream_query
from ragantine.retrieve.dense import SentenceTransformerEmbedder
from ragantine.retrieve.memory import MemoryIndex
from ragantine.retrieve.reranker import CrossEncoderReranker
from ragantine.schemas import QuarantineRecord

HERE = Path(__file__).parent
SNAPSHOT = Path(
    os.environ.get(
        "RAGANTINE_DEMO_SNAPSHOT",
        HERE / "snapshot" if (HERE / "snapshot").exists() else HERE / "demo" / "snapshot",
    )
)
RESULTS = Path(os.environ.get("RAGANTINE_RESULTS", HERE.parent / "results"))
TIER_COLORS = {
    "curated": "#1f6f3f",
    "partner": "#2a5db0",
    "web": "#9a6200",
    "user_upload": "#a8322d",
}
CSS = """
.chunk { border: 1px solid #d0d0d0; border-radius: 6px; padding: 6px 8px; margin: 6px 0; font-size: 0.86em; }
.chunk.blocked { opacity: 0.75; }
.chunk.blocked .snippet { text-decoration: line-through; }
.tier { color: #fff; border-radius: 4px; padding: 1px 6px; font-size: 0.78em; margin-right: 6px; }
.why { color: #a8322d; font-weight: 600; }
.badge { display: inline-block; border-radius: 4px; padding: 2px 8px; margin: 4px 4px 0 0; font-size: 0.85em;
         background: #ececec; color: #222; }
"""


def esc(text: str) -> str:
    return html.escape(text, quote=True)


class Demo:
    def __init__(
        self,
        baseline: Pipeline,
        defended: Pipeline,
        presets: list[dict[str, str]],
        meta: dict[str, Any],
    ) -> None:
        self.baseline, self.defended = baseline, defended
        self.models = baseline.llm.models
        self.presets, self.meta = presets, meta

    @classmethod
    def from_snapshot(cls, snapshot: Path = SNAPSHOT) -> Demo:
        meta = json.loads((snapshot / "snapshot.json").read_text())
        cfg = apply_overrides(load_config(), [f"defenses.d2.variant={meta['d2_variant']}"])
        llm = LLMClient(load_models_config(), Settings(), cache_dir=cfg.paths.llm_cache_dir)
        embedder = SentenceTransformerEmbedder(cfg.embedding)
        reranker = CrossEncoderReranker(cfg.retrieval.rerank_model)
        return cls(
            Pipeline.from_parts(
                cfg, llm, MemoryIndex.load(snapshot / "baseline"), embedder, reranker
            ),
            Pipeline.from_parts(
                cfg, llm, MemoryIndex.load(snapshot / "defended"), embedder, reranker
            ),
            json.loads((snapshot / "presets.json").read_text()),
            meta,
        )

    # ------------------------------------------------------------------ rendering

    def quarantined(self, chunk_id: str) -> QuarantineRecord | None:
        return self.defended.backend.get_quarantine(chunk_id)

    def chunk_html(
        self, chunk_id: str, title: str, tier: str, text: str, why: str | None = None
    ) -> str:
        snippet = esc(text.split("\n\n", 1)[-1][:220]) + ("…" if len(text) > 220 else "")
        reason = f'<div class="why">{esc(why)}</div>' if why else ""
        return (
            f'<div class="chunk{" blocked" if why else ""}"><span class="tier" style="background:'
            f'{TIER_COLORS.get(tier, "#555")}">{esc(tier)}</span><b>{esc(title)}</b> '
            f'<code>{esc(chunk_id)}</code>{reason}<div class="snippet">{snippet}</div></div>'
        )

    def render(self, answer: dict[str, Any] | None, defended: bool, baseline_ids: list[str]) -> str:
        if answer is None:
            return ""
        parts = []
        badges = []
        if defended:
            d5 = [d for d in answer["decisions"] if d["defense_id"] == "d5"]
            if any(d["action"] == "allow" for d in d5):
                badges.append("✔ verified by D5")
            if answer.get("draft") and answer["draft"] != answer["text"]:
                badges.append("✎ rewritten by D5")
        parts.append("".join(f'<span class="badge">{esc(b)}</span>' for b in badges))
        parts.append("<h4>Chunks given to the model</h4>")
        for h in answer["retrieved"]:
            c = h["chunk"]
            parts.append(self.chunk_html(c["chunk_id"], c["title"], c["source_tier"], c["text"]))
        if defended:
            blocked = []
            for d in answer["decisions"]:
                if d["action"] in ("filter", "downweight") and d["target_id"] != "answer":
                    c = self.defended.backend.get(d["target_id"])
                    if c is not None:
                        blocked.append(self.chunk_html(c.chunk_id, c.title, c.source_tier.value, c.text,
                                                       f"{d['defense_id']} {d['action']}: {d['reason']}"))  # fmt: skip
            for cid in baseline_ids:  # what the baseline saw but never reached this index
                rec = self.quarantined(cid)
                if rec is not None:
                    why = "; ".join(
                        f"{d.defense_id} quarantined at ingestion: {d.reason}"
                        for d in rec.decisions
                    )
                    blocked.append(
                        self.chunk_html(
                            cid, rec.chunk.title, rec.chunk.source_tier.value, rec.chunk.text, why
                        )
                    )
            if blocked:
                parts.append("<h4>Blocked by RAGantine</h4>" + "".join(blocked))
        return "\n".join(parts)

    # ------------------------------------------------------------------ querying

    async def ask(
        self, preset: str | None, free_text: str, model: str
    ) -> AsyncIterator[tuple[str, str, str, str]]:
        question = (free_text or "").strip() or next(
            (p["question"] for p in self.presets if p["label"] == preset), ""
        )
        if not question:
            yield "Pick a preset or type a question.", "", "", ""
            return
        queue: asyncio.Queue[tuple[str, str, dict[str, Any]]] = asyncio.Queue()

        async def pump(side: str, pipeline: Pipeline, defenses: list[str]) -> None:
            async for event, data in stream_query(question, pipeline, defenses, model):
                await queue.put((side, event, data))
            await queue.put((side, "end", {}))

        tasks = [asyncio.create_task(pump("base", self.baseline, [])),
                 asyncio.create_task(pump("def", self.defended, list(FULL_STACK)))]  # fmt: skip
        text = {"base": "", "def": ""}
        final: dict[str, dict[str, Any] | None] = {"base": None, "def": None}
        open_sides = 2
        while open_sides:
            side, event, data = await queue.get()
            if event == "token":
                text[side] += data["text"]
            elif event == "done":
                final[side] = data
                text[side] = data["text"]
            elif event == "error":
                text[side] = f"⚠ {data['message']}"
            elif event == "end":
                open_sides -= 1
            base_ids = [h["chunk"]["chunk_id"] for h in (final["base"] or {}).get("retrieved", [])]
            yield (
                text["base"],
                self.render(final["base"], False, []),
                text["def"],
                self.render(final["def"], True, base_ids),
            )
        await asyncio.gather(*tasks)


def results_tab() -> tuple[str, str | None]:
    reports = sorted(RESULTS.glob("report-*/report.md"))
    if not reports:
        return (
            "No results yet. Run the ablation, then `uv run ragantine bench report --runs latest`.",
            None,
        )
    latest = reports[-1]
    charts = sorted(latest.parent.glob("asr_by_attack_*.png"))
    return latest.read_text(), str(charts[0]) if charts else None


def build_ui(demo: Demo) -> gr.Blocks:
    with gr.Blocks(title="RAGantine") as ui:
        gr.Markdown(
            "# RAGantine — quarantine for retrieval\n"
            "Plant poison in a RAG corpus, see whether it wins, then quarantine it. "
            f"Snapshot: {demo.meta['articles']} Wikipedia articles + {demo.meta['poison_docs']} poisoned documents "
            f"(benchmark test split, seed {demo.meta['seed']}; D2 {demo.meta['d2_variant']})."
        )
        with gr.Tab("Ask"):
            with gr.Row():
                preset = gr.Dropdown(
                    [p["label"] for p in demo.presets],
                    label="Poisoned test question (2 per attack)",
                )
                model = gr.Dropdown(
                    sorted(demo.models.models), value=demo.models.defaults.generator, label="Model"
                )
            free = gr.Textbox(
                label="…or ask your own question", placeholder="Overrides the preset when filled"
            )
            go = gr.Button("Ask both pipelines", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Baseline (D0)")
                    base_answer = gr.Markdown()
                    base_chunks = gr.HTML()
                with gr.Column():
                    gr.Markdown("### RAGantine (all defenses)")
                    def_answer = gr.Markdown()
                    def_chunks = gr.HTML()
            go.click(
                demo.ask, [preset, free, model], [base_answer, base_chunks, def_answer, def_chunks]
            )
        with gr.Tab("Results"):
            text, chart = results_tab()
            gr.Markdown(text)
            if chart:
                gr.Image(chart, label="ASR by attack type and configuration", show_label=True)
    return ui


if __name__ == "__main__":
    if not (SNAPSHOT / "snapshot.json").exists():
        raise SystemExit("demo snapshot missing: run `uv run python demo/build_snapshot.py` first")
    build_ui(Demo.from_snapshot()).queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
