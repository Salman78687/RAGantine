"""Render poison documents into text / markdown / html / pdf (SPEC §B5.3).

A5 hides the instruction payload with one technique per sample, rotated:
HTML comment, ``display:none`` span, ``alt`` attribute, zero-width-separated text, or
white 1-pt text in a PDF. The naive baseline loaders keep all of these; D1 strips them.
"""

from __future__ import annotations

import html
from typing import Literal

HidingTechnique = Literal[
    "html_comment", "display_none", "alt_attribute", "zero_width", "pdf_white"
]
HIDING_TECHNIQUES: tuple[HidingTechnique, ...] = (
    "html_comment",
    "display_none",
    "alt_attribute",
    "zero_width",
    "pdf_white",
)
ZERO_WIDTH_SPACE = "\u200b"


def paragraphs(body: str) -> list[str]:
    paras = [p.strip() for p in body.replace("\r\n", "\n").split("\n\n") if p.strip()]
    return paras or [body.strip()]


def insert_middle(body: str, payload: str) -> str:
    """Embed the payload as its own paragraph after the first paragraph (or at the end)."""
    paras = paragraphs(body)
    at = 1 if len(paras) > 1 else len(paras)
    return "\n\n".join([*paras[:at], payload, *paras[at:]])


def zero_width_separate(text: str) -> str:
    return ZERO_WIDTH_SPACE.join(text)


def render_html(
    title: str, body: str, payload: str | None, technique: HidingTechnique | None
) -> str:
    paras = [f"<p>{html.escape(p)}</p>" for p in paragraphs(body)]
    hidden = ""
    if payload is not None:
        esc = html.escape(payload)
        if technique == "html_comment":
            hidden = f"<!-- {payload.replace('--', '-')} -->"
        elif technique == "display_none":
            hidden = f'<span style="display:none">{esc}</span>'
        elif technique == "alt_attribute":
            hidden = f'<img src="figure.png" alt="{esc}">'
        elif technique == "zero_width":
            hidden = f"<p>{html.escape(zero_width_separate(payload))}</p>"
        else:
            raise ValueError(f"technique {technique!r} is not an HTML technique")
    at = 1 if len(paras) > 1 else len(paras)
    content = "\n".join([*paras[:at], hidden, *paras[at:]]) if hidden else "\n".join(paras)
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head>\n<body>\n<h1>{html.escape(title)}</h1>\n"
        f"{content}\n</body></html>\n"
    )


PDF_MARGIN = 72
PDF_FONT_SIZE = 11
PDF_HIDDEN_FONT_SIZE = 1


def render_pdf(title: str, body: str, hidden_payload: str | None = None) -> bytes:
    """A4 PDF with the visible body; ``hidden_payload`` is drawn as white 1-pt text."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    r = page.rect
    rect = pymupdf.Rect(r.x0 + PDF_MARGIN, r.y0 + PDF_MARGIN, r.x1 - PDF_MARGIN, r.y1 - PDF_MARGIN)
    page.insert_textbox(
        rect, f"{title}\n\n" + "\n\n".join(paragraphs(body)), fontsize=PDF_FONT_SIZE
    )
    if hidden_payload:
        bottom = pymupdf.Rect(rect.x0, rect.y1 - 20, rect.x1, rect.y1)
        page.insert_textbox(bottom, hidden_payload, fontsize=PDF_HIDDEN_FONT_SIZE, color=(1, 1, 1))
    data: bytes = doc.tobytes()
    doc.close()
    return data
