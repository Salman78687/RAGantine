"""D1 ingestion sanitizer (SPEC §B6).

Re-parses the raw document and keeps only what a human reader would see:

* HTML/Markdown: drop comments, ``<script>``, ``<style>``, ``<meta>``, attribute text
  (alt, title, aria-*), elements styled ``display:none``, ``visibility:hidden``,
  ``font-size:0``, ``opacity:0`` or text colour equal to background colour, and
  elements with the ``hidden`` attribute or ``aria-hidden=true``.
* PDF: drop spans smaller than 4 pt or within 5% of white.
* All formats: remove zero-width and bidi control characters, then Unicode NFKC.

``hidden = naive_text - sanitized_text``. If at least ``flag_min_hidden_chars``
non-whitespace characters were hidden, the whole document is quarantined (reason
``hidden_content``) so a reviewer sees what was hidden; otherwise ingestion continues
with the sanitized text.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment, Tag

from ragantine.defenses.base import Defense, register
from ragantine.ingest.loaders import as_text, html_to_text, markdown_to_html, normalize_whitespace
from ragantine.schemas import DefenseDecision, RawDocument

INVISIBLE_CHARS = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_STYLE_RULE = re.compile(r"\s*([\w-]+)\s*:\s*([^;]+)")
PDF_MIN_FONT_PT = 4.0
PDF_WHITE_TOLERANCE = 0.05  # "within 5% of white"
EXCERPT_CHARS = 120
DROP_TAGS = ("script", "style", "meta", "noscript", "template", "link")


def _style(tag: Tag) -> dict[str, str]:
    return {k.lower(): v.strip().lower() for k, v in _STYLE_RULE.findall(str(tag.get("style", "")))}


def _is_zero(value: str) -> bool:
    return re.fullmatch(r"0+(\.0+)?(px|pt|em|rem|%)?", value.replace(" ", "")) is not None


def _norm_color(value: str) -> str:
    v = value.replace(" ", "").lower()
    named = {"white": "#ffffff", "#fff": "#ffffff", "black": "#000000", "#000": "#000000",
             "rgb(255,255,255)": "#ffffff", "rgb(0,0,0)": "#000000"}  # fmt: skip
    return named.get(v, v)


def _hidden_by_style(tag: Tag) -> bool:
    s = _style(tag)
    if s.get("display") == "none" or s.get("visibility") in ("hidden", "collapse"):
        return True
    if "font-size" in s and _is_zero(s["font-size"]):
        return True
    if "opacity" in s and _is_zero(s["opacity"]):
        return True
    bg = s.get("background-color") or s.get("background")
    return bool("color" in s and bg and _norm_color(s["color"]) == _norm_color(bg))


def sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for t in soup.find_all(DROP_TAGS):
        t.decompose()
    for t in list(soup.find_all(True)):
        if not isinstance(t, Tag) or t.decomposed:
            continue
        if (
            t.has_attr("hidden")
            or str(t.get("aria-hidden", "")).lower() == "true"
            or _hidden_by_style(t)
        ):
            t.decompose()
    return html_to_text(str(soup), keep_hidden=False)


def _is_whiteish(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    floor = 255 * (1 - PDF_WHITE_TOLERANCE)
    return r >= floor and g >= floor and b >= floor


def sanitize_pdf(data: bytes) -> str:
    import pymupdf

    pages: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            lines: list[str] = []
            info: dict[str, Any] = page.get_text("dict")
            for block in info.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [
                        s["text"]
                        for s in line.get("spans", [])
                        if s.get("size", 0) >= PDF_MIN_FONT_PT
                        and not _is_whiteish(int(s.get("color", 0)))
                    ]
                    if "".join(spans).strip():
                        lines.append("".join(spans))
                lines.append("")
            pages.append("\n".join(lines))
    return normalize_whitespace("\n\n".join(pages))


def strip_invisible(text: str) -> str:
    return unicodedata.normalize("NFKC", INVISIBLE_CHARS.sub("", text))


def hidden_text(naive: str, sanitized: str) -> str:
    """Characters present in the naive text but missing from the sanitized text."""
    a, b = naive, sanitized
    if a == b:
        return ""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return "".join(a[i1:i2] for op, i1, i2, _, _ in sm.get_opcodes() if op in ("delete", "replace"))


@register
class Sanitizer(Defense):
    id = "d1"

    def visible_text(self, doc: RawDocument) -> str:
        if doc.fmt == "html":
            text = sanitize_html(as_text(doc.raw))
        elif doc.fmt == "markdown":
            text = sanitize_html(markdown_to_html(as_text(doc.raw)))
        elif doc.fmt == "pdf":
            if not isinstance(doc.raw, bytes):
                raise TypeError(f"pdf document {doc.doc_id} must carry bytes")
            text = sanitize_pdf(doc.raw)
        else:
            text = normalize_whitespace(as_text(doc.raw))
        return normalize_whitespace(strip_invisible(text))

    def on_load(self, doc: RawDocument, text: str) -> tuple[str, list[DefenseDecision]]:
        sanitized = self.visible_text(doc)
        # NFKC on both sides so ordinary Unicode (km², ligatures) never counts as hidden;
        # zero-width characters stay in the naive side and do count.
        naive = normalize_whitespace(unicodedata.normalize("NFKC", text))
        hidden = hidden_text(naive, sanitized)
        visible_chars = re.sub(r"\s+", "", hidden)
        if len(visible_chars) >= int(self.cfg["flag_min_hidden_chars"]):
            excerpt = hidden.strip()[:EXCERPT_CHARS].encode("unicode_escape").decode("ascii")
            return text, [
                self.decision(
                    stage="ingestion",
                    target_id=doc.doc_id,
                    action="quarantine",
                    score=float(len(visible_chars)),
                    reason=f"hidden_content: {len(visible_chars)} hidden chars, e.g. '{excerpt}'",
                )
            ]
        return sanitized, []
